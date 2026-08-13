# `glzn.data`

Data loading, sampling, encoding, and transform utilities for GLZN.

Essentially, `glzn.data` follows the [WDS](https://github.com/webdataset/webdataset) format with an additional 
custom format for indexing. The goal is to provide faster loading, retrieval, and sampling from locally hosted data, 
maintaining high IO in sequential reads while allowing for fast random access when sampling.

## Module overview

- `dataset.py`  
  Dataset interfaces, including iTar-backed dataset support.
- `sampler.py`  
  Sampling strategies for train/eval iteration.
- `encoders.py`  
  Target/value encoders used in data pipelines.
- `maptrafo.py`  
  Mapping/transform helpers applied to dataset outputs.
- `itar/`  
  Indexed TAR (`.tar` + `.taridx`) backend for scalable sample retrieval.

## iTar backend

The `itar` package provides fast random access into tar shards via an index.

- `itar/entry.py`  
  Index structures and serialization (`iTarState`, row/header dtypes).
- `itar/fold.py`  
  Fold-level filtering and retrieval (`iTarFold`, `iTarRetriever`).
- `itar/utils.py`  
  Utility helpers used by parsing/retrieval.
- `itar/README.md`  
  Detailed data-structure documentation for iTar internals.

## Typical flow

1. Build/load dataset definitions (`dataset.py`).
2. Resolve source storage (filesystem or iTar).
3. Decode/encode fields (`encoders.py`).
4. Apply field-aligned mapping transforms (`maptrafo.py`).
5. Optionally collapse logical arity for packaging.
6. Iterate with sampler policy (`sampler.py`) / collate.

## Logical field model

`iTarDataset` is built around a **fixed logical field arity** from the ordered
`extensions` list. Repeated extension names are intentional and first-class.

```text
physical archive members
    ->
ordered logical fields from extensions
    ->
arity-preserving transform graph  (map / map_tuple / map_all / map_group)
    ->
optional explicit arity collapse  (collapse_arity)
    ->
collate
```

### Physical vs logical extensions

Physical decoder lookup uses the bare extension stem (`jpg`, `cls`, …).
Logical schema positions are `len(extensions)` long, including repeats.

```python
# One on-disk jpg member, three logical image slots, one label.
ds = iTarDataset(..., extensions=["jpg", "jpg", "jpg", "cls"])
sample = ds[0]
# sample is a 4-tuple; sample[0] is sample[1] is sample[2] before transforms
# (same decoded object referenced at each logical slot).
```

Repeated slots initially **alias** the same decoded object. Per-slot transforms
must be **functional / non-mutating**. Independent crops/augs on each slot are
expressed with `map_tuple` / `map_group`, not by cloning in the dataset.

### Stable logical field names

Unique extensions keep their bare name. Repeated extensions are indexed in
appearance order:

```text
extensions = ["jpg", "jpg", "jpg", "cls"]
logical names = ("jpg.0", "jpg.1", "jpg.2", "cls")
```

Access remains positional. Names are available as `ds.logical_field_names` and
are accepted by `collapse_arity` alongside indices.

**Grouping caveat:** `logical_field_names` always describe the configured
`extensions` schema. After `add_grouping()`, the effective output arity follows
the grouping template, which may differ. Name-based `collapse_arity` is
**rejected** in that mode; use positional indices only.

### Field-aligned transforms (arity-preserving)

| API | Meaning |
|-----|---------|
| `map(f)` | Joint map: `f(field_0, …, field_{N-1}) -> N fields` |
| `map_tuple(f0, f1, …)` | One callable per logical position |
| `map_all(f)` | Same unary callable on every field |
| `map_group(f, indices)` | Joint map on selected positions only |

Invariants:

- none of these change logical arity `N`;
- results must be list/tuple (namedtuple allowed) with `len(result) == N`;
- dicts/scalars are rejected (not a free-form sample escape hatch);
- the four operations compose in any order;
- arity is checked after every field map (fails immediately on drift);
- `map(f)` validates that `f` accepts `N` positional parameters (unless `*args`).

```python
from collections import namedtuple

Sample = namedtuple("Sample", ["image", "label"])

ds = iTarDataset(..., extensions=["jpg", "cls"])
ds.map(lambda image, label: Sample(image, int(label)))
ds.map_tuple(train_image_aug, lambda y: y)   # still field-aligned after map
```

### Explicit arity collapse

When a consumer needs a lower-arity nested structure, use
`collapse_arity(*groups)` — a **terminal structural** step, not a field map:

```python
# 2 global + 8 local SSL views as logical jpg slots, then pack.
n_local = 8
ds = iTarDataset(..., extensions=["jpg"] * (2 + n_local))
ds.map_tuple(global1, global2, *([local] * n_local))
ds.collapse_arity(
    [0, 1],                 # or ["jpg.0", "jpg.1"]
    list(range(2, 2 + n_local)),
)
# sample becomes: ((g0, g1), (l0, ..., l7))
```

Rules:

- groups form a partition of all logical fields (every field exactly once);
- duplicates, omissions, and invalid names/indices are rejected;
- after `collapse_arity`, further field maps are rejected;
- name-based groups are unavailable after `add_grouping()` (use indices);
- prefer collators when grouping is naturally a batch concern;
  `collapse_arity` is a convenience so a custom collator is not mandatory.

### SSL-style multi-view example

DINO-style independent views: one physical `jpg`, many logical slots, per-slot
functional transforms; collate packs fields for the training step:

```python
n_views = 2 + local_crops_number
ds = iTarDataset(..., extensions=[image_ext] * n_views)
ds.map_tuple(*view_transforms)  # length n_views
# collate: field tuple -> {"global_crops": [...], "local_crops": [...]}
```

Do **not** use `map(lambda img: {"global_crops": ...})` to abandon field arity.
That free-form packaging path is intentionally not part of the contract.

## Writing iTar datasets

Use `glzn.data.writer.iTarDatasetWriter` to create sharded tar files and binary
`<fold>.taridx` indices directly (no JSON index files).

Example:

```python
from glzn.data.writer import iTarDatasetWriter

with iTarDatasetWriter('my_dataset', '/data', folds=['train', 'val', 'test']) as w:
  w.fold('train').write('sample0001', {'jpg': img, 'cls': label})
  w.fold('val').write('sample1001', {'jpg': val_img, 'cls': val_label})
```

Constraints enforced by the writer:
- no compression (`.tar` only),
- flat names (no path separators),
- sample keys / stems must not contain `.`,
- tar member names must fit standard header name limits (<= 100 bytes).

## Conventions

- Keep public dataset behavior stable (`__len__`, `__getitem__` semantics).
- Prefer additive changes in serialization/index formats.
- Document format changes in `itar/README.md`.
- Avoid breaking extension/decoder mapping compatibility.

## Grouping mode (opt-in)

Grouping is a niche feature and is intentionally not part of standard dataset
initialization. Configure all filtering first, then activate grouping explicitly:

```python
ds = iTarDataset(
  'my_dataset', 
  '/data', 
  fold='train', 
  extensions=[
    'f0.jpg', 'f0.cls', 
    'f1.jpg', 'f1.cls',
    ...
    'f10.jpg', 'f10.cls',
  ]
)
ds = ds.filter_stems([
  'sample0001', 
  'sample0002',
  ...
])
ds = ds.add_grouping(
  ['{0}.jpg', '{0}.cls', '{1}.jpg', '{1}.cls'], 
  grouping_replace=False,
  grouping_seed=42 # For reproducibility
)
```

The example above then provides samples on the form:

```
('sampleXXXX.fY.jpg', 'sampleXXXX.fY.cls', 'sampleXXXX.fZ.jpg', 'sampleXXXX.fZ.cls')
```

where `X` is the sample and `Y`,`Z` are sampled extension prefixes based on the given grouping seed and indices.
This is useful for multipair datasets, such as video frame data, or multipose data such as 3DIEBench.

Rules:
- grouping is activated only via `add_grouping(...)`;
- pseudoextensions are not supported in grouping mode;
- schema-changing filtering (`filter_extensions`) is disallowed after grouping;
- transforms (`map`, `map_all`, `map_group`, `map_tuple`) work in ordinary and grouped modes;
- once transforms are attached, schema-changing operations such as `add_grouping` and `filter_extensions` are disallowed;
- grouping does not change dataset cardinality (`__len__`);
- grouping templates are positional (for example `{0}`, `{1}`) and must use contiguous slot indices from `0`;
- grouping currently requires symmetric slot requirements (all slots must require the same suffix set);
- grouped sampling is deterministic per dataset state, epoch, and sampled index;
- `grouping_seed` can be supplied to `add_grouping(..., grouping_seed=...)` to override internal grouped sampling seed.

See **Logical field model** above for repeated extensions, field maps, and
`collapse_arity`.

Sampling remains fused into `iTarDataset`: the internal physical-storage-aware
permutation defines a global logical order, and optional distributed rank
partitioning consumes that order by rank-striding:

```text
local index ℓ  →  global position g = rank + ℓ * world_size
               →  physical index p = sampler[g]
```

`set_epoch(epoch)` is the epoch-boundary resume primitive; exact mid-epoch
cursor resume is intentionally not part of the dataset API. Do not call
`set_epoch` while a DataLoader iterator is active (workers can mix epochs
within one pass).

Dataset checkpoint state (`state_dict` / `load_state_dict`) is portable
across ranks and process topologies. It restores epoch, seeds, and shuffle
configuration only. Distributed topology is configured separately through
`set_distributed()` and is not overwritten by `load_state_dict()` (legacy
topology keys in older state dicts are ignored).

Stem filtering remains allowed after grouping and after transforms because
it does not change output schema; `filter_extensions` and `add_grouping`
remain schema-changing.

Example of an invalid asymmetric template (fails fast):

```python
ds.add_grouping(['{0}.jpg', '{0}.npy', '{1}.jpg', '{1}.seg16'])
```

The template above is rejected because slot `0` requires `{jpg, npy}` while slot `1` requires `{jpg, seg16}`.

## Browsing samples

Use `BrowserWrapper` together with `browse_dataset` when inspecting iTar samples.
Labels are taken from the same iterable item as the image (no separate labels input).

Example:

```python
from glzn.data.dataset import iTarDataset, BrowserWrapper
from glzn.data.imgbrowser import browse_dataset

ds = iTarDataset('my_dataset', '/data', fold='train', extensions=['jpg', 'cls'])
view = BrowserWrapper(ds, img_ext='jpg', lab_ext='cls')
browse_dataset(view, page_size=24, cols=6, width=128)
```

Notes:
- `BrowserWrapper` validates `img_ext` against `encoders.PIL` supported extensions.
- Browser entries should be either `PIL.Image` or tuples where element `0` is image and element `1` is label.
- Selection workflow in browser:
  - Use `Select` toggles on cards to mark examples.
  - `Only Selected` limits the gallery view to selected items.
  - `Export Selection` creates a downloadable JSON list of selected stems.
  - `Import Selection` loads selections from a JSON list upload.

Selection JSON format:

```json
["sample0001", "sample0002", "sample0100"]
```

Imported entries with missing stems in the current view are ignored.
Legacy verbose object formats are not accepted by import.

## Lookup by stems

Use `iTarDataset.lookup_stems(stems, extensions=None)` to retrieve specific
files by exact stem names without modifying active dataset filters.

`stems` can be either:
- a list of stem strings, or
- a path to a JSON file containing a list of stem strings (for example, an
  export from the browser selection tool).

If `extensions` is omitted, the current real dataset extensions are used.

**Example:**

```python
from glzn.data.dataset import iTarDataset

ds = iTarDataset('my_dataset', '/data', fold='train', extensions=['jpg', 'cls'])
out = ds.lookup_stems(['sample0001', 'sample0002'], ['jpg', 'cls'])

# Path-based input from browser-exported stems JSON
out2 = ds.lookup_stems('selection.json', ['jpg', 'cls'])

# Uses current real dataset extensions by default
out3 = ds.lookup_stems('selection.json')

# out['sample0001']['jpg'] -> decoded PIL image
# out['sample0001']['cls'] -> decoded class label
```

The returned structure is `dict[stem][extension] = decoded_object`.
Missing pairs are omitted as empty/missing extension entries.

## Related docs

- iTar internals: `glzn/data/itar/README.md`
