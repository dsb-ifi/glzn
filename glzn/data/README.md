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
4. Apply mapping transforms (`maptrafo.py`).
5. Iterate with sampler policy (`sampler.py`).

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
- sample keys must not contain `.`,
- tar member names must fit standard header name limits (<= 100 bytes).

## Conventions

- Keep public dataset behavior stable (`__len__`, `__getitem__` semantics).
- Prefer additive changes in serialization/index formats.
- Document format changes in `itar/README.md`.
- Avoid breaking extension/decoder mapping compatibility.

## Related docs

- iTar internals: `glzn/data/itar/README.md`