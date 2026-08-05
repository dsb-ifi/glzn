from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import string
from collections.abc import Mapping as ABCMapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torchvision
import xxhash
from packaging.version import parse as parse_version
from torch.utils.data import Dataset

from .encoders import DEFAULT_DECODERS, PIL, PseudoExtension
from .imgbrowser import browse_dataset
from .itar.fold import iTarFold, iTarRetriever
from .itar.utils import StemHelper, stripext
from .maptrafo import Map, MapAll, MapGrouped, MapTuple
from .sampler import IdentitySampler, MultiFeistelSampler

USE_TV_TENSOR = parse_version(torchvision.__version__) >= parse_version('0.16')
_valid_pseudo_extensions = PseudoExtension._valid_pseudo_extensions
Transform = Callable[..., Any]
Decoder = Callable[[bytes], Any]


# TODO: Possibly Pseudoepoch: A selected artificial number of samples that can be anything, 
# including infinity for a fully stochastic stream-based sampling procedure.


class _ComposeTransforms:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms: tuple[Transform, ...] = tuple(transforms)

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


def _compose(fs: Sequence[Transform]) -> _ComposeTransforms:
    return _ComposeTransforms(fs)


def _finalize_sample(result:Any) -> tuple[Any, ...]:
    '''Normalize transform-pipeline output to a sample field tuple.

    Public contract: the pipeline must yield a list or tuple of logical
    output fields. Strings, mappings, and bare scalars are rejected so they
    are never silently expanded by ``tuple(...)``.
    '''
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    if isinstance(result, (str, bytes, bytearray)) or isinstance(result, ABCMapping):
        raise TypeError(
            'Transform pipeline must return a list or tuple of sample fields; '
            f'got {type(result).__name__}.'
        )
    raise TypeError(
        'Transform pipeline must return a list or tuple of sample fields; '
        f'got {type(result).__name__}. Wrap a single field as a one-element '
        'tuple or list, e.g. ``(value,)``.'
    )


def _parse_decoders(
    fold:iTarFold, overrides:dict[str,Decoder]|None=None
) -> dict[int,Decoder]:
    _dec: dict[str, Decoder] = {**DEFAULT_DECODERS}
    if overrides is None:
        overrides = {}
    _dec.update(overrides)
    return {
        v:_dec[k.split(".")[-1]]
        for k,v in fold.state.ext2id.items()
    }


class _BrowserWrapper:

    def __init__(
        self, dataset:iTarDataset, img_ext:str='jpg', lab_ext:str='cls',
        labeldict:Mapping[Any,Any]|None=None
    ):
        self.dataset = dataset
        self.img_ext = stripext(img_ext).lower()
        self.lab_ext = stripext(lab_ext).lower()
        self.labeldict = labeldict

        supported = PIL.decoder().supported_extensions
        ext_source = supported.keys() if isinstance(supported, dict) else (supported or [])
        valid_pil_ext = {stripext(ext).lower() for ext in ext_source}
        if self.img_ext not in valid_pil_ext:
            curext = ', '.join(sorted(valid_pil_ext))
            raise ValueError(
                f'Invalid image extension {self.img_ext}. '
                f'Valid PIL extensions are: {curext}.'
            )

        try:
            self._imgindex = dataset.extensions.index(self.img_ext)
            self._labindex = dataset.extensions.index(self.lab_ext) if self.lab_ext in dataset.extensions else None
            self._stemindex = (
                dataset.extensions.index('_stem') if '_stem' in dataset.extensions else
                (dataset.extensions.index('_name') if '_name' in dataset.extensions else None)
            )
            self._idxindex = dataset.extensions.index('_idx') if '_idx' in dataset.extensions else None
            self._fidindex = dataset.extensions.index('_fid') if '_fid' in dataset.extensions else None
        except ValueError:
            curext = ', '.join(dataset.extensions)
            raise ValueError(
                f'No current extensions {self.img_ext}. '
                f'Valid current extensions are: {curext}.'
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]
        label = None
        if self._labindex is not None:
            label = sample[self._labindex]
            if self.labeldict is not None:
                label = self.labeldict.get(label, label)

        meta = {'idx': i}
        if self._stemindex is not None:
            meta['stem'] = sample[self._stemindex]
        if self._idxindex is not None:
            meta['idx'] = int(sample[self._idxindex])
        if self._fidindex is not None:
            meta['fid'] = int(sample[self._fidindex])

        return sample[self._imgindex], label, meta


class iTarDataset(Dataset[Any]):

    def __init__(
        self,
        dataset:str,
        loc:str|PathLike[Any],
        fold:str,
        extensions:Sequence[str]|None=None,
        parse_if_missing:bool=False,
        serialize:bool=True,
        idxext:str='taridx',
        prefer_mmap:bool=False,
        internal_seed:int=0,
        buckets_per_shard:int=2,
        enforce_contiguous:bool=False,
        **kwargs
    ):
        kw = {'verbose':True, 'enforce_contiguous':enforce_contiguous}
        kw.update(kwargs)

        # Init root path
        if not isinstance(loc, Path):
            loc = Path(loc)
        self.loc = loc
        self.dataset = dataset
        self.root = loc / dataset
        if not self.root.exists():
            raise FileExistsError(f'Folder {self.root} does not exist!')

        # Init fold and retriever
        self.fold = iTarFold(
            self.root, fold, '_', 'tar', parse_if_missing,
            serialize, idxext, None, **kw
        )
        self.retriever = iTarRetriever(
            self.fold.full_paths, prefer_mmap
        )

        # Init valid extensions
        self._val_ext = list(self.fold.state.ext2id.keys())
        if extensions is None:
            extensions = self._val_ext

        self.extensions = [stripext(e) for e in extensions]
        self.fold.filter_extensions(self.extensions)
        self._sync_extension_state()

        # Init transforms
        self.transforms = []
        self._trafo = _compose(self.transforms)
        self._logical_sample_output = False

        # Init sampling details
        self._epoch = 0
        self._shared_epoch = mp.Value('q', 0)
        self._seed = internal_seed
        self._grouping_seed = internal_seed ^ 0xBAD5EED
        self.buckets_per_shard = max(1, buckets_per_shard)
        self._distributed_rank = 0
        self._distributed_world_size = 1
        self._distributed_drop_last = False
        self._shuffle_enabled = False
        self._shuffle_shard_mixing = False
        self._shuffle_rounds = 3
        self._bucket_size = 1
        self._sampler: MultiFeistelSampler | IdentitySampler
        self._update_fold_state_vars()

        # Grouping is an explicit opt-in mode initialized via add_grouping().
        self.grouping:tuple[str,...]|None = None
        self.grouping_replace = False
        self._group_slots = 0
        self._group_prefixes:tuple[str,...] = tuple()
        self._grouping_active = False

    def __len__(self):
        self._sync_epoch_from_shared()
        return self._distributed_len(self._global_len())

    def __getitem__(self, idx:int) -> Any:
        self._sync_epoch_from_shared()
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f'iTarDataset index out of range, {idx} not in [0, {len(self)}).'
            )
        idx = self._sampler[self._local_to_global_index(idx)]

        if self._grouping_active:
            return self._getitem_grouped(idx)
        return self._getitem_standard(idx)

    def _getitem_standard(self, idx:int) -> Any:
        n = self._nrealext
        idx_start = idx * n
        out = {}
        fid = None
        stem = None

        for row in self.fold.state.arr[idx_start:idx_start+n]:
            extid = row['extid']
            ext = self._extmap[extid]
            fid = row['fid'] if fid is None else fid
            stem = self.retriever.hdrname(row) if stem is None else stem
            out[ext] = self.decoders[extid](bytes(self.retriever.from_row(row)))

        for ext in self._pseu_ext:
            match ext:
                case '_name' | '_stem':
                    out[ext] = stem
                case '_idx':
                    out[ext] = idx
                case '_fid':
                    out[ext] = int(fid) if fid is not None else None

        return self._finalize_transform_output(
            self._trafo([out[e] for e in self.extensions])
        )

    def _getitem_grouped(self, idx:int) -> Any:
        n = self._nrealext
        idx_start = idx * n
        out = {}
        extensions = self._sample_ext_groups(idx)
        extensions_set = set(extensions)

        for row in self.fold.state.arr[idx_start:idx_start+n]:
            extid = row['extid']
            ext = self._extmap[extid]
            if ext not in extensions_set:
                continue
            out[ext] = self.decoders[extid](bytes(self.retriever.from_row(row)))

        return self._finalize_transform_output(
            self._trafo([out[e] for e in extensions])
        )

    @staticmethod
    def supports_tv_tensor() -> bool:
        '''Checks if the dataset supports tv tensors.
        '''
        return USE_TV_TENSOR

    def _add_trafo(self, trafo:Transform) -> 'iTarDataset':
        self.transforms.append(trafo)
        self._trafo = _compose(self.transforms)
        return self

    def _add_field_trafo(self, trafo:Transform, caller:str) -> 'iTarDataset':
        if self._logical_sample_output:
            raise RuntimeError(
                f'{caller} cannot be attached after map(), because map() '
                'produces one logical sample object rather than field-aligned '
                'sample outputs.'
            )
        return self._add_trafo(trafo)

    def _finalize_transform_output(self, result:Any) -> Any:
        if self._logical_sample_output:
            return result
        return _finalize_sample(result)
    
    def _sync_extension_state(self):
        """Resync all extension-derived metadata from self.extensions and fold state."""
        self._real_ext = {e for e in self.extensions if e not in _valid_pseudo_extensions}
        self._pseu_ext = {e for e in self.extensions if e in _valid_pseudo_extensions}
        self._nrealext = len(self._real_ext)
        if self._nrealext == 0:
            raise ValueError('At least one real extension must be provided!')
        self._extmap = {v: k for k, v in self.fold.state.ext2id.items()}
        self.decoders = _parse_decoders(self.fold)
        self._validate_contiguous_row_groups('_sync_extension_state()')

    def _refresh_bucketsize(self):
        computed_size = round(sum(self.shard_bincount) / (len(self.shard_bincount) * self.buckets_per_shard))
        self._bucket_size = max(1, int(computed_size))

    def _update_fold_state_vars(self):
        bincount = self._sample_bincount()
        if bincount.sum() <= 0:
            raise ValueError(
                'Fold state has no samples! ' 
                'This is either due to erroneous filtering or an empty dataset.'
        )
        self.shard_bincount = bincount
        self._refresh_bucketsize()
        self._rebuild_sampler()

    def _assert_grouping_inactive(self, caller:str):
        if self._grouping_active:
            raise RuntimeError(
                f'{caller} is unavailable after add_grouping() activation. '
                f'Configure all filters before enabling grouping.'
            )

    def _assert_schema_mutable(self, caller:str):
        if self.transforms:
            raise RuntimeError(
                f'{caller} cannot change dataset output schema after transforms '
                'have been attached.'
            )

    def _expected_output_arity(self) -> int:
        if self._grouping_active and self.grouping is not None:
            return len(self.grouping)
        return len(self.extensions)

    def _validate_transform_indices(self, indices:Sequence[int], caller:str) -> tuple[int,...]:
        clean = tuple(indices)
        arity = self._expected_output_arity()
        if len(clean) == 0:
            raise ValueError(f'{caller} requires at least one index.')
        for i in clean:
            if i < 0 or i >= arity:
                raise IndexError(
                    f'{caller} index {i} is out of range for output arity {arity}.'
                )
        return clean

    def _global_len(self) -> int:
        return len(self.fold.state.arr) // self._nrealext

    def _distributed_len(self, global_len:int) -> int:
        world_size = self._distributed_world_size
        rank = self._distributed_rank
        if self._distributed_drop_last:
            return global_len // world_size
        if rank >= global_len:
            return 0
        return (global_len - rank + world_size - 1) // world_size

    def _local_to_global_index(self, idx:int) -> int:
        global_idx = self._distributed_rank + idx * self._distributed_world_size
        if self._distributed_drop_last:
            usable = (self._global_len() // self._distributed_world_size) * self._distributed_world_size
            if global_idx >= usable:
                raise IndexError(f'Distributed index {idx} maps outside drop_last range.')
        return global_idx

    def _sample_bincount(self) -> np.ndarray:
        arr = self.fold.state.arr
        if len(arr) % self._nrealext != 0:
            raise ValueError(
                f'Fold row count {len(arr)} is not divisible by '
                f'{self._nrealext} real extensions.'
            )
        sample_fids = arr[::self._nrealext]['fid']
        return np.bincount(sample_fids)

    def _sampler_sizes(self, shard_mixing:bool|None=None) -> list[int]:
        shard_mixing = self._shuffle_shard_mixing if shard_mixing is None else shard_mixing
        Ns:list[int] = self.shard_bincount.tolist()
        if shard_mixing:
            if not self.fold.state.is_contiguous:
                raise ValueError(
                    'Shard mixing requires contiguous fold state! '
                    'Reinitialize dataset with `enforce_contiguous=True`.'
                )
            N = self._bucket_size
            num_Ns = self._global_len() // N
            last_N = self._global_len() % N
            Ns = [N] * num_Ns + ([last_N] if last_N > 0 else [])
        return Ns

    def _rebuild_sampler(self):
        if self._shuffle_enabled:
            self._sampler = MultiFeistelSampler(
                self._sampler_sizes(),
                self._shuffle_rounds,
                self._seed + self._epoch,
            )
        else:
            self._sampler = IdentitySampler(self._global_len())

    def _sync_epoch_from_shared(self):
        epoch_value = int(self._shared_epoch.value)
        if epoch_value != self._epoch:
            self._epoch = epoch_value
            self._rebuild_sampler()

    def _set_epoch_local_and_shared(self, epoch:int):
        if epoch < 0:
            raise ValueError(f'epoch must be non-negative, got {epoch}.')
        self._epoch = int(epoch)
        self._shared_epoch.value = int(epoch)
        self._rebuild_sampler()

    def _stem_label(self, row) -> str:
        crashid = int(row['crashid'])
        if crashid != 0:
            for stem, cid in self.fold.state.crashstem.items():
                if cid == crashid:
                    return stem
        return self.fold.state.hashinfo.get(int(row['keyhash']), str(int(row['keyhash'])))

    def _validate_contiguous_row_groups(self, caller:str) -> None:
        arr = self.fold.state.arr
        n = self._nrealext
        if len(arr) % n != 0:
            raise ValueError(
                f'{caller}: fold row count {len(arr)} is not divisible by '
                f'{n} requested real extensions.'
            )
        expected_ids = {
            self.fold.state.ext2id[e]
            for e in self._real_ext
            if e in self.fold.state.ext2id
        }
        if len(expected_ids) != n:
            raise ValueError(
                f'{caller}: expected {n} unique real extensions, got '
                f'{len(expected_ids)} from {sorted(self._real_ext)}.'
            )
        expected_ext = sorted(self._real_ext)
        for start in range(0, len(arr), n):
            rows = arr[start:start+n]
            keys = {(int(r['keyhash']), int(r['crashid'])) for r in rows}
            if len(keys) != 1:
                stems = [self._stem_label(r) for r in rows]
                raise ValueError(
                    f'{caller}: row group starting at row {start} mixes stems '
                    f'{stems}; expected one contiguous stem block.'
                )
            observed_ids = [int(r['extid']) for r in rows]
            if set(observed_ids) != expected_ids or len(observed_ids) != len(set(observed_ids)):
                observed = [
                    self._extmap.get(int(extid), f'<unknown:{int(extid)}>')
                    for extid in observed_ids
                ]
                raise ValueError(
                    f'{caller}: row group starting at row {start} for stem '
                    f'{self._stem_label(rows[0])} has extensions {observed}; '
                    f'expected exactly once each: {expected_ext}.'
                )

    def _infer_grouping(self, seq_str:tuple[str,...]|None) -> int:
        if seq_str is None:
            return 0
        fmt_str = ','.join(seq_str)
        formatter = string.Formatter()
        fields = [
            field_name for _, field_name, _, _ 
            in formatter.parse(fmt_str) 
            if field_name is not None
        ]
        if not all(f.isdigit() for f in fields):
            raise ValueError(
                f'Invalid grouping format string: {fmt_str}. '
                f'All fields must be digit-only strings.'
            )

        unique_indices = {int(f) for f in fields if f.isdigit()}
        num_slots = len(unique_indices) if unique_indices else 0

        if not all(r == i for r, i in enumerate(sorted(unique_indices))):
            raise ValueError(
                f'Grouping format string fields must be a contiguous '
                f'sequence of integers starting from 0. '
                f'Got indices: {sorted(unique_indices)}.'
            )

        return num_slots

    def _check_grouping_validity(
        self,
        grouping:tuple[str,...],
        group_slots:int,
        grouping_replace:bool,
    ) -> tuple[str,...]:
        if len(self._pseu_ext) > 0:
            raise ValueError(
                'Grouping mode does not support pseudoextensions. '
                f'Remove pseudoextensions first: {sorted(self._pseu_ext)}.'
            )

        if len(grouping) == 0:
            raise ValueError('Grouping cannot be empty.')
        if group_slots <= 0:
            raise ValueError(
                'Grouping must include at least one numeric format field '
                '(e.g. "{0}.jpg").'
            )

        if not all(len(e.split('.')) == 2 for e in self._real_ext):
            raise ValueError(
                'Grouping requires real extensions to have "prefix.suffix" format, '
                'for example "a.jpg" or "b.cls".'
            )

        formatter = string.Formatter()
        prefix_to_suffixes:dict[str, set[str]] = {}
        for ext in self._real_ext:
            prefix, suffix = ext.split('.')
            prefix_to_suffixes.setdefault(prefix, set()).add(suffix)

        slot_suffixes:dict[int, set[str]] = {i: set() for i in range(group_slots)}
        for spec in grouping:
            parts = spec.split('.')
            if len(parts) != 2:
                raise ValueError(
                    f'Invalid grouping spec {spec}. Expected exactly one dot, e.g. "{{0}}.jpg".'
                )
            suffix = parts[1]
            fields = [
                field_name for _, field_name, _, _
                in formatter.parse(spec)
                if field_name is not None
            ]
            if len(fields) == 0:
                raise ValueError(
                    f'Invalid grouping spec {spec}. '
                    'Each grouping element must reference at least one slot.'
                )
            for field in fields:
                slot_suffixes[int(field)].add(suffix)

        slot_candidates:dict[int, tuple[str,...]] = {}
        for slot, req_suffixes in slot_suffixes.items():
            valid = tuple(
                sorted(
                    p for p, suffixes in prefix_to_suffixes.items()
                    if req_suffixes.issubset(suffixes)
                )
            )
            if len(valid) == 0:
                raise ValueError(
                    f'Grouping slot {{{slot}}} is unsatisfiable. '
                    f'No prefix supports required suffixes {sorted(req_suffixes)}.'
                )
            slot_candidates[slot] = valid

        unique_suffix_sets = {
            tuple(sorted(req_suffixes))
            for req_suffixes in slot_suffixes.values()
        }
        if len(unique_suffix_sets) != 1:
            raise ValueError(
                'Grouping currently requires symmetric slot requirements. '
                'Each slot must require the same set of suffixes. '
                'For example: ("{0}.jpg", "{0}.npy", "{1}.jpg", "{1}.npy").'
            )

        prefixes = slot_candidates[0]
        if not grouping_replace:
            if group_slots > len(prefixes):
                raise ValueError(
                    f'Grouping needs {group_slots} unique prefixes but only '
                    f'{len(prefixes)} are available when grouping_replace=False.'
                )

        return prefixes

    def add_grouping(
        self,
        grouping:Sequence[str],
        grouping_replace:bool=False,
        grouping_seed:int|None=None,
    ) -> 'iTarDataset':
        '''Activate grouped extension sampling.

        Parameters
        ----------
        grouping : Sequence[str]
            Grouping format specification using positional fields, for example
            ``['{0}.jpg', '{0}.npy', '{1}.jpg', '{1}.npy']``.
            Fields must be a contiguous sequence starting at 0.
        grouping_replace : bool, optional
            Whether slot prefixes are sampled with replacement, by default False.
        grouping_seed : int | None, optional
            Optional seed override for grouped sampling. If None, keeps the
            existing internal grouping seed.

        Returns
        -------
        iTarDataset
            Dataset instance with grouping mode activated.

        Notes
        -----
        Grouping does not change dataset cardinality (``__len__``).
        Grouped sampling is deterministic per dataset state, epoch, and sampled
        index.
        '''
        self._assert_schema_mutable('add_grouping()')
        group = tuple(grouping)
        group_slots = self._infer_grouping(group)
        prefixes = self._check_grouping_validity(group, group_slots, grouping_replace)

        self.grouping = group
        self.grouping_replace = grouping_replace
        self._group_slots = group_slots
        self._group_prefixes = prefixes
        self._grouping_active = True
        self._grouping_seed = self._grouping_seed if grouping_seed is None else grouping_seed
        return self

    def _sample_ext_groups(self, idx:int) -> tuple[str,...]:
        '''Sample grouped extensions for a single dataset index.

        Sampling is deterministic with respect to grouping seed, epoch, and
        sampled index.
        '''
        if not self._grouping_active or self.grouping is None:
            return tuple(self.extensions)
        s = (self._grouping_seed + self._epoch + idx) & ((1 << 64) - 1)
        _sample = np.random.default_rng(s).choice(
            self._group_prefixes,
            self._group_slots, 
            replace=self.grouping_replace
        ).tolist()
        return tuple(map(lambda s: s.format(*_sample), self.grouping))
        
    def filter_extensions(self, extensions:Sequence[str]):
        '''Filter dataset to specified extensions.

        Parameters
        ----------
        extensions : Sequence[str]
            Iterable of extensions to filter by. Must be a subset of the current extensions.

        Returns
        -------
        iTarDataset
            Updated dataset with filtered extensions.
        '''
        self._assert_schema_mutable('filter_extensions()')
        self._assert_grouping_inactive('filter_extensions()')
        clean = [stripext(e) for e in extensions]
        self.fold.filter_extensions(clean)
        self.extensions = clean
        self._sync_extension_state()
        self._update_fold_state_vars()
        return self
        
    def filter_stems(self, stems:Sequence[str]) -> "iTarDataset":
        '''Filter dataset to specified stems.

        Parameters
        ----------
        stems : Sequence[str]
            Iterable of stem names to filter by. Must be a subset of the current stems.

        Returns
        -------
        iTarDataset
            Updated dataset with filtered stems.

        Notes
        -----
        Stem filtering is schema-neutral: it does not change output field count,
        order, or meaning. It remains allowed after transforms and after
        ``add_grouping()``. Extension filtering and adding grouping remain
        schema-changing and are blocked once transforms are attached (and
        extension filtering stays blocked after grouping).
        '''
        self.fold.filter_stems(stems)
        self._validate_contiguous_row_groups('filter_stems()')
        self._update_fold_state_vars()
        return self

    def filter_stems_by_json(self, path:str|PathLike[Any]) -> "iTarDataset":
        # TODO: Unfortunate naming; clashes with the export functionality of the browser.
        self._assert_grouping_inactive('filter_stems_by_json()')
        if not isinstance(path, Path):
            path = Path(path)

        def _cond(stems) -> list[str]|None:
            if (
                isinstance(stems, list) and
                all(isinstance(k, str) for k in stems)
            ):
                return [StemHelper(s).stem for s in stems]
            return None

        with open(path, 'r') as infile:
            stemlist = _cond(json.load(infile))

        if stemlist is None:
            raise ValueError(f'Invalid list of stems: {path}.')

        return self.filter_stems(stemlist)

    def lookup_stems(
        self,
        stems:Sequence[str]|str|PathLike[Any],
        extensions:Sequence[str]|None=None
    ) -> dict[str, dict[str, Any]]:
        """Retrieve decoded files by exact stem and extension.

        Parameters
        ----------
        stems : Sequence[str] | str | PathLike
            Either an iterable of stem names or a JSON file path containing a
            list of stem strings.
        extensions : Sequence[str], optional
            Iterable of required extensions to retrieve. If omitted, defaults
            to current real dataset extensions.

        Returns
        -------
        dict[str, dict[str, Any]]
            Nested dictionary where `out[stem][ext]` is the decoded object.
            Missing stem/extension pairs are omitted.
        """
        stem_src = stems
        if isinstance(stems, (str, PathLike, Path)):
            path = Path(stems)
            if path.exists() and path.is_file():
                try:
                    with open(path, 'r') as infile:
                        loaded = json.load(infile)
                except Exception as exc:
                    raise ValueError(f'Failed reading stems JSON file {path}: {exc}') from exc

                if not (isinstance(loaded, list) and all(isinstance(k, str) for k in loaded)):
                    raise ValueError(
                        f'Invalid stems JSON file {path}: expected list[str].'
                    )
                stem_src = loaded

        if not isinstance(stem_src, Sequence):
            raise TypeError(
                'Expected stems to be a sequence of strings or a JSON file path.'
            )
        if not all(isinstance(s, str) for s in stem_src):
            raise TypeError(
                'Expected all stems to be strings.'
            )

        stem_list = [StemHelper(s).stem for s in stem_src]
        if len(stem_list) == 0:
            return {}

        if extensions is None:
            extensions = [
                e for e in self.extensions
                if e not in _valid_pseudo_extensions and e in self.fold.state.ext2id
            ]

        ext_list = [stripext(e).lower() for e in extensions]
        if len(ext_list) == 0:
            return {s: {} for s in stem_list}

        ext2id = self.fold.state.ext2id
        unknown = [e for e in ext_list if e not in ext2id]
        if len(unknown) > 0:
            valid = ', '.join(sorted(ext2id.keys()))
            raise ValueError(
                f'Unknown extensions: {unknown}. '
                f'Valid extensions are: {valid}.'
            )

        wanted_stems = set(stem_list)
        arr = self.fold.state.arr
        h_vec = np.fromiter(
            (xxhash.xxh64(s.encode()).intdigest() for s in wanted_stems),
            dtype=arr['keyhash'].dtype
        )
        c_vec = np.fromiter(
            (self.fold.state.crashstem.get(s, 0) for s in wanted_stems),
            dtype=arr['crashid'].dtype
        )
        e_vec = np.fromiter(
            (ext2id[e] for e in ext_list),
            dtype=arr['extid'].dtype
        )

        hits = (
            np.isin(arr['extid'], e_vec) &
            np.isin(arr['keyhash'], h_vec) &
            np.isin(arr['crashid'], c_vec)
        )

        if not hits.any():
            return {s: {} for s in stem_list}

        rows = arr[hits]

        out:dict[str, dict[str, Any]] = {s: {} for s in stem_list}
        decoders = _parse_decoders(self.fold)
        id2ext = {v:k for k,v in ext2id.items()}

        for row in rows:
            stem_on_disk = self.retriever.hdrname(row)
            if stem_on_disk not in wanted_stems:
                continue

            ext = id2ext[int(row['extid'])]
            out[stem_on_disk][ext] = decoders[int(row['extid'])](
                bytes(self.retriever.from_row(row))
            )

        return out

    def map(self, mapping:Transform) -> "iTarDataset":
        '''Takes a mapping and applies it to the tuple of extensions.

        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the logical sample object produced by ``f(<sample>.jpg, <sample>.cls)``.

        ``map()`` is the boundary where archive-field tuple arity stops being
        semantically meaningful. The mapping may return any picklable Python
        object accepted by the downstream DataLoader/collate path. Use
        ``map_tuple()``, ``map_group()``, or ``map_all()`` for field-aligned
        transforms that preserve sample-field tuple semantics.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to the tuple of extensions.

        Returns
        -------
        iTarDataset
            Updated dataset with added transformations.
        '''
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")
        if self._logical_sample_output:
            raise RuntimeError(
                'map() cannot be attached after map(), because map() has '
                'already produced one logical sample object. A second map() '
                'is not currently supported.'
            )

        sig = inspect.signature(mapping)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        n_expected = self._expected_output_arity()
        n_actual = len(params)

        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
        if not has_varargs and n_actual != n_expected:
            raise TypeError(
                f"Mapping function for map() must accept {n_expected} positional arguments, "
                f"but got {n_actual}."
            )
        self._logical_sample_output = True
        return self._add_trafo(Map(mapping))

    def map_all(self, mapping:Transform) -> "iTarDataset":
        '''Takes a mapping and applies it to all extensions.

        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `(f(<sample>.jpg), f(<sample>.cls))`.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to all extensions.

        Returns
        -------
        iTarDataset
            Updated dataset with added transformations.
        '''
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")
        if self._logical_sample_output:
            raise RuntimeError(
                'map_all() cannot be attached after map(), because map() '
                'produces one logical sample object rather than field-aligned '
                'sample outputs.'
            )
        return self._add_field_trafo(MapAll(mapping), 'map_all()')

    def map_group(self, mapping:Transform, indices:Sequence[int]) -> "iTarDataset":
        '''Takes a mapping and applies it to specific indices of extensions.

        For `mapping = f`, `extensions = ['jpg', 'cls'], and `indices = (0,)` this
        will return the tuple `f(<sample>.jpg), <sample>.cls`.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to the tuple of extensions.
        indices : Sequence[int]
            Indices for which to apply the mapping.

        Returns
        -------
        iTarDataset
            Updated dataset with added transformations.
        '''
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")
        if self._logical_sample_output:
            raise RuntimeError(
                'map_group() cannot be attached after map(), because map() '
                'produces one logical sample object rather than field-aligned '
                'sample outputs.'
            )
        return self._add_field_trafo(
            MapGrouped(mapping, self._validate_transform_indices(indices, 'map_group()')),
            'map_group()',
        )

    def map_tuple(self, *maps:Transform) -> "iTarDataset":
        """Applies given mappings to individual extensions of dataset items.

        For `maps = [f1, f2]` and `extensions = ['jpg', 'cls']`, this will return
        the tuple `(f1(<sample>.jpg), f2(<sample>.cls))`.

        NOTE: Certain extensions such as pseudo extensions and class labels
              do not require explicit transforms in order for map_tuple to function
              correctly. In other words, map_tuple can handle the case where
              `maps = [f1]` and `extensions = ['jpg', 'cls']`. This avoids tedious
              constructions where class labels and potential pseudo extensions require
              explicit mappings for the transformations to parse correctly.

              Another example is useful for clarity:
              A case with `maps = [f1, f2]` and `extensions = ['jpg', 'seg16', '_name']`
              would work, since '_name' is a pseudoextension, and doesn't require an
              explicit mapping to infer intended behaviour.

        Parameters
        ----------
        maps : tuple[Callable, ...]
            Tuple of callables for mapping individual extensions.

        Returns
        -------
        iTarDataset
            Updated dataset with added transformations.
        """
        if not all(callable(m) for m in maps):
            raise TypeError("One or more mapping is not callable.")
        if self._logical_sample_output:
            raise RuntimeError(
                'map_tuple() cannot be attached after map(), because map() '
                'produces one logical sample object rather than field-aligned '
                'sample outputs.'
            )
        num_ext = self._expected_output_arity()
        if len(maps) != num_ext:
            raise ValueError(
                f"Incorrect number of transforms provided. "
                f"Expected {num_ext}, got {len(maps)}."
            )
        return self._add_field_trafo(MapTuple(maps), 'map_tuple()')

    def set_distributed(
        self,
        rank:int,
        world_size:int,
        drop_last:bool=False,
    ) -> "iTarDataset":
        """Configure fused rank partitioning after global permutation.

        The dataset first builds one global physical-storage-aware ordering,
        then rank ``r`` consumes positions ``r, r + world_size, ...`` from that
        ordering. This preserves the iTar-locality sampler while making rank
        partitioning explicit and deterministic.
        """
        if world_size < 1:
            raise ValueError(f'world_size must be positive, got {world_size}.')
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f'rank must be in [0, {world_size}), got {rank}.'
            )
        self._distributed_rank = int(rank)
        self._distributed_world_size = int(world_size)
        self._distributed_drop_last = bool(drop_last)
        return self

    def set_epoch(self, epoch:int) -> "iTarDataset":
        """Set epoch-boundary sampling state.

        This is an epoch-boundary resume primitive, not a mid-epoch cursor.
        Worker copies created by a DataLoader observe changes through shared
        epoch state on the next ``__getitem__`` or ``__len__`` call.

        Do not call ``set_epoch`` while a DataLoader iterator over this dataset
        is active: workers refresh on the next sample, so a single pass can
        mix two epochs. Finish the iterator (or drop it), then set the epoch
        before constructing the next iterator.
        """
        self._set_epoch_local_and_shared(epoch)
        return self

    def set_shuffle(
        self,
        enabled:bool=True,
        *,
        shard_mixing:bool=False,
        rounds:int=3,
    ) -> "iTarDataset":
        """Enable or disable the internal physical-storage-aware permutation."""
        if rounds < 1:
            raise ValueError(f'rounds must be positive, got {rounds}.')
        self._shuffle_enabled = bool(enabled)
        self._shuffle_shard_mixing = bool(shard_mixing)
        self._shuffle_rounds = int(rounds)
        self._rebuild_sampler()
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return portable epoch-boundary sampling state.

        Dataset checkpoint state is portable across ranks and process
        topologies. It includes deterministic logical sampling configuration
        only (epoch, seeds, shuffle, bucket sizing). Distributed topology
        (``rank``, ``world_size``, ``drop_last``) is runtime process state
        owned by ``set_distributed()`` and is not included.
        """
        return {
            'epoch': self._epoch,
            'seed': self._seed,
            'grouping_seed': self._grouping_seed,
            'shuffle_enabled': self._shuffle_enabled,
            'shuffle_shard_mixing': self._shuffle_shard_mixing,
            'shuffle_rounds': self._shuffle_rounds,
            'buckets_per_shard': self.buckets_per_shard,
        }

    def load_state_dict(self, state:Mapping[str, Any]) -> "iTarDataset":
        """Restore portable epoch-boundary sampling state.

        Restores logical sampling configuration (epoch, seeds, shuffle,
        buckets). Live runtime handles such as the shared epoch cell,
        retrievers, and memory maps are not serialized.

        Distributed topology is configured separately through
        ``set_distributed()`` and is never overwritten here. Legacy state
        dictionaries that still contain ``rank`` / ``world_size`` /
        ``drop_last`` keys are accepted but those keys are ignored.
        """
        self._seed = int(state.get('seed', self._seed))
        self._grouping_seed = int(state.get('grouping_seed', self._grouping_seed))
        self.buckets_per_shard = int(state.get('buckets_per_shard', self.buckets_per_shard))
        self._shuffle_enabled = bool(state.get('shuffle_enabled', self._shuffle_enabled))
        self._shuffle_shard_mixing = bool(state.get('shuffle_shard_mixing', self._shuffle_shard_mixing))
        self._shuffle_rounds = int(state.get('shuffle_rounds', self._shuffle_rounds))
        self._refresh_bucketsize()
        self._set_epoch_local_and_shared(int(state.get('epoch', self._epoch)))
        return self
    
    def browse(
        self, img_ext:str='jpg', lab_ext:str='cls', 
        labeldict:Mapping[Any,Any]|None=None, page_size:int=24, 
        cols:int=6, width:int=128
    ):
        '''Renders a notebook-based browser for the dataset.

        The browser is a convenient way to visually inspect the dataset and its samples. 
        It displays the images along with their corresponding labels (if available) 
        in a grid format, with support for pagination and selecting specific samples 
        for export.

        Parameters
        ----------
        img_ext : str, optional
            Extension to use for images, by default 'jpg'.
        lab_ext : str, optional
            Extension to use for labels, by default 'cls'.
        labeldict : Mapping[Any, Any], optional
            Optional mapping to convert label values, by default None.

        '''
        return browse_dataset(
            _BrowserWrapper(self, img_ext, lab_ext, labeldict),
            page_size=page_size,
            cols=cols,
            width=width,
        )
    
    @contextmanager
    def shufflecontext(
        self, seed:int|None=None, shard_mixing:bool=False, rounds:int=3
    ):
        seed = self._seed + self._epoch if seed is None else seed
        old = (
            self._shuffle_enabled,
            self._shuffle_shard_mixing,
            self._shuffle_rounds,
            self._sampler,
        )
        try:
            self._sampler = MultiFeistelSampler(self._sampler_sizes(shard_mixing), rounds, seed)
            yield
        finally:
            (
                self._shuffle_enabled,
                self._shuffle_shard_mixing,
                self._shuffle_rounds,
                self._sampler,
            ) = old
            self.set_epoch(self._epoch + 1)
