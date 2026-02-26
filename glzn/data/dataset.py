from __future__ import annotations
import string, torchvision, pkg_resources, json, functools, inspect, xxhash, numpy as np
from contextlib import contextmanager
from torch.utils.data import Dataset
from pathlib import Path
from os import PathLike
from typing import Any, Callable, Sequence, Mapping

from .encoders import DEFAULT_DECODERS, PseudoExtension, PIL
from .maptrafo import MapAll, MapGrouped, MapTuple, DefaultIdentity
from .sampler import IdentitySampler, MultiFeistelSampler
from .imgbrowser import browse_dataset
from .itar.fold import iTarFold, iTarRetriever
from .itar.utils import StemHelper, stripext

USE_TV_TENSOR = pkg_resources.parse_version(torchvision.__version__) >= pkg_resources.parse_version('0.16')
_valid_pseudo_extensions = PseudoExtension._valid_pseudo_extensions


# TODO: Possibly Pseudoepoch: A selected artificial number of samples that can be anything, 
# including infinity for a fully stochastic stream-based sampling procedure.

def _compose(fs):
    def c(f,g):
        def c2(x):
            return g(f(x))
        return c2
    return functools.reduce(c, fs, DefaultIdentity())


def _parse_decoders(
    fold:iTarFold, overrides:dict[str,Callable]|None=None
) -> dict[int,Callable]:
    _dec = {**DEFAULT_DECODERS}
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
        except:
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


class iTarDataset(Dataset):

    def __init__(
        self,
        dataset:str,
        loc:str|PathLike,
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

        # Init sampling details
        self._epoch = 0
        self._seed = internal_seed
        self._grouping_seed = internal_seed ^ 0xBAD5EED
        self.buckets_per_shard = max(1, buckets_per_shard)
        self._update_fold_state_vars()

        # Grouping is an explicit opt-in mode initialized via add_grouping().
        self.grouping:tuple[str,...]|None = None
        self.grouping_replace = False
        self._group_slots = 0
        self._group_prefixes:tuple[str,...] = tuple()
        self._grouping_active = False

    def __len__(self):
        return len(self.fold.state.arr) // self._nrealext

    def __getitem__(self, idx:int) -> tuple:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f'iTarDataset index out of range, {idx} not in [0, {len(self)}).'
            )
        idx = self._sampler[idx]

        if self._grouping_active:
            return self._getitem_grouped(idx)
        return self._getitem_standard(idx)

    def _getitem_standard(self, idx:int) -> tuple:
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

        return tuple(self._trafo([out[e] for e in self.extensions]))

    def _getitem_grouped(self, idx:int) -> tuple:
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

        return tuple(self._trafo([out[e] for e in extensions]))

    @staticmethod
    def supports_tv_tensor() -> bool:
        '''Checks if the dataset supports tv tensors.
        '''
        return USE_TV_TENSOR

    def _add_trafo(self, trafo:Callable) -> 'iTarDataset':
        self.transforms.append(trafo)
        self._trafo = _compose(self.transforms)
        return self
    
    def _sync_extension_state(self):
        """Resync all extension-derived metadata from self.extensions and fold state."""
        self._real_ext = {e for e in self.extensions if e not in _valid_pseudo_extensions}
        self._pseu_ext = {e for e in self.extensions if e in _valid_pseudo_extensions}
        self._nrealext = len(self._real_ext)
        if self._nrealext == 0:
            raise ValueError('At least one real extension must be provided!')
        self._extmap = {v: k for k, v in self.fold.state.ext2id.items()}
        self.decoders = _parse_decoders(self.fold)

    def _refresh_bucketsize(self):
        computed_size = round(sum(self.shard_bincount) / (len(self.shard_bincount) * self.buckets_per_shard))
        self._bucket_size = max(1, int(computed_size))

    def _update_fold_state_vars(self):
        bincount = self.fold.bincount
        if bincount.sum() <= 0:
            raise ValueError(
                'Fold state has no samples! ' 
                'This is either due to erroneous filtering or an empty dataset.'
            )
        self.shard_bincount = bincount
        self._sampler = IdentitySampler(len(self))
        self._refresh_bucketsize()

    def _assert_grouping_inactive(self, caller:str):
        if self._grouping_active:
            raise RuntimeError(
                f'{caller} is unavailable after add_grouping() activation. '
                f'Configure all filters before enabling grouping.'
            )

    def _assert_grouping_active_for_map(self, caller:str):
        if not self._grouping_active:
            raise RuntimeError(
                f'{caller} requires active grouping mode. '
                f'Call add_grouping(...) before adding transforms.'
            )

    def _expected_output_arity(self) -> int:
        if self._grouping_active and self.grouping is not None:
            return len(self.grouping)
        return len(self.extensions)

    def _has_distinct_slot_assignment(
        self,
        slot_candidates:dict[int, tuple[str,...]]
    ) -> bool:
        slots = sorted(slot_candidates.keys(), key=lambda i: len(slot_candidates[i]))
        used:set[str] = set()

        def _dfs(k:int) -> bool:
            if k == len(slots):
                return True
            slot = slots[k]
            for prefix in slot_candidates[slot]:
                if prefix in used:
                    continue
                used.add(prefix)
                if _dfs(k + 1):
                    return True
                used.remove(prefix)
            return False

        return _dfs(0)

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

        prefixes = tuple(sorted(prefix_to_suffixes.keys()))
        if not grouping_replace:
            if group_slots > len(prefixes):
                raise ValueError(
                    f'Grouping needs {group_slots} unique prefixes but only '
                    f'{len(prefixes)} are available when grouping_replace=False.'
                )
            if not self._has_distinct_slot_assignment(slot_candidates):
                raise ValueError(
                    'Grouping is unsatisfiable with grouping_replace=False. '
                    'No unique prefix assignment satisfies all slot requirements.'
                )

        return prefixes

    def add_grouping(
        self,
        grouping:Sequence[str],
        grouping_replace:bool=False,
        grouping_seed:int|None=None,
    ) -> 'iTarDataset':
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
        '''
        self._assert_grouping_inactive('filter_stems()')
        self.fold.filter_stems(stems)
        self._update_fold_state_vars()
        return self

    def filter_stems_by_json(self, path:str|PathLike) -> "iTarDataset":
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
        stems:Sequence[str],
        extensions:Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Retrieve decoded files by exact stem and extension.

        Parameters
        ----------
        stems : Sequence[str]
            Iterable of stem names to look up.
        extensions : Sequence[str]
            Iterable of required extensions to retrieve.

        Returns
        -------
        dict[str, dict[str, Any]]
            Nested dictionary where `out[stem][ext]` is the decoded object.
            Missing stem/extension pairs are omitted.
        """
        stem_list = [StemHelper(s).stem for s in stems]
        if len(stem_list) == 0:
            return {}

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

    def map(self, mapping:Callable) -> "iTarDataset":
        '''Takes a mapping and applies it to the tuple of extensions.

        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `f(<sample>.jpg, <sample>.cls)`.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to the tuple of extensions.

        Returns
        -------
        iTarDataset
            Updated dataset with added transformations.
        '''
        self._assert_grouping_active_for_map('map()')
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")

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
        return self._add_trafo(mapping)

    def map_all(self, mapping:Callable) -> "iTarDataset":
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
        self._assert_grouping_active_for_map('map_all()')
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")
        return self._add_trafo(MapAll(mapping))

    def map_group(self, mapping:Callable, indices:Sequence[int]) -> "iTarDataset":
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
        self._assert_grouping_active_for_map('map_group()')
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")
        return self._add_trafo(MapGrouped(mapping, indices))

    def map_tuple(self, *maps:Sequence[Callable]) -> "iTarDataset":
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
        self._assert_grouping_active_for_map('map_tuple()')
        if not all(callable(m) for m in maps):
            raise TypeError("One or more mapping is not callable.")
        num_ext = self._expected_output_arity()
        if len(maps) != num_ext:
            raise ValueError(
                f"Incorrect number of transforms provided. "
                f"Expected {num_ext}, got {len(maps)}."
            )
        return self._add_trafo(MapTuple(maps))
    
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
        Ns:list[int] = self.fold.bincount.tolist()
        if shard_mixing: 
            if not self.fold.state.is_contiguous:
                raise ValueError(
                    'Shard mixing requires contiguous fold state! ' 
                    'Reinitialize dataset with `enforce_contiguous=True`.'
                )
            N = self._bucket_size
            num_Ns = len(self) // N
            last_N = len(self) % N
            Ns = [N] * num_Ns + ([last_N] if last_N > 0 else [])
        
        try:
            self._sampler = MultiFeistelSampler(Ns, rounds, seed)
            yield
        finally:
            self._sampler = IdentitySampler(len(self))
            self._epoch += 1

