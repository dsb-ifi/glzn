from __future__ import annotations
import torch, torchvision, os, pkg_resources, json, functools, inspect
from contextlib import contextmanager
from torch.utils.data import Dataset
from pathlib import Path
from os import PathLike
from typing import Callable, Sequence

from .encoders import DEFAULT_DECODERS, PseudoExtension
from .maptrafo import MapAll, MapGrouped, MapTuple, DefaultIdentity
from .sampler import JointFeistelSampler
from .itar.fold import iTarFold, iTarRetriever
from .itar.utils import StemHelper, stripext

USE_TV_TENSOR = pkg_resources.parse_version(torchvision.__version__) >= pkg_resources.parse_version('0.16')
_valid_pseudo_extensions = PseudoExtension._valid_pseudo_extensions


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
        seed:int=0,
        shuffle_rows:bool=True,
        shuffle_shard_mixing:bool=True,
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


        # Check contiguity for shard mixing
        if shuffle_shard_mixing and not self.fold.state.is_contiguous:
            raise ValueError(
                'Shard mixing requires contiguous fold state! ' 
                'Rerun with `enforce_contiguous=True` or `shuffle_shard_mixing=False`.'
            )
        
        # Init sampling details
        self._epoch = 0
        self._seed = seed
        self.shuffle_rows = shuffle_rows
        self.shuffle_shard_mixing = shuffle_shard_mixing
        self.buckets_per_shard = max(1, buckets_per_shard)
        self._update_fold_state_vars()

    def __len__(self):
        return len(self.fold.state.arr) // self._nrealext

    def __getitem__(self, idx:int) -> tuple:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f'iTarDataset index out of range, {idx} not in [0, {len(self)}).'
            )

        n = self._nrealext
        idx_start = idx * n
        out = {}
        fid = None
        stem = None

        for row in self.fold.state.arr[idx_start:idx_start+n]:
            extid = row['extid']
            fid = row['fid'] if fid is None else fid
            stem = self.retriever.hdrname(row) if stem is None else stem
            out[self._extmap[extid]] = self.decoders[extid](bytes(self.retriever.from_row(row)))

        for ext in self._pseu_ext:
            match ext:
                case '_name' | '_stem':
                    out[ext] = stem
                case '_idx':
                    out[ext] = idx
                case '_fid':
                    out[ext] = int(fid) if fid is not None else None

        return tuple(self._trafo([out[e] for e in self.extensions]))

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
        self._refresh_bucketsize()
        
    def filter_extensions(self, extensions:Sequence[str]):
        clean = [stripext(e) for e in extensions]
        self.fold.filter_extensions(clean)
        self.extensions = clean
        self._sync_extension_state()
        self._update_fold_state_vars()
        if self.shuffle_shard_mixing and not self.fold.state.is_contiguous:
            raise ValueError(
                'Shard mixing requires contiguous fold state after filtering. '
                'Reinitialize with `enforce_contiguous=True` or `shuffle_shard_mixing=False`.'
            )
        return self
        
    def filter_stems(self, stems:Sequence[str]) -> "iTarDataset":
        self.fold.filter_stems(stems)
        self._update_fold_state_vars()
        if self.shuffle_shard_mixing and not self.fold.state.is_contiguous:
            raise ValueError(
                'Shard mixing requires contiguous fold state after filtering. '
                'Reinitialize with `enforce_contiguous=True` or `shuffle_shard_mixing=False`.'
            )
        return self

    def filter_stems_by_json(self, path:str|PathLike) -> "iTarDataset":
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
        if not callable(mapping):
            raise TypeError("Provided mapping is not callable.")

        sig = inspect.signature(mapping)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        n_expected = len(self.extensions)
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
        if not all(callable(m) for m in maps):
            raise TypeError("One or more mapping is not callable.")
        num_ext = len(self.extensions)
        if len(maps) == num_ext:
            self._add_trafo(MapTuple(maps))
            return self

        req_extensions = [
            (e, 1 - DEFAULT_DECODERS[e]._default_to_identity)
            for e in self.extensions
        ]
        num_req = sum(map(lambda x: x[-1], req_extensions))

        if len(maps) != num_req:
            raise ValueError(
                f"Incorrect number of transforms provided. "
                f"Expected {len(self.extensions)} | {num_req}, got {len(maps)}."
            )

        padmaps = []
        j = 0
        for _, d in req_extensions:
            if d:
                padmaps.append(maps[j])
                j += 1
            else:
                padmaps.append(DefaultIdentity())

        return self._add_trafo(MapTuple(padmaps))
    
    @contextmanager
    def shufflecontext(self):
        raise NotImplementedError()  
        

