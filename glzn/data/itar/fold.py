import os, time, xxhash, mmap
from functools import lru_cache
import numpy as np

from collections import defaultdict
from os import PathLike
from pathlib import Path, PurePath
from typing import Sequence

from .entry import parse_tar, iTarState, _BLK
from .utils import StemHelper


@lru_cache(maxsize=128)
def _mmap_helper(path:str) -> mmap.mmap:
    with open(path,'rb') as file:
        return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    

def _pread_helper(path:PathLike, offset:int, length:int) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, length, offset)
    finally:
        os.close(fd)

class iTarDirectory:

    def __init__(self, root:PathLike, delimiter:str='_', extension:str='tar'):
        self.root = root
        self.delimiter = delimiter
        self.extension = extension
        self._folds = defaultdict(list)
        for f in os.listdir(self.root):
            if not f.endswith("." + extension):
                continue
            try:
                key, val = f[:-len(extension)-1].split(delimiter)
            except ValueError:
                raise ValueError(
                    f'Invalid delimiter `{delimiter}` for `{f}`.'
                )
            self._folds[key].append(val)

        for k in self._folds:
            self._folds[k] = sorted(self._folds[k])

    @property
    def folds(self) -> list[str]:
        return list(self._folds.keys())

    @property
    def nfolds(self) -> int:
        return sum([len(v) for v in self._folds.values()])

    def has_taridx(self, idxext:str):
        for fold in self.folds:
            path = os.path.join(self.root, f'{fold}.{idxext}')
            if not os.path.isfile(path):
                return False
        return True

    def get_fold(self, fold):
        if fold not in self.folds:
            return []
        return [fold + self.delimiter + d for d in self._folds[fold]]

    def parse_folder(
        self, bufsize:int=8 << 20, verbose:bool=False, serialize:bool=False,
        idxext:str='taridx'
    ) -> dict[str, iTarState]:
        total = self.nfolds
        done = 0
        t_start = time.time()
        dct = {}
        if verbose:
            msg = f'Indexing {self.root}'
            print(f' {msg.strip()} '.center(80, '-'))
        for fold in self.folds:
            state = None
            for stem in self.get_fold(fold):
                path = os.path.join(self.root, f'{stem}.{self.extension}')
                t0 = time.time()
                state = parse_tar(Path(path), state=state, bufsize=bufsize)
                slice_time = time.time() - t0
                done += 1
                if verbose:
                    ela = time.time() - t_start
                    eta = (ela / done) * (total - done) if done else 0.0
                    pct = 100.0 * done / total
                    print(f'{stem:36s} | {slice_time:6.3f}s | {ela:7.1f}s | '
                          f'{pct:5.1f}%  eta {eta:7.1f}s')
            if state is None:
                raise ValueError(f'Missing stem entries for fold {fold}.')
            state = state.finalize()
            if not serialize:
                dct[fold] = state
            else:
                path = os.path.join(self.root, f'{fold}.{idxext}')
                state.save(Path(path))

        if verbose:
            msg = f'Indexing Complete'
            print(f' {msg.strip()} '.center(80, '-'))
            print(f'{total} files indexed in {time.time() - t_start:.2f}s')
        return dct


class iTarFold:

    _pseudoextensions = ['_idx', '_name', '_stem', '_fid']

    def __init__(
        self, root:PathLike, fold:str='train', delimiter:str='_',
        extension:str='tar', parse_if_missing:bool=False, serialize:bool=True,
        idxext:str='taridx', filter_extensions:Sequence[str]|None=None,
        enforce_contiguous:bool=False, **kwargs
    ):
        self._dir = iTarDirectory(root, delimiter=delimiter, extension=extension)
        self.root = root
        self.fold = fold
        self.delimiter = delimiter
        self.extension = extension
        self.idxext = idxext
        self._paths = self._dir.get_fold(fold)
        if len(self._paths) == 0:
            valids = ', '.join(self._dir.folds)
            raise ValueError(f'Invalid fold {fold}. Valid folds are: {valids}.')

        path = os.path.join(root, f'{fold}.{idxext}')
        if self._dir.has_taridx(idxext):
            self.state = iTarState.load(Path(path))
        else:
            if not parse_if_missing:
                msg = (
                    f'Missing {fold}.{idxext} in {root}. ' +
                    'Rerun with `parse_is_missing=True`.'
                )
                raise ValueError(msg)
            if not serialize:
                self.state = self._dir.parse_folder(**kwargs)[fold]
            else:
                self._dir.parse_folder(idxext=idxext, serialize=serialize, **kwargs)
                self.state = iTarState.load(Path(path))

        self.filter_extensions(filter_extensions)
        if enforce_contiguous:
            self.state = self.state.enforce_contiguous()

    @property
    def _combined(self) -> list[str]:
        return self._pseudoextensions + list(self.state.ext2id.keys())
    
    @property
    def full_paths(self) -> list[Path]:
        return [Path(self.path_from_idx(i)) for i in range(len(self._paths))]
    
    @property
    def bincount(self) -> np.ndarray:
        return np.bincount(self.state.arr['fid'])
    
    def path_from_idx(self, idx: int) -> PathLike:
        path = os.path.join(self.root, self._paths[idx] + f'.{self.extension}')
        return Path(path)

    def filter_extensions(self, required:Sequence[str]|None) -> "iTarFold":
        if required is None or len(required) == 0:
            return self

        state = self.state
        arr = state.arr
        n = len(arr)

        # Convert to set to filter duplicates
        real = {e for e in required if e in state.ext2id}
        unknown = [e for e in required if e not in self._combined]

        if len(unknown) > 0:
            raise ValueError(
                f'Unknown extensions: {unknown}. '
                f'Valid extensions: {self._combined}.'
            )
        
        if len(real) == 0:
            return self
        
        req_ids = np.fromiter(
            (state.ext2id[e] for e in real),
            dtype=state.arr['extid'].dtype
        )

        order = slice(None)
        if not state.is_contiguous:
            order = np.lexsort((arr['crashid'], arr['keyhash']))

        sorted_ext  = arr['extid'][order]
        key_change = np.empty(n, dtype=bool)
        key_change[0] = True
        key_change[1:] = (
            (arr['keyhash'][order][1:] != arr['keyhash'][order][:-1]) |
            (arr['crashid'][order][1:] != arr['crashid'][order][:-1])
        )
        idx = np.flatnonzero(key_change)
        counts = np.diff(np.append(idx, n))

        is_req = np.isin(sorted_ext, req_ids)
        grp_hits = np.add.reduceat(is_req, idx)
        keep_grp = grp_hits == len(req_ids)

        if not keep_grp.any():
            raise RuntimeError(
                f"{self.fold}: no stem has all of {required}"
            )

        keep_row_sorted = np.repeat(keep_grp, counts) & is_req

        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[order] = keep_row_sorted # unsort if non-contiguous

        new_arr = arr[keep_mask].copy()
        self.state = state._replace(arr=new_arr, pos=len(new_arr))
        return self

    def filter_stems(self, stems:Sequence[str], verify:bool=True, strict:bool=False) -> "iTarFold":
        if not stems:
            return self

        state  = self.state
        arr    = state.arr
        h2c    = state.crashstem
        wanted = []

        for s in stems:
            h = xxhash.xxh64(s.encode()).intdigest()
            c = h2c.get(s, 0)
            wanted.append((h, c, s))

        h_vec = np.fromiter((w[0] for w in wanted), dtype=arr['keyhash'].dtype)
        c_vec = np.fromiter((w[1] for w in wanted), dtype=arr['crashid'].dtype)

        hits  = np.isin(arr['keyhash'], h_vec) & np.isin(arr['crashid'], c_vec)
        if not hits.any():
            if strict:
                raise RuntimeError(f'{self.fold}: none of {len(stems)} stems found')
            return self

        cand = arr[hits]

        if verify:
            keep = np.empty(len(cand), dtype=bool)
            for i, r in enumerate(cand):
                fid, off = int(r['fid']), int(r['offset'])
                path     = self.path_from_idx(fid)
                try:
                    mm = _mmap_helper(path)
                    raw   = mm[off : off + 100]
                except (OSError, ValueError):
                    raw = _pread_helper(path, off, 100)
                stem_on_disk = StemHelper.from_hdr(raw).stem
                keep[i] = stem_on_disk in stems
            cand = cand[keep]

            if not len(cand):
                if strict:
                    raise RuntimeError(f'{self.fold}: verification failed')
                return self

        new_arr = cand.copy()
        self.state = state._replace(arr=new_arr, pos=len(new_arr))
        return self
    

class iTarRetriever:
    """Simple retrieval with iTar.
    
    Parameters
    ----------
    paths : Sequence[str | Path]
        The shard paths in fid order (0 .. len(paths)-1).
    prefer_mmap : bool, default True
        Toggle mmap. When False we always pread.
    """

    def __init__(self, paths: Sequence[Path], prefer_mmap: bool = True):
        if not all(p.exists() for p in paths):
            raise ValueError
        self._paths = paths
        self._prefer_mmap = prefer_mmap

    def get(self, fid: int, offset: int, size: int) -> memoryview:
        path = Path(self._paths[fid])
        if self._prefer_mmap:
            try:
                mm = _mmap_helper(path)
                return memoryview(mm)[offset:offset+size]
            except (OSError, ValueError):
                pass

        return memoryview(_pread_helper(path, offset, size))

    def from_row(self, row) -> memoryview:
        return self.get(int(row["fid"]), int(row["offset"]) + _BLK, int(row["size"]))
    
    def hdrname(self, row) -> str:
        return StemHelper.from_hdr(
            bytes(self.get(int(row["fid"]), int(row["offset"]), 100))
        ).stem