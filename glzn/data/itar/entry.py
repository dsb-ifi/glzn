"""
itar.entry
----------

Defines the core data structures and logic for indexing tar files in the iTar system.

- Provides numpy dtype definitions for offset and header records.
- Implements the iTarState NamedTuple, which holds all metadata and offset information for a parsed tar or set of tars.
- Contains functions for parsing tar files into offset indices, handling crash stems (duplicate keys), and serializing/deserializing the index.
- Ensures efficient random access to samples within tar shards by storing offsets, sizes, extension IDs, and crash IDs.

This module is the backbone for all higher-level fold and dataset logic in iTar.
"""

from typing import NamedTuple
import os, mmap, xxhash, struct, numpy as np
from pathlib import Path
from os import PathLike
from .utils import StemHelper

_BLK   = 512
_ARR_DT = np.dtype([
    ('fid',    '<u2'),
    ('offset', '<u8'),
    ('size',   '<u8'),
    ('extid',  '<u2'),
    ('crashid','<u4'),
    ('keyhash','<u8'),
])
_HDR_DT = np.dtype([
    ('magic',     'S8'),
    ('major',     '<u2'),
    ('minor',     '<u2'),
    ('rec_size',  '<u2'),
    ('hdr_size',  '<u2'),
    ('n_stems',   '<u8'),
    ('n_rows',    '<u8'),
    ('n_ext',     '<u4'),
    ('n_crash',   '<u4'),
    ('off_crash', '<u8'),
    ('off_arr',   '<u8'),
    ('flags',     '<u1'),
    ('reserved',  '<V7'),
])

def _rows_contiguous(arr: np.ndarray, n_stems: int) -> bool:
    n = len(arr)
    if n < 2:
        return True
    diff = (
        (arr['keyhash'][1:] != arr['keyhash'][:-1]) |
        (arr['crashid'][1:] != arr['crashid'][:-1])
    )
    nruns = int(diff.sum()) + 1
    return nruns == n_stems

def _scan(buf:bytes|mmap.mmap, fid:int, st:"iTarState") -> "iTarState":
    """
    Scans a tar file buffer and updates the iTarState with offset records.

    Parameters
    ----------
    buf : bytes or mmap.mmap
        Buffer containing tar file data.
    fid : int
        File ID for the current tar shard.
    st : iTarState
        Current parse state to update.

    Returns
    -------
    iTarState
        Updated parse state with new offset records.
    """    
    off = 0
    while True:
        hdr = buf[off:off + _BLK]
        if len(hdr) < _BLK or hdr == b'\0' * _BLK:
            break
        size = int(hdr[124:136].split(b'\0', 1)[0] or b'0', 8)
        fn = StemHelper.from_hdr(hdr)
        stem, ext = fn.stem, fn.suffix_no_dot
        key = xxhash.xxh64(stem.encode()).intdigest()
        extid = st.ext2id.setdefault(ext, len(st.ext2id))
        st.seen_stems.add(stem)

        canon = st.hashinfo.get(key)
        if canon is None:
            st.hashinfo[key] = stem
            crashid = 0
        elif stem == canon:
            crashid = 0
        else:
            crashid = st.crashstem.get(stem)
            if crashid is None:
                crashid = st.next_crash + 1
                st = st._replace(next_crash=crashid)
                st.crashstem[stem] = crashid

        st = st.ensure(st.pos + 1)
        st.arr[st.pos] = (fid, off, size, extid, crashid, key)
        st = st._replace(pos=st.pos + 1)

        off += _BLK + ((size + _BLK - 1) // _BLK) * _BLK

    return st


class iTarState(NamedTuple):
    """
    Represents the complete parsed state of one or more tar files.

    Stores:
    - arr: numpy array of offset records for each sample (fid, offset, size, extid, crashid, keyhash).
    - ext2id: mapping from extension string to integer ID.
    - crashstem: mapping from duplicate stem names to crash IDs.
    - hashinfo: mapping from keyhash to canonical stem name.
    - next_crash: next available crash ID.
    - n_stems: number of unique stems in the dataset.
    - seen_stems: set of all stem names encountered.
    - is_contiguous: whether the samples are stored contiguously by stem.

    Provides methods for serialization, deserialization, expansion, and finalization.
    """    
    arr:            np.ndarray
    pos:            int
    ext2id:         dict[str, int]
    crashstem:      dict[str, int]
    hashinfo:       dict[int, str]
    next_crash:     int
    n_stems:        int|None
    seen_stems:     set[str]
    is_contiguous:  bool|None

    @staticmethod
    def empty(initial_rows: int = 1 << 20) -> "iTarState":
        return iTarState(
            arr        = np.empty(initial_rows, dtype=_ARR_DT),
            pos        = 0,
            ext2id     = {},
            crashstem  = {},
            hashinfo   = {},
            next_crash = 0,
            n_stems    = None,
            seen_stems = set(),
            is_contiguous  = None,
        )

    @property
    def next_fid(self) -> int:
        return self.arr['fid'][self.pos-1] + 1

    @property
    def _crash_block_bytes(self) -> bytes:
        lst = sorted(self.crashstem.keys(), key=self.crashstem.get) # type: ignore
        out = '\n'.join(lst)
        return out.encode('utf8')

    @property
    def _ext_block_bytes(self) -> bytes:
        lst = sorted(self.ext2id.keys(), key=self.ext2id.get) # type: ignore
        out = '\n'.join(lst)
        return out.encode('utf8')

    @property
    def _hdr(self) -> np.ndarray:
        if self.n_stems is None or self.is_contiguous is None:
            raise ValueError('State not finalized before writing header!')
        ext_bytes = self._ext_block_bytes
        crash_bytes = self._crash_block_bytes

        hdr = np.zeros(1, dtype=_HDR_DT)
        hdr['magic']     = b'TARIDX'
        hdr['major']     = 1
        hdr['minor']     = 0
        hdr['rec_size']  = 32
        hdr['hdr_size']  = 64
        hdr['n_stems']   = self.n_stems
        hdr['n_rows']    = self.arr.shape[0]
        hdr['n_ext']     = len(self.ext2id)
        hdr['n_crash']   = len(self.crashstem)
        hdr['off_crash'] = 64 + len(ext_bytes)
        hdr['off_arr']   = 64 + len(ext_bytes) + len(crash_bytes)
        hdr['flags']     = 1 if self.is_contiguous else 0

        return hdr

    def __bytes__(self):
        return (
            self._hdr.tobytes() +
            self._ext_block_bytes +
            self._crash_block_bytes +
            self.arr.tobytes()
        )

    def save(self, path:PathLike):
        '''Serializes a ParseState to file.

        NOTE: Ideally use the .taridx extension.

        Parameters
        ----------
        path : PathLike
            Path to serialize ParseState object.
        '''
        with open(path, 'wb') as outfile:
            outfile.write(bytes(self))

    @classmethod
    def load(cls, path:PathLike) -> "iTarState":
        '''Loads a ParseState from serialized file.

        NOTE: Ideally uses the .taridx extension.

        Parameters
        ----------
        path : PathLike
            Path to file containing a serialized ParseState object.

        Returns
        -------
        ParseState
            A serialized ParseState object.
        '''
        hdrsize = _HDR_DT.itemsize
        with open(path, 'rb') as infile:
            hdr = np.frombuffer(infile.read(hdrsize), _HDR_DT)
            if hdr['magic'] != b'TARIDX':
                raise ValueError('Invalid ParseState tar index file!')

            n_ext_bytes = int(hdr['off_crash'][0] - hdrsize)
            extlist = infile.read(n_ext_bytes).decode('utf8').split()
            ext2id = {k:v for v,k in enumerate(extlist)}

            n_crash_bytes = int(hdr['off_arr'][0] - hdrsize - n_ext_bytes)
            crashlist = infile.read(n_crash_bytes).decode('utf8').split()
            crashstem = {k:v+1 for v,k in enumerate(crashlist)}

            n_stems = int(hdr['n_stems'][0])
            is_contiguous = bool(int(hdr['flags'][0]) & 1)
            arr = np.frombuffer(infile.read(), _ARR_DT)

        if hdr['n_rows'][0] != len(arr):
            raise ValueError(
                'Error in deserializing {path}: n_rows in data do not match header!'
            )

        if hdr['n_ext'][0] != len(ext2id):
            raise ValueError(
                'Error in deserializing {path}: n_ext does not match extensions in header!'
            )

        return cls(
            arr, len(arr), ext2id, crashstem,
            {}, len(crashstem), n_stems, set(), is_contiguous
        )

    def ensure(self, need: int) -> "iTarState":
        '''Expands current capacity if size < requirements.

        Parameters
        ----------
        need : int
            Required capacity.

        Returns
        -------
        ParseState
            A new parse state with expanded capacity.
        '''
        if need < self.arr.shape[0]:
            return self
        bigger = np.empty(max(self.arr.shape[0] * 2, need + 1), dtype=_ARR_DT)
        bigger[:self.pos] = self.arr[:self.pos]
        return self._replace(arr=bigger)

    def finalize(self) -> "iTarState":
        '''Finalizes a parse state, truncating rows and removing crash table and stem set.

        Returns
        -------
        ParseState
            A finalized parse state.
        '''
        n_stems = len(self.seen_stems)
        truncarr = self.arr[:self.pos].copy()
        is_contiguous = _rows_contiguous(truncarr, n_stems)
        return self._replace(
            arr=truncarr,
            hashinfo={},
            n_stems=n_stems,
            seen_stems=set(),
            is_contiguous=is_contiguous
        )
    
    def enforce_contiguous(self) -> "iTarState":
        if self.is_contiguous:
            return self

        order = np.lexsort((self.arr['crashid'], self.arr['keyhash']))
        new_arr = self.arr[order].copy()
        return self._replace(arr=new_arr, is_contiguous=True)    


def parse_tar(
    path: PathLike,
    *,
    state: iTarState | None = None,
    bufsize: int = 8 << 20,
) -> iTarState:
    '''Convert a tar into a ParseState of offset indices.

    Parameters
    ----------
    path : PathLike
        Path of tar to parse.
    state : ParseState, optional
        Current ParseState, required for parsing multiple files.
    bufsize : int
        Buffer size for parsing.

    Returns
    -------
    ParseState
        A finalized ParseState including offsets, extension and crash tables.
    '''
    fid = 0 if state is None else state.next_fid
    st = state or iTarState.empty()
    p = Path(path)
    with open(p, "rb") as f:
        try:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.madvise(mmap.MADV_SEQUENTIAL)
                mm.madvise(mmap.MADV_WILLNEED)               
                st = _scan(mm, fid, st)
        except (OSError, ValueError, BufferError):
            data = bytearray()
            while chunk := f.read(bufsize):
                data.extend(chunk)
            st = _scan(data, fid, st)
    return st


