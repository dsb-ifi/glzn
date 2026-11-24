"""Read training windows from Equinor SGZ seismic cubes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from seismic_zfp.read import SgzReader


@dataclass(frozen=True)
class SgzWindow:
    """A window in zero-based (inline, crossline, sample) index order."""

    path:str|Path
    origin:tuple[int, int, int]
    shape:tuple[int, int, int]


class SgzDataset(Dataset[np.ndarray]):
    """Read float32 windows from 3D post-stack SGZ cubes.

    Each worker opens its own reader on demand and keeps the most recently
    used cube open. Windows retain their input order.
    """

    def __init__(self, windows:Sequence[SgzWindow]):
        self._reader:SgzReader|None = None
        self._path:Path|None = None
        self._pid:int|None = None
        self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx:int) -> np.ndarray:
        window = self.windows[idx]
        origin = tuple(map(index, window.origin))
        shape = tuple(map(index, window.shape))
        if len(origin) != 3 or len(shape) != 3:
            raise ValueError('SGZ origin and shape must each contain three integers.')
        if min(origin) < 0 or min(shape) <= 0:
            raise ValueError('SGZ origin must be nonnegative and shape must be positive.')
        inline, crossline, sample = origin
        n_inlines, n_crosslines, n_samples = shape
        reader = self._reader_for(Path(window.path).expanduser())
        crop = reader.read_subvolume(
            inline, inline + n_inlines,
            crossline, crossline + n_crosslines,
            sample, sample + n_samples,
            multithreading=False,
        )
        # The reader can return a view of cached decompressed blocks.
        return crop.copy()

    def _reader_for(self, path:Path) -> SgzReader:
        pid = os.getpid()
        if self._reader is not None and self._pid == pid and self._path == path:
            return self._reader

        self.close()
        try:
            from seismic_zfp.read import SgzReader
        except ModuleNotFoundError as exc:
            if exc.name != 'seismic_zfp':
                raise
            raise ImportError('Install GLZN with the seismic extra to read SGZ cubes.') from exc

        reader = SgzReader(str(path))
        if not reader.is_3d:
            reader.close()
            raise ValueError('SgzDataset requires a 3D post-stack cube.')
        self._reader, self._path, self._pid = reader, path, pid
        return reader

    def __getstate__(self):
        return {**vars(self), '_reader': None, '_path': None, '_pid': None}

    def close(self) -> None:
        """Close the cached reader in the current process."""
        if self._reader is not None:
            self._reader.close()
            self._reader, self._path, self._pid = None, None, None

    def __del__(self) -> None:
        self.close()
