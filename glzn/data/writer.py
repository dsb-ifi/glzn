from __future__ import annotations

import getpass
import os
import time
import xxhash

from pathlib import Path
from os import PathLike
from typing import Any, Callable, Iterable, Optional

from .encoders import DEFAULT_ENCODERS
from .itar.entry import iTarState
from .itar.utils import stripext


_BLK = 512

def _normalize_ext(ext: str) -> str:
	out = stripext(ext).lower()
	if out == '':
		raise ValueError('Extension cannot be empty.')
	return out


def _check_flatname(name: str, what: str):
	if '/' in name or '\\' in name:
		raise ValueError(f'{what} must be flat (no path separators): {name!r}.')


def _octal(value: int, width: int) -> bytes:
	if value < 0:
		raise ValueError('Octal fields do not support negative values.')
	txt = f'{value:o}'.encode('ascii')
	if len(txt) > width - 1:
		raise ValueError(f'Value {value} does not fit in {width} byte octal field.')
	return txt.rjust(width - 1, b'0') + b'\0'


def _tar_header(
	name: str,
	size: int,
	*,
	mode: int,
	uid: int,
	gid: int,
	mtime: int,
	uname: str,
	gname: str,
) -> bytes:
	_check_flatname(name, 'Tar member name')
	name_bytes = name.encode('utf-8')
	if len(name_bytes) > 100:
		raise ValueError(
			f'Tar member name exceeds 100 bytes and would require longname extension: {name!r}.'
		)

	hdr = bytearray(_BLK)
	hdr[0:100] = name_bytes.ljust(100, b'\0')
	hdr[100:108] = _octal(mode, 8)
	hdr[108:116] = _octal(uid, 8)
	hdr[116:124] = _octal(gid, 8)
	hdr[124:136] = _octal(size, 12)
	hdr[136:148] = _octal(mtime, 12)
	hdr[148:156] = b'        '
	hdr[156:157] = b'0'
	hdr[257:263] = b'ustar\0'
	hdr[263:265] = b'00'
	hdr[265:297] = uname.encode('utf-8')[:32].ljust(32, b'\0')
	hdr[297:329] = gname.encode('utf-8')[:32].ljust(32, b'\0')

	cksum = sum(hdr)
	hdr[148:156] = f'{cksum:06o}\0 '.encode('ascii')
	return bytes(hdr)


class _TarShardWriter:

	def __init__(
		self,
		path: PathLike,
		*,
		username: str,
		groupname: str,
		mode: int,
	):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self.fileobj = open(self.path, 'wb')
		self.username = username
		self.groupname = groupname
		self.mode = mode

	@property
	def bytes_written(self) -> int:
		return self.fileobj.tell()

	def add_member(self, name: str, payload: bytes) -> tuple[int, int]:
		if not isinstance(payload, (bytes, bytearray, memoryview)):
			raise TypeError('Payload must be bytes-like.')

		data = bytes(payload)
		offset = self.fileobj.tell()
		now = int(time.time())
		hdr = _tar_header(
			name,
			len(data),
			mode=self.mode,
			uid=0,
			gid=0,
			mtime=now,
			uname=self.username,
			gname=self.groupname,
		)
		self.fileobj.write(hdr)
		self.fileobj.write(data)

		pad = (-len(data)) % _BLK
		if pad:
			self.fileobj.write(b'\0' * pad)

		return offset, len(data)

	def close(self):
		if self.fileobj.closed:
			return
		self.fileobj.write(b'\0' * (_BLK * 2))
		self.fileobj.close()


class iTarFoldWriter:

	def __init__(
		self,
		root: PathLike,
		fold: str,
		*,
		shard_pattern: str = '%04d',
		delimiter: str = '_',
		shard_maxfiles: int = 100000,
		shard_maxsize: float = 3e9,
		start: int = 0,
		username: Optional[str] = None,
		groupname: str = 'defaultgroup',
		mode: int = 0o444,
		override_encoders: Optional[Iterable[tuple[str, Callable[[Any], bytes]]]] = None,
		enforce_contiguous: bool = True,
	):
		if delimiter == '' or len(delimiter) != 1:
			raise ValueError('Delimiter must be exactly one character.')
		if delimiter in fold:
			raise ValueError(
				f'Fold {fold!r} cannot contain delimiter {delimiter!r} because iTarDirectory '
				'expects filenames to split into exactly two parts.'
			)
		_check_flatname(fold, 'Fold name')

		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self.fold = fold
		self.shard_pattern = shard_pattern
		self.delimiter = delimiter
		self.shard_maxfiles = shard_maxfiles
		self.shard_maxsize = shard_maxsize
		self.start = start
		self.username = username if username is not None else getpass.getuser()
		self.groupname = groupname
		self.mode = mode
		self.enforce_contiguous = enforce_contiguous

		self.encoders = {**DEFAULT_ENCODERS}
		if override_encoders is not None:
			for ext, fn in override_encoders:
				if not callable(fn):
					raise TypeError(f'Encoder for {ext!r} is not callable.')
				self.encoders[f'.{_normalize_ext(ext)}'] = fn

		self.state = iTarState.empty()

		self._next_shard = start
		self._cur_shard: _TarShardWriter | None = None
		self._cur_fid: int | None = None
		self._cur_count = 0
		self._closed = False
		self.total_count = 0

		self._open_new_shard()

	def _shard_name(self, idx: int) -> str:
		suffix = self.shard_pattern % idx
		if self.delimiter in suffix:
			raise ValueError(
				f'Shard suffix {suffix!r} contains delimiter {self.delimiter!r}. '
				'This breaks iTarDirectory filename parsing.'
			)
		return f'{self.fold}{self.delimiter}{suffix}.tar'

	def _shard_path(self, idx: int) -> Path:
		return self.root / self._shard_name(idx)

	def _open_new_shard(self):
		if self._cur_shard is not None:
			self._cur_shard.close()

		idx = self._next_shard
		self._next_shard += 1
		self._cur_fid = idx - self.start
		self._cur_shard = _TarShardWriter(
			self._shard_path(idx),
			username=self.username,
			groupname=self.groupname,
			mode=self.mode,
		)
		self._cur_count = 0

	def _resolve_crashid(self, stem: str) -> int:
		keyhash = xxhash.xxh64(stem.encode()).intdigest()
		canon = self.state.hashinfo.get(keyhash)
		if canon is None:
			self.state.hashinfo[keyhash] = stem
			return 0
		if canon == stem:
			return 0

		crashid = self.state.crashstem.get(stem)
		if crashid is None:
			crashid = self.state.next_crash + 1
			self.state.crashstem[stem] = crashid
			self.state = self.state._replace(next_crash=crashid)
		return crashid

	def _append_row(self, fid: int, offset: int, size: int, extid: int, crashid: int, keyhash: int):
		self.state = self.state.ensure(self.state.pos + 1)
		self.state.arr[self.state.pos] = (fid, offset, size, extid, crashid, keyhash)
		self.state = self.state._replace(pos=self.state.pos + 1)

	def write(self, key: str | dict[str, Any], sample: Optional[dict[str, Any]] = None):
		if self._closed:
			raise RuntimeError(f'{self.__class__.__name__} is already closed.')

		if isinstance(key, dict):
			if sample is not None:
				raise ValueError('When first argument is a sample dict, second argument must be None.')
			obj = key
			if '__key__' not in obj:
				raise ValueError("Sample dict must include '__key__'.")
			keystr = str(obj['__key__'])
			payloads = {k: v for k, v in obj.items() if k != '__key__'}
		else:
			if sample is None:
				raise ValueError('Sample payload dict is required.')
			keystr = key
			payloads = sample

		if keystr == '':
			raise ValueError('Sample key cannot be empty.')
		_check_flatname(keystr, 'Sample key')
		if '.' in keystr:
			raise ValueError('Sample key cannot contain dots. Stem parsing uses first dot as separator.')
		if len(payloads) == 0:
			raise ValueError('Sample payload dict cannot be empty.')

		assert self._cur_shard is not None and self._cur_fid is not None
		if (
			self._cur_count >= self.shard_maxfiles or
			self._cur_shard.bytes_written >= self.shard_maxsize
		):
			self._open_new_shard()
			assert self._cur_shard is not None and self._cur_fid is not None

		keyhash = xxhash.xxh64(keystr.encode()).intdigest()
		crashid = self._resolve_crashid(keystr)
		self.state.seen_stems.add(keystr)

		for ext, obj in payloads.items():
			ext_nodot = _normalize_ext(ext)
			ext_with_dot = f'.{ext_nodot}'
			encoder = self.encoders.get(ext_with_dot)
			if encoder is None:
				raise KeyError(f'No encoder registered for extension {ext_with_dot!r}.')

			encoded = encoder(obj)
			if not isinstance(encoded, (bytes, bytearray, memoryview)):
				raise TypeError(
					f'Encoder for {ext_with_dot!r} returned non-bytes payload: {type(encoded)}.'
				)

			member_name = f'{keystr}.{ext_nodot}'
			offset, size = self._cur_shard.add_member(member_name, bytes(encoded))

			extid = self.state.ext2id.setdefault(ext_nodot, len(self.state.ext2id))
			self._append_row(self._cur_fid, offset, size, extid, crashid, keyhash)

		self._cur_count += 1
		self.total_count += 1

	@property
	def taridx_path(self) -> Path:
		return self.root / f'{self.fold}.taridx'

	def close(self):
		if self._closed:
			return

		if self._cur_shard is not None:
			self._cur_shard.close()
			self._cur_shard = None

		self.state = self.state.finalize()
		if self.enforce_contiguous and not self.state.is_contiguous:
			raise RuntimeError(
				f'Fold {self.fold!r} was written with non-contiguous rows. '
				'Ensure all members for one key are written together before next key.'
			)
		self.state.save(self.taridx_path)
		self._closed = True

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()


class iTarDatasetWriter:

	def __init__(
		self,
		dataset_name: str,
		loc: str | PathLike,
		*,
		folds: Iterable[str],
		shard_pattern: str = '%04d',
		delimiter: str = '_',
		shard_maxfiles: int = 100000,
		shard_maxsize: float = 3e9,
		start: int = 0,
		username: Optional[str] = None,
		groupname: str = 'defaultgroup',
		mode: int = 0o444,
		override_encoders: Optional[Iterable[tuple[str, Callable[[Any], bytes]]]] = None,
		enforce_contiguous: bool = True,
	):
		self.dataset_name = dataset_name
		self.loc = Path(loc)
		if not self.loc.exists():
			raise FileExistsError(f'Location does not exist: {self.loc}.')

		self.root = self.loc / dataset_name
		self.root.mkdir(parents=True, exist_ok=True)

		self.shard_pattern = shard_pattern
		self.delimiter = delimiter
		self.shard_maxfiles = shard_maxfiles
		self.shard_maxsize = shard_maxsize
		self.start = start
		self.username = username
		self.groupname = groupname
		self.mode = mode
		self.override_encoders = override_encoders
		self.enforce_contiguous = enforce_contiguous

		self._fold_writers: dict[str, iTarFoldWriter] = {}
		self._closed = False

		for fold in folds:
			self.fold(fold)

	def fold(self, name: str) -> iTarFoldWriter:
		if self._closed:
			raise RuntimeError(f'{self.__class__.__name__} is already closed.')
		if name not in self._fold_writers:
			self._fold_writers[name] = iTarFoldWriter(
				root=self.root,
				fold=name,
				shard_pattern=self.shard_pattern,
				delimiter=self.delimiter,
				shard_maxfiles=self.shard_maxfiles,
				shard_maxsize=self.shard_maxsize,
				start=self.start,
				username=self.username,
				groupname=self.groupname,
				mode=self.mode,
				override_encoders=self.override_encoders,
				enforce_contiguous=self.enforce_contiguous,
			)
		return self._fold_writers[name]

	@property
	def folds(self) -> tuple[str, ...]:
		return tuple(self._fold_writers.keys())

	def close(self):
		if self._closed:
			return
		for writer in self._fold_writers.values():
			writer.close()
		self._closed = True

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.close()

