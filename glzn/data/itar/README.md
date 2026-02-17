# iTar data structures

This document describes the binary/index structures used by iTar-backed datasets.

## Core index record (`_ARR_DT`)

Defined in [glzn/data/itar/entry.py](glzn/data/itar/entry.py), each row is one tar member:

- `fid: uint16` — shard/file id (`train_0000.tar` -> `0`, etc.)
- `offset: uint64` — byte offset of tar header
- `size: uint64` — payload size in bytes
- `extid: uint16` — integer id for extension (mapped by `ext2id`)
- `crashid: uint32` — disambiguator for stem hash collisions
- `keyhash: uint64` — xxhash64 of stem (filename without extension)

Payload bytes are read at `offset + 512` (`_BLK`).

## Serialized header (`_HDR_DT`)

Also in [glzn/data/itar/entry.py](glzn/data/itar/entry.py), the taridx header contains:

- magic/version (`magic`, `major`, `minor`)
- shape info (`n_rows`, `n_stems`, `n_ext`, `n_crash`)
- byte offsets to extension block, crash block, and array block
- `flags` bit 0 = rows contiguous by `(keyhash, crashid)`

## `iTarState`

[`glzn.data.itar.entry.iTarState`](glzn/data/itar/entry.py) is the in-memory + serializable index state:

- `arr`: structured array of `_ARR_DT`
- `ext2id`: extension -> integer id
- `crashstem`: colliding stem string -> `crashid`
- `hashinfo`: canonical stem per hash (parse-time helper)
- `seen_stems`: parse-time unique stem tracking
- `n_stems`, `is_contiguous`: finalized metadata

Lifecycle:
1. `empty()` allocates initial buffers
2. `parse_tar(...)` appends rows
3. `finalize()` truncates + computes metadata
4. `save()` / `load()` writes/reads `.taridx`

## Fold and retrieval contracts

- [`glzn.data.itar.fold.iTarFold`](glzn/data/itar/fold.py) owns one fold index (`train`, `val`, ...), filtering rows by extension/stem.
- [`glzn.data.itar.fold.iTarRetriever`](glzn/data/itar/fold.py) reads bytes from `(fid, offset, size)` tuples, optionally via mmap.
- [`glzn.data.dataset.iTarDataset`](glzn/data/dataset.py):
  - assumes per-sample rows are grouped by selected real extensions
  - decodes using `extid -> decoder`
  - supports pseudo extensions (`_idx`, `_name`, `_stem`, `_fid`)

## Notes for contributors

- Keep `_ARR_DT` and `_HDR_DT` backward-compatible unless bumping format version.
- Any change to row ordering assumptions should be validated against dataset indexing (`__len__`, `__getitem__`) in [glzn/data/dataset.py](glzn/data/dataset.py).
- For new extension types, ensure `ext2id`/decoder mapping remains stable.

## Related docs

- Binary format specification: [glzn/data/itar/TARIDX_FORMAT.md](glzn/data/itar/TARIDX_FORMAT.md)