# TARIDX binary format specification

This document specifies the on-disk `.taridx` format used by iTar.

It reflects the current implementation in [glzn/data/itar/entry.py](entry.py), and is intended to support:

- compatibility across versions,
- independent readers/writers,
- safe refactors with explicit invariants.

## 1) Scope

A `.taridx` file stores metadata and random-access row records for one fold/shard collection of tar members.

Each row identifies one tar member payload by:

- tar shard id (`fid`),
- tar header byte offset (`offset`),
- payload byte size (`size`),
- extension id (`extid`),
- stem collision id (`crashid`),
- hash of stem (`keyhash`).

## 2) Versioning

Header fields:

- `magic = b"TARIDX"` (stored in an 8-byte field)
- `major` and `minor`

Current writer version:

- `major = 1`
- `minor = 0`

Compatibility policy:

- Readers MUST reject unknown `magic`.
- Readers MAY accept newer `minor` within same `major` if all required fields are understood.
- `major` changes indicate potential breaking layout/semantic changes.

## 3) Endianness and scalar encoding

All numeric fields are little-endian.

- `uint16` = `<u2`
- `uint32` = `<u4`
- `uint64` = `<u8`

No compression is used.

## 4) File layout

The file is a strict concatenation of 4 regions:

1. fixed-size header (`_HDR_DT`, 64 bytes),
2. extension block (UTF-8 text),
3. crash-stem block (UTF-8 text),
4. row array (`_ARR_DT` records, 32 bytes each).

### 4.1 Region offsets

Let:

- `hdr_size = 64`
- `ext_bytes = serialized extension block length`
- `crash_bytes = serialized crash block length`

Then:

- extension block starts at byte `64`
- crash block starts at byte `off_crash = 64 + ext_bytes`
- row array starts at byte `off_arr = 64 + ext_bytes + crash_bytes`

The row region continues to EOF.

## 5) Header record (`_HDR_DT`)

The header is exactly one struct with this layout:

| Field | Type | Size | Meaning |
|---|---:|---:|---|
| `magic` | `S8` | 8 | Magic bytes (`TARIDX` + NUL padding) |
| `major` | `<u2` | 2 | Major format version |
| `minor` | `<u2` | 2 | Minor format version |
| `rec_size` | `<u2` | 2 | Row record size (currently 32) |
| `hdr_size` | `<u2` | 2 | Header size (currently 64) |
| `n_stems` | `<u8` | 8 | Number of unique stems |
| `n_rows` | `<u8` | 8 | Number of row records |
| `n_ext` | `<u4` | 4 | Number of extensions |
| `n_crash` | `<u4` | 4 | Number of crash stems |
| `off_crash` | `<u8` | 8 | Byte offset of crash block |
| `off_arr` | `<u8` | 8 | Byte offset of row array |
| `flags` | `<u1` | 1 | Bit flags |
| `reserved` | `V7` | 7 | Reserved bytes (MUST be ignored by readers) |

Flag bits:

- bit 0 (`flags & 0x01`): rows are contiguous by `(keyhash, crashid)` groups.
- bits 1..7: reserved for future use.

## 6) Extension block

UTF-8 text containing extension names in id order, joined by newline (`\n`):

`ext_0 + "\n" + ext_1 + ... + ext_(n_ext-1)`

Notes:

- No mandatory trailing newline.
- Empty block is valid when `n_ext == 0`.
- Reader reconstructs ids by order of entries.

## 7) Crash-stem block

UTF-8 text containing crash stem strings in crash-id order (for ids `1..n_crash`), joined by newline.

`crash_1 + "\n" + crash_2 + ... + crash_n`

Notes:

- `crashid = 0` is reserved for non-colliding/canonical stems.
- As with extension block, no mandatory trailing newline.
- Reader reconstructs ids by order and assigns id `index + 1`.

## 8) Row record (`_ARR_DT`)

Each row is exactly 32 bytes:

| Field | Type | Size | Meaning |
|---|---:|---:|---|
| `fid` | `<u2` | 2 | Tar shard id |
| `offset` | `<u8` | 8 | Byte offset of tar header block (512-byte tar header) |
| `size` | `<u8` | 8 | Payload size in bytes |
| `extid` | `<u2` | 2 | Extension id into extension table |
| `crashid` | `<u4` | 4 | Collision disambiguator (`0` canonical, `>=1` crash table) |
| `keyhash` | `<u8` | 8 | `xxhash64(stem)` integer digest |

Payload location in tar file:

- payload begins at `offset + 512`
- payload length is `size`

## 9) Ordering and grouping invariants

Primary invariant for dataset indexing:

- rows SHOULD be grouped contiguously by sample key `(keyhash, crashid)`.

When the invariant holds, header `flags` bit 0 MUST be set.

If rows are not contiguous:

- readers may still parse the index,
- higher-level dataset behavior may require reordering (`force_contiguous` in current implementation).

## 10) Reader requirements

A compliant reader MUST:

1. parse exactly one 64-byte header,
2. verify `magic`,
3. slice extension/crash blocks using `off_crash` and `off_arr`,
4. parse row records from `off_arr` to EOF as 32-byte structs,
5. verify row count matches `n_rows`.

A robust reader SHOULD additionally validate:

- `hdr_size == 64`,
- `rec_size == 32`,
- `off_crash >= 64`,
- `off_arr >= off_crash`,
- `n_ext` equals decoded extension count,
- `n_crash` equals decoded crash-stem count,
- all `extid < n_ext`.

## 11) Writer requirements

A compliant writer MUST:

- emit little-endian structs exactly as specified,
- emit extension and crash blocks in id order,
- keep `n_*` fields consistent with emitted data,
- set offsets from actual emitted byte lengths,
- set `flags` bit 0 according to contiguous grouping state.

## 12) Error handling guidance

Recommended failure modes:

- invalid magic/version/layout: fail fast with explicit format error,
- count/offset mismatch: fail as corrupted/inconsistent index,
- unsupported future major version: fail with compatibility error.

## 13) Worked layout example (symbolic)

For `n_ext=2`, `n_crash=1`, `n_rows=3`:

- Header (64 bytes)
- Extension block: `"jpg\njson"` (8 bytes)
- Crash block: `"duplicate_stem"` (14 bytes)
- Row array: `3 * 32 = 96` bytes

Computed offsets:

- `off_crash = 64 + 8 = 72`
- `off_arr = 72 + 14 = 86`
- total file size = `86 + 96 = 182` bytes

## 14) Change management

Any format-affecting change MUST update:

- this specification,
- version fields (`major`/`minor`) as appropriate,
- reader compatibility notes and migration guidance.
