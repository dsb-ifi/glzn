"""Summaries of recent update records from schema-v1 JSONL logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import SEEK_END
from pathlib import Path

from .schema import LogRecord, parse_record

_TAIL_BLOCK_SIZE = 64 * 1024


@dataclass(frozen=True)
class LogSummary:
    latest: LogRecord | None
    records: int
    updates_per_minute: float | None
    amp_skips: int
    nonfinite: int
    invalid_lines: int


def _tail_lines(path: Path, limit: int) -> list[bytes]:
    with path.open("rb") as log:
        offset = log.seek(0, SEEK_END)
        chunks = []
        newlines = 0
        while offset and newlines <= limit:
            size = min(offset, _TAIL_BLOCK_SIZE)
            offset -= size
            log.seek(offset)
            chunk = log.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")

    # The writer terminates each complete record with a newline.
    return b"".join(reversed(chunks)).split(b"\n")[-limit - 1:-1]


def _continues(previous: LogRecord, current: LogRecord) -> bool:
    return (
        current.run_id == previous.run_id
        and current.rank == previous.rank
        and current.world_size == previous.world_size
        and current.time >= previous.time
        and current.fullstep >= previous.fullstep
        and (current.epoch, current.iteration) > (previous.epoch, previous.iteration)
    )


def summarize_log(path: str | Path, *, window: int = 100) -> LogSummary:
    """Summarize at most ``window`` complete lines from the end of a log.

    Counts and rate describe the latest sequence of forward-moving records.
    A rollback in a resumed run starts a new sequence. Skipped attempts
    contribute elapsed time while leaving the successful update count fixed.
    The result also counts invalid lines encountered within the window.
    """
    if window < 1:
        raise ValueError("window must be positive.")

    records: list[LogRecord] = []
    invalid_lines = 0
    for line in _tail_lines(Path(path), window):
        try:
            record = parse_record(json.loads(line))
        except ValueError:
            invalid_lines += 1
            continue
        if records and not _continues(records[-1], record):
            records.clear()
        records.append(record)

    if not records:
        return LogSummary(None, 0, None, 0, 0, invalid_lines)

    first, latest = records[0], records[-1]
    elapsed = latest.time - first.time
    rate = (latest.fullstep - first.fullstep) * 60 / elapsed if elapsed > 0 else None
    return LogSummary(
        latest=latest,
        records=len(records),
        updates_per_minute=rate,
        amp_skips=sum(record.step_skipped for record in records),
        nonfinite=sum(len(record.nonfinite or {}) for record in records),
        invalid_lines=invalid_lines,
    )
