"""LogCollator: normalize → LogRecordV1 → sink fan-out."""
from __future__ import annotations

import time as time_mod
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import LogRecordV1, build_log_record
from .sinks import JSONLSink, LogSink, StdoutSink


class LogCollator:
    """Build versioned update records and fan them out to sinks.

    The collator does not compute task metrics, average microbatches, inspect
    model I/O, or run distributed collectives. Callers supply already-correct
    **update-level** scalar metrics.

    ``b.metrics`` on a Processor batch must contain finalized effective-update
    metrics written by the task on the closing microbatch. The processor does
    not combine per-microbatch metric mappings.
    """

    def __init__(
        self,
        *,
        run_id: str,
        rank: int = 0,
        world_size: int = 1,
        sinks: Sequence[LogSink] | None = None,
    ):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string.")
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}.")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}.")
        self.run_id = run_id
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.sinks: list[LogSink] = list(sinks or ())
        self._closed = False

    @classmethod
    def local(
        cls,
        *,
        run_id: str,
        root: str | Path,
        rank: int = 0,
        world_size: int = 1,
        logfoldername: str = "log",
        stdout: bool = True,
        rank_local: bool = False,
        metric_allowlist: Sequence[str] | None = None,
    ) -> "LogCollator":
        """Construct a collator with canonical JSONL (+ optional stdout / rank-local)."""
        root = Path(root)
        log_dir = root / logfoldername
        sinks: list[LogSink] = [
            JSONLSink(
                log_dir / "metrics.jsonl",
                rank=rank,
                world_size=world_size,
                enabled=(rank == 0),
            )
        ]
        if rank_local:
            sinks.append(
                JSONLSink(
                    log_dir / "rank" / f"{rank}.jsonl",
                    rank=rank,
                    world_size=world_size,
                    enabled=True,
                )
            )
        if stdout:
            sinks.append(
                StdoutSink(
                    rank=rank,
                    enabled=(rank == 0),
                    metric_allowlist=metric_allowlist,
                )
            )
        return cls(run_id=run_id, rank=rank, world_size=world_size, sinks=sinks)

    def log_update(
        self,
        *,
        epoch: int,
        iteration: int,
        fullstep: int,
        update_succeeded: bool,
        step_skipped: bool,
        metrics: Mapping[str, Any] | None = None,
        dims: Mapping[str, Any] | None = None,
        time: float | None = None,
    ) -> LogRecordV1:
        """Normalize, validate one LogRecordV1, fan the same frozen instance out."""
        if self._closed:
            raise RuntimeError("LogCollator is closed.")
        record = build_log_record(
            time=time if time is not None else time_mod.time(),
            run_id=self.run_id,
            epoch=epoch,
            iteration=iteration,
            fullstep=fullstep,
            rank=self.rank,
            world_size=self.world_size,
            update_succeeded=update_succeeded,
            step_skipped=step_skipped,
            metrics=metrics,
            dims=dims,
        )
        for sink in self.sinks:
            sink.log(record)
        return record

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                sink.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> "LogCollator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
