"""Log sinks: JSONL persistence and optional stdout presentation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TextIO

from .schema import LogRecord

_STDOUT_LOGGER_NAME = "glzn.log.stdout"


class LogSink(Protocol):
    def log(self, record: LogRecord) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JSONLSink:
    """Append-only line-buffered JSONL writer for schema-v1 records.

    Default policy: only rank 0 writes the canonical ``metrics.jsonl``.
    Optional rank-local diagnostics write under ``rank/<rank>.jsonl``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        enabled: bool | None = None,
    ):
        self.path = Path(path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.enabled = (self.rank == 0) if enabled is None else bool(enabled)
        self._fh: TextIO | None = None
        self._closed = False
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Hold the handle open; line buffering so each write is a full line.
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def log(self, record: LogRecord) -> None:
        if not self.enabled or self._fh is None:
            return
        if self._closed:
            raise RuntimeError(f"JSONLSink for {self.path} is closed.")
        payload = record.model_dump_jsonl_payload()
        line = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._fh.write(line)
        self._fh.write("\n")

    def flush(self) -> None:
        if self._fh is not None and not self._closed:
            self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "JSONLSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class StdoutSink:
    """Compact rank-0 stdout view of the same normalized records."""

    def __init__(
        self,
        *,
        rank: int = 0,
        enabled: bool | None = None,
        metric_allowlist: Sequence[str] | None = None,
        max_metrics: int = 8,
    ):
        self.rank = int(rank)
        self.enabled = (self.rank == 0) if enabled is None else bool(enabled)
        self.metric_allowlist = (
            None if metric_allowlist is None else tuple(metric_allowlist)
        )
        self.max_metrics = int(max_metrics)
        self._closed = False
        self._logger = self._get_logger()

    @staticmethod
    def _get_logger() -> logging.Logger:
        logger = logging.getLogger(_STDOUT_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # Avoid duplicate handlers across reloads / repeated construction.
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger

    def _format_metric(self, record: LogRecord, key: str, value: Any) -> str:
        nf = record.nonfinite or {}
        if key in nf:
            kind = nf[key]
            if kind == "nan":
                return f"{key}=nan"
            if kind == "posinf":
                return f"{key}=+inf"
            if kind == "neginf":
                return f"{key}=-inf"
        if value is None:
            return f"{key}=None"
        if isinstance(value, float):
            return f"{key}={value:.6g}"
        return f"{key}={value}"

    def log(self, record: LogRecord) -> None:
        if not self.enabled or self._closed:
            return
        payload = record.model_dump(mode="json", exclude_none=True)
        metrics: Mapping[str, Any] = payload.get("metrics", {})
        if self.metric_allowlist is not None:
            keys = [k for k in self.metric_allowlist if k in metrics]
        else:
            keys = list(metrics.keys())[: self.max_metrics]

        parts = [
            f"epoch={record.epoch}",
            f"iter={record.iteration}",
            f"fullstep={record.fullstep}",
            f"ok={int(record.update_succeeded)}",
            f"skip={int(record.step_skipped)}",
        ]
        dims = payload.get("dims") or {}
        if dims:
            dims_s = ",".join(f"{k}={v}" for k, v in dims.items())
            parts.append(f"dims[{dims_s}]")
        for key in keys:
            parts.append(self._format_metric(record, key, metrics[key]))
        self._logger.info(" ".join(parts))

    def flush(self) -> None:
        for handler in self._logger.handlers:
            handler.flush()

    def close(self) -> None:
        self._closed = True
        self.flush()

    def __enter__(self) -> "StdoutSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
