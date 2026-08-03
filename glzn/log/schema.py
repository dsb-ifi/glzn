"""Schema v1: normalization stage + frozen Pydantic boundary."""
from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

# Optional NumPy support without a hard dependency.
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

import torch

SCHEMA_VERSION: Literal[1] = 1
EVENT_UPDATE: Literal["update"] = "update"
PHASE_TRAIN: Literal["train"] = "train"

NonFiniteKind = Literal["nan", "posinf", "neginf"]
DimensionScalar = bool | int | float | str | None
MetricScalar = bool | int | float | None

_NAME_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NONFINITE_KINDS = frozenset({"nan", "posinf", "neginf"})


class SchemaError(ValueError):
    """Raised when a log record or field violates schema v1."""


def validate_name(name: str, *, kind: str) -> str:
    if not isinstance(name, str) or not name:
        raise SchemaError(f"{kind} name must be a non-empty string, got {name!r}.")
    if name.startswith("/") or name.endswith("/") or "//" in name:
        raise SchemaError(f"malformed {kind} name {name!r}.")
    parts = name.split("/")
    if any(not _NAME_PART_RE.fullmatch(part) for part in parts):
        raise SchemaError(f"malformed {kind} name {name!r}.")
    return name


def _is_numpy_scalar(value: Any) -> bool:
    if np is None:
        return False
    return isinstance(value, np.generic)


def normalize_scalar(value: Any, *, allow_str: bool, allow_none: bool, field: str) -> Any:
    """Normalize a value to a Python scalar (no tensors/arrays remain)."""
    if value is None:
        if allow_none:
            return None
        raise SchemaError(f"{field} rejects None.")

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{field} rejects non-finite float {value!r}.")
        return float(value)
    if allow_str and isinstance(value, str):
        return value

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise SchemaError(
                f"{field} rejects non-scalar tensor with shape {tuple(value.shape)}."
            )
        return normalize_scalar(
            value.detach().cpu().reshape(()).item(),
            allow_str=allow_str,
            allow_none=allow_none,
            field=field,
        )

    if _is_numpy_scalar(value):
        return normalize_scalar(
            value.item(),
            allow_str=allow_str,
            allow_none=allow_none,
            field=field,
        )

    if np is not None and isinstance(value, np.ndarray):
        raise SchemaError(f"{field} rejects NumPy arrays; use a scalar metric/dim.")

    if isinstance(value, (list, tuple, dict, set, Mapping)):
        raise SchemaError(
            f"{field} rejects nested/sequence values of type {type(value).__name__}."
        )

    raise SchemaError(f"{field} rejects unsupported type {type(value).__name__}.")


def classify_nonfinite(value: float) -> NonFiniteKind:
    if math.isnan(value):
        return "nan"
    if math.isinf(value) and value > 0:
        return "posinf"
    if math.isinf(value) and value < 0:
        return "neginf"
    raise SchemaError(f"expected non-finite float, got {value!r}.")


def normalize_dims(dims: Mapping[str, Any] | None) -> dict[str, DimensionScalar]:
    """Task dims → plain Python scalars (non-finite rejected)."""
    raw = {} if dims is None else dict(dims)
    out: dict[str, DimensionScalar] = {}
    for key, value in raw.items():
        name = validate_name(str(key), kind="dimension")
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaError(f"dimension {name!r} rejects non-finite float.")
        if torch.is_tensor(value) and value.numel() == 1:
            item = value.detach().cpu().reshape(()).item()
            if isinstance(item, float) and not math.isfinite(item):
                raise SchemaError(f"dimension {name!r} rejects non-finite float.")
            value = item
        out[name] = normalize_scalar(
            value, allow_str=True, allow_none=True, field=f"dimension {name!r}"
        )
    return out


def normalize_metrics(
    metrics: Mapping[str, Any] | None,
) -> tuple[dict[str, MetricScalar], dict[str, NonFiniteKind]]:
    """Task metrics → Python scalars + nonfinite sidecar (no tensors remain)."""
    raw = {} if metrics is None else dict(metrics)
    out: dict[str, MetricScalar] = {}
    nonfinite: dict[str, NonFiniteKind] = {}
    for key, value in raw.items():
        name = validate_name(str(key), kind="metric")
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise SchemaError(f"metric {name!r} rejects non-scalar tensor.")
            value = value.detach().cpu().reshape(()).item()
        elif _is_numpy_scalar(value):
            value = value.item()
        elif np is not None and isinstance(value, np.ndarray):
            raise SchemaError(f"metric {name!r} rejects NumPy arrays.")
        elif isinstance(value, str):
            raise SchemaError(
                f"metric {name!r} rejects strings; put categorical values in dims."
            )
        elif isinstance(value, (list, tuple, dict, set, Mapping)):
            raise SchemaError(f"metric {name!r} rejects nested/sequence values.")

        if isinstance(value, float) and not math.isfinite(value):
            out[name] = None
            nonfinite[name] = classify_nonfinite(value)
            continue
        if value is None:
            out[name] = None
            continue
        out[name] = normalize_scalar(
            value, allow_str=False, allow_none=False, field=f"metric {name!r}"
        )
    return out, nonfinite


class LogRecordV1(BaseModel):
    """Frozen schema-v1 update record (Pydantic v2 validation boundary).

    Pydantic is an implementation and validation mechanism. The serialized
    JSON object is the public contract and must stay stable across refactors.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1] = 1
    time: float
    run_id: str
    event: Literal["update"] = "update"
    phase: Literal["train"] = "train"

    epoch: int
    iteration: int
    fullstep: int

    rank: int
    world_size: int

    update_succeeded: bool
    step_skipped: bool

    dims: dict[str, DimensionScalar] = Field(default_factory=dict)
    metrics: dict[str, MetricScalar] = Field(default_factory=dict)
    nonfinite: dict[str, NonFiniteKind] | None = None

    @field_validator("run_id")
    @classmethod
    def _run_id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("run_id must be a non-empty string.")
        return v

    @field_validator("time")
    @classmethod
    def _time_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("time must be a finite float.")
        return v

    @field_validator("epoch", "iteration", "fullstep", "rank")
    @classmethod
    def _nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0.")
        return v

    @field_validator("world_size")
    @classmethod
    def _world_size_pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("world_size must be >= 1.")
        return v

    @field_validator("nonfinite", mode="before")
    @classmethod
    def _empty_nonfinite_to_none(cls, v: Any) -> Any:
        if v is not None and len(v) == 0:
            return None
        return v

    @field_validator("dims")
    @classmethod
    def _dims_names(cls, v: dict[str, DimensionScalar]) -> MappingProxyType:
        for key in v:
            validate_name(str(key), kind="dimension")
        return MappingProxyType(dict(v))

    @field_validator("metrics")
    @classmethod
    def _metrics_names(cls, v: dict[str, MetricScalar]) -> MappingProxyType:
        for key in v:
            validate_name(str(key), kind="metric")
        return MappingProxyType(dict(v))

    @field_validator("nonfinite")
    @classmethod
    def _freeze_nonfinite(
        cls, v: dict[str, NonFiniteKind] | None
    ) -> MappingProxyType | None:
        if v is None:
            return None
        return MappingProxyType(dict(v))

    @field_serializer("dims", "metrics", "nonfinite")
    def _serialize_maps(self, v: Any) -> Any:
        # MappingProxyType is not a default pydantic JSON type; dump as plain dict.
        if v is None:
            return None
        return dict(v)

    @model_validator(mode="after")
    def _cross_field(self) -> "LogRecordV1":
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank must be < world_size, got rank={self.rank}, world_size={self.world_size}."
            )
        # Schema v1 update records are exactly one of:
        #   success: update_succeeded=True,  step_skipped=False
        #   AMP skip: update_succeeded=False, step_skipped=True
        # Reject both-true and both-false (no defined failed-but-not-skipped case).
        if self.update_succeeded == self.step_skipped:
            raise ValueError(
                "schema v1 requires update_succeeded XOR step_skipped "
                f"(got update_succeeded={self.update_succeeded}, "
                f"step_skipped={self.step_skipped})."
            )

        if self.nonfinite is not None:
            for key, kind in self.nonfinite.items():
                if kind not in _NONFINITE_KINDS:
                    raise ValueError(f"invalid nonfinite kind {kind!r} for {key!r}.")
                if key not in self.metrics:
                    raise ValueError(f"nonfinite key {key!r} missing from metrics.")
                if self.metrics[key] is not None:
                    raise ValueError(
                        f"metrics[{key!r}] must be None when listed in nonfinite."
                    )
        return self

    def model_dump_jsonl_payload(self) -> dict[str, Any]:
        """Plain dict for strict JSON serialization (schema-v1 public shape)."""
        return self.model_dump(mode="json", exclude_none=True)


# Public alias used by older call sites / tests that still say LogRecord.
LogRecord = LogRecordV1


def build_log_record(
    *,
    time: float,
    run_id: str,
    epoch: int,
    iteration: int,
    fullstep: int,
    rank: int,
    world_size: int,
    update_succeeded: bool,
    step_skipped: bool,
    metrics: Mapping[str, Any] | None = None,
    dims: Mapping[str, Any] | None = None,
    schema_version: int = SCHEMA_VERSION,
    event: str = EVENT_UPDATE,
    phase: str = PHASE_TRAIN,
) -> LogRecordV1:
    """Normalize task values, then construct a validated LogRecordV1."""
    if schema_version != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version {schema_version}; writer supports {SCHEMA_VERSION}."
        )
    if event != EVENT_UPDATE:
        raise SchemaError(f"schema v1 only supports event={EVENT_UPDATE!r}, got {event!r}.")
    if phase != PHASE_TRAIN:
        raise SchemaError(f"schema v1 only supports phase={PHASE_TRAIN!r}, got {phase!r}.")

    dims_n = normalize_dims(dims)
    metrics_n, nonfinite = normalize_metrics(metrics)

    try:
        return LogRecordV1(
            schema_version=1,
            time=float(time),
            run_id=run_id,
            event="update",
            phase="train",
            epoch=epoch,
            iteration=iteration,
            fullstep=fullstep,
            rank=rank,
            world_size=world_size,
            update_succeeded=update_succeeded,
            step_skipped=step_skipped,
            dims=dims_n,
            metrics=metrics_n,
            nonfinite=nonfinite or None,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise SchemaError(str(exc)) from exc
