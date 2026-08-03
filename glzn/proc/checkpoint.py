from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .step import StepTracker


def save_checkpoint(
    path: str | Path,
    *,
    state: Mapping[str, Any],
) -> None:
    """Save a small named-state checkpoint.

    Values with ``state_dict()`` are stored by state dict. ``StepTracker`` is
    stored through ``to_dict()`` after ``assert_checkpointable()`` so open
    accumulation windows cannot be saved as exact resume points. Other values
    are stored directly.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "glzn.checkpoint.v1",
        "state": {name: _pack_state(obj) for name, obj in state.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    state: Mapping[str, Any] | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring named mutable objects.

    When ``state`` is provided, objects exposing ``load_state_dict`` are restored
    in place. ``StepTracker`` entries are returned because it is immutable.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    saved = payload["state"]
    if state is None:
        return {name: _unpack_state(value) for name, value in saved.items()}

    restored: dict[str, Any] = {}
    for name, value in saved.items():
        if name not in state:
            restored[name] = _unpack_state(value)
            continue
        target = state[name]
        if hasattr(target, "load_state_dict"):
            target.load_state_dict(value["state_dict"], strict=strict)
        else:
            restored[name] = _unpack_state(value)
    return restored


def _pack_state(obj: Any) -> Any:
    if isinstance(obj, StepTracker):
        # to_dict() enforces update-boundary / validation-only checkpoint policy.
        return {"kind": "step_tracker", "value": obj.to_dict()}
    if hasattr(obj, "assert_checkpointable"):
        obj.assert_checkpointable()
    if hasattr(obj, "state_dict"):
        return {"kind": "state_dict", "state_dict": obj.state_dict()}
    if hasattr(obj, "model_dump"):
        return {"kind": "pydantic", "value": obj.model_dump(mode="json")}
    return {"kind": "raw", "value": obj}


def _unpack_state(value: Any) -> Any:
    kind = value.get("kind")
    if kind == "step_tracker":
        return StepTracker.from_dict(value["value"])
    if kind == "state_dict":
        return value["state_dict"]
    return value.get("value")
