from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


class Phase(str, Enum):
    TRAIN = "train"
    VAL = "val"


@dataclass(frozen=True, slots=True)
class StepState:
    epoch: int
    phase: Phase
    train_iter: int
    val_iter: int
    microstep: int
    fullstep: int

    train_iters_per_epoch: int
    val_iters_per_epoch: int
    accum_steps: int
    microbatch_size: int
    total_epochs: int

    def __post_init__(self):
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {self.accum_steps}")
        if self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1, got {self.microbatch_size}")
        if self.train_iters_per_epoch < 0:
            raise ValueError(f"train_iters_per_epoch must be >= 0, got {self.train_iters_per_epoch}")
        if self.val_iters_per_epoch < 0:
            raise ValueError(f"val_iters_per_epoch must be >= 0, got {self.val_iters_per_epoch}")
        if self.epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {self.epoch}")
        if self.train_iter < 0:
            raise ValueError(f"train_iter must be >= 0, got {self.train_iter}")
        if self.val_iter < 0:
            raise ValueError(f"val_iter must be >= 0, got {self.val_iter}")
        if self.microstep < 0:
            raise ValueError(f"microstep must be >= 0, got {self.microstep}")
        if self.fullstep < 0:
            raise ValueError(f"fullstep must be >= 0, got {self.fullstep}")
        if self.total_epochs < 0:
            raise ValueError(f"total_epochs must be >= 0, got {self.total_epochs}")
        # Open train windows keep microstep strictly inside the active bucket.
        # Closed windows have microstep == 0. Validation never accumulates.
        if self.is_train and self.microstep > 0 and self.microstep >= self.bucket_size:
            raise ValueError(
                f"microstep {self.microstep} is outside open accumulation window "
                f"of size {self.bucket_size}."
            )
        if not self.is_train and self.microstep != 0:
            raise ValueError(
                f"validation phase requires microstep == 0, got {self.microstep}."
            )

    @property
    def is_train(self) -> bool:
        return self.phase is Phase.TRAIN

    @property
    def phase_iter(self) -> int:
        return self.train_iter if self.is_train else self.val_iter

    @property
    def phase_iters_per_epoch(self) -> int:
        return self.train_iters_per_epoch if self.is_train else self.val_iters_per_epoch

    @property
    def phase_remaining(self) -> int:
        return max(0, self.phase_iters_per_epoch - self.phase_iter)

    @property
    def bucket_size(self) -> int:
        """Accumulation window size for the currently open window.

        The window is sized when it opens and must not shrink under an open
        ``microstep`` counter. Horizon is the micros already finished in this
        window plus the remaining phase iterations (including the upcoming
        one): ``phase_remaining + microstep``. Then
        ``bucket_size = min(accum_steps, horizon)``.
        """
        if not self.is_train:
            return 1
        horizon = self.phase_remaining + self.microstep
        if horizon <= 0:
            return 1
        return max(1, min(self.accum_steps, horizon))

    @property
    def is_update_step(self) -> bool:
        """Whether the next train microbatch completes the open accum window.

        ``microstep`` is the number of microbatches already finished in the
        current window (0 at the start of a window). The upcoming microbatch
        is the last in the window when ``microstep + 1 == bucket_size``.

        ``Processor`` must evaluate this **before** ``next_micro()``. After a
        successful or skipped optimizer update, ``on_update`` resets
        ``microstep`` to 0. Non-update microbatches call ``next_micro()`` only.
        """
        if not self.is_train:
            return False
        return (self.microstep + 1) == self.bucket_size

    @property
    def nominal_batch_size(self) -> int:
        return self.microbatch_size * self.accum_steps

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.bucket_size

    @property
    def normalized_epoch_progress(self) -> float:
        total = self.train_iters_per_epoch + self.val_iters_per_epoch
        if total <= 0:
            return 0.0
        if self.is_train:
            return self.train_iter / total
        return (self.train_iters_per_epoch + self.val_iter) / total

    @property
    def total_train_steps(self) -> int:
        return self.total_epochs * ceildiv(self.train_iters_per_epoch, self.accum_steps)

    @property
    def global_progress(self) -> float:
        total = self.total_train_steps
        if total <= 0:
            return 0.0
        return min(1.0, self.fullstep / total)

    @property
    def is_final_epoch(self) -> bool:
        return self.epoch >= self.total_epochs - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "phase": self.phase.value,
            "train_iter": self.train_iter,
            "val_iter": self.val_iter,
            "microstep": self.microstep,
            "fullstep": self.fullstep,
            "train_iters_per_epoch": self.train_iters_per_epoch,
            "val_iters_per_epoch": self.val_iters_per_epoch,
            "accum_steps": self.accum_steps,
            "microbatch_size": self.microbatch_size,
            "total_epochs": self.total_epochs,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StepState":
        return StepState(
            epoch=int(d["epoch"]),
            phase=Phase(d["phase"]),
            train_iter=int(d["train_iter"]),
            val_iter=int(d["val_iter"]),
            microstep=int(d["microstep"]),
            fullstep=int(d["fullstep"]),
            train_iters_per_epoch=int(d["train_iters_per_epoch"]),
            val_iters_per_epoch=int(d["val_iters_per_epoch"]),
            accum_steps=int(d["accum_steps"]),
            microbatch_size=int(d["microbatch_size"]),
            total_epochs=int(d["total_epochs"]),
        )


@dataclass(frozen=True, slots=True)
class StepTelemetry:
    runstart: float
    epochstart: float
    phasestart: float
    last_update_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runstart": self.runstart,
            "epochstart": self.epochstart,
            "phasestart": self.phasestart,
            "last_update_time": self.last_update_time,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StepTelemetry":
        return StepTelemetry(
            runstart=float(d["runstart"]),
            epochstart=float(d["epochstart"]),
            phasestart=float(d["phasestart"]),
            last_update_time=(None if d.get("last_update_time") is None else float(d["last_update_time"])),
        )


@dataclass(frozen=True, slots=True)
class StepTracker:

    s: StepState
    t: StepTelemetry

    @staticmethod
    def init(
        *,
        runstart: float,
        epochstart: float,
        trainsamples: int,
        valsamples: int,
        microbatch_size: int,
        accum_steps: int,
        total_epochs: int,
        start_epoch: int = 0,
    ) -> "StepTracker":
        train_iters = ceildiv(trainsamples, microbatch_size) if trainsamples > 0 else 0
        val_iters = ceildiv(valsamples, microbatch_size) if valsamples > 0 else 0
        s = StepState(
            epoch=start_epoch,
            phase=Phase.TRAIN,
            train_iter=0,
            val_iter=0,
            microstep=0,
            fullstep=0,
            train_iters_per_epoch=train_iters,
            val_iters_per_epoch=val_iters,
            accum_steps=accum_steps,
            microbatch_size=microbatch_size,
            total_epochs=total_epochs,
        )
        t = StepTelemetry(runstart=runstart, epochstart=epochstart, phasestart=epochstart, last_update_time=None)
        return StepTracker(s=s, t=t)

    def switch_phase(self, phase: Phase, *, now: float) -> "StepTracker":
        s = self.s
        if s.phase is phase:
            return self
        if s.is_train and s.microstep != 0:
            raise RuntimeError(
                "Cannot leave the training phase with an open accumulation window "
                f"(microstep={s.microstep}). Finish or flush the remaining "
                "microbatches so the final partial window updates, or restore a "
                "checkpoint taken at an update boundary."
            )
        if phase is Phase.TRAIN:
            s2 = replace(s, phase=Phase.TRAIN, microstep=0)
        else:
            s2 = replace(s, phase=Phase.VAL, microstep=0)
        t2 = replace(self.t, phasestart=now)
        return StepTracker(s=s2, t=t2)

    def next_micro(self) -> "StepTracker":
        s = self.s
        if not s.is_train:
            return self
        return StepTracker(s=replace(s, microstep=s.microstep + 1), t=self.t)

    def on_update(self, *, step_succeeded: bool, now: float) -> "StepTracker":
        """Close the current accumulation window after an optimizer attempt.

        ``fullstep`` advances only when ``step_succeeded`` is true (successful
        optimizer update). AMP overflow / scaler skip leaves ``fullstep``
        unchanged but still resets ``microstep`` so the window does not remain
        half-open. Call only when ``is_update_step`` is true.
        """
        s = self.s
        if not s.is_train:
            return self
        if not s.is_update_step:
            return self
        fullstep = s.fullstep + (1 if step_succeeded else 0)
        t2 = replace(self.t, last_update_time=(now if step_succeeded else self.t.last_update_time))
        return StepTracker(s=replace(s, microstep=0, fullstep=fullstep), t=t2)

    def advance_iter(self) -> "StepTracker":
        s = self.s
        if s.phase is Phase.TRAIN:
            return StepTracker(s=replace(s, train_iter=s.train_iter + 1), t=self.t)
        return StepTracker(s=replace(s, val_iter=s.val_iter + 1), t=self.t)

    def next_epoch(self, *, now: float) -> "StepTracker":
        s = self.s
        if s.is_train and s.microstep != 0:
            raise RuntimeError(
                "Cannot advance epoch with an open accumulation window "
                f"(microstep={s.microstep})."
            )
        s2 = replace(
            s,
            epoch=s.epoch + 1,
            phase=Phase.TRAIN,
            train_iter=0,
            val_iter=0,
            microstep=0,
        )
        t2 = replace(self.t, epochstart=now, phasestart=now)
        return StepTracker(s=s2, t=t2)

    def assert_checkpointable(self) -> None:
        """Reject trackers that are not exact resume points.

        Training may be checkpointed only with a closed accumulation window
        (``microstep == 0``), including after an AMP-skipped update attempt.
        Validation is always checkpointable. Gradients are never serialized.
        """
        if self.s.is_train and self.s.microstep != 0:
            raise RuntimeError(
                "Cannot checkpoint during an open accumulation window "
                f"(phase=train, microstep={self.s.microstep}). "
                "Exact resume is supported only at update boundaries "
                "(microstep == 0) or during validation."
            )

    def to_dict(self) -> Dict[str, Any]:
        self.assert_checkpointable()
        return {"step": self.s.to_dict(), "telemetry": self.t.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StepTracker":
        return StepTracker(s=StepState.from_dict(d["step"]), t=StepTelemetry.from_dict(d["telemetry"]))