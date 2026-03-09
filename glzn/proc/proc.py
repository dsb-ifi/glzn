from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn

from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer

from ..log.collator import LogCollator
from .ema import EMA
from .sched import Scheduler
from .step import Phase, StepState, StepTracker
from .wrap import ScheduledEMA, ScheduledOptimizer

TensorSequence = Tensor | Sequence[Tensor]
CallableContext = Callable[[], ContextManager[Any]]


class ProcDeps(NamedTuple):
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: Scheduler | None = None
    wd_scheduler: Scheduler | None = None
    lr_group_schedulers: dict[str | int, Scheduler] | None = None
    wd_group_schedulers: dict[str | int, Scheduler] | None = None
    scaler: GradScaler | None = None
    ema: EMA | None = None
    ema_scheduler: Scheduler | None = None


@dataclass
class _BatchContext:
    processor: "Processor"
    tracker: StepTracker
    inputs: Optional[TensorSequence]
    targets: Optional[TensorSequence]
    context: CallableContext
    phase: Optional[Phase]
    now: Optional[float]
    logging_kwargs: dict[str, Any]

    outputs: Optional[TensorSequence] = None
    loss: Optional[Tensor] = None
    updated_tracker: Optional[StepTracker] = None
    step_skipped: bool = False
    last_lr: Optional[float] = None

    def __enter__(self) -> "_BatchContext":
        phase = self.phase or self.tracker.s.phase
        at = self.now if self.now is not None else time.time()
        self.tracker = self.tracker.switch_phase(phase, now=at)
        return self

    def __exit__(self, exc_type, _exc_value, _traceback):
        if exc_type is not None:
            return False
        self.updated_tracker, self.step_skipped, self.last_lr = self.processor._process_batch(
            tracker=self.tracker,
            loss=self.loss,
            inputs=self.inputs,
            outputs=self.outputs,
            targets=self.targets,
            context=self.context,
            **self.logging_kwargs,
        )
        return False


class Processor:

    def __init__(
        self,
        deps: ProcDeps,
        *,
        logger: LogCollator | None = None,
        gradient_clipping: float | None = None,
        max_step_skipped: int = 25,
    ):
        self.deps = deps
        self.logger = logger
        self.gradient_clipping = gradient_clipping
        self.max_step_skipped = max_step_skipped
        self._acc_skipped = 0
        self.scheduled_optimizer = ScheduledOptimizer(
            optimizer=deps.optimizer,
            lr_scheduler=deps.lr_scheduler,
            wd_scheduler=deps.wd_scheduler,
            lr_group_schedulers=deps.lr_group_schedulers,
            wd_group_schedulers=deps.wd_group_schedulers,
        )
        self.scheduled_ema = (
            None if deps.ema is None
            else ScheduledEMA(
                ema=deps.ema, 
                momentum_scheduler=deps.ema_scheduler
            )
        )

    @property
    def cancel_run(self) -> bool:
        return self._acc_skipped > self.max_step_skipped

    @staticmethod
    def _opt_params(optimizer: Optimizer):
        return [p for group in optimizer.param_groups for p in group["params"]]

    @staticmethod
    def _clean(tseq: Optional[TensorSequence]) -> Optional[TensorSequence]:
        if torch.is_tensor(tseq) and isinstance(tseq, Tensor):
            return tseq.detach().cpu()
        if isinstance(tseq, Sequence):
            return [t.detach().cpu() if torch.is_tensor(t) else t for t in tseq]
        return tseq

    def _backward(self, loss: Tensor) -> None:
        if self.deps.scaler is None:
            loss.backward()
            return
        self.deps.scaler.scale(loss).backward()

    def _clip_gradients(self) -> None:
        if self.gradient_clipping is None:
            return
        if self.deps.scaler is not None:
            self.deps.scaler.unscale_(self.deps.optimizer)
        clip_grad_norm_(self._opt_params(self.deps.optimizer), self.gradient_clipping)

    def _optimizer_step(self, step_state: StepState) -> bool:
        if self.deps.scaler is None:
            self.scheduled_optimizer.step(step_state)
            return True

        self.scheduled_optimizer.apply(step_state)
        old_scale = self.deps.scaler.get_scale()
        self.deps.scaler.step(self.scheduled_optimizer.optimizer)
        self.deps.scaler.update()
        return self.deps.scaler.get_scale() >= old_scale

    def _log_batch(
        self,
        *,
        tracker: StepTracker,
        loss: Tensor,
        inputs: Optional[TensorSequence],
        outputs: Optional[TensorSequence],
        targets: Optional[TensorSequence],
        step_skipped: bool,
        last_lr: Optional[float],
        **logging_kwargs,
    ) -> None:
        if self.logger is None:
            return
        self.logger(
            time=time.time(),
            epoch=tracker.s.epoch,
            iteration=tracker.s.phase_iter,
            phase=tracker.s.phase.value,
            fullstep=tracker.s.fullstep,
            microstep=tracker.s.microstep,
            loss=loss.item(),
            inputs=self._clean(inputs),
            outputs=self._clean(outputs),
            targets=self._clean(targets),
            step_skipped=step_skipped,
            last_lr=last_lr,
            training=tracker.s.is_train,
            **logging_kwargs,
        )

    def _process_batch(
        self,
        *,
        tracker: StepTracker,
        loss: Optional[Tensor],
        inputs: Optional[TensorSequence],
        outputs: Optional[TensorSequence],
        targets: Optional[TensorSequence],
        context: CallableContext,
        **logging_kwargs,
    ) -> tuple[StepTracker, bool, Optional[float]]:
        if loss is None:
            raise ValueError("Batch loss must be populated inside processor.batch() context.")

        step_skipped = False
        last_lr: Optional[float] = None
        updated = tracker
        log_tracker = updated

        if updated.s.is_train:
            with context():
                self._backward(loss)

            updated = updated.next_micro()
            if updated.s.is_update_step:
                self._clip_gradients()

                step_succeeded = self._optimizer_step(updated.s)
                step_skipped = not step_succeeded

                now = time.time()
                updated = updated.on_update(step_succeeded=step_succeeded, now=now)

                self.deps.optimizer.zero_grad(set_to_none=True)

                if step_succeeded and self.scheduled_ema is not None:
                    self.scheduled_ema.update_parameters(self.deps.model, updated.s)

                if self.deps.optimizer.param_groups:
                    last_lr = float(self.deps.optimizer.param_groups[0].get("lr", 0.0))
        else:
            step_skipped = False
            if self.deps.optimizer.param_groups:
                last_lr = float(self.deps.optimizer.param_groups[0].get("lr", 0.0))

        self._acc_skipped = (self._acc_skipped + int(step_skipped)) * int(step_skipped)
        log_tracker = updated
        updated = updated.advance_iter()

        self._log_batch(
            tracker=log_tracker,
            loss=loss,
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            step_skipped=step_skipped,
            last_lr=last_lr,
            **logging_kwargs,
        )
        return updated, step_skipped, last_lr

    def batch(
        self,
        *,
        tracker: StepTracker,
        inputs: Optional[TensorSequence] = None,
        targets: Optional[TensorSequence] = None,
        phase: Optional[Phase] = None,
        now: Optional[float] = None,
        context: CallableContext = nullcontext,
        **logging_kwargs,
    ) -> _BatchContext:
        return _BatchContext(
            processor=self,
            tracker=tracker,
            inputs=inputs,
            targets=targets,
            context=context,
            phase=phase,
            now=now,
            logging_kwargs=logging_kwargs,
        )



