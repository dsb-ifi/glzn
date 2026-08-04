from __future__ import annotations

import time
import warnings
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Mapping, NamedTuple, Protocol, Sequence, cast

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


CallableContext = Callable[[], ContextManager[Any]]
EMASource = Callable[[nn.Module], nn.Module]
Scalar = bool | int | float | None


class UpdateHook(Protocol):
    def on_update(
        self,
        *,
        model: nn.Module,
        step_state: StepState,
    ) -> None:
        ...


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
    ema_source: EMASource | None = None
    update_hooks: Sequence[UpdateHook] = ()


@dataclass
class Batch:
    processor: Processor
    phase: Phase | None = None
    now: float | None = None
    context: CallableContext = nullcontext

    metrics: dict[str, Scalar] = field(default_factory=dict)
    dims: dict[str, Scalar] = field(default_factory=dict)
    update_attempted: bool = False
    update_succeeded: bool = False
    step_skipped: bool = False
    grad_norm: Tensor | None = None
    last_lr: float | None = None
    state_at_update: StepState | None = None

    _stack: ExitStack = field(default_factory=ExitStack, init=False)
    _tracker_before_open: StepTracker | None = field(default=None, init=False)
    _time: float | None = field(default=None, init=False)

    def __enter__(self) -> Batch:
        self.processor._open_batch(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._stack.__exit__(exc_type, exc, tb)
            if exc_type is not None:
                self.processor._handle_batch_exception(self)
                return False
            try:
                self.processor._close_batch(self)
            except BaseException:
                self.processor._handle_batch_exception(self, force_abort=True)
                raise
            return False
        finally:
            self.processor._active_batch = None

    def backward(self, loss: Tensor) -> None:
        if self.processor._active_batch is not self:
            raise RuntimeError("Batch.backward() called outside its active batch.")
        self.processor.backward(loss)

    def autocast(self) -> ContextManager[Any]:
        return self.processor.autocast()

    @property
    def state(self) -> StepState:
        return self.processor.s

    @property
    def is_train(self) -> bool:
        return self.state.is_train

    @property
    def is_update_step(self) -> bool:
        return self.state.is_update_step

    @property
    def bucket_size(self) -> int:
        return self.state.bucket_size

    @property
    def fullstep(self) -> int:
        return self.state.fullstep


class Processor:
    """Training processor that owns the mutable tracker reference.

    Task semantics stay outside proc. Proc owns the mechanical lifecycle:
    accumulation scaling, autocast, DDP ``no_sync``, optimizer/scaler updates,
    EMA timing, update-level logging, and checkpointable optimization state.
    It is sequential and not thread-safe: one batch context may be active at a
    time.

    Validation batches advance the validation iterator only; validation metric
    aggregation/logging is intentionally not implemented here yet.
    """

    def __init__(
        self,
        deps: ProcDeps,
        *,
        tracker: StepTracker,
        logger: LogCollator | None = None,
        gradient_clipping: float | None = None,
        autocast_device_type: str | None = None,
        autocast_dtype: torch.dtype | None = None,
        autocast_enabled: bool = False,
        max_step_skipped: int = 25,
    ):
        self.deps = deps
        self.tracker = tracker
        self.logger = logger
        self.gradient_clipping = gradient_clipping
        self.autocast_device_type = autocast_device_type
        self.autocast_dtype = autocast_dtype
        self.autocast_enabled = autocast_enabled
        self.max_step_skipped = max_step_skipped
        self._acc_skipped = 0
        self._loss_sum: Tensor | None = None
        self._loss_count = 0
        self._backward_called = False
        self._aborted = False
        self._active_batch: Batch | None = None
        self._check_amp_configuration()
        self.scheduled_optimizer = ScheduledOptimizer(
            optimizer=deps.optimizer,
            lr_scheduler=deps.lr_scheduler,
            wd_scheduler=deps.wd_scheduler,
            lr_group_schedulers=deps.lr_group_schedulers,
            wd_group_schedulers=deps.wd_group_schedulers,
        )
        self.scheduled_ema = (
            None
            if deps.ema is None
            else ScheduledEMA(
                ema=deps.ema,
                momentum_scheduler=deps.ema_scheduler,
                source=deps.ema_source,
            )
        )

    @property
    def s(self) -> StepState:
        return self.tracker.s

    @property
    def cancel_run(self) -> bool:
        return self._acc_skipped > self.max_step_skipped

    def batch(
        self,
        *,
        phase: Phase | None = None,
        now: float | None = None,
        context: CallableContext = nullcontext,
    ) -> Batch:
        return Batch(
            processor=self,
            phase=phase,
            now=now,
            context=context,
        )

    def autocast(self) -> ContextManager[Any]:
        if not self.autocast_enabled:
            return nullcontext()
        device_type = self.autocast_device_type or self._model_device_type()
        return torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def backward(self, loss: Tensor) -> None:
        if self._active_batch is None:
            raise RuntimeError("Processor.backward() must be called inside batch().")
        if not self.s.is_train:
            raise RuntimeError("Processor.backward() is only valid in train phase.")
        if self._backward_called:
            raise RuntimeError("Processor.backward() may be called only once per batch.")
        scaled = loss / float(self.s.bucket_size)
        with torch.autocast(
            device_type=self.autocast_device_type or self._model_device_type(),
            enabled=False,
        ):
            if self.deps.scaler is None:
                scaled.backward()
            else:
                self.deps.scaler.scale(scaled).backward()
        detached = loss.detach().float()
        self._loss_sum = detached if self._loss_sum is None else self._loss_sum + detached
        self._loss_count += 1
        self._backward_called = True

    def state_dict(self) -> dict[str, Any]:
        self.assert_checkpointable()
        state: dict[str, Any] = {
            "tracker": self.tracker.to_dict(),
            "optimizer": self.deps.optimizer.state_dict(),
            "acc_skipped": self._acc_skipped,
        }
        if self.deps.scaler is not None:
            state["scaler"] = self.deps.scaler.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        self.tracker = StepTracker.from_dict(state["tracker"])
        self.deps.optimizer.load_state_dict(state["optimizer"])
        if self.deps.scaler is not None:
            if "scaler" not in state:
                if strict:
                    raise RuntimeError("Processor checkpoint is missing scaler state.")
            else:
                self.deps.scaler.load_state_dict(state["scaler"])
        self._acc_skipped = int(state.get("acc_skipped", 0))
        self._loss_sum = None
        self._loss_count = 0
        self._backward_called = False
        self._active_batch = None
        self._aborted = False

    def assert_checkpointable(self) -> None:
        if self._aborted:
            raise RuntimeError("Cannot checkpoint an aborted Processor.")
        self.tracker.assert_checkpointable()
        if self._active_batch is not None:
            raise RuntimeError("Cannot checkpoint during an active proc batch.")
        if self._loss_count != 0:
            raise RuntimeError("Cannot checkpoint with an open proc loss window.")

    def next_epoch(self, *, now: float | None = None) -> None:
        self.tracker = self.tracker.next_epoch(now=time.time() if now is None else now)

    def _open_batch(self, batch: Batch) -> None:
        if self._aborted:
            raise RuntimeError(
                "Processor is not reusable after an exception following "
                "backward(). Restore from checkpoint or rebuild it."
            )
        if self._active_batch is not None:
            raise RuntimeError("Nested Processor.batch() contexts are not supported.")
        phase = batch.phase or self.s.phase
        at = batch.now if batch.now is not None else time.time()
        batch._tracker_before_open = self.tracker
        batch._time = at
        self.tracker = self.tracker.switch_phase(phase, now=at)
        batch.update_attempted = self.s.is_train and self.s.is_update_step
        self._backward_called = False
        self._active_batch = batch

        if self.s.is_train and not self.s.is_update_step:
            no_sync = getattr(self.deps.model, "no_sync", None)
            if callable(no_sync):
                batch._stack.enter_context(cast(ContextManager[Any], no_sync()))
        batch._stack.enter_context(self.autocast())
        batch._stack.enter_context(batch.context())

    def _close_batch(self, batch: Batch) -> None:
        if self.s.is_train:
            self._close_train_batch(batch)
        else:
            self.tracker = self.tracker.advance_iter()

    def _close_train_batch(self, batch: Batch) -> None:
        if not self._backward_called:
            raise RuntimeError("Train batch closed without calling Processor.backward(loss).")
        if self.s.is_update_step:
            batch.state_at_update = self.s
            if self._loss_count != batch.state_at_update.bucket_size:
                raise RuntimeError(
                    "Loss accumulation count does not match active bucket size: "
                    f"{self._loss_count} != {batch.state_at_update.bucket_size}."
                )
            batch.grad_norm = self._prepare_gradients()
            step_succeeded = self._optimizer_step(self.s)
            batch.update_succeeded = step_succeeded
            batch.step_skipped = not step_succeeded
            self.tracker = self.tracker.on_update(
                step_succeeded=step_succeeded,
                now=self._batch_time(batch),
            )
            self.deps.optimizer.zero_grad(set_to_none=True)
            if step_succeeded and self.scheduled_ema is not None:
                self.scheduled_ema.update_parameters(self.deps.model, batch.state_at_update)
            if step_succeeded:
                for hook in self.deps.update_hooks:
                    hook.on_update(model=self.deps.model, step_state=batch.state_at_update)
            batch.last_lr = self._last_lr()
            self._acc_skipped = (self._acc_skipped + int(batch.step_skipped)) * int(batch.step_skipped)
            self._emit_update_log(batch)
            self._clear_loss_window()
        else:
            self.tracker = self.tracker.next_micro()
            batch.last_lr = self._last_lr()
        self.tracker = self.tracker.advance_iter()

    def _prepare_gradients(self) -> Tensor | None:
        if not self._needs_grad_norm():
            return None
        if self.deps.scaler is not None:
            self.deps.scaler.unscale_(self.deps.optimizer)
        max_norm = float("inf") if self.gradient_clipping is None else self.gradient_clipping
        grad_norm = clip_grad_norm_(self._opt_params(), max_norm)
        return grad_norm.detach()

    def _optimizer_step(self, step_state: StepState) -> bool:
        if self.deps.scaler is None:
            self.scheduled_optimizer.step(step_state)
            return True
        self.scheduled_optimizer.apply(step_state)
        old_scale = self.deps.scaler.get_scale()
        self.deps.scaler.step(self.scheduled_optimizer.optimizer)
        self.deps.scaler.update()
        return self.deps.scaler.get_scale() >= old_scale

    def _emit_update_log(self, batch: Batch) -> None:
        if self.logger is None:
            return
        metrics: dict[str, Scalar] = dict(batch.metrics)
        if self._loss_sum is None:
            raise RuntimeError("Cannot log loss/total without accumulated loss.")
        metrics.setdefault("loss/total", self._loss_sum.item() / float(self._loss_count))
        if batch.last_lr is not None:
            metrics.setdefault("optim/lr", batch.last_lr)
        if batch.grad_norm is not None:
            metrics.setdefault("optim/grad_norm", batch.grad_norm.item())
        self.logger.log_update(
            time=self._batch_time(batch),
            epoch=self.s.epoch,
            iteration=batch.state_at_update.phase_iter if batch.state_at_update is not None else self.s.phase_iter,
            fullstep=self.s.fullstep,
            update_succeeded=batch.update_succeeded,
            step_skipped=batch.step_skipped,
            metrics=metrics,
            dims=batch.dims,
        )

    def _clear_loss_window(self) -> None:
        self._loss_sum = None
        self._loss_count = 0

    def _opt_params(self) -> list[nn.Parameter]:
        return [p for group in self.deps.optimizer.param_groups for p in group["params"]]

    def _last_lr(self) -> float | None:
        if not self.deps.optimizer.param_groups:
            return None
        return float(self.deps.optimizer.param_groups[0].get("lr", 0.0))

    def _model_device_type(self) -> str:
        try:
            return next(self.deps.model.parameters()).device.type
        except StopIteration:
            return "cpu"

    def _needs_grad_norm(self) -> bool:
        return self.gradient_clipping is not None or self.logger is not None

    def _batch_time(self, batch: Batch) -> float:
        return batch._time if batch._time is not None else time.time()

    def _handle_batch_exception(self, batch: Batch, *, force_abort: bool = False) -> None:
        if force_abort:
            self._aborted = True
            return
        if self._backward_called or self._loss_count != 0:
            self._aborted = True
            return
        if batch._tracker_before_open is not None:
            self.tracker = batch._tracker_before_open

    def _check_amp_configuration(self) -> None:
        if not self.autocast_enabled:
            return
        dtype = self.autocast_dtype
        if dtype is None:
            dtype = torch.get_autocast_dtype(self.autocast_device_type or self._model_device_type())
        if dtype is torch.float16 and self.deps.scaler is None:
            warnings.warn(
                "float16 autocast is enabled without a GradScaler; this is "
                "usually unsafe except for explicitly managed cases.",
                stacklevel=2,
            )
        if dtype is torch.bfloat16 and self.deps.scaler is not None:
            warnings.warn(
                "bfloat16 autocast usually does not need a GradScaler; keeping "
                "both is allowed but uncommon.",
                stacklevel=2,
            )
