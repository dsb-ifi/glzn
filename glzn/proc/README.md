# `glzn.proc`

Context-based training and validation processing with support for:

- gradient accumulation via `StepTracker`
- gradient clipping
- AMP gradient scaling (`GradScaler`)
- scheduled optimizer updates (global and per-group)
- scheduled EMA updates
- optional batch-level logging through `LogCollator`

The `proc` package is intentionally split into composable modules:

- `step.py`: immutable state and telemetry (`StepState`, `StepTracker`)
- `sched.py`: scalar schedules based on `fullstep`
- `wrap.py`: wrappers that apply schedules to optimizer/EMA/loss
- `proc.py`: orchestration layer (`Processor`) with context-based batch flow

---

## Core Concepts

1. `StepTracker` is the source of truth for loop progress.
2. `Processor.batch(...)` returns a context object where you set `outputs` and `loss`.
3. On context exit, `Processor` performs the train/val action automatically.
4. `tracker = b.updated_tracker` is required after every batch.

Train batches can trigger backward/update. Validation batches never update optimizer/EMA.

---

## Minimal Example

```python
import time

import torch

from glzn.proc.proc import ProcDeps, Processor
from glzn.proc.step import Phase, StepTracker

model = ...
optimizer = ...
loss_fn = ...
train_loader = ...
val_loader = ...

total_epochs = 10
accum_steps = 2
microbatch_size = 32

tracker = StepTracker.init(
    runstart=time.time(),
    epochstart=time.time(),
    trainsamples=len(train_loader.dataset),
    valsamples=len(val_loader.dataset),
    microbatch_size=microbatch_size,
    accum_steps=accum_steps,
    total_epochs=total_epochs,
)

proc = Processor(
    ProcDeps(model=model, optimizer=optimizer),
    gradient_clipping=1.0,
)

for _epoch in range(total_epochs):
    model.train()
    for inputs, targets in train_loader:
        with proc.batch(
            tracker=tracker,
            inputs=inputs,
            targets=targets,
            phase=Phase.TRAIN,
        ) as b:
            b.outputs = model(inputs)
            b.loss = loss_fn(b.outputs, targets)
        tracker = b.updated_tracker

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            with proc.batch(
                tracker=tracker,
                inputs=inputs,
                targets=targets,
                phase=Phase.VAL,
            ) as b:
                b.outputs = model(inputs)
                b.loss = loss_fn(b.outputs, targets)
            tracker = b.updated_tracker

    tracker = tracker.next_epoch(now=time.time())
```

---

## AMP and Scheduler Example

```python
import torch

from glzn.proc.ema import EMA
from glzn.proc.proc import ProcDeps, Processor
from glzn.proc.sched import Scheduler

model = ...
optimizer = ...

lr_scheduler = Scheduler(total_steps=10000, base_val=3e-4, end_val=1e-5, main_schedule="cosine")
wd_scheduler = Scheduler(total_steps=10000, base_val=1.0, end_val=0.2, main_schedule="cosine")
ema = EMA(model, decay=0.999)
ema_scheduler = Scheduler(total_steps=10000, base_val=1.0, end_val=1.0, main_schedule="none")
scaler = torch.amp.GradScaler("cuda")

proc = Processor(
    ProcDeps(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        scaler=scaler,
        ema=ema,
        ema_scheduler=ema_scheduler,
    ),
    gradient_clipping=1.0,
)
```

When AMP is enabled, optimizer updates are executed through the scaler and skipped safely on overflow.

---

## `ProcDeps` Reference

- `model`: training model used for forward/backward and EMA source updates
- `optimizer`: raw optimizer
- `lr_scheduler`: optional global learning-rate scheduler
- `wd_scheduler`: optional global weight-decay scheduler
- `lr_group_schedulers`: optional per-group LR schedulers keyed by group name/index
- `wd_group_schedulers`: optional per-group WD schedulers keyed by group name/index
- `scaler`: optional `torch.amp.GradScaler`
- `ema`: optional `EMA`
- `ema_scheduler`: optional scheduler for EMA momentum factor

---

## Logging Hooks

Pass a `LogCollator` into `Processor(..., logger=...)` to log one entry per processed batch.

The processor emits keys such as:

- `epoch`, `iteration`, `phase`, `fullstep`, `microstep`
- `loss`, `last_lr`, `step_skipped`, `training`
- `inputs`, `outputs`, `targets` (detached to CPU when tensors)

---

## Update-boundary clock

These fields share one successful-update clock:

| Field / consumer | Meaning |
|------------------|---------|
| `microstep` | Microbatches finished in the open accumulation window (0 after each update attempt) |
| `fullstep` | Number of **successful** optimizer updates completed so far |
| `phase_iter` (`train_iter` / `val_iter`) | Batches finished in the current phase |
| `is_update_step` | True when the **upcoming** microbatch completes the window: `(microstep + 1) == bucket_size` |
| LR / WD schedules | Applied at update time with the pre-update `fullstep` |
| EMA momentum + `update_hooks` | Receive the same pre-update `StepState` as the optimizer schedules |
| AMP skip | Does **not** advance `fullstep`; still resets `microstep` and zeros grads |

`bucket_size = min(accum_steps, phase_remaining + microstep)` so the open
window does not shrink mid-accumulation, and a short final window still flushes.

### Logging

Each log row is emitted after the train action for that batch and **before**
`phase_iter` advances:

| Key | Meaning |
|-----|---------|
| `iteration` | 0-based index of the batch just processed |
| `microstep` / `fullstep` | counters **after** the batch action |
| schedule index (LR/WD/EMA/hooks) | pre-update `fullstep` (not logged directly) |
| `last_lr` | active optimizer group LR after the batch (reapplied only on update attempts) |
| `step_skipped` | true only when an update was attempted and the scaler rejected it |

Example with `accum_steps=2`, `train_iters=3`:

```text
iter 0: microstep 1, fullstep 0, no update
iter 1: microstep 0, fullstep 1, successful update
iter 2: microstep 0, fullstep 2, successful tail update
```

### Phase transitions

Leaving train (for validation or `next_epoch`) with `microstep != 0` **raises**.
A completed training phase always flushes its final partial window, so a normal
train → val handoff has `microstep == 0`. An open window means an interrupted
iterator, early manual phase switch, or bad bookkeeping — not a silent discard.

### Checkpoint resume

`StepTracker` serializes counters (`microstep`, `fullstep`, iters, epoch, …).
It does **not** serialize `.grad` accumulation buffers.

**Exact resume is enforced at save time** via `StepTracker.assert_checkpointable()`
(also invoked by `to_dict()` / `save_checkpoint`):

- training: `microstep == 0` required (closed window after success **or** AMP skip)
- validation: always allowed

Saving mid-window fails with a clear `RuntimeError`. Do not add gradient
serialization for mid-window resume.

---

## Notes and Gotchas

1. Always assign `tracker = b.updated_tracker` after each batch.
2. `StepTracker.next_epoch(...)` is not implicit; call it at the end of every epoch.
3. Validation batches still require a loss for logging/consistency in current implementation.
4. With gradient accumulation, updates happen only when `StepState.is_update_step` is true **before** counting the microbatch.
5. `Processor.cancel_run` can be polled to stop if too many consecutive skipped updates occur.
6. Mid-window checkpoints and mid-window phase exits are rejected; see **Checkpoint resume** and **Phase transitions**.