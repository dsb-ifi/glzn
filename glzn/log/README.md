# `glzn.log`

Local-first, vendor-neutral training metrics for HPC compute nodes.

**The schema is the public contract. Sinks are replaceable adapters.**

Schema v1 is implemented as a frozen **Pydantic v2** model (`LogRecordV1`).
Pydantic is an implementation and validation mechanism. The serialized JSON
schema is the public contract. Future internal model refactors must preserve
schema-v1 JSON compatibility.

JSONL is the canonical durable interchange format. WandB, Aim, and other
integrations are out of scope here; they should consume the same records later.

---

## Intended path

```text
task / training loop
    → computes semantically correct update-level scalar metrics
    → Processor identifies an optimizer update attempt
    → LogCollator builds one versioned record
    → sinks receive the same normalized record
        → JSONLSink  (canonical)
        → StdoutSink (optional view)
        → future vendor adapters
```

The logger validates and routes metrics. It does **not** compute task metrics,
average microbatches, inspect model I/O, or run distributed reductions.

---

## Schema version 1

One JSON object per **optimizer update attempt** (including AMP skips).

Non-update accumulation microbatches emit **no** durable records.

### Record shape

```python
{
    "schema_version": 1,
    "time": float,              # Unix seconds
    "run_id": str,
    "event": "update",
    "phase": "train",
    "epoch": int,
    "iteration": int,           # 0-based train-phase batch index that closed the window
    "fullstep": int,            # == StepTracker.fullstep after the attempt
    "rank": int,
    "world_size": int,
    "update_succeeded": bool,   # XOR with step_skipped
    "step_skipped": bool,       # success=(true,false); AMP skip=(false,true)
    "dims": dict[str, Scalar],
    "metrics": dict[str, MetricScalar],
    "nonfinite": dict[str, "nan"|"posinf"|"neginf"]  # optional
}
```

### Frozen reserved top-level keys

```text
schema_version time run_id event phase epoch iteration fullstep
rank world_size update_succeeded step_skipped dims metrics nonfinite
```

Metrics live only in `metrics`. Dimensions live only in `dims`.

### `fullstep` (checkpoint-aligned)

```text
fullstep = number of successful optimizer updates completed
```

This is exactly `StepTracker.fullstep` after the update attempt, and the same
field serialized in checkpoints.

- Increments only on **successful** optimizer updates.
- **Non-decreasing but not unique**: AMP-skipped attempts keep the previous
  `fullstep`.
- **Training-progress axis, not a unique record id.**

Logical update-record identifier within a training phase:

```text
(epoch, iteration)
```

where `iteration` is the zero-based phase-batch index of the microbatch that
closed the accumulation window.

Schema v1 requires **exactly one** of success or AMP skip per record:

```text
update_succeeded XOR step_skipped
```

Valid pairs:

```text
success:  update_succeeded=true,  step_skipped=false
AMP skip: update_succeeded=false, step_skipped=true
```

`(true, true)` and `(false, false)` are rejected (no defined failed-but-not-skipped case).

Example sequence:

```text
success → fullstep=1, update_succeeded=true,  step_skipped=false
AMP skip → fullstep=1, update_succeeded=false, step_skipped=true
success → fullstep=2, update_succeeded=true,  step_skipped=false
```

Internal LR/WD/EMA/hook schedules still use the **pre-update** `fullstep` inside
`Processor`. That schedule index is not a public log field.

### Metric and dimension names

Flat slash-separated paths, e.g. `loss/total`, `optim/lr/head`, `time/batch_s`.

Reject empty names, leading/trailing `/`, and `//`.

Use `dims` for series dimensions (not nested maps):

```python
{"dataset": "imagenet_v2", "scope": "reduced"}
```

Suggested `scope` values: `local` | `replicated` | `reduced`.  
`glzn.log` never runs collectives.

### Scalar contract

- **dims**: `None`, `bool`, `int`, finite `float`, `str`, scalar tensor/NumPy.
- **metrics**: `None`, `bool`, `int`, finite `float`, scalar tensor/NumPy.
- No strings in metrics (categoricals → dims).
- No non-scalar tensors, arrays, sequences, nested maps.

### Non-finite encoding

Strict JSON (`allow_nan=False`). Non-finite metrics:

```python
"metrics": {"loss/total": null},
"nonfinite": {"loss/total": "nan"}  # or posinf / neginf
```

Genuine `None` has no `nonfinite` entry. Empty sidecar is omitted.  
Non-finite values are rejected in `dims`.

### `optim/lr` convention

When the processor injects learning rate:

```text
optim/lr = parameter group 0 learning rate
```

Not the unique model-wide LR. Callers may also supply `optim/lr/head`, etc.

---

## Task ownership vs logger ownership

**Task / training loop owns** loss decomposition, entropy/prototype stats, center
norms, effective-batch aggregation, algorithm-required distributed reductions,
and any nonlinear sufficient-statistic metrics.

**The task supplies finalized effective-update metrics.** The processor does
**not** aggregate microbatch metric mappings. Typically the task keeps running
state over the accumulation window and assigns `b.metrics` on the closing
microbatch only.

**`glzn.log` owns** schema construction, scalar normalization (torch/NumPy →
Python), Pydantic validation, routing, JSONL persistence, and stdout presentation.

Update-level losses (e.g. `loss/total`) must be **task-supplied**. The processor
never logs the final microbatch loss tensor as the update loss.

---

## Sinks

### `JSONLSink` (canonical)

Default path:

```text
<root>/<logfoldername>/metrics.jsonl
```

- Rank 0 writes canonical file; other ranks do not by default.
- Optional rank-local: `<root>/<logfoldername>/rank/<rank>.jsonl`.
- Open once with `open(..., "a", encoding="utf-8", buffering=1)`.
- One compact JSON object + `\n` per record; no per-record reopen; no `fsync`.

Line buffering makes newline-terminated records visible after `log()` under
normal process operation and protects against ordinary process loss of the
Python userspace buffer. It does **not** guarantee durability across node or
kernel failure.

### `StdoutSink`

Optional rank-0 compact one-line view of the same normalized record. JSONL
remains canonical.

---

## Usage

```python
from glzn.log import LogCollator
from glzn.proc import ProcDeps, Processor

logger = LogCollator.local(
    run_id="exp001",
    root="/scratch/runs/exp001",
    rank=0,
    world_size=1,
    stdout=True,
)

proc = Processor(ProcDeps(model=model, optimizer=opt), logger=logger)

with proc.batch(tracker=tracker, phase=Phase.TRAIN) as b:
    b.outputs = model(x)
    b.loss = loss_fn(b.outputs, y) / accum_steps
    # Only emitted if this microbatch closes an accumulation window:
    b.metrics = {
        "loss/total": update_level_loss,  # task-defined effective-batch loss
        "time/batch_s": batch_seconds,
    }
    b.dims = {"scope": "local"}
tracker = b.updated_tracker
```

---

## Processor emission rules

- Capture `update_attempted = is_update_step` **before** `on_update` / `next_micro`.
- Emit **exactly one** record when an update is attempted (success or AMP skip).
- Emit **nothing** on non-update accumulation microbatches.
- Do not pass inputs/outputs/targets to the logger; no `.cpu()` of training tensors for logging; no per-microbatch `loss.item()` for durable logs.

---

## Non-goals (this package)

WandB/Aim, upload/replay CLI, schema migrations, DuckDB, async writers,
background samplers, artifacts/images/histograms, reducer registries,
logging-only distributed collectives, automatic task metric computation.
