from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import Tensor

Scalar = bool | int | float | None


@dataclass
class WeightedScalar:
    """Accumulate weighted scalar tensors and sync only at finalization."""

    total: Tensor | None = None
    count: int = 0
    _reduced: bool = False

    def add(self, value: Tensor, *, weight: int = 1) -> None:
        if value.ndim != 0:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}.")
        self.add_sum(value.detach().float() * float(weight), count=weight)

    def add_sum(self, value_sum: Tensor, *, count: int) -> None:
        if self._reduced:
            raise RuntimeError("Cannot add to WeightedScalar after all_reduce_().")
        if value_sum.ndim != 0:
            raise ValueError(
                f"Expected scalar tensor, got shape {tuple(value_sum.shape)}."
            )
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        detached = value_sum.detach().float()
        self.total = detached if self.total is None else self.total + detached
        self.count += count

    def all_reduce_(
        self,
        group: dist.ProcessGroup | None = None,
    ) -> "WeightedScalar":
        """SUM-reduce the accumulated total and count across distributed ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return self
        if self._reduced:
            raise RuntimeError("WeightedScalar.all_reduce_() was already called.")

        backend = distributed_backend(group)
        device = collective_device(backend, self.total)
        if self.total is None:
            self.total = torch.zeros((), dtype=torch.float32, device=device)
        elif self.total.device != device and backend == "nccl":
            self.total = self.total.to(device)

        count = torch.tensor(
            self.count,
            dtype=torch.int64,
            device=collective_count_device(backend, device),
        )
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
        self.count = int(count.item())
        self._reduced = True
        return self

    def clear(self) -> None:
        self.total = None
        self.count = 0
        self._reduced = False

    def mean_tensor(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / float(self.count)

    def mean(self) -> float:
        return float(self.mean_tensor().cpu())


@dataclass
class MultiDenominatorMetricWindow:
    """Named weighted metrics with explicit local/global diagnostic counts."""

    empty_error: str = "Cannot finalize an empty metric window."
    metrics: dict[str, WeightedScalar] = field(default_factory=dict)
    local_counts: dict[str, int] = field(default_factory=dict)
    global_counts: dict[str, int] = field(default_factory=dict)
    micros: int = 0

    def add(self, name: str, value: Tensor, *, weight: int) -> None:
        metric = self.metrics.setdefault(name, WeightedScalar())
        metric.add(value, weight=weight)

    def add_sum(self, name: str, value_sum: Tensor, *, count: int) -> None:
        metric = self.metrics.setdefault(name, WeightedScalar())
        metric.add_sum(value_sum, count=count)

    def add_local_count(self, name: str, count: int) -> None:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        self.local_counts[name] = self.local_counts.get(name, 0) + count

    def add_global_count(self, name: str, count: int) -> None:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        self.global_counts[name] = self.global_counts.get(name, 0) + count

    def clear(self) -> None:
        self.metrics.clear()
        self.local_counts.clear()
        self.global_counts.clear()
        self.micros = 0

    def finalize(
        self,
        group: dist.ProcessGroup | None = None,
    ) -> dict[str, Scalar]:
        if self.micros == 0:
            raise RuntimeError(self.empty_error)
        out = {name: _finalize_metric(metric, group) for name, metric in self.metrics.items()}
        out.update(self._finalize_global_counts(group))
        out.update(self.local_counts)
        return out

    def _finalize_global_counts(
        self,
        group: dist.ProcessGroup | None,
    ) -> dict[str, int]:
        if not self.global_counts:
            return {}
        if not dist.is_available() or not dist.is_initialized():
            return dict(self.global_counts)

        backend = distributed_backend(group)
        reference = self._reference_total()
        device = collective_count_device(
            backend,
            collective_device(backend, reference),
        )
        names = list(self.global_counts)
        counts = torch.tensor(
            [self.global_counts[name] for name in names],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        return {
            name: int(counts[idx].item())
            for idx, name in enumerate(names)
        }

    def _reference_total(self) -> Tensor | None:
        for metric in self.metrics.values():
            if metric.total is not None:
                return metric.total
        return None


def _finalize_metric(
    metric: WeightedScalar,
    group: dist.ProcessGroup | None,
) -> Scalar:
    metric.all_reduce_(group=group)
    if metric.total is None or metric.count == 0:
        return None
    return metric.mean()


def distributed_backend(group: dist.ProcessGroup | None) -> str:
    try:
        return str(dist.get_backend(group))
    except ValueError:
        return "gloo"


def collective_device(backend: str, reference: Tensor | None) -> torch.device:
    if backend == "nccl":
        if reference is not None and reference.device.type == "cuda":
            return reference.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("NCCL metric reduction requires CUDA.")
    return torch.device("cpu") if reference is None else reference.device


def collective_count_device(
    backend: str,
    preferred: torch.device,
) -> torch.device:
    if backend == "nccl":
        if preferred.type == "cuda":
            return preferred
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("NCCL metric count reduction requires CUDA.")
    return torch.device("cpu")
