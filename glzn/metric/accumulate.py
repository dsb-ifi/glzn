from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import Tensor

from glzn.run import (
    all_reduce_sum,
    collective_count_device,
    collective_device,
    distributed_enabled,
)

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
        if not distributed_enabled():
            return self
        if self._reduced:
            raise RuntimeError("WeightedScalar.all_reduce_() was already called.")

        device = collective_device(self.total, group=group)
        if self.total is None:
            self.total = torch.zeros((), dtype=torch.float32, device=device)
        elif self.total.device != device:
            self.total = self.total.to(device)

        count = torch.tensor(
            self.count,
            dtype=torch.int64,
            device=collective_count_device(device, group=group),
        )
        self.total = all_reduce_sum(self.total, group=group)
        count = all_reduce_sum(count, group=group)
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
        if not distributed_enabled():
            return dict(self.global_counts)

        reference = self._reference_total()
        device = collective_count_device(collective_device(reference, group=group), group=group)
        names = list(self.global_counts)
        counts = torch.tensor(
            [self.global_counts[name] for name in names],
            dtype=torch.int64,
            device=device,
        )
        counts = all_reduce_sum(counts, group=group)
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

