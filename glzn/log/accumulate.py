from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class WeightedScalar:
    """Accumulate weighted scalar tensors and sync only at finalization."""

    total: Tensor | None = None
    count: int = 0
    _reduced: bool = False

    def add(self, value: Tensor, *, weight: int = 1) -> None:
        if self._reduced:
            raise RuntimeError("Cannot add to WeightedScalar after all_reduce_().")
        if value.ndim != 0:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}.")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        detached = value.detach().float() * float(weight)
        self.total = detached if self.total is None else self.total + detached
        self.count += weight

    def all_reduce_(
        self,
        group: dist.ProcessGroup | None = None,
    ) -> "WeightedScalar":
        """SUM-reduce the accumulated total and count across distributed ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return self
        if self._reduced:
            raise RuntimeError("WeightedScalar.all_reduce_() was already called.")
        if self.total is None:
            self._reduced = True
            return self

        count = torch.tensor(
            self.count,
            dtype=torch.int64,
            device=self.total.device,
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
