from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class WeightedScalar:
    """Accumulate weighted scalar tensors and sync only at finalization."""

    total: Tensor | None = None
    count: int = 0

    def add(self, value: Tensor, *, weight: int = 1) -> None:
        if value.ndim != 0:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}.")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        detached = value.detach().float() * float(weight)
        self.total = detached if self.total is None else self.total + detached
        self.count += weight

    def clear(self) -> None:
        self.total = None
        self.count = 0

    def mean_tensor(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / float(self.count)

    def mean(self) -> float:
        return float(self.mean_tensor().cpu())
