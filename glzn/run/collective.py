from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_backend(group: dist.ProcessGroup | None = None) -> str:
    if not distributed_enabled():
        return "gloo"
    try:
        return str(dist.get_backend(group))
    except ValueError:
        return "gloo"


def collective_device(
    reference: Tensor | None = None,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.device:
    if not distributed_enabled():
        return torch.device("cpu") if reference is None else reference.device
    backend = distributed_backend(group)
    if backend == "nccl":
        if reference is not None and reference.device.type == "cuda":
            return reference.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("NCCL collectives require CUDA tensors.")
    return torch.device("cpu") if reference is None else reference.device


def collective_count_device(
    preferred: torch.device,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.device:
    if not distributed_enabled():
        return preferred
    backend = distributed_backend(group)
    if backend == "nccl":
        if preferred.type == "cuda":
            return preferred
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("NCCL count collectives require CUDA tensors.")
    return torch.device("cpu")


def all_reduce_sum(
    value: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    if not distributed_enabled():
        return value
    device = collective_device(value, group=group)
    reduced = value if value.device == device else value.to(device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
    return reduced
