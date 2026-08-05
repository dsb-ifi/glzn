from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    collate_fn: Callable[[list[object]], object] | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Build a DataLoader while respecting worker-dependent PyTorch rules."""

    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers == 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def move_to_device(
    value: object,
    device: torch.device | str,
    *,
    non_blocking: bool,
) -> object:
    """Recursively move tensor leaves while preserving common containers."""

    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: move_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    return value
