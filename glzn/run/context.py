from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .collective import collective_device

T = TypeVar("T")


@dataclass(frozen=True)
class DDPConfig:
    find_unused_parameters: bool = False


@dataclass
class RunContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool = False

    @classmethod
    def from_config(cls, cfg: object) -> "RunContext":
        env_rank = _optional_env_int("RANK")
        env_world_size = _optional_env_int("WORLD_SIZE")
        env_local_rank = _optional_env_int("LOCAL_RANK")
        rank = _resolve_cfg_env_int("rank", getattr(cfg, "rank", None), env_rank, default=0)
        world_size = _resolve_cfg_env_int(
            "world_size",
            getattr(cfg, "world_size", None),
            env_world_size,
            default=1,
        )
        local_rank = _resolve_cfg_env_int(
            "local_rank",
            getattr(cfg, "local_rank", None),
            env_local_rank,
            default=rank,
        )
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must lie in [0, {world_size}), got {rank}.")

        device = _resolve_device(str(getattr(cfg, "device", "cpu")))
        if device.type == "cuda" and world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

        if world_size == 1:
            return cls(
                enabled=False,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                device=device,
            )
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable but world_size > 1.")

        owns_group = False
        if not dist.is_initialized():
            dist.init_process_group(
                backend=str(getattr(cfg, "ddp_backend")),
                init_method=str(getattr(cfg, "ddp_url", "env://")),
                rank=rank,
                world_size=world_size,
            )
            owns_group = True
        elif dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError(
                "Initialized process group disagrees with config/env: "
                f"dist rank/world={dist.get_rank()}/{dist.get_world_size()}, "
                f"cfg/env rank/world={rank}/{world_size}."
            )
        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            owns_process_group=owns_group,
        )

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.owns_process_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def resolve_run_id(
        self,
        *,
        run_name: str,
        custom_run_id: str | None,
        now_fn: Callable[[], float] = time.time,
    ) -> str:
        if not self.enabled:
            return custom_run_id or f"{run_name}-{int(now_fn())}"
        if custom_run_id is not None:
            values: list[str | None] = [None for _ in range(self.world_size)]
            dist.all_gather_object(values, custom_run_id)
            if any(value != custom_run_id for value in values):
                raise RuntimeError(f"custom_run_id differs across ranks: {values}.")
            return custom_run_id
        objects: list[str | None] = [
            f"{run_name}-{int(now_fn())}" if self.rank == 0 else None
        ]
        dist.broadcast_object_list(objects, src=0)
        run_id = objects[0]
        if not isinstance(run_id, str):
            raise RuntimeError("Failed to broadcast run_id from rank 0.")
        return run_id

    def prepare(
        self,
        model: nn.Module,
        ddp: DDPConfig | None = None,
    ) -> nn.Module:
        if not self.enabled:
            return model
        ddp = ddp or DDPConfig()
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=ddp.find_unused_parameters,
            )
        return DistributedDataParallel(
            model,
            find_unused_parameters=ddp.find_unused_parameters,
        )

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def require_equal(self, label: str, **values: int) -> None:
        if not self.enabled:
            return
        names = list(values)
        local = torch.tensor(
            [values[name] for name in names],
            dtype=torch.int64,
            device=collective_device(None),
        )
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        table = torch.stack(gathered)
        if torch.all(table == table[0]).item():
            return
        rows = [
            {
                name: int(table[rank, idx].item())
                for idx, name in enumerate(names)
            }
            for rank in range(self.world_size)
        ]
        raise RuntimeError(
            f"{label} must match across ranks. Uneven-input DDP/join semantics "
            f"are not implemented. Per-rank values: {rows}."
        )

    def rank_zero(self, operation: str, fn: Callable[[], T]) -> T | None:
        result: T | None = None
        local_error: str | None = None
        local_exc: BaseException | None = None
        if self.rank == 0:
            try:
                result = fn()
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
                local_exc = exc
        failures = self.collect_failures(local_error)
        self.raise_failures(operation, failures, cause=local_exc if self.rank == 0 else None)
        return result

    def all_ranks(self, operation: str, fn: Callable[[], T]) -> T | None:
        result: T | None = None
        local_error: str | None = None
        local_exc: BaseException | None = None
        try:
            result = fn()
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            local_exc = exc
        failures = self.collect_failures(local_error)
        self.raise_failures(operation, failures, cause=local_exc)
        return result

    def collect_failures(self, local_error: str | None) -> list[tuple[int, str]]:
        if not self.enabled:
            return [] if local_error is None else [(self.rank, local_error)]
        gathered: list[tuple[int, str] | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            gathered,
            None if local_error is None else (self.rank, local_error),
        )
        return [failure for failure in gathered if failure is not None]

    @staticmethod
    def raise_failures(
        operation: str,
        failures: Sequence[tuple[int, str]],
        *,
        cause: BaseException | None = None,
    ) -> None:
        if not failures:
            return
        details = "; ".join(f"rank {rank}: {message}" for rank, message in failures)
        error = RuntimeError(f"{operation} failed on distributed rank(s): {details}")
        if cause is not None:
            raise error from cause
        raise error


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device='cuda' but CUDA is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("cfg.device='mps' but MPS is not available.")
    return torch.device(name)


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value is None else int(value)


def _resolve_cfg_env_int(
    name: str,
    cfg_value: int | None,
    env_value: int | None,
    *,
    default: int,
) -> int:
    if cfg_value is not None and env_value is not None and cfg_value != env_value:
        raise RuntimeError(
            f"{name} config value {cfg_value} disagrees with environment value {env_value}."
        )
    if env_value is not None:
        return env_value
    if cfg_value is not None:
        return cfg_value
    return default
