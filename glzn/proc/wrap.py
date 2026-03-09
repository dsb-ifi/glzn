import warnings

from typing import Callable

from .step import StepState
from .optim import Optimizer
from .sched import Scheduler
from .ema import EMA


class ScheduledOptimizer:

    def __init__(
        self,
        optimizer:Optimizer, 
        lr_scheduler:Scheduler|None=None,
        wd_scheduler:Scheduler|None=None,
        lr_group_schedulers:dict[str | int, Scheduler] | None=None,
        wd_group_schedulers:dict[str | int, Scheduler] | None=None,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.lr_group_schedulers = lr_group_schedulers or {}
        self.wd_group_schedulers = wd_group_schedulers or {}
        self._base_lrs = [float(group.get('lr', 0.0)) for group in optimizer.param_groups]
        self._base_wds = [float(group.get('weight_decay', 0.0)) for group in optimizer.param_groups]
        self._lr_group_scales = [float(group.get('lr_scale', 1.0)) for group in optimizer.param_groups]
        self._wd_group_scales = [float(group.get('wd_scale', 1.0)) for group in optimizer.param_groups]
        self._group_names = [str(group.get('group_name', f'group_{i}')) for i, group in enumerate(optimizer.param_groups)]
        
        # Check if schedulers are normalized, if not, normalize and reinitialize
        for (sch, name) in [(self.lr_scheduler, "LR"), (self.wd_scheduler, "WD")]:
             if sch is not None and not sch.normalize:
                warnings.warn(f"Unnormalized {name} scheduler detected. Normalizing and reinitializing schedule.")
                sch.normalize = True
                sch.initialize_schedule()

        for sch in self.lr_group_schedulers.values():
            if sch is not None and not sch.normalize:
                warnings.warn("Unnormalized per-group LR scheduler detected. Normalizing and reinitializing schedule.")
                sch.normalize = True
                sch.initialize_schedule()

        for sch in self.wd_group_schedulers.values():
            if sch is not None and not sch.normalize:
                warnings.warn("Unnormalized per-group WD scheduler detected. Normalizing and reinitializing schedule.")
                sch.normalize = True
                sch.initialize_schedule()

    @staticmethod
    def _resolve_group_scheduler(
        schedulers:dict[str | int, Scheduler],
        group_name:str,
        group_index:int,
    ) -> Scheduler | None:
        if group_name in schedulers:
            return schedulers[group_name]
        if group_index in schedulers:
            return schedulers[group_index]
        index_key = str(group_index)
        if index_key in schedulers:
            return schedulers[index_key]
        return None

    def apply(self, step_state:StepState):
        if len(self.optimizer.param_groups) != len(self._base_lrs):
            raise RuntimeError("Optimizer param_groups changed after ScheduledOptimizer initialization.")

        global_lr_factor = self.lr_scheduler(step_state) if self.lr_scheduler is not None else 1.0
        global_wd_factor = self.wd_scheduler(step_state) if self.wd_scheduler is not None else 1.0

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self._base_lrs[i]
            base_wd = self._base_wds[i]
            lr_scale = self._lr_group_scales[i]
            wd_scale = self._wd_group_scales[i]
            group_name = self._group_names[i]

            lr_group_scheduler = self._resolve_group_scheduler(self.lr_group_schedulers, group_name, i)
            wd_group_scheduler = self._resolve_group_scheduler(self.wd_group_schedulers, group_name, i)

            lr_group_factor = lr_group_scheduler(step_state) if lr_group_scheduler is not None else 1.0
            wd_group_factor = wd_group_scheduler(step_state) if wd_group_scheduler is not None else 1.0

            param_group['lr'] = base_lr * global_lr_factor * lr_group_factor * lr_scale
            param_group['weight_decay'] = base_wd * global_wd_factor * wd_group_factor * wd_scale

    def step(self, step_state:StepState):
        self.apply(step_state)
        self.optimizer.step()


class ScheduledEMA:

    def __init__(self, ema:EMA, momentum_scheduler:Scheduler|None=None):
        self.ema = ema
        self._base_momentum = ema.decay
        self.momentum_scheduler = momentum_scheduler
        if self.momentum_scheduler is not None and not self.momentum_scheduler.normalize:
            warnings.warn("Unnormalized momentum scheduler detected. Normalizing and reinitializing schedule.")
            self.momentum_scheduler.normalize = True
            self.momentum_scheduler.initialize_schedule()

    def update_parameters(self, model, step_state:StepState):
        if self.momentum_scheduler is not None:
            self.ema.decay = self.momentum_scheduler(step_state) * self._base_momentum
        self.ema.update_parameters(model)


class ScheduledLoss:

    def __init__(self, loss_fn, loss_scheduler:Scheduler|None=None):
        self.loss_fn = loss_fn
        self.loss_scheduler = loss_scheduler
        if self.loss_scheduler is not None and not self.loss_scheduler.normalize:
            warnings.warn("Unnormalized loss scheduler detected. Normalizing and reinitializing schedule.")
            self.loss_scheduler.normalize = True
            self.loss_scheduler.initialize_schedule()

    def weighted_loss(self, step_state:StepState) -> Callable:
        if self.loss_scheduler is not None:
            loss_weight = self.loss_scheduler(step_state)
        else:
            loss_weight = 1.0
        
        def weighted_loss_fn(*args, **kwargs):
            return self.loss_fn(*args, **kwargs) * loss_weight
        
        return weighted_loss_fn