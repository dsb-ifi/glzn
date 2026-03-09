import torch

from typing import Callable
from .step import StepState

def linear(_:float):
    def _linear(start:float, end:float, num_steps:int):
        return torch.linspace(start, end, num_steps)
    return _linear

def powerlaw(power:float):
    def _powerlaw(start:float, end:float, num_steps:int):
        return start + (end - start) * torch.linspace(0, 1, num_steps) ** power
    return _powerlaw

def cosine(_:float):
    def _cosine(start:float, end:float, num_steps:int):
        return end + (start - end) * (1 + torch.cos(torch.linspace(0, torch.pi, num_steps))) / 2
    return _cosine

def exponential(lambda_:float):
    def _exponential(start:float, end:float, num_steps:int):
        sgn = 1 if end > start else -1
        return start + (end - start) * (1 - torch.exp(sgn * lambda_ * torch.linspace(0, 1, num_steps))) / (1 - torch.tensor(sgn * lambda_).exp())
    return _exponential if lambda_ != 0 else linear(lambda_)

def logistic(lambda_:float):
    _k = torch.pi * lambda_ / 2
    _s = torch.sigmoid(torch.tensor(-_k))
    _d = 1 - 2 * _s
    def _logistic(start:float, end:float, num_steps:int):
        return start + (end - start) * torch.sigmoid(torch.linspace(-_k, _k, num_steps)).sub(_s) / _d
    return _logistic


SchedulerFn = Callable[[float], Callable[[float, float, int], torch.Tensor]]
_SCHEDULER_FNS : dict[str, SchedulerFn] = {
    'linear': linear,
    'powerlaw': powerlaw,
    'cosine': cosine,
    'exponential': exponential,
    'logistic': logistic,
    'none': linear
}


# TODO: Maybe add support for MultiCycleScheduler in the future

class Scheduler:

    def __init__(
        self, 
        total_steps:int, 
        warmup_ratio:float = 5/300,
        base_val:float = 3e-3, 
        start_val:float = 1e-8,
        end_val:float = 1e-7,
        main_schedule:str|SchedulerFn = 'cosine', 
        warmup_schedule:str|SchedulerFn = 'linear',
        normalize:bool = True,
        **kwargs
    ):
        self.base_val = base_val
        self.start_val = start_val
        self.end_val = end_val
        self.total_steps = total_steps
        self.warmup_steps = int(round(total_steps * warmup_ratio)) if warmup_schedule is not None else 0
        self.main_steps = total_steps - self.warmup_steps
        self._max_val = max(base_val, start_val, end_val)
        self.normalize = normalize

        warmup_schedule_lambda = kwargs.get('warmup_schedule_lambda', 2)
        main_schedule_lambda = kwargs.get('main_schedule_lambda', 2)

        self.set_warmup_schedule(warmup_schedule, warmup_schedule_lambda)
        self.set_main_schedule(main_schedule, main_schedule_lambda)
        self.initialize_schedule()

    def initialize_schedule(self):
        self.recalculate_schedule(self.start_val, self.base_val, self.end_val, self.normalize)

    def recalculate_schedule(self, start_val:float, base_val:float, end_val:float, normalize:bool=True):
        _warmup_schedule = self.warmup_fn(start_val, base_val, self.warmup_steps)
        _main_schedule = self.main_fn(base_val, end_val, self.main_steps)
        self.schedule = torch.cat((_warmup_schedule, _main_schedule))
        if normalize:
            self.schedule = self.schedule / self._max_val

    def set_warmup_schedule(self, warmup_schedule:str|SchedulerFn, warmup_schedule_lambda:float = 2):
        if isinstance(warmup_schedule, str):
            assert warmup_schedule in _SCHEDULER_FNS, f"Unsupported warmup schedule: {warmup_schedule}"
            warmup_schedule = _SCHEDULER_FNS[warmup_schedule]
        self.warmup_fn = warmup_schedule(warmup_schedule_lambda)

    def set_main_schedule(self, main_schedule:str|SchedulerFn, main_schedule_lambda:float = 2):
        if isinstance(main_schedule, str):
            assert main_schedule in _SCHEDULER_FNS, f"Unsupported main schedule: {main_schedule}"
            main_schedule = _SCHEDULER_FNS[main_schedule]
        self.main_fn = main_schedule(main_schedule_lambda)

    def __getitem__(self, step:int) -> float:
        if step < self.total_steps:
            return self.schedule[step].item()
        else:
            return self.end_val
    
    def __call__(self, step_state:StepState) -> float:
        step = step_state.fullstep
        return self[step]

