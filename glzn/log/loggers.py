import torch, time
from typing import Dict, Any, Sequence, Dict

StrDict = Dict[str, Any]

class AbstractLogger:

    def __init__(self, keywords:Sequence[str], *args, **kwargs):
        self.keywords = keywords

    def _filterfn(self, pair) -> bool:
        key, _ = pair
        if key in self.keywords:
            return True
        return False

    def filter_logs(self, **logging_kwargs) -> StrDict:
        return dict(filter(self._filterfn, logging_kwargs.items()))
    
    def __call__(self, **logging_kwargs) -> StrDict:
        return self.log(**self.filter_logs(**logging_kwargs))

    def log(self, **logging_kwargs) -> StrDict:
        return logging_kwargs
    

class ProgressLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__([
            'STATUS', 'epoch', 'iteration', 'training', 
            'step_skipped', 'cfg'
        ])


class LossLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['loss'])


class LRLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['last_lr'])


class GPULogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__([])
    
    @staticmethod
    def getmem(d=None) -> float:
        '''Returns reserved memory on device.

        Args:
            d (torch.device): A torch device.
        
        Returns:
            float: Currently reserved memory on device.
        '''
        if d is not None and d.type == 'cpu':
                return 0
        a, b = torch.cuda.mem_get_info(d)
        return (b-a) / (1024**2)
    
    def log(self, **logging_kwargs):
        return {'gpumem': self.getmem()}


class AccuracyLogger(AbstractLogger):

    def __init__(self, top_k:int, *args, **kwargs):
        super().__init__(['outputs', 'targets'])
        self.logname = f'Acc{top_k}'
        self.top_k = top_k

    def unpack(self, name, dict):
        val = dict.get(name, None)
        if isinstance(val, tuple) or isinstance(val, list):
            if len(val) != 1:
                raise ValueError(
                    f'AccuracyLogger expects single output and target tensor. '
                    f'Got {type(val)} of length {len(val)}.'
                )
            return self.unpack(name, {name:val[0]})
        return val

    def log(self, **logging_kwargs) -> StrDict:
        outputs = self.unpack('outputs', logging_kwargs)
        targets = self.unpack('targets', logging_kwargs)
        if outputs is None and targets is None:
            return {}
        if not torch.is_tensor(outputs) or not torch.is_tensor(targets):
            raise ValueError(
                'AccuracyLogger expects single output and target tensor. '
                f'Got {type(outputs)=} {type(targets)=}'
            )
        if not outputs.shape[0] == targets.shape[0]:
            raise ValueError(
                'AccuracyLogger expects input and output tensor of same batch dimension. '
                f'Got {outputs.shape=} {targets.shape=}'
            )
        nb = len(targets)
        if targets.ndim == 1:
            labelidx = targets[:,None]
        else:
            labelidx = targets.topk(1, dim=-1).indices

        acc = (outputs.topk(self.top_k, dim=-1).indices == labelidx).count_nonzero().div(nb)
        return {self.logname: acc}
    

class DeltaTimeLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['time'])
        self.logname = 'timedelta'
        self.prev_time = time.time()

    def log(self, **logging_kwargs):
        cur_time = logging_kwargs.get('time', None)
        if cur_time is None:
            return {}
        delta_time = cur_time - self.prev_time
        self.prev_time = cur_time
        return {self.logname: delta_time}
    