import torch
from typing import Union
from torch.optim.swa_utils import AveragedModel

class EMA(AveragedModel):

    def __init__(self, model, decay:float, device:Union[str, int, torch.device]="cpu"):
        self.decay = decay
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        super().__init__(model, self.device, self.ema_avg, use_buffers=True)

    @torch.no_grad
    def ema_avg(self, avg_model_param, model_param, num_averaged):
        return self.decay * avg_model_param + (1 - self.decay) * model_param
    