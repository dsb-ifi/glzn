from __future__ import annotations
from typing import Optional, Literal
from .basecfg import BaseConfig

class RunConfig(BaseConfig):
    '''Standard GLZN RunConfig.

    Attributes
    ----------
    batch_size : int
        Batch size for training. 
    epochs : int
        Number of training epochs. 
    start_epoch : int
        Start training from given epoch
    run_train : bool
        Flag for performing training.
    run_test : bool
        Flag for performing validation.
    detect_anomaly : bool
        Set anomaly detection, for debugging.
    num_workers : int
        Number of workers for data loading.
    prefetch_factor : int
        Prefetch factor for data loading.
    loader_drop_last : bool
        Whether to drop the last incomplete batch in data loaders.
    shuffle_train : bool
        Whether to shuffle training data.
    input_extensions : list[str]
        File extensions to consider for input data.
    target_extensions : list[str]
        File extensions to consider for target data.
    optimizer : str
        Optimizer type for training.
    base_lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    norm_decay : float
        Normalization decay for the optimizer.
    momentum : float
        Momentum for the optimizer.
    opt_epsilon : float
        Epsilon for the optimizer.
    opt_betas : tuple[float, float]
        Betas for the optimizer.
    grad_clip : float
        Gradient clipping value.
    accumulation_steps : int
        Number of steps for gradient accumulation.
    lr_scheduler : str
        Learning rate scheduler type.
    lr_scheduler_warmup_type : str
        Type of learning rate warmup schedule.
    lr_scheduler_warmup_epochs : int
        Number of warmup epochs for the learning rate scheduler.
    lr_stop : float
        Final learning rate for the scheduler.
    lr_start : float
        Initial learning rate for the scheduler.
    llrd : float
        Learning rate decay factor for the scheduler.
    wd_scheduler : str
        Weight decay scheduler type.
    wd_scheduler_warmup_type : str
        Type of weight decay warmup schedule.
    wd_scheduler_warmup_epochs : int
        Number of warmup epochs for the weight decay scheduler.
    wd_stop : float
        Final weight decay for the scheduler.
    wd_start : float
        Initial weight decay for the scheduler.
    from_checkpoint : str | None
        Path to checkpoint or pretrained weights.
    load_model_only : bool
        Flag to load only model weights from checkpoint.
    num_rolling_checkpoints : int
        Number of rolling checkpoints to keep.
    logging_interval : int
        Interval for logging training progress (in iterations).
    '''
    batch_size:int = 256
    epochs:int = 300
    start_epoch:int = 0
    run_train:bool = True
    run_test:bool = True
    detect_anomaly:bool = False
    num_workers:int = 4
    prefetch_factor:int = 2
    loader_drop_last:bool = True
    shuffle_train:bool = True
    input_extensions:list[str] = ['jpg']
    target_extensions:list[str] = ['cls']
    optimizer:Literal['adamw','sgd','lamb','cadamw'] = 'adamw'
    base_lr:float = 1e-3
    weight_decay:float = 1e-2
    norm_decay:float = 1e-2
    momentum:float = 0.9
    opt_epsilon:float = 1e-8
    opt_betas:tuple[float, float] = (0.9, 0.999)
    grad_clip:float = 3.0
    accumulation_steps:int = 1
    lr_scheduler:Optional[Literal['cosine','linear']] = 'cosine'
    lr_scheduler_warmup_type:Literal['linear','cosine'] = 'linear'
    lr_scheduler_warmup_epochs:int = 5
    lr_stop:float = 1e-7
    lr_start:float = 1e-6
    llrd:float = 0.0
    wd_scheduler:Optional[Literal['cosine','linear', None]] = None
    wd_scheduler_warmup_type:Literal['linear','cosine'] = 'linear'
    wd_scheduler_warmup_epochs:int = 0
    wd_stop:float = 1e-4
    wd_start:float = 1e-3
    from_checkpoint:str | None = None
    load_model_only:bool = False
    num_rolling_checkpoints:int = 5
    logging_interval:int = 100

