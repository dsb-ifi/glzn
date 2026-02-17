import os, time, json, torch, logging
from typing import Sequence

from .loggers import (
    AbstractLogger, ProgressLogger, LossLogger, LRLogger, 
    GPULogger, DeltaTimeLogger, AccuracyLogger
)

LoggerSequence = Sequence[AbstractLogger]


def _getrunlog():
    logger = logging.getLogger('glzn.log')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logfmt = logging.Formatter('%(levelname)s:%(asctime)s | %(message)s')
    ch.setFormatter(logfmt)
    logger.addHandler(ch)
    return logger


applog = _getrunlog()


class LogCollator:

    def __init__(
        self,
        runid:str, # TODO: We probably don't need this to be required.
        root:str,
        rank:int,
        local_rank:int,
        loggers:LoggerSequence,
        logfoldername:str='log',
        stdout:bool=False,
        drop_debug_entries:Sequence[str]=['time', 'cfg'],
    ):
        self.runid = runid
        self.loggers = loggers
        self.save_folder = os.path.join(root, logfoldername)
        self.file_name = f"{runid}_{rank}_{local_rank}.jsonl"
        self.rank = rank
        self.local_rank = local_rank
        self.stdout = stdout
        self.drop_debug_entries = drop_debug_entries
        os.makedirs(self.save_folder, exist_ok=True)
        self.file_path = os.path.join(self.save_folder, self.file_name)

    @property
    def _use_applog(self) -> bool:
        return self.stdout and self.rank == 0 and self.local_rank == 0

    def get_entries(self, **logging_kwargs):
        def t2item(val):
            if torch.is_tensor(val):
                if val.numel() == 1:
                    return val.item()
                return val.tolist()
            return val
        return {
            k:t2item(v) for logger in self.loggers 
            for k, v in logger(**logging_kwargs).items()
        }

    def __call__(self, **logging_kwargs):
        timestamp = logging_kwargs.get('time', time.time())
        log_entry = {
            'time': timestamp, 
            **self.get_entries(**logging_kwargs)
        }
        mode = 'a' if os.path.isfile(self.file_path) else 'w'
        if self._use_applog:
            applog.debug(
                ' '.join([
                    f"{k}={v}" for k,v in log_entry.items() 
                    if k not in self.drop_debug_entries
                ])
            )
        with open(self.file_path, mode) as log_file:
            log_file.write(json.dumps(log_entry))
            log_file.write('\n')

    @classmethod
    def standard_logger(
        cls, runid:str, root:str, rank:int, local_rank:int,
        stdout:bool, logfoldername:str='log'
    ):
        loggers = [
            ProgressLogger(), DeltaTimeLogger(), LossLogger(), 
            AccuracyLogger(1), AccuracyLogger(5), LRLogger(),
            GPULogger()
        ]
        return cls(
            runid, root, rank, local_rank, loggers, 
            logfoldername, stdout
        )