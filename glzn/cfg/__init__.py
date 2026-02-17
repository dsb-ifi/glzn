from importlib.metadata import version as _v
from .runcfg import RunConfig
from .basecfg import BaseConfig

__all__ = ["RunConfig", "BaseConfig"]
# __version__ = _v("glzn")
del _v