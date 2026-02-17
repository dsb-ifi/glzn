from __future__ import annotations
import argparse, json, os, sys, re, toml, yaml, inspect
from pathlib import Path
from typing import List, Literal, Optional, TypeVar
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError, field_validator, model_validator, ValidationInfo
from numpydoc.docscrape import NumpyDocString

from .typeparse import _argparse_meta, _is_nullable, _NULL_STRINGS

T_BaseCfg = TypeVar("T_BaseCfg", bound="BaseConfig")

class BaseConfig(BaseSettings):
    '''Base configuration for GLZN runs.

    Configuration hierarchy (later sources override earlier ones):
    1. Default values in field definitions
    2. Environment variables (case insensitive, no prefix)
    3. Configuration files (YAML/TOML/JSON)
    4. Command line arguments

    Attributes
    ----------
    model : str
        Name or path of model to use.
    data_path : str
        Path to dataset directory.
    dataset : str
        Name of dataset to use.
    cfgfile : List[Path], optional
        Paths to config files (JSON/YML/TOML) with overrides.
    savedir : Path, optional
        Directory for checkpoints and logs (default: $HOME).
    device : {'cuda', 'cpu'}, optional
        Compute device (default: 'cuda').
    ddp_backend : str, optional
        DDP backend to use (default: 'nccl').
    ddp_url : str, optional
        URL for DDP process group (default: 'env://').
    ddp_master_address : str, optional
        Master node address (env: MASTER_ADDR).
    ddp_master_port : str, optional
        Master node port (env: MASTER_PORT).
    world_size : int, optional
        Total number of processes (env: WORLD_SIZE).
    rank : int, optional
        Global rank of current process (env: RANK).
    local_world_size : int, optional
        Number of processes on this node (env: LOCAL_WORLD_SIZE).
    local_rank : int, optional
        Local rank on this node (env: LOCAL_RANK).
    custom_run_id : str, optional
        Custom identifier for this run (env: RUN_ID).
    custom_project_id : str, optional
        Project identifier for logging (env: PROJECT_ID).
    dtype : {'float32', 'bfloat16', 'float16'}, optional
        Data type for model parameters (default: 'float32').
    '''
    model:str           = Field(..., description='Name or path of model to use.')
    data_path:str       = Field(..., description='Path to dataset directory.')
    dataset:str         = Field(..., description='Name of dataset to use.')
    cfgfile:List[Path]  = Field(
        default_factory=list, alias='CFGFILE',
        description='Paths to config files (JSON/YML/TOML) with overrides.'
    )
    savedir:Path = Field(
        default_factory=Path.home,
        description='Directory for checkpoints and logs (default: $HOME).'
    )
    device:Literal['cuda', 'cpu']       = Field(
        'cuda', description='Compute device (default: `cuda`).'
    )
    ddp_backend:str                     = Field(
        'nccl', description='DDP backend to use (default: `nccl`).'
    )
    ddp_url:str                         = Field(
        'env://', description='URL for DDP process group (default: `env://`).'
    )
    ddp_master_address:Optional[str]    = Field(
        None, alias='MASTER_ADDR', description='Master node address (env: MASTER_ADDR).',
    )
    ddp_master_port:Optional[str]       = Field(
        None, alias='MASTER_PORT', description='Master node port (env: MASTER_PORT)',
    )
    world_size:Optional[int]            = Field(
        None, alias='WORLD_SIZE', description='Total number of processes (env: WORLD_SIZE).',
    )
    rank:Optional[int]                  = Field(
        None, alias='RANK', description='Global rank of current process (env: RANK).',
    )
    local_world_size:Optional[int]      = Field(
        None, alias='LOCAL_WORLD_SIZE', description='Number of processes on this node (env: LOCAL_WORLD_SIZE).',
    )
    local_rank:Optional[int]            = Field(
        None, alias='LOCAL_RANK', description='Local rank on this node (env: LOCAL_RANK).',
    )
    custom_run_id:Optional[str]         = Field(
        None, alias='RUN_ID', description='Custom identifier for this run (env: RUN_ID).',
    )
    custom_project_id:Optional[str]     = Field(
        None, alias='PROJECT_ID', description='Project identifier for logging (env: PROJECT_ID).',
    )
    dtype:Literal['float32', 'bfloat16', 'float16'] = Field(
        'float32', description='Data type for model parameters (default: `float32`).'
    )

    # Pydantic settings configuration
    model_config = {
        'env_prefix': '', 
        'case_sensitive': False, 
        'extra': 'forbid',
        'populate_by_name':True,
    }

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_strings(cls, data: dict):
        """
        Convert values like "none" or "null" to actual None **iff**
        the corresponding field is allowed to be None.
        """
        for name, value in list(data.items()):
            if not (isinstance(value, str) and value.lower() in _NULL_STRINGS):
                continue
            f = cls.model_fields.get(name)
            if f and _is_nullable(f):
                data[name] = None
        return data   

    @classmethod
    def _apply_doc_descriptions(cls) -> None:
        """Fill in Field.description from the class’ NumPy docstring."""
        raw = inspect.getdoc(cls) or ""
        parsed = NumpyDocString(raw)
        attr_map = {n: " ".join(d).strip() for n, _t, d in parsed["Attributes"]}

        for name, field in cls.model_fields.items():
            if not field.description and name in attr_map:
                field.description = attr_map[name]

    @field_validator('cfgfile', mode='before')
    @classmethod
    def _split_env_cfg(cls, v, info: ValidationInfo):
        if isinstance(v, str):
            return [Path(p.strip()) for chunk in v.split(',') for p in chunk.split()]
        return v

    @classmethod
    def parse(cls:type[T_BaseCfg], argv: list[str] | None = None) -> T_BaseCfg:
        cls._apply_doc_descriptions()
        cli_dict = cls._parse_cli(sys.argv[1:] if argv is None else argv)

        env_cfg_paths = cls._cfgfiles_from_env()
        cli_cfg_paths = cli_dict.get('cfgfile') or []
        cfg_paths = [*env_cfg_paths, *cli_cfg_paths] or None

        file_dict = cls._load_cfg_files(cfg_paths)
        merged = {**file_dict, **cli_dict}
        if cfg_paths is not None:
            merged['cfgfile'] = cfg_paths
        try:
            return cls(**merged)
        except ValidationError as e:
            print('Config validation failed:\n', file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _cfgfiles_from_env() -> list[Path]:
        return [Path(p) for p in re.split(r'[,\s]+', os.getenv('CFGFILE', '').strip()) if p]

    @staticmethod
    def _load_cfg_files(paths: list[Path] | None) -> dict:
        merged: dict = {}
        if paths is None:
            return merged
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            data = (
                yaml.safe_load if p.suffix in {'.yml', '.yaml'}
                else toml.loads if p.suffix == '.toml'
                else json.loads
            )(p.read_text())
            if not isinstance(data, dict):
                raise TypeError(f'{p} did not parse to a mapping')
            merged.update(data)
        return merged

    @classmethod
    def _parse_cli(cls, argv: list[str]) -> dict:
        parser = argparse.ArgumentParser()
        for name, field in cls.model_fields.items():
            arg_name = f"--{name.replace('_', '-')}"
            meta_kw, _ = _argparse_meta(field.annotation)
            kw = {"help": field.description or "", **meta_kw}

            # NOTE: We do not let argparse inject defaults for absent flags.
            # Requiredness is enforced after merging all sources (Pydantic).
            kw["default"] = argparse.SUPPRESS

            parser.add_argument(arg_name, **kw)

        ns = parser.parse_args(argv)
        return vars(ns)