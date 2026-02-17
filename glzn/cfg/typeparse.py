from __future__ import annotations
from typing import Union, get_origin, get_args, Tuple, Literal, List
from pydantic.fields import FieldInfo
from argparse import BooleanOptionalAction
from pathlib import Path

_PRIMITIVES = {str, int, float, Path}
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "nan"}

def _unwrap_optional(ann:FieldInfo) -> Tuple[FieldInfo, bool]:
    if get_origin(ann) is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def _is_nullable(ann:FieldInfo) -> bool:
    if ann is None:
        return True
    if get_origin(ann) is Union:
        return type(None) in get_args(ann)
    return False

def _argparse_meta(ann):
    """
    Given a field annotation, return (argparse_kwargs, pydantic_target_type).
    argparse_kwargs contains only the things argparse needs to know *upfront*
    (`type`, `nargs`, `choices`, `action`).  Everything else is deferred to
    Pydantic for validation.
    """
    kw = {}
    origin = get_origin(ann)
    
    if origin in (list, List):
        elem_type, _ = _unwrap_optional(get_args(ann)[0])
        kw["nargs"] = "*"
        kw["type"]  = elem_type if elem_type in _PRIMITIVES else str
        return kw, ann

    if origin in (tuple, Tuple) and len(get_args(ann)) == 2 and get_args(ann)[1] is ...:
        # Tuple[T, ...]  (variable length) → let argparse collect many, join w/ ','
        elem_type, _ = _unwrap_optional(get_args(ann)[0])
        kw["nargs"] = "*"
        kw["type"]  = elem_type if elem_type in _PRIMITIVES else str
        return kw, ann

    if origin is Literal:
        choices = list(get_args(ann))
        if _is_nullable(ann):
            choices += list(_NULL_STRINGS)
        # unwrap None:  Optional[Literal[...]] already handled earlier
        if all(isinstance(c, str) for c in choices):
            kw["choices"] = choices
            kw["type"]    = str
        elif all(isinstance(c, int) for c in choices):
            kw["choices"] = choices
            kw["type"]    = int
        else:
            kw["choices"] = choices
            # mixed types → leave argparse type=str and let Pydantic figure it out
        return kw, ann

    if origin is Union:
        args = set(get_args(ann))
        args.discard(type(None))
        if len(args) == 1:
            # Union[int] degenerated → treat like plain int
            return _argparse_meta(args.pop())
        if args <= _PRIMITIVES:
            # heterogeneous primitives → accept str; Pydantic will cast/validate
            kw["type"] = str
        return kw, ann

    if ann is bool:
        kw["action"] = BooleanOptionalAction
    elif ann in _PRIMITIVES:
        kw["type"] = ann

    return kw, ann

