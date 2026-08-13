"""Field-aligned and structural transforms for iTar logical field tuples."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence
from typing import Any, Callable


def finalize_fields(result: Any, expected_arity: int) -> tuple[Any, ...]:
    """Normalize field-transform output and enforce logical arity.

    Accepts list/tuple (including namedtuple). Rejects mappings and scalars so
    they are never silently expanded. Arity must equal ``expected_arity``.
    """
    if isinstance(result, tuple):
        fields = result
    elif isinstance(result, list):
        fields = tuple(result)
    elif isinstance(result, (str, bytes, bytearray)) or isinstance(result, ABCMapping):
        raise TypeError(
            "Transform pipeline must return a list or tuple of sample fields; "
            f"got {type(result).__name__}."
        )
    else:
        raise TypeError(
            "Transform pipeline must return a list or tuple of sample fields; "
            f"got {type(result).__name__}. Wrap a single field as a one-element "
            "tuple or list, e.g. ``(value,)``."
        )
    if len(fields) != expected_arity:
        raise ValueError(
            f"Transform pipeline must preserve logical arity {expected_arity}; "
            f"got {len(fields)} fields."
        )
    return fields


class ArityPreserving:
    """Run a field transform then enforce logical arity (immediate check)."""

    def __init__(self, transform: Callable, expected_arity: int):
        if not callable(transform):
            raise TypeError("ArityPreserving transform must be callable.")
        if expected_arity < 1:
            raise ValueError(
                f"expected_arity must be positive, got {expected_arity}."
            )
        self.transform = transform
        self.expected_arity = int(expected_arity)

    def __call__(self, y: Sequence[object]) -> tuple:
        return finalize_fields(self.transform(y), self.expected_arity)


class Map:
    """Joint field map: ``f(*fields) -> N fields`` (arity-preserving)."""

    def __init__(self, mapping: Callable):
        if not callable(mapping):
            raise TypeError("Map mapping must be callable.")
        self.mapping = mapping

    def __call__(self, y: Sequence[object]) -> object:
        return self.mapping(*y)


class MapTuple:
    """Per-position maps: ``(f0(y0), f1(y1), ...)``."""

    def __init__(self, maps: Sequence[Callable]):
        if not all(callable(m) for m in maps):
            raise TypeError("All MapTuple maps must be callable.")
        self.maps = tuple(maps)

    def __call__(self, y: Sequence[object]) -> tuple:
        if len(y) != len(self.maps):
            raise ValueError(
                f"MapTuple expected {len(self.maps)} fields, got {len(y)}."
            )
        return tuple(fn(x) for fn, x in zip(self.maps, y))


class MapAll:
    """Apply one map to every field: ``(f(y0), f(y1), ...)``."""

    def __init__(self, mapping: Callable):
        if not callable(mapping):
            raise TypeError("MapAll mapping must be callable.")
        self.mapping = mapping

    def __call__(self, y: Sequence[object]) -> tuple:
        return tuple(self.mapping(x) for x in y)


class MapGrouped:
    """Apply a joint map to selected indices; leave others unchanged."""

    def __init__(self, mapping: Callable, indices: Sequence[int]):
        if not callable(mapping):
            raise TypeError("MapGrouped mapping must be callable.")
        if not all(isinstance(i, int) for i in indices):
            raise TypeError("MapGrouped indices must be integers.")
        self.mapping = mapping
        self.indices = tuple(indices)

    def __call__(self, y: Sequence[object]) -> tuple:
        grouped_inputs = tuple(y[i] for i in self.indices)
        transformed_outputs = self.mapping(*grouped_inputs)
        if not isinstance(transformed_outputs, tuple):
            transformed_outputs = (transformed_outputs,)
        if len(transformed_outputs) != len(self.indices):
            raise ValueError(
                f"map_group expected {len(self.indices)} outputs for indices "
                f"{self.indices}, got {len(transformed_outputs)}."
            )
        output = list(y)
        for idx, value in zip(self.indices, transformed_outputs):
            output[idx] = value
        return tuple(output)


class CollapseArity:
    """Structural grouping of logical fields into a lower-arity sample.

    Each group becomes one nested tuple; group order is preserved. This is
    intentionally *not* a field map — it changes logical arity.
    """

    def __init__(self, groups: Sequence[Sequence[int]]):
        cleaned: list[tuple[int, ...]] = []
        for group in groups:
            cleaned.append(tuple(int(i) for i in group))
        self.groups = tuple(cleaned)

    def __call__(self, y: Sequence[object]) -> tuple:
        return tuple(tuple(y[i] for i in group) for group in self.groups)


class DefaultIdentity:
    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        return args
