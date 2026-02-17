from typing import Callable, Sequence

class MapTuple:

    def __init__(self, maps):
        assert all([isinstance(m, Callable) for m in maps])
        self.maps = maps

    def fnapply(self, fn, x):
        return fn(x)

    def __call__(self, y):
        return list(map(self.fnapply, self.maps, y))


class MapAll:

    def __init__(self, mapping):
        assert isinstance(mapping, Callable)
        self.mapping = mapping

    def __call__(self, y):
        return list(map(self.mapping, y))


class MapGrouped:

    def __init__(self, mapping: Callable, indices: Sequence[int]):
        assert isinstance(mapping, Callable)
        assert all(isinstance(i, int) for i in indices)
        self.mapping = mapping
        self.indices = tuple(indices)

    def __call__(self, y):
        grouped_inputs = tuple(y[i] for i in self.indices)
        transformed_outputs = self.mapping(*grouped_inputs)
        if not isinstance(transformed_outputs, tuple):
            transformed_outputs = (transformed_outputs,)
        output = list(y)
        for idx, value in zip(self.indices, transformed_outputs):
            output[idx] = value
        return tuple(output)


class DefaultIdentity:

    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        return args

