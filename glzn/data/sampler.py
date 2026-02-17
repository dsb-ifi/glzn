import random
import numpy as np
from typing import Sequence

# TODO: Add fuzzy testing, random N, random rounds, random seed.

class FeistelSampler:
    
    def __init__(self, N:int, rounds:int=3, init_seed:int=0):
        self.N, self.rounds = N, rounds
        w = (N - 1).bit_length()
        self.half = (w + 1) // 2
        self.mask = (1 << self.half) - 1
        self.refesh_keys(init_seed)

    def refesh_keys(self, seed):
        rng = random.Random(seed)
        r = self.rounds
        bits = self.half
        mask = (1 << bits) - 1
        self.keys = np.array(
            [rng.getrandbits(bits) & mask for _ in range(r)], 
            dtype=np.uint64
        )
    
    def _genperm(self, i:int) -> int:
        N, half, mask, keys, F = self.N, self.half, self.mask, self.keys, self._F
        L, R = i >> half, i & mask
        for k in keys:
            L, R = R, L ^ F(R, int(k))
        y = (L << half) | R

        # Cycle-walk for miss
        while y >= N:
            L, R = y >> half, y & mask
            for k in keys:
                L, R = R, L ^ F(R, int(k))
            y = (L << half) | R
        return y

    def _genpermvec(self, a:np.ndarray) -> np.ndarray:
        assert isinstance(a, np.ndarray)
        assert a.ndim == 1
        assert a.dtype == np.uint64
        L = a >> self.half
        R = a & self.mask
        for k in self.keys:
            T = self._F(R, k)
            L, R = R, L ^ T
        y = (L << self.half) | R
        over = y >= self.N
        while over.any():
            o = y[over]
            L = o >> self.half
            R = o & self.mask
            for k in self.keys:
                T = self._F(R, k)
                L, R = R, L ^ T
            y[over] = (L << self.half) | R
            over = y >= self.N
        return y

    def _F(self, x, k):
        # Fast bijective mixing on half-words
        return (x * 0x9E3779B97F4A7C15 + k) & self.mask

    def __getitem__(self, i:int) -> int:
        return self._genperm(i)

    def __iter__(self):
        for i in range(self.N):
            yield self._genperm(i)

    def __len__(self):
        return self.N

    def randperm(self) -> np.ndarray:
        return self._genpermvec(np.arange(self.N, dtype=np.uint64))


class JointFeistelSampler:

    def __init__(self, Ns:Sequence[int], rounds:int=3, init_seed:int=0):
        self.rounds = rounds
        self.refresh_samplers(Ns, init_seed, rounds)

    def refresh_samplers(self, Ns:Sequence[int], init_seed:int, rounds:int|None=None):
        seeds = [init_seed + i for i in range(len(Ns))]
        rounds = self.rounds if rounds is None else rounds
        self.cums = np.cumsum(Ns)
        self.samplers = [
            FeistelSampler(N, rounds, seed) for N, seed in zip(Ns, seeds)
        ]

    def _bucket(self, i:int) -> int:
        cums = self.cums
        if i < 0 or i >= cums[-1]:
            raise IndexError(f"Index {i} out of bounds for cumulative sizes {cums}")
        return int(np.searchsorted(cums, i, side='right') - 1)

    def __getitem__(self, i:int) -> int:
        j = self._bucket(i)
        local_d = self.cums[j-1] if j > 0 else 0
        return self.samplers[j][i - local_d] + local_d

    def randperm(self) -> np.ndarray:
        return np.concatenate([
            sampler.randperm() + np.uint64(self.cums[i-1] if i > 0 else 0)
            for i, sampler in enumerate(self.samplers)
        ])


class ShardSampler:

    def __init__(
        self, 
        shuffle:bool,
        shuffle_rows:bool,
        shuffle_mixing:bool,
        shard_bincount:Sequence[int],
        init_bucketsize:int|None
    ):
        self.shuffle = shuffle
        self.shuffle_rows = shuffle_rows and shuffle
        self.shuffle_mixing = shuffle_mixing and shuffle
        self.shard_bincount = shard_bincount

        self.N = sum(shard_bincount)
        self.E = self.N / len(shard_bincount)
        self.bucketsize = None
        if init_bucketsize is None and shuffle_mixing:
            self.bucketsize = int(round(self.E / 2))

        

