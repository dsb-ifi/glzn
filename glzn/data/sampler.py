import random
import numpy as np
from typing import Sequence

# TODO: Add fuzzy testing, random N, random rounds, random seed.

def __F(x, k):
    # Fast bijective mixing on half-words
    return (x * 0x9E3779B97F4A7C15 + k) & mask

def feistel(i:int, N:int, half:int, mask:int, keys:np.ndarray):
    '''Feistel permutation for a single integer.
    
    Parameters
    ----------
    i : int
        The input integer to permute. Must be in the range [0, N-1].
    N : int
        The size of the permutation. Must be a positive integer.
    half : int
        The number of bits in the half-word. Must be a positive integer.
    mask : int
        The bitmask for the half-word. Must be (1 << half) - 1.
    keys : np.ndarray
        The round keys for the Feistel network. Must be a 1D array of integers with length equal to the number of rounds.
    
    Returns
    -------
    int
        The permuted integer in the range [0, N-1].
    '''
    L, R = i >> half, i & mask
    for k in keys:
        L, R = R, L ^ __F(R, int(k))
    y = (L << half) | R

    # Cycle-walk for miss
    while y >= N:
        L, R = y >> half, y & mask
        for k in keys:
            L, R = R, L ^ __F(R, int(k))
        y = (L << half) | R
    return y


def feistelvec(
    a:np.ndarray, N:int, half:int, mask:int, keys:np.ndarray
) -> np.ndarray:
    '''Feistel permutation for a vector of integers.

    Parameters
    ----------
    a : np.ndarray
        The input array of integers to permute. Must be a 1D array of non-negative integers.
    N : int
        The size of the permutation. Must be a positive integer.
    half : int
        The number of bits in the half-word. Must be a positive integer.
    mask : int
        The bitmask for the half-word. Must be (1 << half) - 1.
    keys : np.ndarray
        The round keys for the Feistel network. Must be a 1D array of integers with length equal to the number of rounds.

    Returns
    -------
    np.ndarray
        The permuted array of integers, with the same shape as the input array. All values are in the range [0, N-1].
    '''
    L = a >> half
    R = a & mask
    for k in keys:
        T = __F(R, int(k))
        L, R = R, L ^ T
    y = (L << half) | R
    over = y >= N
    while over.any():
        o = y[over]
        L = o >> half
        R = o & mask
        for k in keys:
            T = __F(R, int(k))
            L, R = R, L ^ T
        y[over] = (L << half) | R
        over = y >= N
    return y

class IdentitySampler:

    def __init__(self, N:int):
        assert N > 0, "Size N must be a positive integer."
        self.N = N

    def __getitem__(self, i:int) -> int:
        if i < 0 or i >= self.N:
            raise IndexError(f"Index {i} out of bounds for size {self.N}")
        return i

    def __iter__(self):
        for i in range(self.N):
            yield i

    def __len__(self):
        return self.N


class FeistelSampler:
    
    def __init__(self, N:int, rounds:int=3, init_seed:int=0):
        assert N > 0, "Size N must be a positive integer."
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

    def __getitem__(self, i:int) -> int:
        return feistel(i, self.N, self.half, self.mask, self.keys)

    def __call__(self, i:int) -> int:
        return self.__getitem__(i)

    def __iter__(self):
        for i in range(self.N):
            yield self.__getitem__(i)

    def __len__(self):
        return self.N

    def randperm(self) -> np.ndarray:
        arr = np.arange(self.N, dtype=np.uint64)
        return feistelvec(arr, self.N, self.half, self.mask, self.keys)


class MultiFeistelSampler:

    def __init__(
        self, Ns:Sequence[int], rounds:int=3, init_seed:int=0,
        shuffle_outer:bool=False
    ):
        assert all(N > 0 for N in Ns), "All sizes must be positive integers."
        self.Ns = Ns
        self.cums = np.cumsum(Ns)
        self.num_Ns = len(Ns)
        self.rounds = rounds
        w = np.array([(N - 1).bit_length() for N in Ns], dtype=np.uint64)
        self.half = (w + 1) // 2
        self.mask = (1 << self.half) - 1
        self.bucket_order = IdentitySampler(self.num_Ns)
        if shuffle_outer:
            self.bucket_order = FeistelSampler(self.num_Ns, rounds, 0)
        self.refesh_keys(init_seed)
    
    def refesh_keys(self, seed:int):
        rng = random.Random(seed)
        r = self.rounds
        bits = self.half
        mask = self.mask
        self.keys = np.array([[
                rng.getrandbits(bits[i]) & mask[i] 
                for i in range(self.num_Ns)
            ] for _ in range(r)
        ], dtype=np.uint64)
        self.bucket_order.refesh_keys(seed + 1) 
    
    def _bucket(self, i:int) -> int:
        cums = self.cums
        if i < 0 or i >= cums[-1]:
            raise IndexError(f"Index {i} out of bounds for cumulative sizes {cums}")
        j = int(np.searchsorted(cums, i, side='right') - 1)
        return self.bucket_order[j]

    def __getitem__(self, i:int) -> int:
        j = self._bucket(i)
        N, half, mask = self.Ns[j], self.half[j], self.mask[j]
        keys = self.keys[:, j]
        local_d = self.cums[j-1] if j > 0 else 0
        return feistel(i - local_d, N, half, mask, keys) + local_d

    def __call__(self, i:int) -> int:
        return self.__getitem__(i)
    
    def __iter__(self):
        for i in range(self.cums[-1]):
            yield self.__getitem__(i)

    def __len__(self):
        return self.cums[-1]

    def randperm(self) -> np.ndarray:
        return np.concatenate([
            feistelvec(
                np.arange(N, dtype=np.uint64), N, self.half[j], self.mask[j], self.keys[:, j]
            ) + np.uint64(self.cums[j-1] if j > 0 else 0)
            for j, N in enumerate(self.Ns)
        ])
        

