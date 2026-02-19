import random
import numpy as np
from typing import Sequence

# TODO: Add fuzzy testing, random N, random rounds, random seed.

np.seterr(over='ignore')

_MULT = np.uint64(0x9E3779B97F4A7C15)

def __F(x: np.uint64, k: np.uint64, mask: np.uint64) -> np.uint64:
    # Fast bijective mixing on half-words natively in 64-bit
    return (x * _MULT + k) & mask

def feistel(i: int, N: int, half: int, mask: int, keys: np.ndarray) -> int:
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
        The round keys for the Feistel network. Must be a 1D array of integers 
        with length equal to the number of rounds.
    
    Returns
    -------
    int
        The permuted integer in the range [0, N-1].
    '''
    y = np.uint64(i)
    u_N = np.uint64(N)
    u_half = np.uint64(half)
    u_mask = np.uint64(mask)
    
    while True:
        L, R = y >> u_half, y & u_mask
        for k in keys:
            L, R = R, L ^ __F(R, k, u_mask)
        y = (L << u_half) | R
        if y < u_N:
            break
            
    return int(y)


def feistelvec(
    a: np.ndarray, N: int, half: int, mask: int, keys: np.ndarray
) -> np.ndarray:
    '''Feistel permutation for a vector of integers.

    Parameters
    ----------
    a : np.ndarray
        The input array of integers to permute. Must be a 1D array of 
        non-negative integers.
    N : int
        The size of the permutation. Must be a positive integer.
    half : int
        The number of bits in the half-word. Must be a positive integer.
    mask : int
        The bitmask for the half-word. Must be (1 << half) - 1.
    keys : np.ndarray
        The round keys for the Feistel network. Must be a 1D array of integers 
        with length equal to the number of rounds.

    Returns
    -------
    np.ndarray
        The permuted array of integers, with the same shape as the input array. 
        All values are in the range [0, N-1].
    '''
    y = a.astype(np.uint64)
    u_N = np.uint64(N)
    u_half = np.uint64(half)
    u_mask = np.uint64(mask)

    L = y >> u_half
    R = y & u_mask
    for k in keys:
        L, R = R, L ^ __F(R, k, u_mask) # type: ignore
    y = (L << u_half) | R
    
    over = y >= u_N
    while over.any():
        o = y[over]
        L = o >> u_half
        R = o & u_mask
        for k in keys:
            L, R = R, L ^ __F(R, k, u_mask) # type: ignore
        y[over] = (L << u_half) | R
        over = y >= u_N
        
    return y


class IdentitySampler:
    '''Sampler that returns the identity permutation of integers from 0 to N-1.

    Parameters
    ----------
    N : int
        The size of the permutation. Must be a positive integer.

    Attributes
    ----------
    N : int
        The size of the permutation.

    Methods
    -------
    __getitem__(i)
        Returns the integer at index i, which is simply i.
    __iter__()
        Returns an iterator over the integers from 0 to N-1.
    __len__()
        Returns the size N of the permutation.
    '''
    def __init__(self, N: int):
        assert N > 0, "Size N must be a positive integer."
        self.N = N

    def __getitem__(self, i: int) -> int:
        if i < 0 or i >= self.N:
            raise IndexError(f"Index {i} out of bounds for size {self.N}")
        return i
    
    def __call__(self, i: int) -> int:
        return self.__getitem__(i)

    def __iter__(self):
        for i in range(self.N):
            yield i

    def __len__(self):
        return self.N

    def randperm(self) -> np.ndarray:
        return np.arange(self.N, dtype=np.uint64)
    
    def refesh_keys(self, seed):
        pass


class FeistelSampler:
    '''FeistelSampler generates a pseudorandom permutation of integers from 0 to 
    N-1 using a Feistel network. It supports multiple rounds of mixing and can be
    seeded for reproducibility. The permutation is deterministic and can be 
    accessed via indexing or iteration. The randperm method returns the entire 
    permuted array.

    Parameters
    ----------
    N : int
        The size of the permutation. Must be a positive integer.
    rounds : int, optional
        The number of rounds in the Feistel network. More rounds generally
        lead to better mixing but may be slower. Default is 3.
    init_seed : int, optional
        The initial seed for generating the round keys. Must be a non-negative
        integer. Default is 0.
    
    Attributes
    ----------
    N : int
        The size of the permutation.
    rounds : int
        The number of rounds in the Feistel network.
    half : int
        The number of bits in the half-word, calculated as (w + 1) // 2 where w 
        is the bit length of N-1.
    mask : int
        The bitmask for the half-word, calculated as (1 << half) - 1.
    keys : np.ndarray
        The round keys for the Feistel network, generated from the initial seed.

    Methods
    -------
    __getitem__(i)
        Returns the permuted integer at index i.
    __call__(i)
        Alias for __getitem__(i).
    __iter__()
        Returns an iterator over the permuted integers from 0 to N-1.
    __len__()
        Returns the size N of the permutation.
    randperm()
        Returns a numpy array containing the permuted integers from 0 to N-1.
    
    '''
    def __init__(self, N: int, rounds: int=3, init_seed: int=0):
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

    def __getitem__(self, i: int) -> int:
        return feistel(i, self.N, self.half, self.mask, self.keys)

    def __call__(self, i: int) -> int:
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
    '''MultiFeistelSampler generates a pseudorandom permutation of integers in 
    [0, N-1] using a Feistel network. It supports multiple rounds of mixing and 
    can be seeded for reproducibility. The permutation is deterministic and can 
    be accessed via indexing or iteration. The randperm method returns the 
    entire permuted array.

    Parameters
    ----------
    N : int
        The size of the permutation. Must be a positive integer.
    rounds : int, optional
        The number of rounds in the Feistel network. More rounds generally
        lead to better mixing but may be slower. Default is 3.
    init_seed : int, optional
        The initial seed for generating the round keys. Must be a non-negative
        integer. Default is 0.
    
    Attributes
    ----------
    N : int
        The size of the permutation.
    rounds : int
        The number of rounds in the Feistel network.
    half : int
        The number of bits in the half-word, calculated as (w + 1) // 2 where w 
        is the bit length of N-1.
    mask : int
        The bitmask for the half-word, calculated as (1 << half) - 1.
    keys : np.ndarray
        The round keys for the Feistel network, generated from the initial seed.

    Methods
    -------
    __getitem__(i)
        Returns the permuted integer at index i.
    __call__(i)
        Alias for __getitem__(i).
    __iter__()
        Returns an iterator over the permuted integers from 0 to N-1.
    __len__()
        Returns the size N of the permutation.
    randperm()
        Returns a numpy array containing the permuted integers from 0 to N-1.
    
    '''
    def __init__(
        self, Ns: Sequence[int], rounds: int=3, init_seed: int=0,
        shuffle_outer: bool=False
    ):
        assert all(N > 0 for N in Ns), "All sizes must be positive integers."
        self.Ns = Ns
        # Force cumulative sums to uint64 for safe later addition
        self.cums = np.cumsum(Ns, dtype=np.uint64) 
        self.num_Ns = len(Ns)
        self.rounds = rounds
        w = np.array([(N - 1).bit_length() for N in Ns], dtype=np.uint64)
        self.half = (w + 1) // 2
        # Explicit uint64 shifting for array masks
        self.mask = (np.uint64(1) << self.half) - np.uint64(1)
        self.bucket_order = IdentitySampler(self.num_Ns)
        if shuffle_outer:
            self.bucket_order = FeistelSampler(self.num_Ns, rounds, 0)
        self.refesh_keys(init_seed)
    
    def refesh_keys(self, seed: int):
        rng = random.Random(seed)
        r = self.rounds
        bits = self.half
        mask = self.mask
        self.keys = np.array([[
                # Cast bits[i] to native int to satisfy Python's random module
                rng.getrandbits(int(bits[i])) & mask[i] 
                for i in range(self.num_Ns)
            ] for _ in range(r)
        ], dtype=np.uint64)
        self.bucket_order.refesh_keys(seed + 1) 
    
    def _bucket(self, i: int) -> tuple[int, int]:
        if i < 0 or i >= self.cums[-1]:
            raise IndexError(f"Index {i} out of bounds")
        j = int(np.searchsorted(self.cums, i, side='right'))
        i_val = int(i) - int(self.cums[j-1]) if j > 0 else int(i)
        j_val = int(self.bucket_order[j])
        return i_val, j_val

    def __getitem__(self, idx: int) -> int:
        i, j = self._bucket(idx)
        d = int(self.cums[j-1]) if j > 0 else 0
        N, half, mask = self.Ns[j], int(self.half[j]), int(self.mask[j])
        keys = self.keys[:, j]
        return feistel(i, N, half, mask, keys) + d

    def __call__(self, i: int) -> int:
        return self.__getitem__(i)
    
    def __iter__(self):
        for i in range(int(self.cums[-1])):
            yield self.__getitem__(i)

    def __len__(self):
        return int(self.cums[-1])

    def randperm(self) -> np.ndarray:
        order = self.bucket_order.randperm()
        Narr = np.array(self.Ns, dtype=np.uint64)[order]
        return np.concatenate([
            feistelvec(
                np.arange(N, dtype=np.uint64), N, int(self.half[j]), 
                int(self.mask[j]), self.keys[:, j]
            ) + (self.cums[j-1] if j > 0 else np.uint64(0))
            for j, N in zip(order, Narr)
        ])