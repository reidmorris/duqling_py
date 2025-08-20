"""Various sampling schema to use as duqling inputs"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats.qmc import LatinHypercube

class Sampler(ABC):
    @abstractmethod
    def sample(self, n: int, d: int, ranges: np.ndarray=None, seed: int=None) -> np.ndarray:
        pass

class LHSSampler(Sampler):
    def sample(self, n, d, ranges=None, seed=None):
        sampler = LatinHypercube(d, seed=seed)
        samples = sampler.random(n)
        if ranges is not None:
            samples = ranges[:, 0] + samples * (ranges[:, 1] - ranges[:, 0])
        return samples
