"""Various sampling schema to use as duqling inputs"""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
from scipy.stats.qmc import Halton, LatinHypercube, PoissonDisk, Sobol


class Sampler(ABC):
    """Represents a heuristic for sampling batches of input data for duqling functions"""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.default_rng = rng

    def sample(
        self,
        n: int,
        d: int,
        ranges: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Public interface that handles sampling and value normalization.

        Args:
            n: the number of samples
            d: the number of input dimensions
            ranges: an array of tuples corresponding to the
                    upper and lower bound of each input dimension
            seed: a random seed for reproducible results
        """
        rng = (
            np.random.default_rng(seed)
            if self.default_rng is None
            else self.default_rng
        )
        samples = self._sample_unit_cube(n, d, rng)
        return self._normalize(samples, ranges)

    @abstractmethod
    def _sample_unit_cube(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        pass

    def _normalize(
        self, samples: np.ndarray, ranges: Optional[np.ndarray]
    ) -> np.ndarray:
        """Transform samples from [0,1] to specified ranges."""
        if ranges is None:
            return samples
        return ranges[:, 0] + samples * (ranges[:, 1] - ranges[:, 0])


class QMCSampler(Sampler):
    """
    Represents a heuristic for Quasi-Monte Carlo sampling methods supported in SciPy.
    ref: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    """

    def __init__(
        self,
        optimization: Optional[Literal["random-cd", "lloyd"]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(rng=rng)
        self.optimization = optimization

    def _sample_unit_cube(self, n, d, rng):
        sampler = self._qmc_sampler(d, rng)
        return sampler.random(n)

    @abstractmethod
    def _qmc_sampler(self, d: int, rng: np.random.Generator):
        pass


class LHSSampler(QMCSampler):
    """Latin hypercube sampler."""

    def __init__(self, strength: int = 1, scramble: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.scramble = scramble

    def _qmc_sampler(self, d, rng):
        return LatinHypercube(
            d=d,
            scramble=self.scramble,
            optimization=self.optimization,
            rng=rng,
            strength=self.strength,
        )


class SobolSampler(QMCSampler):
    """Sobol sequence sampler."""

    def __init__(self, bits: Optional[int] = None, scramble: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.scramble = scramble

    def _qmc_sampler(self, d, rng):
        return Sobol(
            d=d,
            scramble=self.scramble,
            optimization=self.optimization,
            rng=rng,
            bits=self.bits,
        )


class HaltonSampler(QMCSampler):
    """Halton sequence sampler."""

    def __init__(self, scramble: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.scramble = scramble

    def _qmc_sampler(self, d, rng):
        return Halton(
            d=d, scramble=self.scramble, optimization=self.optimization, rng=rng
        )


class PoissonDiskSampler(QMCSampler):
    """Poisson disk sampler."""

    def __init__(
        self,
        radius: float = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: int = 30,
        l_bounds: Optional[np.ndarray] = None,
        u_bounds: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.radius = radius
        self.hypersphere = hypersphere
        self.ncandidates = ncandidates
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

    def _qmc_sampler(self, d, rng):
        return PoissonDisk(
            d=d,
            radius=self.radius,
            hypersphere=self.hypersphere,
            ncandidates=self.ncandidates,
            l_bounds=self.l_bounds,
            u_bounds=self.u_bounds,
            optimization=self.optimization,
            rng=rng,
        )


class UniformSampler(Sampler):
    """Uniform distribution sampler."""

    def _sample_unit_cube(self, n, d, rng):
        return rng.uniform(size=(n, d))


class NormalSampler(Sampler):
    """Normal distribution sampler."""

    def __init__(self, mean: float = 0.0, std_dev: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std_dev = std_dev

    def _sample_unit_cube(self, n, d, rng):
        return rng.normal(size=(n, d), loc=self.mean, scale=self.std_dev)


class GridSampler(Sampler):
    """
    Grid-based sampler.
    Note: this scheme will create (floor(sqrt(n)))^2 samples,
          which is approximately, but not necessarily, n.
    """

    def _sample_unit_cube(self, n, d, seed=None):
        # floor(sqrt(n)) equally spaced points along each input dim
        axes = [np.linspace(0, 1, int(np.sqrt(n))) for _ in range(d)]
        # the cartesian product of these points (creating a grid of (floor(sqrt(n)))^2 ~ n points)
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)
