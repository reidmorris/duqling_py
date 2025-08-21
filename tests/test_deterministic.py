"""Tests for deterministic functions in Duqling implementations."""
import os
import numpy as np
import pytest

from duqling_py.duqling import Duqling as DuqlingPy
from duqling_py.duqling_r import DuqlingR
from duqling_py.sampling import LHSSampler

duq_r = DuqlingR()
duq_py = DuqlingPy()
lhs = LHSSampler()

deterministic_fnames = list(duq_r.quack(stochastic='n').fname)
deterministic_fnames.append('stochastic_piston') # `stochastic_piston` is basically deterministic


def get_special_kwargs(fname: str) -> dict:
    """Return special kwargs for specific functions."""
    if fname == "stochastic_piston":
        return {"Ta_generate": lambda: 13, "P0_generate": lambda: 0.5}
    return {}


class TestFunctionOutputs:
    """Test consistency of deterministic functions between implementations."""


    @pytest.fixture(params=deterministic_fnames, ids=deterministic_fnames)
    def fname(self, request):
        return request.param


    @pytest.fixture(params=[False, True], ids=["original_scale", "unit_scale"])
    def scale01(self, request):
        return request.param


    def test_outputs_match(self, fname, scale01):
        """Verify both implementations produce identical outputs."""
        info = duq_py.quack(fname)
        input_dim = info["input_dim"]
        input_range = info["input_range"] if not scale01 else None

        X = lhs.sample(1000, input_dim, input_range)
        kwargs = get_special_kwargs(fname)

        y_r  = duq_r .duq(X=X, f=fname, scale01=scale01, **kwargs)
        y_py = duq_py.duq(X=X, f=fname, scale01=scale01, **kwargs)

        assert y_r.shape == y_py.shape, "shape mismatch"
        assert np.allclose(y_r, y_py, rtol=1e-9, atol=1e-10), f"{fname} (scale01={scale01})"


    def test_deterministic(self, fname):
        """Verify outputs are deterministic, and by proxy, that functions don't mutate input X."""
        info = duq_py.quack(fname)
        X = lhs.sample(10, info["input_dim"], info["input_range"])
        kwargs = get_special_kwargs(fname)

        y1 = duq_py.duq(X=X, f=fname, scale01=False, **kwargs)
        y2 = duq_py.duq(X=X, f=fname, scale01=False, **kwargs)

        assert np.array_equal(y1, y2, equal_nan=True)


    def test_finite_outputs(self, fname):
        """Verify all outputs are finite values."""
        info = duq_py.quack(fname)
        X = lhs.sample(100, info["input_dim"], info["input_range"])
        y = duq_py.duq(X=X, f=fname, scale01=False, **get_special_kwargs(fname))

        assert np.all(np.isfinite(y)), f"{fname} contains NaN or Inf"


class TestEdgeCases:
    """Test boundary conditions and edge cases."""


    @pytest.fixture(params=deterministic_fnames, ids=deterministic_fnames)
    def fname(self, request):
        return request.param


    def test_boundary_values_finite(self, fname):
        """Verify functions produce finite values at boundary inputs."""
        info = duq_py.quack(fname)
        input_range = info["input_range"]
        kwargs = get_special_kwargs(fname)

        # lower bound check
        X_lower = input_range[:, 0].reshape(1,-1)
        y_lower = duq_py.duq(X=X_lower, f=fname, scale01=False, **kwargs)
        assert np.all(np.isfinite(y_lower)), f"{fname} contains NaN or Inf at lower bound"

        # upper bound check
        X_upper = input_range[:, 1].reshape(1,-1)
        y_upper = duq_py.duq(X=X_upper, f=fname, scale01=False, **kwargs)
        assert np.all(np.isfinite(y_upper)), f"{fname} contains NaN or Inf at upper bound"


    def test_boundary_values_equal(self, fname):
        """Verify functions produce finite values at boundary inputs."""
        info = duq_py.quack(fname)
        input_range = info["input_range"]
        kwargs = get_special_kwargs(fname)

        # lower bound check
        X_lower = input_range[:, 0].reshape(1,-1)
        y_lower_r  = duq_r .duq(X=X_lower, f=fname, scale01=False, **kwargs)
        y_lower_py = duq_py.duq(X=X_lower, f=fname, scale01=False, **kwargs)
        assert np.allclose(y_lower_r, y_lower_py), (
            f"implementations of {fname} disagree at the lower bound"
        )

        # upper bound check
        X_upper = input_range[:, 1].reshape(1,-1)
        y_upper_r  = duq_r .duq(X=X_upper, f=fname, scale01=False, **kwargs)
        y_upper_py = duq_py.duq(X=X_upper, f=fname, scale01=False, **kwargs)
        assert np.allclose(y_upper_r, y_upper_py), (
            f"implementations of {fname} disagree at the upper bound"
        )


    def test_invalid_shapes(self):
        """Verify invalid input shapes raise value errors."""
        fname = np.random.choice(deterministic_fnames)
        info = duq_py.quack(fname)
        X_too_large = lhs.sample(1, info["input_dim"] + 1)
        with pytest.raises(ValueError):
            duq_py.duq(X=X_too_large, f=fname, scale01=False)

        if info["input_dim"] > 1:
            X_too_small = lhs.sample(1, info["input_dim"] - 1)
            with pytest.raises(ValueError):
                duq_py.duq(X=X_too_small, f=fname, scale01=False)
