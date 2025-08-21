"""Tests for stochastic functions in Duqling implementations."""
import numpy as np
import pytest
from scipy import stats

from duqling_py.duqling import Duqling as DuqlingPy
from duqling_py.duqling_r import DuqlingR
from duqling_py.sampling import LHSSampler

duq_r = DuqlingR()
duq_py = DuqlingPy()
lhs = LHSSampler()

stochastic_fnames = list(duq_r.quack(stochastic='y').fname)

class TestStochasticFunctions:
    """Test stochastic function behavior."""


    @pytest.fixture(params=stochastic_fnames, ids=stochastic_fnames)
    def fname(self, request):
        return request.param


    @pytest.fixture(params=[False, True], ids=["original_scale", "unit_scale"])
    def scale01(self, request):
        return request.param


    def test_deterministic_with_fixed_seeds(self, fname):
        """Verify stochastic functions are deterministic with fixed seeds."""
        info = duq_py.quack(fname)
        X = lhs.sample(10, info["input_dim"], info["input_range"])

        seed = np.random.randint(100)
        y1 = duq_py.duq(X=X, f=fname, seed=seed)
        y2 = duq_py.duq(X=X, f=fname, seed=seed)

        assert np.array_equal(y1, y2), f"{fname} fails to produce reproducable results"


    def test_variability(self, fname):
        """
        Verify stochastic functions produce variable outputs with different random seeds.
        Note: some functions have outputs that are partially deterministic 
        (i.e., deterministic up until a point).
        """

        # `stochastic_piston` actually behaves fairly deterministically
        if fname == 'stochastic_piston':
            return

        info = duq_py.quack(fname)
        n_samples = 5
        X = lhs.sample(n_samples, info["input_dim"], info["input_range"])

        n_runs = 30
        outputs = np.array([duq_py.duq(X=X, f=fname, seed=i) for i in range(n_runs)])
        variance = np.var(outputs, axis=0)

        assert np.max(variance) > 1e-10, f"{fname} behaves deterministically."


    def test_output_shapes_match(self, fname, scale01):
        """Verify both implementations produce arrays with identical shapes."""
        info = duq_py.quack(fname)
        input_dim = info["input_dim"]
        input_range = info["input_range"] if not scale01 else None

        X = lhs.sample(100, input_dim, input_range)
        y_r  = duq_r .duq(X=X, f=fname, scale01=scale01)
        y_py = duq_py.duq(X=X, f=fname, scale01=scale01)

        assert y_r.shape == y_py.shape, "shape mismatch"


    def test_output_bounds_finite(self, fname):
        """Verify stochastic outputs are finite."""
        info = duq_py.quack(fname)
        X = lhs.sample(100, info["input_dim"], info["input_range"])

        n_runs = 10
        outputs = np.array([duq_py.duq(X=X, f=fname, seed=i) for i in range(n_runs)])

        assert np.all(np.isfinite(outputs)), f"{fname} produced NaN or Inf"


    def test_implementation_consistency_statistical(self, fname):
        """Verify R and Python implementations have similar statistical properties."""
        info = duq_py.quack(fname)
        n_samples = 5
        X = lhs.sample(n_samples, info["input_dim"], info["input_range"])

        n_runs = 1000
        outputs_r  = []
        outputs_py = []
        # run the function on a given handful of samples a bunch of times
        for _ in range(n_runs):
            outputs_r .append(duq_r .duq(X=X, f=fname))
            outputs_py.append(duq_py.duq(X=X, f=fname))
        outputs_r  = np.array(outputs_r)
        outputs_py = np.array(outputs_py)

        # we want `outputs_*` to have shape (n_runs, n_samples, output_dim).
        # however, funcs with `uni` response types have shape (n_runs, n_samples)
        if outputs_r.ndim == 2:
            outputs_r  = np.expand_dims(outputs_r,  axis=2)
            outputs_py = np.expand_dims(outputs_py, axis=2)

        for sample_idx in range(n_samples):
            sample_r  =  outputs_r[:, sample_idx, :].flatten()
            sample_py = outputs_py[:, sample_idx, :].flatten()

            # compare means
            mean_r  = np.mean(sample_r)
            mean_py = np.mean(sample_py)

            assert np.allclose(mean_r, mean_py, rtol=1e-1), (
                f"{fname} has sufficiently different means"
                f"   R: {mean_r}\n"
                f"  Py: {mean_py}"
            )

            # compare std devs
            std_r  = np.std(sample_r)
            std_py = np.std(sample_py)

            assert np.allclose(std_r, std_py, rtol=1e-1, atol=2e-1), (
                f"{fname} has sufficiently different standard deviations:\n"
                f"   R: {std_r}\n"
                f"  Py: {std_py}"
            )


    def test_distribution_similarity(self, fname):
        """Use Kolmogorov-Smirnov test to verify distributions are similar."""
        info = duq_py.quack(fname)
        X = lhs.sample(1, info["input_dim"], info["input_range"])

        n_samples = 500
        outputs_r  = []
        outputs_py = []
        for _ in range(n_samples):
            y_r  = duq_r .duq(X=X, f=fname)
            y_py = duq_py.duq(X=X, f=fname)

            outputs_r .append(y_r .flatten()[0])
            outputs_py.append(y_py.flatten()[0])

        ks_statistic, p_value = stats.ks_2samp(outputs_r, outputs_py)

        assert p_value > 0.01, (
            f"{fname}: Distributions significantly different (p={p_value:.4f}). "
            f"KS statistic: {ks_statistic:.4f}"
        )


class TestStochasticEdgeCases:
    """
    Edge case tests specific to stochastic functions.
    Note: some stochastic functions are deterministic at edge case inputs.
    """


    @pytest.fixture(params=stochastic_fnames, ids=stochastic_fnames)
    def fname(self, request):
        return request.param


    def test_boundary_values_finite(self, fname):
        """Verify functions produce finite values at boundary inputs."""
        info = duq_py.quack(fname)
        input_range = info["input_range"]

        # lower bound check
        X_lower = input_range[:, 0].reshape(1,-1)
        y_lower = duq_py.duq(X=X_lower, f=fname, scale01=False)
        assert np.all(np.isfinite(y_lower)), f"{fname} contains NaN or Inf at lower bound"

        # upper bound check
        X_upper = input_range[:, 1].reshape(1,-1)
        y_upper = duq_py.duq(X=X_upper, f=fname, scale01=False)
        assert np.all(np.isfinite(y_upper)), f"{fname} contains NaN or Inf at upper bound"
