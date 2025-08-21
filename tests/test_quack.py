"""Test function metadata queried using the `quack` method."""
import numpy as np
import pytest

from duqling_py.duqling import Duqling as DuqlingPy
from duqling_py.duqling_r import DuqlingR
from duqling_py.sampling import LHSSampler

duq_r = DuqlingR()
duq_py = DuqlingPy()

all_fnames = list(duq_r.quack().fname)


def normalize_quack_info(info: dict) -> dict:
    """Remove keys that are inconsistent between implementations."""
    return {k: v for k, v in info.items() if k not in {"stochastic", "output_dim"}}


class TestQuackMetadata:
    """Test function metadata consistency between implementations."""

    @pytest.fixture(params=all_fnames, ids=all_fnames)
    def fname(self, request):
        return request.param


    def test_fnames_match(self):
        """Verify that the test functions form a 1:1 correspondence between implementations"""
        duqling_funcs_r  = set(duq_r .quack().fname)
        duqling_funcs_py = set(duq_py.quack().fname)
        assert duqling_funcs_r == duqling_funcs_py, (
            "Missing function implementations:\n"
            f"   R: {duqling_funcs_py - duqling_funcs_r}\n"
            f"  Py: {duqling_funcs_r  - duqling_funcs_py}"
        )


    def test_keys_match(self, fname):
        """Verify metadata keys are identical."""
        info_r  = normalize_quack_info(duq_r .quack(fname))
        info_py = normalize_quack_info(duq_py.quack(fname))

        assert info_r.keys() == info_py.keys(), (
            f"{fname} has different keys\n"
            f"   R-only: {info_r .keys() - info_py.keys()}\n"
            f"  Py-only: {info_py.keys() -  info_r.keys()}"
        )


    def test_values_match(self, fname):
        """Verify metadata values are equivalent."""
        info_r  = normalize_quack_info(duq_r .quack(fname))
        info_py = normalize_quack_info(duq_py.quack(fname))

        for key, val_r in info_r.items():
            val_py = info_py[key]

            if isinstance(val_r, np.ndarray):
                assert np.allclose(val_r, val_py, rtol=1e-10), f"{fname}.{key}: arrays differ"
            else:
                assert val_r == val_py, f"{fname}.{key}: {val_r!r} != {val_py!r}"


    def test_input_dim_and_range_agree(self, fname):
        """Verify that the `input_dim` = length(`input_range`)"""
        info = duq_py.quack(fname)
        assert info['input_dim'] == len(info['input_range']), (
            f"{fname} input dim ({info['input_dim']}) does not match"
            f"the number of input ranges ({len(info['input_range'])})"
        )
