"""
Ripples Function.
"""

import warnings

import numpy as np

from ..utils import register_function


def ripples(x, scale01=True, fixed_coeff=True, input_dims=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if fixed_coeff and input_dims != 2:
        warnings.warn("Cannot have fixed coefficients when input_dims != 2.")

    if fixed_coeff and input_dims == 2:
        W = np.array(
            [
                [-0.138, 1.905],
                [0.348, 1.520],
                [1.628, -1.533],
                [-1.452, 0.185],
                [0.955, -1.440],
            ]
        )
    else:
        W = np.random.normal(-2, 2, (5, input_dims))

    z = W @ np.asarray(x)[:input_dims]
    return np.sin((np.arange(1, 6) * 2 + 1) * np.pi * z).sum() / 20


# Register function with metadata
register_function(
    fname="ripples",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1]]),
)
