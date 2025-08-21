"""
Constant Functions.
"""

import numpy as np

from ..utils import register_function


def const_fn(x, scale01=True):
    """
    Constant Function

    Always returns 0, regardless of input.

    Parameters
    ----------
    x : array_like
        Inputs of any dimension.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Always returns 0.
    """
    return 0


def const_fn3(x, scale01=True):
    """
    Constant Function (3D)

    Always returns 0, regardless of input. Meant for 3-dimensional inputs.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Always returns 0.
    """
    return const_fn(x)


def const_fn15(x, scale01=True):
    """
    Constant Function (15D)

    Always returns 0, regardless of input. Meant for 15-dimensional inputs.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 15.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Always returns 0.
    """
    return const_fn(x)


# Register functions with metadata
register_function(
    fname="const_fn",
    input_dim=1,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1]]),
)

register_function(
    fname="const_fn3",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1], [0, 1]]),
)

register_function(
    fname="const_fn15",
    input_dim=15,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((15, 2))
    + np.array([[0, 1]]),  # Create 15x2 array with [0,1] ranges
)
