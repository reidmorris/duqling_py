"""
Rosenbrock's Banana Function.
"""

import numpy as np

from ..utils import register_function


def banana(x, scale01=True, ab=None):
    """
    Rosenbrock's Banana Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.
    ab : array_like, optional
        Hyperparameters to Rosenbrock's banana. Default is [1, 100].

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    A non-convex function introduced by Howard H. Rosenbrock in 1960.
    It is a difficult optimization test-problem, with a unique minimum at (1,1).

    References
    ----------
    Rosenbrock, HoHo. "An automatic method for finding the greatest or least value
    of a function." The computer journal 3.3 (1960): 175-184.
    """
    if ab is None:
        ab = [1, 100]

    if scale01:
        RR = np.array([[-2, 2], [-1, 3]])
        x = x[:2] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    a = ab[0]
    b = ab[1]
    z = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    return z


# Register function with metadata
register_function(
    fname="banana",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[-2, 2], [-1, 3]]),
)
