"""
Short Column Function.
"""

import numpy as np

from ..utils import register_function


def short_column(x, scale01=True):
    """
    Short Column

    Dimension 5.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 5.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    The Short Column function models a short column with uncertain material properties
    and dimensions subject to uncertain loads. Y is the yield stress, M is the bending
    moment, P is the axial force, b is the width of the cross section, and h is the
    depth of the cross section. Note that sometimes M and P are given a correlated
    prior (correlation coefficient = 0.5).

    For more details on the Cantilever Beam, see the VLSE.

    References
    ----------
    Eldred, M. S., et al. "Investigation of reliability method formulations in DAKOTA/UQ."
    Structure and Infrastructure Engineering 3.3 (2007): 199-213.

    Kuschel, Norbert, and Rüdiger Rackwitz. "Two basic problems in reliability-based
    structural optimization." Mathematical Methods of Operations Research 46 (1997): 309-333.
    """
    if scale01:
        RR = np.array([[4.8, 5.1], [400, 3600], [100, 900], [3, 7], [10, 20]])
        x = x[:5] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    Y = x[0]
    M = x[1]
    P = x[2]
    b = x[3]
    h = x[4]

    res = 1 - 4 * M / (b * h**2 * Y) - P**2 / (b**2 * h**2 * Y**2)
    return res


# Register function with metadata
register_function(
    fname="short_column",
    input_dim=5,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[4.8, 5.1], [400, 3600], [100, 900], [3, 7], [10, 20]]),
)
