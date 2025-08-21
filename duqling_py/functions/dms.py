"""
Denison, Mallick and Smith Bivariate Test Functions.
"""

import numpy as np

from ..utils import register_function


def dms_simple(x, scale01=True):
    """
    DMS Simple Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    One of the five test functions from Denison, Mallick and Smith (1998).

    References
    ----------
    Denison, David GT, Bani K. Mallick, and Adrian FM Smith. "Bayesian mars."
    Statistics and Computing, 8.4 (1998): 337-346.
    """
    return 10.391 * ((x[0] - 0.4) * (x[1] - 0.6) + 0.36)


def dms_radial(x, scale01=True):
    """
    DMS Radial Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    One of the five test functions from Denison, Mallick and Smith (1998).

    References
    ----------
    Denison, David GT, Bani K. Mallick, and Adrian FM Smith. "Bayesian mars."
    Statistics and Computing, 8.4 (1998): 337-346.
    """
    r = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
    return 24.234 * (r * (0.75 - r))


def dms_harmonic(x, scale01=True):
    """
    DMS Harmonic Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    One of the five test functions from Denison, Mallick and Smith (1998).

    References
    ----------
    Denison, David GT, Bani K. Mallick, and Adrian FM Smith. "Bayesian mars."
    Statistics and Computing, 8.4 (1998): 337-346.
    """
    xx1 = x[0] - 0.5
    xx2 = x[1] - 0.5
    return 42.659 * (0.1 + xx1 * (0.05 + xx1**4 - 10 * xx1**2 * xx2**2 + 5 * xx2**4))


def dms_additive(x, scale01=True):
    """
    DMS Additive Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    One of the five test functions from Denison, Mallick and Smith (1998).

    References
    ----------
    Denison, David GT, Bani K. Mallick, and Adrian FM Smith. "Bayesian mars."
    Statistics and Computing, 8.4 (1998): 337-346.
    """
    return 1.3356 * (
        1.5 * (1 - x[0])
        + np.exp(2 * x[0] - 1) * np.sin(3 * np.pi * (x[0] - 0.6) ** 2)
        + np.exp(3 * (x[1] - 0.5)) * np.sin(4 * np.pi * (x[1] - 0.9) ** 2)
    )


def dms_complicated(x, scale01=True):
    """
    DMS Complicated Interaction Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    One of the five test functions from Denison, Mallick and Smith (1998).

    References
    ----------
    Denison, David GT, Bani K. Mallick, and Adrian FM Smith. "Bayesian mars."
    Statistics and Computing, 8.4 (1998): 337-346.
    """
    return 1.9 * (
        1.35
        + np.exp(x[0])
        * np.sin(13 * (x[0] - 0.6) ** 2)
        * np.exp(-x[1])
        * np.sin(7 * x[1])
    )


# Register functions with metadata
for func_name in [
    "dms_simple",
    "dms_radial",
    "dms_harmonic",
    "dms_additive",
    "dms_complicated",
]:
    register_function(
        fname=func_name,
        input_dim=2,
        input_cat=False,
        response_type="uni",
        stochastic="n",
        input_range=np.array([[0, 1], [0, 1]]),
    )
