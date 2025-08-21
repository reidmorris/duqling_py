"""
The Friedman Functions.
"""

import numpy as np

from ..utils import register_function


def friedman(x, scale01=True):
    """
    The Friedman Function

    Dimensions: 5. The Friedman function is often used with additional, inert inputs.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 5.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        A scalar response.

    Notes
    -----
    For details on the Friedman function, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Friedman, J. H. (1991). Multivariate adaptive regression splines. The annals
    of statistics, 19(1), 1-67.
    """
    return (
        10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]
    )


def friedman10(x, scale01=True):
    """
    The Friedman10 Function

    Same as Friedman function, but with 10 input dimensions (only first 5 are active).

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 10.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        A scalar response.
    """
    return friedman(x, scale01)


def friedman20(x, scale01=True):
    """
    The Friedman20 Function

    Same as Friedman function, but with 20 input dimensions (only first 5 are active).

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 20.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        A scalar response.
    """
    return friedman(x, scale01)


# Register functions with metadata
register_function(
    fname="friedman",
    input_dim=5,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
)

register_function(
    fname="friedman10",
    input_dim=10,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    ),
)

register_function(
    fname="friedman20",
    input_dim=20,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((20, 2))
    + np.array([[0, 1]]),  # Create 20x2 array with [0,1] ranges
)
