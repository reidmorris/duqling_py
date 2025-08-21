"""
Lim et al Test Functions.
"""

import numpy as np

from ..utils import register_function


def lim_polynomial(x, scale01=True):
    """
    Lim et al Polynomial Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Two similar functions from Lim et al (2002). One is a polynomial the other is a
    non-polynomial but both functions have similar shape and behavior.
    For details, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design and analysis
    of computer experiments when the output is highly correlated over the input space.
    Canadian Journal of Statistics, 30(1), 109-126.
    """
    x1 = x[0]
    x2 = x[1]

    term1 = (5 / 2) * x1 - (35 / 2) * x2
    term2 = (5 / 2) * x1 * x2 + 19 * x2**2
    term3 = -(15 / 2) * x1**3 - (5 / 2) * x1 * x2**2
    term4 = -(11 / 2) * x2**4 + (x1**3) * (x2**2)

    y = 9 + term1 + term2 + term3 + term4
    return y


def lim_non_polynomial(x, scale01=True):
    """
    Lim et al Non-Polynomial Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Two similar functions from Lim et al (2002). One is a polynomial the other is a
    non-polynomial but both functions have similar shape and behavior.
    For details, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design and analysis
    of computer experiments when the output is highly correlated over the input space.
    Canadian Journal of Statistics, 30(1), 109-126.
    """
    x1 = x[0]
    x2 = x[1]

    fact1 = 30 + 5 * x1 * np.sin(5 * x1)
    fact2 = 4 + np.exp(-5 * x2)

    y = (fact1 * fact2 - 100) / 6
    return y


# Register functions with metadata
register_function(
    fname="lim_polynomial",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1]]),
)

register_function(
    fname="lim_non_polynomial",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1]]),
)
