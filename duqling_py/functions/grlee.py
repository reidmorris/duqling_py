"""
Gramacy and Lee Test Functions.
"""

import numpy as np

from ..utils import register_function


def grlee1(x, scale01=True):
    """
    Gramacy and Lee 1D Test Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 1.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Functions from three Gramacy and Lee papers. For details, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Gramacy, R. B., & Lee, H. K. (2012). Cases for the nugget in modeling computer
    experiments. Statistics and Computing, 22(3), 713-722.
    """
    if scale01:
        x = 0.5 + 2 * x[0]
    else:
        x = x[0]

    term1 = np.sin(10 * np.pi * x) / (2 * x)
    term2 = (x - 1) ** 4

    y = term1 + term2
    return y


def grlee2(x, scale01=True):
    """
    Gramacy and Lee 2D Test Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Functions from three Gramacy and Lee papers. For details, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Gramacy, R. B., & Lee, H. K. (2008). Gaussian processes and limiting linear models.
    Computational Statistics & Data Analysis, 53(1), 123-136.
    """
    if scale01:
        x = 8 * x[:2] - 2
    else:
        x = x[:2]

    y = x[0] * np.exp(-x[0] ** 2 - x[1] ** 2)
    return y


def grlee6(x, scale01=True):
    """
    Gramacy and Lee 6D Test Function

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 6.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Functions from three Gramacy and Lee papers. For details, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Gramacy, R. B., & Lee, H. K. (2009). Adaptive design and analysis of supercomputer
    experiments. Technometrics, 51(2).
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]

    term1 = np.exp(np.sin((0.9 * (x1 + 0.48)) ** 10))
    term2 = x2 * x3
    term3 = x4

    y = term1 + term2 + term3
    return y


# Register functions with metadata
register_function(
    fname="grlee1",
    input_dim=1,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1]]),
)

register_function(
    fname="grlee2",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[-2, 6], [-2, 6]]),
)

register_function(
    fname="grlee6",
    input_dim=6,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((6, 2)) + np.array([[0, 1]]),
)
