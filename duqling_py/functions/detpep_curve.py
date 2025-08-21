"""
Dette & Pepelyshev Curved Functions.
"""

import numpy as np

from ..utils import register_function


def detpep_curve(x, scale01=True):
    """
    The Dette & Pepelyshev Curved Function

    Dimensions 3.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    This highly curved function is used for the comparison of computer experiment designs.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Dette, Holger, and Andrey Pepelyshev. "Generalized Latin hypercube design for
    computer experiments." Technometrics 52.4 (2010): 421-429.
    """
    ya = 4 * (x[0] - 2 + 8 * x[1] - x[1] ** 2) ** 2
    yb = (3 - 4 * x[1]) ** 2 + 16 * np.sqrt(x[2] + 1) * (2 * x[2] - 1) ** 2

    return ya + yb


def detpep8(x, scale01=True):
    """
    The Dette & Pepelyshev 8-Dim Function

    Dimensions 8.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 8.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Function is highly curved in first 3 dimensions, less so for inputs 4-8.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Dette, Holger, and Andrey Pepelyshev. "Generalized Latin hypercube design for
    computer experiments." Technometrics 52.4 (2010): 421-429.
    """
    ya = 4 * (x[0] - 2 + 8 * x[1] - x[1] ** 2) ** 2
    yb = (3 - 4 * x[1]) ** 2 + 16 * np.sqrt(x[2] + 1) * (2 * x[2] - 1) ** 2

    yc = 0
    for j in range(3, 8):
        yc += (j + 1) * np.log(1 + np.sum(x[2 : j + 1]))

    return ya + yb + yc


def welch20(x, scale01=True):
    """
    The Welch Screening Function

    Dimensions 20.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 20.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    A challenging high-dimensional problem used for variable screening purposes.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., & Morris, M. D. (1992).
    Screening, predicting, and computer experiments. Technometrics, 34(1), 15-25.
    """
    if scale01:
        x = x - 0.5

    # Explicitly extract variables
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    x10 = x[9]
    x11 = x[10]
    x12 = x[11]
    x13 = x[12]
    x14 = x[13]
    x15 = x[14]
    x16 = x[15]
    x17 = x[16]
    x18 = x[17]
    x19 = x[18]
    x20 = x[19]

    term1 = 5 * x12 / (1 + x1)
    term2 = 5 * (x4 - x20) ** 2
    term3 = x5 + 40 * x19**3 - 5 * x19
    term4 = 0.05 * x2 + 0.08 * x3 - 0.03 * x6
    term5 = 0.03 * x7 - 0.09 * x9 - 0.01 * x10
    term6 = -0.07 * x11 + 0.25 * x13**2 - 0.04 * x14
    term7 = 0.06 * x15 - 0.01 * x17 - 0.03 * x18

    y = term1 + term2 + term3 + term4 + term5 + term6 + term7
    return y


# Register functions with metadata
register_function(
    fname="detpep_curve",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1], [0, 1]]),
)

register_function(
    fname="detpep8",
    input_dim=8,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((8, 2)) + np.array([[0, 1]]),
)

register_function(
    fname="welch20",
    input_dim=20,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
        ]
    ),
)
