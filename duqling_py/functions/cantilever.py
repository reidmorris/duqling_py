"""
Cantilever Beam Functions.
"""

import numpy as np

from ..utils import register_function


def cantilever_D(x, scale01=True, L=100, D0=2.2535):
    """
    The Cantilever Beam Function - Displacement

    Dimensions: 6. The Cantilever Beam has two outputs: stress (S) and
    displacement (D). This can also be treated as a single function with bivariate output.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 6.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.
    L : float, optional
        Length of beam. Default is 100 inches.
    D0 : float, optional
        Displacement tolerance. Default is 2.2535 inches (Constraint: D < D_0).

    Returns
    -------
    float
        Displacement response.

    Notes
    -----
    For details on the Cantilever Beam, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Sues, Robert, Mohammad Aminpour, and Youngwon Shin. "Reliability based MDO
    for aerospace systems." 19th AIAA Applied Aerodynamics Conference. 2000.

    Wu, Y-T., et al. "Safety-factor based approach for probability-based design
    optimization." 19th AIAA applied aerodynamics conference. 2001.
    """
    if scale01:
        RR = np.array(
            [
                [40000 - 2000 * 4, 40000 + 2000 * 4],
                [2.9e7 - 1.45e6 * 4, 2.9e7 + 1.45e6 * 4],
                [500 - 100 * 4, 500 + 100 * 4],
                [1000 - 100 * 4, 1000 + 100 * 4],
                [4 - 0.25 * 4, 4 + 0.25 * 4],
                [2 - 0.25 * 4, 2 + 0.25 * 4],
            ]
        )
        x = x[:6] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    R = x[0]
    E = x[1]
    X = x[2]
    Y = x[3]
    w = x[4]
    t = x[5]

    term1 = 4 * L**3 / (E * w * t)
    term2 = np.sqrt(Y**2 / t**4 + X**2 / w**4)
    res = term1 * term2
    res = min(res, D0)
    return res


def cantilever_S(x, scale01=True):
    """
    The Cantilever Beam Function - Stress

    Dimensions: 6. The Cantilever Beam has two outputs: stress (S) and
    displacement (D). This can also be treated as a single function with bivariate output.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 6.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.

    Returns
    -------
    float
        Stress response.

    Notes
    -----
    For details on the Cantilever Beam, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Sues, Robert, Mohammad Aminpour, and Youngwon Shin. "Reliability based MDO
    for aerospace systems." 19th AIAA Applied Aerodynamics Conference. 2000.

    Wu, Y-T., et al. "Safety-factor based approach for probability-based design
    optimization." 19th AIAA applied aerodynamics conference. 2001.
    """
    if scale01:
        RR = np.array(
            [
                [40000 - 2000 * 4, 40000 + 2000 * 4],
                [2.9e7 - 1.45e6 * 4, 2.9e7 + 1.45e6 * 4],
                [500 - 100 * 4, 500 + 100 * 4],
                [1000 - 100 * 4, 1000 + 100 * 4],
                [4 - 0.25 * 4, 4 + 0.25 * 4],
                [2 - 0.25 * 4, 2 + 0.25 * 4],
            ]
        )
        x = x[:6] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    R = x[0]
    E = x[1]
    X = x[2]
    Y = x[3]
    w = x[4]
    t = x[5]

    res = 600 * Y / (w * t**2) + 600 * X / (w**2 * t)
    res = min(res, R)
    return res


# Register functions with metadata
register_function(
    fname="cantilever_D",
    input_dim=6,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [40000 - 2000 * 4, 40000 + 2000 * 4],
            [2.9e7 - 1.45e6 * 4, 2.9e7 + 1.45e6 * 4],
            [500 - 100 * 4, 500 + 100 * 4],
            [1000 - 100 * 4, 1000 + 100 * 4],
            [4 - 0.25 * 4, 4 + 0.25 * 4],
            [2 - 0.25 * 4, 2 + 0.25 * 4],
        ]
    ),
)

register_function(
    fname="cantilever_S",
    input_dim=6,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [40000 - 2000 * 4, 40000 + 2000 * 4],
            [2.9e7 - 1.45e6 * 4, 2.9e7 + 1.45e6 * 4],
            [500 - 100 * 4, 500 + 100 * 4],
            [1000 - 100 * 4, 1000 + 100 * 4],
            [4 - 0.25 * 4, 4 + 0.25 * 4],
            [2 - 0.25 * 4, 2 + 0.25 * 4],
        ]
    ),
)
