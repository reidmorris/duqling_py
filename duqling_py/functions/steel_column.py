"""
Steel Column Function.
"""

import numpy as np

from ..utils import register_function


def steel_column(x, scale01=True, b=300, d=20, h=300):
    """
    Steel Column

    Dimension 9. Models the "reliability" of a steel column.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 9.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    b : float, optional
        Mean flange breadth. Default is 300.
    d : float, optional
        Flange thickness. Default is 20.
    h : float, optional
        Profile height. Default is 300.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    This function is often used to explore the trade-off between cost and reliability
    of a steel column. The cost is b*d + 5*h which are parameters giving the mean
    flange breadth, flange thickness, and profile height, respectively. Non-normal
    distributions are often used for the priors.

    For details, see the VLSE.

    References
    ----------
    Kuschel, Norbert, and Rüdiger Rackwitz. "Two basic problems in reliability-based
    structural optimization." Mathematical Methods of Operations Research 46 (1997): 309-333.
    """
    if scale01:
        RR = np.array(
            [
                [400 - 35 * 4, 400 + 35 * 4],
                [500000 - 50000 * 4, 500000 + 50000 * 4],
                [600000 - 90000 * 4, 600000 + 90000 * 4],
                [600000 - 90000 * 4, 600000 + 90000 * 4],
                [b - 3 * 4, b + 3 * 4],
                [d - 2 * 4, d + 2 * 4],
                [h - 5 * 4, h + 5 * 4],
                [30 - 10 * 2.999, 30 + 10 * 4],
                [210000 - 4200 * 4, 210000 + 4200 * 4],
            ]
        )
        x = x[:9] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    Fs = x[0]
    P1 = x[1]
    P2 = x[2]
    P3 = x[3]
    B = x[4]
    D = x[5]
    H = x[6]
    F0 = x[7]
    E = x[8]

    Eb = np.pi**2 * E * B * D * H**2 / (2 * (7500) ** 2)
    P = P1 + P2 + P3
    term1 = 1 / (2 * B * D)
    term2 = F0 * Eb / (B * D * H * (Eb - P))

    res = Fs - P * (term1 + term2)
    return res


# Register function with metadata
register_function(
    fname="steel_column",
    input_dim=9,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [400 - 35 * 4, 400 + 35 * 4],
            [500000 - 50000 * 4, 500000 + 50000 * 4],
            [600000 - 90000 * 4, 600000 + 90000 * 4],
            [600000 - 90000 * 4, 600000 + 90000 * 4],
            [300 - 3 * 4, 300 + 3 * 4],
            [20 - 2 * 4, 20 + 2 * 4],
            [300 - 5 * 4, 300 + 5 * 4],
            [30 - 10 * 2.999, 30 + 10 * 4],
            [210000 - 4200 * 4, 210000 + 4200 * 4],
        ]
    ),
)
