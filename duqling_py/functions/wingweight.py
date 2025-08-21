"""
Wing Weight Function.
"""

import numpy as np

from ..utils import register_function


def wingweight(x, scale01=True):
    """
    Wing Weight Function

    Dimensions: 10. Models a light aircraft wing.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 10.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.

    Returns
    -------
    float
        The response is the wing's weight.

    Notes
    -----
    For details on this function, see the VLSE.

    Parameter ranges:
    - x1 = Sw: wing area (ft^2), [150, 200]
    - x2 = Wfw: weight of fuel in the wing (lb), [220, 300]
    - x3 = A: aspect ratio, [6, 10]
    - x4 = Lam: quarter chord sweep (degrees), [-10, 10]
    - x5 = q: dynamic pressure at cruise (lb/ft^2), [16, 45]
    - x6 = lam: taper ratio, [0.5, 1]
    - x7 = tc: aerofoil thickness to chord ratio, [0.08, 0.18]
    - x8 = Nz: ultimate load factor, [2.5, 6]
    - x9 = Wdg: flight design gross weight (lb), [1700, 2500]
    - x10 = Wp: paint weight (lb/ft^2), [0.025, 0.08]

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via
    surrogate modelling: a practical guide. Wiley.

    Moon, H. (2010). Design and Analysis of Computer Experiments for Screening
    Input Variables (Doctoral dissertation, Ohio State University).
    """
    if scale01:
        RR = np.array(
            [
                [150, 200],
                [220, 300],
                [6, 10],
                [-10, 10],
                [16, 45],
                [0.5, 1.0],
                [0.08, 0.18],
                [2.5, 6.0],
                [1700, 2500],
                [0.025, 0.08],
            ]
        )
        x = x.copy()
        x = x[:10] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    Sw = x[0]
    Wfw = x[1]
    A = x[2]
    LamCaps = x[3] * (np.pi / 180)  # Convert to radians
    q = x[4]
    lam = x[5]
    tc = x[6]
    Nz = x[7]
    Wdg = x[8]
    Wp = x[9]

    fact1 = 0.036 * Sw**0.758 * Wfw**0.0035
    fact2 = (A / ((np.cos(LamCaps)) ** 2)) ** 0.6
    fact3 = q**0.006 * lam**0.04
    fact4 = (100 * tc / np.cos(LamCaps)) ** (-0.3)
    fact5 = (Nz * Wdg) ** 0.49

    term1 = Sw * Wp

    y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
    return y


# Register function with metadata
register_function(
    fname="wingweight",
    input_dim=10,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [150, 200],
            [220, 300],
            [6, 10],
            [-10, 10],
            [16, 45],
            [0.5, 1.0],
            [0.08, 0.18],
            [2.5, 6.0],
            [1700, 2500],
            [0.025, 0.08],
        ]
    ),
)
