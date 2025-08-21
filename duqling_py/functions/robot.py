"""
Robot Arm Function.
"""

import numpy as np

from ..utils import register_function


def robot(x, scale01=True):
    """
    Robot Arm Function

    Dimensions: 8. Used commonly in neural network papers, models the position
    of a robot arm which has four segments. While the shoulder is fixed at the
    origin, the four segments each have length Li, and are positioned at an angle θi
    (with respect to the horizontal axis), for i = 1, …, 4.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 8.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.

    Returns
    -------
    float
        The response is the distance from the end of the robot arm to the origin,
        on the (u, v)-plane.

    Notes
    -----
    For details on this function, see the VLSE.

    Parameter ranges:
    - x1 = theta1: angle of 1st arm segment, [0, 2*pi]
    - x2 = theta2: angle of 2nd arm segment, [0, 2*pi]
    - x3 = theta3: angle of 3rd arm segment, [0, 2*pi]
    - x4 = theta4: angle of 4th arm segment, [0, 2*pi]
    - x5 = L1: length of 1st arm segment, [0, 1]
    - x6 = L2: length of 2nd arm segment, [0, 1]
    - x7 = L3: length of 3rd arm segment, [0, 1]
    - x8 = L4: length of 4th arm segment, [0, 1]

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer experiments:
    an empirical comparison of kriging with MARS and projection pursuit regression.
    Quality Engineering, 19(4), 327-338.
    """
    if scale01:
        RR = np.array(
            [
                [0, 2 * np.pi],
                [0, 2 * np.pi],
                [0, 2 * np.pi],
                [0, 2 * np.pi],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )
        x = x[:8] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    theta = x[0:4]
    L = x[4:8]

    # Create cumulative sum of angles
    sumtheta = np.zeros(4)
    for i in range(4):
        sumtheta[i] = np.sum(theta[0 : i + 1])

    u = np.sum(L * np.cos(sumtheta))
    v = np.sum(L * np.sin(sumtheta))

    y = np.sqrt(u**2 + v**2)
    return y


# Register function with metadata
register_function(
    fname="robot",
    input_dim=8,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    ),
)
