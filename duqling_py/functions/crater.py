"""
Crater Depth Function.
"""

import numpy as np

from ..utils import register_function


def crater(x, scale01=True, const=0.0195, power=0.45):
    """
    Crater Depth

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 7.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval.
    const : float, optional
        Constant for the crater model. Default is 0.0195 per the "original R&C formula".
    power : float, optional
        Power parameter for the crater model. Default is 0.45 per the "original R&C formula".

    Returns
    -------
    float
        Penetration depth (in)

    Notes
    -----
    A physics-based function for predicting crater depth resulting from high-velocity impacts,
    such as those experienced by the Space Shuttle Columbia during reentry.

    Input variables include:
    - x1 = L: length of projectile (in)
    - x2 = d: diameter of projectile (in)
    - x3 = rho_p: density of projectile (lb/in^3)
    - x4 = V: normal velocity (in/s)
    - x5 = V_star: velocity required to break through tile coating (in/s)
    - x6 = S_t: compressive tile strength (psi)
    - x7 = rho_t: tile density (lb/in^3)

    The const and power inputs default to 0.0195 and 0.45 per the "original R&C formula"
    but can be varied; Stellingwerf et al. set these to 0.67 and 0.0086.

    References
    ----------
    Stellingwerf, R., Robinson, J., Richardson, S., Evans, S., Stallworth, R., & Hovater, M. (2004).
    Foam-on-tile impact modeling for the STS-107 investigation. In 45th AIAA/ASME/ASCE/AHS/ASC
    Structures, Structural Dynamics & Materials Conference (p. 1881).
    """
    if scale01:
        RR = np.array(
            [
                [0.01, 20],
                [0.01, 6],
                [0.0001, 0.04],
                [730, 3200],
                [1500, 2500],
                [25, 77],
                [0.005, 0.0055],
            ]
        )
        x = x[:7] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    # Variable definitions
    L = x[0]  # length of projectile (in)
    d = x[1]  # diameter of projectile (in)
    rho_p = x[2]  # density of projectile (lb/in^3)
    V = x[3]  # normal velocity (in/s)
    V_star = x[4]  # velocity required to break through the tile coating (in/s)
    S_t = x[5]  # compressive strength of tile (psi)
    rho_t = x[6]  # density of tile (lb/in^3)

    res = (
        const
        * (L / d) ** power
        * d
        * rho_p**0.27
        * max(V - V_star, 0) ** (2 / 3)
        * S_t ** (-0.25)
        * rho_t ** (-1 / 6)
    )
    return res


# Register function with metadata
register_function(
    fname="crater",
    input_dim=7,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [0.01, 20],
            [0.01, 6],
            [0.0001, 0.04],
            [730, 3200],
            [1500, 2500],
            [25, 77],
            [0.005, 0.0055],
        ]
    ),
)
