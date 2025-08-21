"""
Sulfur Function.
"""

import numpy as np

from ..utils import register_function


def sulfur(x, scale01=True):
    """
    Sulfur

    Dimension 9. Models the radiative forcing of sulfur with uncertain inputs.
    High degree of interaction (but additive in log space).

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 9.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    This function models the radiative forcing of sulfur by sulfate aerosols, in W/m^2.
    Note that all priors are lognormal.

    For details, see the VLSE. NOTE: There is a problem with dimensional analysis of
    the Tatang paper. There is a hidden (or missing) W in the denominator. As such,
    the results do not match the paper perfectly. Use this function at your own risk.

    References
    ----------
    Tatang, Menner A., et al. "An efficient method for parametric uncertainty analysis
    of numerical geophysical models." Journal of Geophysical Research: Atmospheres
    102.D18 (1997): 21925-21932

    Charlson, Robert J., et al. "Climate forcing by anthropogenic aerosols."
    Science 255.5043 (1992): 423-430.
    """
    if scale01:
        RR = np.array(
            [
                [0.76 * 1.2 ** (-3), 0.76 * 1.2**3],
                [0.39 * 1.1 ** (-3), 0.39 * 1.1**3],
                [0.85 * 1.1 ** (-3), 0.85 * 1.1**3],
                [0.3 * 1.3 ** (-3), 0.3 * 1.3**3],
                [5.0 * 1.4 ** (-3), 5.0 * 1.4**3],
                [1.7 * 1.2 ** (-3), 1.7 * 1.2**3],
                [71 * 1.15 ** (-3), 71 * 1.15**3],
                [0.5 * 1.5 ** (-3), 0.5 * 1.5**3],
                [5.5 * 1.5 ** (-3), 5.5 * 1.5**3],
            ]
        )
        x = x[:9] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]

    Tr = x[0]
    Ac_c = x[1]
    Rs_c = x[2]
    beta = x[3]
    psi_e = x[4]
    f_psi = x[5]
    Q = x[6]
    Y = x[7]
    L = x[8]

    # Constants
    S0 = 1366  # solar constant
    A = 5.1e14  # surface area of the earth

    res = (
        -1
        / 2
        * S0**2
        * Ac_c
        * Tr**2
        * Rs_c**2
        * beta
        * psi_e
        * f_psi
        * 3
        * Q
        * Y
        * L
        / A
        * 10**12
        / 365
    )
    res = (
        res / 1000
    )  # Just an attempt to match the original paper. See note in documentation.

    return res


# Register function with metadata
register_function(
    fname="sulfur",
    input_dim=9,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [
            [0.76 * 1.2 ** (-3), 0.76 * 1.2**3],
            [0.39 * 1.1 ** (-3), 0.39 * 1.1**3],
            [0.85 * 1.1 ** (-3), 0.85 * 1.1**3],
            [0.3 * 1.3 ** (-3), 0.3 * 1.3**3],
            [5.0 * 1.4 ** (-3), 5.0 * 1.4**3],
            [1.7 * 1.2 ** (-3), 1.7 * 1.2**3],
            [71 * 1.15 ** (-3), 71 * 1.15**3],
            [0.5 * 1.5 ** (-3), 0.5 * 1.5**3],
            [5.5 * 1.5 ** (-3), 5.5 * 1.5**3],
        ]
    ),
)
