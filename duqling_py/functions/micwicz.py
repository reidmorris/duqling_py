"""
Multivalley (Michalewicz) Function.
"""

import numpy as np

from ..utils import register_function


def multivalley(x, scale01=True, m=10, active_dim=None):
    """
    The Multivalley Function (Michalewicz Function)

    Dimension: Default is 2, but can be any dimension. For inert variables,
    active_dim must be specified.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are
        internally adjusted to their native range. Default is True.
    m : float, optional
        A larger m leads to a more difficult search (steeper valleys/ridges). Default is 10.
    active_dim : int, optional
        Only the first min(active_dim, len(x)) variables will be used. All other variables are inert.
        Default is len(x).

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    Also called the Michalewicz function. For details on this function, see the VLSE.

    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments:
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada,
    accessed May 13 (2013): 2015.

    Molga, M., & Smutnicki, C. Test functions for optimization needs (2005). Retrieved
    June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    """
    if active_dim is None:
        active_dim = len(x)

    p = min(active_dim, len(x))

    if scale01:
        RR = np.zeros((p, 2)) + np.array([[0, 2 * np.pi]])
        x = x[:p] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    else:
        x = x[:p]

    y = 0
    for i in range(p):
        y -= np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * m)

    return y


# Register function with metadata
register_function(
    fname="multivalley",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 2 * np.pi], [0, 2 * np.pi]]),
)
