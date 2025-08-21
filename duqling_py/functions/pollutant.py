"""
Pollutant spill in a channel (4‑D input, space‑time field output)
Port of `pollutant()` from pollutant.R
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..utils import register_function

# ‑‑ parameter ranges used when scale01 = True (rows = variables) ‑‑
_RR = np.array(
    [
        [7.00, 13.00],  # M   : mass of pollutant [kg] spilled at each site
        [0.02, 0.12],  # D   : diffusion rate     [m² s⁻¹]
        [0.01, 3.00],  # L   : location of 2nd spill [m]
        [30.01, 30.295],
    ]  # tau : time of 2nd spill    [s]
)


def _default_space_time(space, time):
    if space is None:
        space = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    else:
        space = np.asarray(space, dtype=float)

    if time is None:
        # seq(0.3, 60, 0.3) in R
        time = np.arange(0.3, 60.0 + 1e-12, 0.3, dtype=float)
    else:
        time = np.asarray(time, dtype=float)

    return space, time


def pollutant(
    x: ArrayLike,
    /,
    *,
    scale01: bool = True,
    space: ArrayLike | None = None,
    time: ArrayLike | None = None,
):
    """
    Concentration field produced by two instantaneous spills in a 1-D channel.

    Parameters
    ----------
    x : length-4 array-like
        (M, D, L, tau) in that order.  If *scale01* is True the four values are
        assumed to be in [0, 1] and are mapped to their native ranges.
    scale01 : bool, default True
        Whether to apply the unit-box → native-range transform to *x*.
    space, time : array-like or None
        Locations *s* (m) and times *t* (s) at which to evaluate the field.
        If omitted the VLSE defaults are used.

    Returns
    -------
    • If ``len(space) == len(time) == 1`` → float
    • If exactly one of them has length 1 → 1-D array
    • Otherwise → 2-D array shaped (len(space), len(time))

    Notes
    -----
    Formula (Bliznyuk et al., 2008):

        term₁(s,t) = M / √(4 π D t) ⋅ exp(-s² / (4 D t))
        term₂(s,t) = I(t > τ) ⋅
                     M / √(4 π D (t-τ)) ⋅ exp(-(s-L)² / (4 D (t-τ)))
        C(s,t)     = √(4 π) ⋅ [term₁ + term₂]
    """
    x = np.asarray(x, dtype=float).copy()
    if x.size < 4:
        raise ValueError("x must contain at least 4 numbers (M, D, L, tau).")

    # 0‑1 → native scaling
    if scale01:
        x[:4] = x[:4] * (_RR[:, 1] - _RR[:, 0]) + _RR[:, 0]

    M, D, L, tau = x[:4]
    space, time = _default_space_time(space, time)

    # vectorised implementation
    s_mat, t_mat = np.meshgrid(space, time, indexing="ij")  # rows = s, cols = t

    # first spill at s = 0, t = 0 (but R reference shifts time by 30 s)
    term1 = (
        M / np.sqrt(4.0 * np.pi * D * t_mat) * np.exp(-(s_mat**2) / (4.0 * D * t_mat))
    )

    # second spill, only contributes when t > tau
    mask = t_mat > tau
    term2 = np.zeros_like(term1)
    dt = t_mat[mask] - tau
    term2[mask] = (
        M
        / np.sqrt(4.0 * np.pi * D * dt)
        * np.exp(-((s_mat[mask] - L) ** 2) / (4.0 * D * dt))
    )

    C = np.sqrt(4.0 * np.pi) * (term1 + term2)

    # conform to R's return conventions
    if C.shape == (1, 1):
        return float(C[0, 0])
    if C.shape[0] == 1:
        return C[0, :]
    if C.shape[1] == 1:
        return C[:, 0]
    return C.T.flatten("F")


# Register function with metadata
register_function(
    fname="pollutant",
    input_dim=4,
    input_cat=False,
    response_type="func",
    stochastic="n",
    input_range=np.array(
        [[7, 13], [0.02, 0.12], [0.01, 3], [30.01, 30.295]]  # M  # D  # L  # tau
    ),
)
