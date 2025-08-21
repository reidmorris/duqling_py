"""
Discrete Time Stochastic SIRS Model with Demography.
"""

from typing import Optional
import numpy as np
from ..utils import register_function

def dts_sirs(x, scale01=True, Tf=90, N0=1000, seed: Optional[int]=None, rng: Optional[np.random.Generator] = None):
    rng = np.random.default_rng(seed) if rng is None else rng

    S = np.full(Tf, np.nan, dtype=int); I = S.copy(); R = S.copy(); N = S.copy()
    N[0] = N0
    I[0] = np.round(x[0] * N0).astype(int)
    S[0] = np.floor(x[1] * (N0 - I[0])).astype(int)
    R[0] = N[0] - S[0] - I[0]

    for t in range(1, Tf):
        births  = rng.poisson(x[4] * N[t-1])
        deaths  = rng.binomial([S[t-1], I[t-1], R[t-1]], x[5:8])
        infect  = rng.binomial(S[t-1] - deaths[0],
                                        1 - np.exp(-x[2] * I[t-1] / N[t-1]))
        recove  = rng.binomial(I[t-1] - deaths[1], x[3])
        resusc  = rng.binomial(R[t-1] - deaths[2], x[8])

        S[t] = S[t-1] + births - deaths[0] - infect + resusc
        I[t] = I[t-1]          - deaths[1] + infect - recove
        R[t] = R[t-1]          - deaths[2] + recove - resusc
        N[t] = S[t] + I[t] + R[t]
        if N[t] <= 0:
            break

    return np.column_stack((S, I, R))

# Register function with metadata
register_function(
    fname="dts_sirs",
    input_dim=9,
    input_cat=False,
    response_type="func",
    stochastic="y",
    input_range=np.zeros((9, 2)) + np.array([[0, 1]])  # Create 9x2 array with [0,1] ranges
)
