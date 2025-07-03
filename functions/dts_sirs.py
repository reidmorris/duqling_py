"""
Discrete Time Stochastic SIRS Model with Demography.
"""

import warnings
import numpy as np
from ..utils import register_function

def dts_sirs(x, Tf=90, N0=1000):
    if x[0] + x[1] > 1:
        warnings.warn(f"I0 will be reduced from {x[1]} to {1 - x[0]} due to S0 constraint")
        x[1] = 1 - x[0]

    S = np.full(Tf, np.nan, dtype=int); I = S.copy(); R = S.copy(); N = S.copy()
    N[0] = N0
    S[0], I[0] = np.round(N0 * np.array(x[:2])).astype(int)
    R[0] = N[0] - S[0] - I[0]

    for t in range(1, Tf):
        births  = np.random.poisson(x[4] * N[t-1])
        deaths  = np.random.binomial([S[t-1], I[t-1], R[t-1]], x[5:8])
        infect  = np.random.binomial(S[t-1] - deaths[0],
                                        1 - np.exp(-x[2] * I[t-1] / N[t-1]))
        recove  = np.random.binomial(I[t-1] - deaths[1], x[3])
        resusc  = np.random.binomial(R[t-1] - deaths[2], x[8])

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
