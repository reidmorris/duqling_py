import numpy as np
from ..utils import register_function

def ebola(x, scale01=True, location="liberia"):
    x = np.array(x, dtype=float)

    if scale01:
        RR = np.array([
            [0.1, 0.4], [0.1, 0.4], [0.05, 0.2], [0.41, 1],
            [0.0276, 0.1702], [0.081, 0.21], [0.25, 0.5], [0.0833, 0.7]
        ])
        x[:8] = x[:8] * (RR[:,1] - RR[:,0]) + RR[:,0]

    if location == "sierra_leone":
        x[4] = 0.0275 + ((x[4] - 0.0276) / (0.1702 - 0.0276)) * (0.1569 - 0.0275)
        x[5] = 0.081 + ((x[5] - 0.081) / (0.21 - 0.081)) * (0.384 - 0.1236)

    beta1, beta2, beta3, rho, gamma1, gamma2, omega, psi = x[:8]
    R0 = (beta1 + beta2 * rho * gamma1 / omega + beta3 * psi / gamma2) / (gamma1 + psi)
    return R0

register_function(
    fname='ebola',
    input_dim=8,
    input_cat=False,
    response_type='uni',
    stochastic='n',
    input_range=np.array([
        [0.1,    0.4],
        [0.1,    0.4],
        [0.05,   0.2],
        [0.41,   1],
        [0.0276, 0.1702],
        [0.081,  0.21],
        [0.25,   0.5],
        [0.0833, 0.7]
    ])
)