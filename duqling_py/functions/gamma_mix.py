import numpy as np
from scipy.stats import gamma

from ..utils import register_function


def gamma_mix(x, scale01=True):
    x = x.copy()
    bounds = np.array([[0, 50], [1, 10], [0.5, 5], [1, 10], [0.5, 5], [0, 1], [0, 10]])

    if scale01:
        x[:7] = x[:7] * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    loc, a1, b1, a2, b2, w, m = x[:7]
    z = loc - m
    d1 = gamma.pdf(z, a=a1, scale=1 / b1)
    d2 = gamma.pdf(z, a=a2, scale=1 / b2)
    y = (w * d1 + (1 - w) * d2) * (z >= 0)
    return y


register_function(
    fname="gamma_mix",
    input_dim=7,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array(
        [[0, 50], [1, 10], [0.5, 5], [1, 10], [0.5, 5], [0, 1], [0, 10]]
    ),
)
