import numpy as np

from ..utils import register_function


def rabbits(x, scale01=True):
    x = x.copy()

    if scale01:
        bounds = np.array([[0, 1], [0, 1], [0.5, 3]])  # P0  # t  # r
        x[:3] = x[:3] * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    P0, t, r = x[:3]
    res = (P0 * np.exp(r * t)) / (1 + P0 * (np.exp(r * t) - 1))
    return res


register_function(
    fname="rabbits",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1], [0.5, 3]]),
)
