import numpy as np
from ..utils import register_function

def permdb(x, scale01=True, d=16, beta=0.5):
    x = np.asarray(x, dtype=np.float64).copy()

    if scale01:
        x[:d] = x[:d] * (2 * d) - d  # scale from [0,1] to [-d, d]

    ii = np.arange(1, d + 1, dtype=np.float64)
    jj = np.tile(ii, (d, 1))
    xxmat = np.tile(x[:d], (d, 1))

    inner = np.sum((jj**ii[:, None] + beta) * ((xxmat / jj)**ii[:, None] - 1), axis=1)
    outer = np.sum(inner**2)

    return outer

register_function(
    fname='permdb',
    input_dim=16,
    input_cat=False,
    response_type='uni',
    stochastic='n',
    input_range=np.tile([-16, 16], (16, 1))
)
