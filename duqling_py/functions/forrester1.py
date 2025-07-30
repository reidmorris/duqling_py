import numpy as np
from ..utils import register_function

def forrester1(x, scale01 = True, A = 1, B = 0, C = 0):
    fx = (6 * x[0] - 0.5)**2 * np.sin(12 * x[0] - 4)
    return A * fx + B * (x - 0.5) - C

def forrester1_low_fidelity(x, scale01 = True):
    return forrester1(x, scale01 = scale01, A = 0.5, B = 10, C = -5)

register_function(
    fname='forrester1',
    input_dim=1,
    input_cat=False,
    response_type='uni',
    stochastic='n',
    input_range=np.array([
        [0, 1]
    ])
)

register_function(
    fname='forrester1_low_fidelity',
    input_dim=1,
    input_cat=False,
    response_type='uni',
    stochastic='n',
    input_range=np.array([
        [0, 1]
    ])
)
