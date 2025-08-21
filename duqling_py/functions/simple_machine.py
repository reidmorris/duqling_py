"""
Simple Machine Functions.
"""

import numpy as np

from ..utils import register_function


def simple_machine(x, scale01=True, effort=np.arange(1, 11), friction=0.1, base_eff=0):
    x = list(x)
    if len(x) < 2:
        x.append(friction)
    if len(x) < 3:
        x.append(base_eff)

    if scale01:
        x[0] *= 3
        x[1] *= 50
        x[2] = x[2] * 2 - 1

    return effort * x[0] / (1 + effort * x[1]) + x[2]


def simple_machine_cm(x, scale01=True, effort=np.arange(1, 11), order=3):
    x = list(x) + [0] * (5 - len(x))  # pad to length-5
    if order == 1:
        x[1:5] = [1e9, 0, 1e9, 0]
    elif order == 2:
        x[3:5] = [1e9, 0]

    if scale01:
        x[0] *= 3
        for i in (1, 3):
            x[i] *= 10
        for i in (2, 4):
            x[i] *= 3

    e = np.asarray(effort, float)
    y1 = x[0] * e * (e < x[1])
    y2 = (x[0] * x[1] + x[2] * (e - x[1])) * ((e >= x[1]) & (e < x[3]))
    y3 = (x[0] * x[1] + x[2] * (x[3] - x[1]) + x[4] * (e - x[3])) * (e >= x[3])
    return y1 + y2 + y3


# Register functions with metadata
register_function(
    fname="simple_machine",
    input_dim=3,
    input_cat=False,
    response_type="func",
    stochastic="n",
    input_range=np.array([[0, 3], [0, 10], [-1, 1]]),
)

register_function(
    fname="simple_machine_cm",
    input_dim=5,
    input_cat=False,
    response_type="func",
    stochastic="n",
    input_range=np.array([[0, 3], [0, 10], [0, 3], [0, 10], [0, 3]]),
)
