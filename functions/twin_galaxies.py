"""
Synthetic Twin Galaxy Function.
"""

import numpy as np
from .grlee import grlee2
from .lim import lim_polynomial
from .ripples import ripples
from ..utils import register_function

def twin_galaxies(x, scale01=True, fixed_coeff=True, seed=None):
    x = np.asarray(x, float).copy()
    if not scale01:
        x[0] /= 360
        x[1] = (x[1] + 90) / 180

    y  = 22 / 40 * lim_polynomial(x, scale01=True)
    y += 5 * grlee2(x, scale01=True)
    y += ripples(x, scale01=True, fixed_coeff=fixed_coeff, seed=seed)
    return y

# Register function with metadata
register_function(
    fname="twin_galaxies",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 360],
        [-90, 90]
    ])
)
