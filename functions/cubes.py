"""
Discontinuous Cube Functions.
"""

import numpy as np
from ..utils import register_function

def cube3(x, scale01=True):
    """
    Discontinuous Cube Function (3D)
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    A 3-dimensional test function equal to the sum of 3 indicator functions:
    f(x) = 1(0.2 < x1 < 0.5)1(0.1 < x2 < 0.6)1(x3 < 0.4) + 
           1(0.3 < x1 < 0.85)1(0.5 < x2 < 0.9)1(0.6 < x3) + 
           1(0.35 < x1 < 0.45)1(x2 < 0.75)
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    
    c1 = ((0.2 < x1) & (x1 < 0.5) & (0.1 < x2) & (x2 < 0.6) & (x3 < 0.4))
    c2 = ((0.3 < x1) & (x1 < 0.85) & (0.5 < x2) & (x2 < 0.9) & (x3 < 0.8))
    c3 = ((0.35 < x1) & (x1 < 0.45) & (x2 < 0.75))
    
    return float(c1) + float(c2) + float(c3)

def cube3_rotate(x, scale01=True):
    """
    Rotated Discontinuous Cube Function (3D)
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    Equal to cube3(z1,z2,z3) after applying the rotation:
    3z1 = x1 + x2 + x3
    3z2 = 1 + 2x1 - x2 + x3/10
    z3 = x3
    """
    z1 = (x[0] + x[1] + x[2]) / 3
    z2 = (1 + 2*x[0] - x[1] + x[2]/10) / 3
    z3 = x[2]
    
    return cube3([z1, z2, z3])

def cube5(x, scale01=True):
    """
    Discontinuous Cube Function (5D)
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 5.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. No effect for this function.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    A 5-dimensional test function equal to:
    cube3(x1,x2,x3) + prod_{i=1:5}1(0.25 < xi < 0.75) + 5*1(0.33 < x5)
    """
    base = cube3(x[:3])
    
    # Check if all 5 dimensions are within [0.25, 0.75]
    cube_indicator = 1.0
    for i in range(5):
        if not (0.25 < x[i] < 0.75):
            cube_indicator = 0.0
            break
    
    # Indicator for x5 > 0.33
    x5_indicator = 5.0 if x[4] > 1/3 else 0.0
    
    return base + cube_indicator + x5_indicator

# Register functions with metadata
register_function(
    fname="cube3",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1],
        [0, 1]
    ])
)

register_function(
    fname="cube3_rotate",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1],
        [0, 1]
    ])
)

register_function(
    fname="cube5",
    input_dim=5,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
)
