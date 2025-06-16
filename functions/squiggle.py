"""
The Squiggle Function.
"""

import numpy as np
from ..utils import register_function

def squiggle(x, scale01=True, sigma=0.05):
    """
    The Squiggle Function
    
    Dimensions: 2
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    sigma : float, optional
        A scaling parameter. Default is 0.05.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    A bivariate squiggle function which has regions of low activity as well as 
    high activity regions.
    """
    mu = np.sin(2*np.pi*x[0]**2)/4 - x[0]/10 + 0.5
    pref = 1.0 / (np.sqrt(2*np.pi) * sigma)      # <─ add this line
    return pref * np.exp(-(x[1]-mu)**2/(2*sigma**2)) * x[0] * x[1]

# Register function with metadata
register_function(
    fname="squiggle",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1]
    ])
)
