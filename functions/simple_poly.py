"""
Simple Polynomial Function from Rumsey et al 2023.
"""

import numpy as np
from ..utils import register_function

def simple_poly(x, beta=1/9, scale01=True):
    """
    Simple Polynomial Function from Rumsey et al 2023
    
    Dimensions: 2.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    beta : float, optional
        Coefficient parameter. Default is 1/9.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    
    Returns
    -------
    float
        A scalar response.
    
    Notes
    -----
    A simple polynomial function.
    
    References
    ----------
    Rumsey, Kellin N., et al. "Co-Active Subspace Methods for the Joint Analysis 
    of Adjacent Computer Models." arXiv preprint arXiv:2311.18146 (2023).
    """
    return x[0]**2 + x[0]*x[1] + beta*x[1]**3

# Register function with metadata
register_function(
    fname="simple_poly",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1]
    ])
)
