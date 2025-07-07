"""
The 100D Function.
"""

import numpy as np
from ..utils import register_function

def onehundred(x, scale01=True, M=100):
    """
    The 100D Function
    
    Dimension: 100
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 100.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    M : int, optional
        The number of active variables (default M=100, (54 <= M <= 100)).
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    The 100-dimensional function is a high-dimensional function derived from a more 
    generic example detailed below. This function was designed for sensitivity analysis: 
    the first-order sensitivity indices of the input variables generally nonlinearly 
    increase with their index, with certain variables having especially high sensitivity 
    (Lüthen et al, 2021).
    
    References
    ----------
    Lüthen, Nora, Stefano Marelli, and Bruno Sudret. "Sparse polynomial chaos expansions: 
    Literature survey and benchmark." SIAM/ASA Journal on Uncertainty Quantification 
    9.2 (2021): 593-649.
    
    UQLab - The Framework for Uncertainty Quantification. "Sensitivity: Sobol' indices 
    of a high-dimensional function." Retrieved July 3, 2024, from 
    https://www.uqlab.com/sensitivity-high-dimension.
    """
    if scale01:
        lb = np.ones(100)
        ub = np.ones(100) * 2
        ub[19] = 3  # Special case for x20
        RR = np.column_stack((lb, ub))
        x = x[:100] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    linear = np.arange(1, len(x) + 1)
    select = (linear <= M).astype(float)
    
    term1 = 3 - 5 / M * np.sum(linear * select * x[:len(linear)])
    term2 = 1 / M * np.sum(linear * select * x[:len(linear)]**3)
    term3 = 1 / (3 * M) * np.sum(linear * select * np.log(x[:len(linear)]**2 + x[:len(linear)]**4))
    term4 = x[0] * x[1]**2 + x[1] * x[3] - x[2] * x[4] + x[50] + x[49] * x[53]**2
    
    res = term1 + term2 + term3 + term4
    return res

# Register function with metadata
register_function(
    fname="onehundred",
    input_dim=100,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.column_stack((np.ones(100), np.ones(100) * 2 + (np.arange(100) == 19)))
)
