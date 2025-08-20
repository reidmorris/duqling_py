"""
The 100D Function.
"""

import warnings
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

register_function(
    fname="onehundred",
    input_dim=100,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.column_stack((np.ones(100), np.ones(100) * 2 + (np.arange(100) == 19)))
)

def d_onehundred(x, scale01: bool = True, M: int = 100) -> np.ndarray:
    """
    Gradient of the 100D Function

    Input dimension: 100
    Output dimension: 100 (first M entries populated)

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 100.
    scale01 : bool, optional
        When True, inputs are expected on [0,1] and are internally scaled.
    M : int, optional
        Number of active variables (default 100; must be at least 55).

    Returns
    -------
    np.ndarray
        Gradient vector of length M.
    """
    x = x.copy()
    if M < 55:
        warnings.warn("M must be at least 55; setting M=55", RuntimeWarning)
        M = 55

    # Scaling bounds
    lb = np.ones(100, dtype=np.float64)
    ub = np.full(100, 2.0, dtype=np.float64)
    ub[19] = 3.0  # x20 upper bound is 3

    if scale01:
        # Scale first 100 dims from [0,1] to [lb, ub]
        x[:100] = x[:100] * (ub - lb) + lb

    # Allocate result
    res = np.empty(M, dtype=np.float64)

    # Linear gradient terms
    for k in range(1, M + 1):
        xi = x[k - 1]
        t1 = -5 * k / M
        t2 = 3 * k * (xi ** 2) / M
        denom = xi ** 3 + xi
        t3 = k * (4 * (xi ** 2) + 2) / denom / (3 * M)
        res[k - 1] = t1 + t2 + t3

    # Interaction terms
    res[0]  += x[1] ** 2
    res[1]  += 2.0 * x[0] * x[1] + x[3]
    res[2]  -= x[4]
    res[3]  += x[1]
    res[4]  -= x[2]
    if M >= 50:
        res[49] += x[53] ** 2
    if M >= 51:
        res[50] += 1.0
    if M >= 54:
        res[53] += 2.0 * x[53] * x[49]

    # Chain rule
    if scale01:
        res *= (ub[:M] - lb[:M])

    return res

register_function(
    fname="d_onehundred",
    input_dim=100,
    input_cat=False,
    response_type="multi",
    stochastic="n",
    input_range=np.column_stack([np.ones(100), np.where(np.arange(100) == 19, 3.0, 2.0)])
)
