"""
The G-function Functions.
"""

import numpy as np
from ..utils import register_function

def Gfunction(x, scale01=True, a=None):
    if a is None:
        p = x.size
        a = np.arange(p) / 4
    u = (np.abs(4 * x - 2) + a) / (1 + a)
    return np.prod(u)

def Gfunction6(x, scale01=True, a=None):
    """
    The G-function (6D)
    
    Dimensions: 6. A multiplicative function.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 6.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    a : array_like, optional
        Parameter vector indicating the importance of variables (lower value is more important).
        Default is [0, 0, 6.52, 6.52, 6.52, 6.52].
    
    Returns
    -------
    float
        A scalar response.
    
    Notes
    -----
    The G function is often used as an integrand for various numerical estimation methods. 
    The exact value of the integral of this function in any dimensions is 1. The a_i values 
    indicate the importance of a variable (lower value is more important).
    
    For details on the G function, see the VLSE.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Gfunction, J. H. (1991). Multivariate adaptive regression splines. The annals 
    of statistics, 19(1), 1-67.
    """
    if a is None:
        a = np.array([0, 0, 6.52, 6.52, 6.52, 6.52])
    a = np.asarray(a, dtype=float)
    x = x[:6]
    u = (np.abs(4 * x - 2) + a) / (1 + a)
    return np.prod(u)

def Gfunction12(x, scale01=True, a=None):
    """
    The G-function (12D)
    
    Dimensions: 12. A multiplicative function.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 12.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    a : array_like, optional
        Parameter vector indicating the importance of variables (lower value is more important).
        Default is [0, 0, 6.52, 6.52, 6.52, 6.52, 9, 9, 15, 25, 50, 99].
    
    Returns
    -------
    float
        A scalar response.
    
    Notes
    -----
    The G function is often used as an integrand for various numerical estimation methods. 
    The exact value of the integral of this function in any dimensions is 1. The a_i values 
    indicate the importance of a variable (lower value is more important).
    
    For details on the G function, see the VLSE.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Gfunction, J. H. (1991). Multivariate adaptive regression splines. The annals 
    of statistics, 19(1), 1-67.
    """
    if a is None:
        a = np.array([0, 0, 6.52, 6.52, 6.52, 6.52, 9, 9, 15, 25, 50, 99])
    a = np.asarray(a, dtype=float)
    x = x[:12]
    u = (abs(4 * x - 2) + a) / (1 + a)
    res = np.prod(u)
    return res

def Gfunction18(x, scale01=True, a=None):
    """
    The G-function (18D)
    
    Dimensions: 18. A multiplicative function.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 18.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    a : array_like, optional
        Parameter vector indicating the importance of variables (lower value is more important).
        Default is [0, 0, 0, 0, 1, 1, 1, 1, 9, 9, 9, 9, 18, 18, 18, 18, 99, 99].
    
    Returns
    -------
    float
        A scalar response.
    
    Notes
    -----
    The G function is often used as an integrand for various numerical estimation methods. 
    The exact value of the integral of this function in any dimensions is 1. The a_i values 
    indicate the importance of a variable (lower value is more important).
    
    For details on the G function, see the VLSE.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Gfunction, J. H. (1991). Multivariate adaptive regression splines. The annals 
    of statistics, 19(1), 1-67.
    """
    if a is None:
        a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 9, 9, 9, 9, 18, 18, 18, 18, 99, 99])
    a = np.asarray(a, dtype=float)
    x = x[:18]
    u = (abs(4 * x - 2) + a) / (1 + a)
    res = np.prod(u)
    return res

# Register functions with metadata

register_function(
    fname="Gfunction",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((3, 2)) + np.array([[0, 1]])
)

register_function(
    fname="Gfunction6",
    input_dim=6,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((6, 2)) + np.array([[0, 1]])
)

register_function(
    fname="Gfunction12",
    input_dim=12,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((12, 2)) + np.array([[0, 1]])
)

register_function(
    fname="Gfunction18",
    input_dim=18,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.zeros((18, 2)) + np.array([[0, 1]])
)
