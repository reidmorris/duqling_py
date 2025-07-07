"""
The Park Functions.
"""

import numpy as np
from ..utils import register_function

def park4(x, scale01=True):
    """
    The Park4 Function
    
    Dimensions: 4.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 4.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    Intended to be a simple four-dim test function. Xiong et al. (2013) use 
    park4_low_fidelity as a low fidelity version.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Park, Jeong Soo. Tuning complex computer codes to data and optimal designs. 
    University of Illinois at Urbana-Champaign, 1991.
    
    Xiong, Shifeng, Peter ZG Qian, and CF Jeff Wu. "Sequential design and analysis 
    of high-accuracy and low-accuracy computer codes." Technometrics 55.1 (2013): 37-46.
    """
    term1 = x[0] / 2 * (np.sqrt(1 + (x[1] + x[2]**2) * x[3] / x[0]**2) - 1)
    term2 = (x[0] + 3 * x[3]) * np.exp(1 + np.sin(x[2]))
    
    res = term1 + term2
    return res

def park4_low_fidelity(x, scale01=True):
    """
    The Park4 Low Fidelity Function
    
    Dimensions: 4.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 4.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    Low fidelity version of park4 function.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Park, Jeong Soo. Tuning complex computer codes to data and optimal designs. 
    University of Illinois at Urbana-Champaign, 1991.
    
    Xiong, Shifeng, Peter ZG Qian, and CF Jeff Wu. "Sequential design and analysis 
    of high-accuracy and low-accuracy computer codes." Technometrics 55.1 (2013): 37-46.
    """
    term1 = x[0] / 2 * (np.sqrt(1 + (x[1] + x[2]**2) * x[3] / x[0]**2) - 1)
    term2 = (x[0] + 3 * x[3]) * np.exp(1 + np.sin(x[2]))
    term3 = -2 * x[0] + x[1]**2 + x[2]**2 + 1/2
    
    res = (1 + np.sin(x[0]) / 10) * (term1 + term2) + term3
    return res

# Register functions with metadata
for func_name in ["park4", "park4_low_fidelity"]:
    register_function(
        fname=func_name,
        input_dim=4,
        input_cat=False,
        response_type="uni",
        stochastic="n",
        input_range=np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
        ])
    )
