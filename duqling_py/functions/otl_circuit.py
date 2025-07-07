"""
OTL Circuit Function
"""

import numpy as np
from ..utils import register_function

def circuit(x, scale01=True):
    """
    OTL Circuit Function
    
    Dimensions: 6 The OTL Circuit function models an output transformerless push-pull circuit.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 6.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    
    Returns
    -------
    float
        The response Vm is midpoint voltage
    
    Notes
    -----
    For details on this function, see the VLSE.
    
    Parameter ranges:
    - x[0] = Rb1: resistance b1 (K-Ohms), [50, 150]
    - x[1] = Rb2: resistance b2 (K-Ohms), [25, 70]
    - x[2] = Rf: resistance f (K-Ohms), [0.5, 3]
    - x[3] = Rc1: resistance c1 (K-Ohms), [1.2, 2.5]
    - x[4] = Rc2: resistance c2 (K-Ohms), [0.25, 1.2]
    - x[5] = beta: current gain (Amps), [50, 300]
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: test functions and datasets." 
    Simon Fraser University, Burnaby, BC, Canada, accessed May 13 (2013): 2015.
    
    Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer experiments: an empirical comparison 
    of kriging with MARS and projection pursuit regression. Quality Engineering, 19(4), 327-338.
    """
    if scale01:
        # Define parameter ranges
        RR = np.array([
            [50, 150],     # Rb1
            [25, 70],      # Rb2
            [0.5, 3],      # Rf
            [1.2, 2.5],    # Rc1
            [0.25, 1.2],   # Rc2
            [50, 300]      # beta
        ])
        # Scale inputs from [0,1] to their native ranges
        x = x[:6] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    # Extract parameters
    Rb1 = x[0]
    Rb2 = x[1]
    Rf = x[2]
    Rc1 = x[3]
    Rc2 = x[4]
    beta = x[5]
    
    # Calculate midpoint voltage
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    
    term1a = (Vb1 + 0.74) * beta * (Rc2 + 9)
    term1b = beta * (Rc2 + 9) + Rf
    term1 = term1a / term1b
    
    term2a = 11.35 * Rf
    term2b = beta * (Rc2 + 9) + Rf
    term2 = term2a / term2b
    
    term3a = 0.74 * Rf * beta * (Rc2 + 9)
    term3b = (beta * (Rc2 + 9) + Rf) * Rc1
    term3 = term3a / term3b
    
    Vm = term1 + term2 + term3
    return Vm

# Register function with metadata
register_function(
    fname="circuit",
    input_dim=6,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [50, 150],     # Rb1
        [25, 70],      # Rb2
        [0.5, 3],      # Rf
        [1.2, 2.5],    # Rc1
        [0.25, 1.2],   # Rc2
        [50, 300]      # beta
    ])
)
