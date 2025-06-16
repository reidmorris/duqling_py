"""
The Low-Fidelity variant of the Borehole Function.
"""

import numpy as np
from .borehole import borehole
from ..utils import register_function

def borehole_low_fidelity(x, scale01=True):
    """
    The Low Fidelity Borehole Function
    
    Dimensions 8. Low fidelity version of the borehole function (Xiong 2013). 
    See borehole for more details.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 8.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    
    Returns
    -------
    float
        Flow through a borehole.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and analysis of 
    high-accuracy and low-accuracy computer codes. Technometrics, 55(1), 37-46.
    """
    return borehole(x, scale01, adjust_fidelity=1)

register_function(
    fname="borehole_low_fidelity",
    input_dim=8,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0.05, 0.15],
        [100, 50000],
        [63070, 115600],
        [990, 1110],
        [63.1, 116],
        [700, 820],
        [1120, 1680],
        [9855, 12045]
    ])
)