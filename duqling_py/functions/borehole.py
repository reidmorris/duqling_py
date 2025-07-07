"""
The Borehole Function
"""

import numpy as np
from ..utils import register_function

def borehole(x, scale01=True, adjust_fidelity=0):
    """
    The Borehole Function
    
    Dimensions: 8. The Borehole function models water flow through a borehole. 
    Its simplicity and quick evaluation makes it a commonly used function for 
    testing a wide variety of methods in computer experiments.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 8.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    adjust_fidelity : float, optional
        Default value of 0 corresponds to the usual borehole function. Value 
        of 1 corresponds to the low-fidelity version described in the VLSE (Xiong 2013).
    
    Returns
    -------
    float
        Flow through a borehole.
    
    Notes
    -----
    For details on the borehole function, see the VLSE.
    
    Parameter ranges:
    - x1 = rw: radius of borehole (m), [0.05, 0.15]
    - x2 = r: radius of influence (m), [100, 50000]
    - x3 = Tu: transmissivity of borehole (m^2/yr), [63070, 115600]
    - x4 = Hu: potentiometric head of upper aquifer (m), [990, 1110]
    - x5 = Tl: transmissivity of lower aquifer (m2/yr), [63.1, 116]
    - x6 = Hl: potentiometric head of lower aquifer (m), [700, 820]
    - x7 = L: length of borehole (m), [1120, 1680]
    - x8 = Kw: hydraulic conductivity of borehole (m/yr), [9855, 12045]
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty analysis of a 
    borehole scenario comparing Latin Hypercube Sampling and deterministic 
    sensitivity approaches (No. BMI/ONWI-516). Battelle Memorial Inst., Columbus, 
    OH (USA). Office of Nuclear Waste Isolation.
    
    Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and analysis of 
    high-accuracy and low-accuracy computer codes. Technometrics, 55(1), 37-46.
    """
    if scale01:
        RR = np.array([
            [0.05, 0.15],
            [100, 50000],
            [63070, 115600],
            [990, 1110],
            [63.1, 116],
            [700, 820],
            [1120, 1680],
            [9855, 12045]
        ])
        x = x[:8] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    rw = x[0]
    r = x[1]
    Tu = x[2]
    Hu = x[3]
    Tl = x[4]
    Hl = x[5]
    L = x[6]
    Kw = x[7]
    
    frac1 = (2 * np.pi - adjust_fidelity * 1.283185) * Tu * (Hu - Hl)
    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + adjust_fidelity / 2 + frac2a + frac2b)
    
    y = frac1 / frac2
    return y

# Register functions with metadata
register_function(
    fname="borehole",
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
