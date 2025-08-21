"""
Piston Simulation Functions.
"""

from typing import Optional
import numpy as np
from ..utils import register_function

def piston(x, scale01=True):
    """
    Piston Simulation Function
    
    Dimensions: 7. The Piston Simulation function models the circular motion 
    of a piston within a cylinder. It involves a chain of nonlinear functions.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 7.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    
    Returns
    -------
    float
        The response C is cycle time (the time it takes to complete one cycle), in seconds.
    
    Notes
    -----
    For details on the piston function, see the VLSE.
    
    Parameter ranges:
    - x1 = M: piston weight (kg), [30, 60]
    - x2 = S: piston surface area (m^2), [0.005, 0.020]
    - x3 = V0: initial gas volume (m^3), [0.002, 0.010]
    - x4 = k: spring coefficient (N/m), [1000, 5000]
    - x5 = P0: atmospheric pressure (N/m^2), [90000, 110000]
    - x6 = Ta: ambient temperature (K), [290, 296]
    - x7 = T0: filling gas temperature (K), [340, 360]
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Kenett, R., & Zacks, S. (1998). Modern industrial statistics: design and 
    control of quality and reliability. Pacific Grove, CA: Duxbury press.
    """
    if scale01:
        RR = np.array([
            [30, 60],
            [0.005, 0.020],
            [0.002, 0.010],
            [1000, 5000],
            [90000, 110000],
            [290, 296],
            [340, 360]
        ])
        x = x[:7] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    M = x[0]
    S = x[1]
    V0 = x[2]
    k = x[3]
    P0 = x[4]
    Ta = x[5]
    T0 = x[6]
    
    Aterm1 = P0 * S
    Aterm2 = 19.62 * M
    Aterm3 = -k * V0 / S
    A = Aterm1 + Aterm2 + Aterm3
    
    Vfact1 = S / (2 * k)
    Vfact2 = np.sqrt(A**2 + 4 * k * (P0 * V0 / T0) * Ta)
    V = Vfact1 * (Vfact2 - A)
    
    fact1 = M
    fact2 = k + (S**2) * (P0 * V0 / T0) * (Ta / (V**2))
    
    C = 2 * np.pi * np.sqrt(fact1 / fact2)
    return C

def stochastic_piston(x, scale01=True, Ta_generate=None, P0_generate=None, seed: Optional[int] = None,
                      rng: Optional[np.random.Generator] = None):
    """
    Stochastic Piston Simulation Function
    
    Dimensions: 5. The Piston Simulation function models the circular motion 
    of a piston within a cylinder. It involves a chain of nonlinear functions.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 5.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    Ta_generate : callable, optional
        The generating distribution for the ambient temperature (on (0,1) scale).
        Default is a beta distribution with alpha=10, beta=15.
    P0_generate : callable, optional
        The generating distribution for the atmospheric pressure (on (0,1) scale).
        Default is a uniform distribution between 0.49 and 0.51.
    
    Returns
    -------
    float
        The response C is cycle time (the time it takes to complete one cycle), in seconds.
    
    Notes
    -----
    For details on the piston function, see the VLSE.
    
    Parameter ranges:
    - x1 = M: piston weight (kg), [30, 60]
    - x2 = S: piston surface area (m^2), [0.020, 0.045]
    - x3 = V0: initial gas volume (m^3), [0.002, 0.010]
    - x4 = k: spring coefficient (N/m), [1000, 5000]
    - x5 = T0: filling gas temperature (K), [340, 360]
    
    This is a stochastic version of the piston function. Note that the range 
    of the piston surface area is different from the canonical piston function. 
    The distribution of the response is heavily influenced by the stochastic 
    process governing ambient temperature. The response distribution is less 
    homoskedastic for larger ranges of atmospheric pressure.
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Kenett, R., & Zacks, S. (1998). Modern industrial statistics: design and 
    control of quality and reliability. Pacific Grove, CA: Duxbury press.
    """
    # Default generating functions if not provided
    rng = np.random.default_rng(seed) if rng is None else rng
    if Ta_generate is None:
        Ta_generate = lambda: rng.beta(10, 15)
    if P0_generate is None:
        P0_generate = lambda: rng.uniform(0.49, 0.51)
    
    Ta = Ta_generate()
    P0 = P0_generate()
    
    # Combine inputs with stochastic components
    full_x = np.zeros(7)
    full_x[0:4] = x[0:4]
    full_x[4] = P0
    full_x[5] = Ta
    full_x[6] = x[4]
    
    if scale01:
        RR = np.array([
            [30, 60],
            [0.005, 0.020],
            [0.002, 0.010],
            [1000, 5000],
            [90000, 110000],
            [290, 296],
            [340, 360]
        ])
        full_x = full_x * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    M = full_x[0]
    S = full_x[1]
    V0 = full_x[2]
    k = full_x[3]
    P0 = full_x[4]
    Ta = full_x[5]
    T0 = full_x[6]
    
    Aterm1 = P0 * S
    Aterm2 = 19.62 * M
    Aterm3 = -k * V0 / S
    A = Aterm1 + Aterm2 + Aterm3
    
    Vfact1 = S / (2 * k)
    Vfact2 = np.sqrt(A**2 + 4 * k * (P0 * V0 / T0) * Ta)
    V = Vfact1 * (Vfact2 - A)
    
    fact1 = M
    fact2 = k + (S**2) * (P0 * V0 / T0) * (Ta / (V**2))
    
    C = 2 * np.pi * np.sqrt(fact1 / fact2)
    return C

# Register functions with metadata
register_function(
    fname="piston",
    input_dim=7,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [30, 60],
        [0.005, 0.020],
        [0.002, 0.010],
        [1000, 5000],
        [90000, 110000],
        [290, 296],
        [340, 360]
    ])
)

register_function(
    fname="stochastic_piston",
    input_dim=5,
    input_cat=False,
    response_type="uni",
    stochastic="y",
    input_range=np.array([
        [30, 60],
        [0.02, 0.045],
        [0.002, 0.010],
        [1000, 5000],
        [340, 360]
    ])
)
