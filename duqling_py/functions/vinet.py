"""
The Vinet Equation of State.
"""

import numpy as np
from ..utils import register_function

def vinet(x, scale01=True, density=None):
    """
    The Vinet Equation of State
    
    Dimensions 3. The Pressure as a function of density.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    density : array_like, optional
        A vector of densities to evaluate. Default is np.arange(15.5, 17.6, 0.1).
    
    Returns
    -------
    ndarray
        Pressure values.
    
    Notes
    -----
    For details on the Vinet EOS, see Vinet (1989) or Brown & Hund (2018).
    The parameter ranges correspond to a range of reasonable values for Tantalum.
    
    - x1 = B0: Bulk modulus (GPa), [195.6, 223.8]
    - x2 = B0p: 1st derivative of B0, [2.9, 4.9]
    - x3 = rho0: initial density (g / cm^3), [16.15, 16.95]
    
    References
    ----------
    Vinet, Pascal, et al. "Universal features of the equation of state of solids." 
    Journal of Physics: Condensed Matter 1.11 (1989): 1941.
    
    Brown, Justin L., and Lauren B. Hund. "Estimating material properties under 
    extreme conditions by using Bayesian model calibration with functional outputs." 
    Journal of the Royal Statistical Society Series C: Applied Statistics 
    67.4 (2018): 1023-1045
    
    Rumsey, Kellin, et al. "Dealing with measurement uncertainties as nuisance 
    parameters in Bayesian model calibration." SIAM/ASA Journal on Uncertainty 
    Quantification 8.4 (2020): 1287-1309.
    """
    if density is None:
        density = np.arange(15.5, 17.6, 0.1)[:-1]
    
    if scale01:
        RR = np.array([
            [195.6, 223.8],
            [2.9, 4.9],
            [16.15, 16.95]
        ])
        x = x[:3] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    B0 = x[0]
    B0p = x[1]
    rho0 = x[2]
    
    eta = (rho0/density)**(1/3)
    term1 = 3 * B0 * (1 - eta) / eta**2
    term2 = np.exp(1.5 * (B0p - 1) * (1 - eta))
    
    y = term1 * term2
    return y

# Register function with metadata
register_function(
    fname="vinet",
    input_dim=3,
    input_cat=False,
    response_type="func",
    stochastic="n",
    input_range=np.array([
        [195.6, 223.8],
        [2.9, 4.9],
        [16.15, 16.95]
    ])
)
