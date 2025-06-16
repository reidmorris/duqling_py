"""
Beam Deflection Function.
"""

import numpy as np
from ..utils import register_function

def beam_deflection(x, scale01=True, p=None):
    """
    Beam Deflection
    
    Dimensions 5. Measures the deflection of a length-L beam with a uniform 
    load at location t away from origin.
    
    Parameters
    ----------
    x : array_like
        Inputs: Uniform load density (P, N/m), Elastic Modulus (E, Mpa), 
        beam length (L, m), beam width (w, m), beam thickness (h, m).
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval and are 
        internally adjusted to their native range. Default is True.
    p : array_like, optional
        Vector of numbers between 0 and 1. p*L is location from the origin 
        (left most side of beam) at which deflection is measured. 
        Default is seq(0, 1, length.out=20).
    
    Returns
    -------
    array_like
        The "output" vector is given by -P/(24EI)ell(ell^3 - 2*L*ell^2 + L^3) 
        where I = w*h^3/12 and ell = p*L.
    
    Notes
    -----
    A modified version of the beam deflection problem from the intro of Joseph (2024).
    
    References
    ----------
    Joseph, Roshan. "Rational Kriging". JASA, (2024)
    """
    if p is None:
        p = np.linspace(0, 1, 20)
    p = np.asarray(p, dtype=float)
    if scale01:
        RR = np.array([
            [1.9e-4, 1.5e6],
            [7000, 14000],
            [0.5, 1.5],
            [0.076, 0.305],
            [0.013, 0.152]
        ])
        x = x[:5] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    P = x[0]     # Load density (N/m)
    E = x[1]*1e6 # Elastic modulus (MPa)
    L = x[2]     # Beam length (m)
    w = x[3]     # Beam width (m)
    h = x[4]     # Beam thickness (m)
    
    I = w*h**3/12 # Area Moment of Inertia
    xx = p*L
    y = -P/(24*E*I)*xx*(xx**3 - 2*L*xx**2 + L**3)
    return y

# Register function with metadata
register_function(
    fname="beam_deflection",
    input_dim=5,
    input_cat=False,
    response_type="func",
    stochastic="n",
    input_range=np.array([
        [1.9e-4, 1.5e6],
        [7000, 14000],
        [0.5, 1.5],
        [0.076, 0.305],
        [0.013, 0.152]
    ])
)
