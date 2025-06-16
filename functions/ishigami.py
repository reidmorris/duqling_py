"""
Ishigami Function.
"""

import numpy as np
from ..utils import register_function

def ishigami(x, scale01=True, ab=None):
    """
    Ishigami Function
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    ab : array_like, optional
        Hyperparameters to the Ishigami function. Default is [7, 0.1].
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    Often used as an example, because it exhibits strong nonlinearity and nonmonotonicity. 
    It has a peculiar dependence on x3. By default, the distribution of the inputs are 
    x_i ~ Uniform(-pi, pi).
    
    References
    ----------
    Ishigami, T., & Homma, T. (1990). An importance quantification technique in 
    uncertainty analysis for computer models. In Uncertainty Modeling and Analysis, 1990.
    """
    if ab is None:
        ab = [7, 0.1]
    
    if scale01:
        RR = np.array([
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ])
        x = x[:3] * (RR[:, 1] - RR[:, 0]) + RR[:, 0]
    
    res = np.sin(x[0]) + ab[0] * np.sin(x[1])**2 + ab[1] * x[2]**4 * np.sin(x[0])
    return res

# Register function with metadata
register_function(
    fname="ishigami",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi]
    ])
)
