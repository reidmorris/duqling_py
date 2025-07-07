"""
Mock Ignition Function
"""

import numpy as np
from scipy.stats import norm
from ..utils import register_function

def ignition(x, scale01=True):
    """
    Mock Ignition Function
    
    Dimensions: 10 The complex mock ignition function from Section E of Hatfield et al. 2019.
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 10.
    scale01 : bool, optional
        (No effect here). When True, inputs are expected to be given on unit interval 
        and are internally adjusted to their native range. Default is True.
    
    Returns
    -------
    float
        The response is mock yield of an ignition cliff.
    
    Notes
    -----
    From Hatfield et al. 2019, the mock-yield is r^5(1 + 100000 + (1+erf(10(r-2)))) 
    where r is the radius from the origin in this 10-D space.
    
    References
    ----------
    Hatfield, P., Rose, S., Scott, R., Almosallam, I., Roberts, S., & Jarvis, M. (2019). 
    Using sparse Gaussian processes for predicting robust inertial confinement fusion 
    implosion yields. IEEE Transactions on Plasma Science, 48(1), 14-21.
    """
    # Calculate the radius from origin in 10D space
    r = np.sqrt(np.sum(x[:10]**2))
    
    # Calculate the mock yield using the formula
    # In R: pnorm(sqrt(2)*10*(r-2))
    # In Python: we use scipy.stats.norm.cdf()
    return np.log10(r**5 * (1 + 100000 * (2 * norm.cdf(np.sqrt(2) * 10 * (r - 2)))))

# Register function with metadata
register_function(
    fname="ignition",
    input_dim=10,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1] for _ in range(10)
    ])
)
