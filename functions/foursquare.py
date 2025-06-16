"""
The Foursquare Function.
"""

import numpy as np
from ..utils import register_function

def foursquare(x, scale01=True, ftype="all"):
    """
    The Foursquare Function
    
    Dimensions: 2
    
    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 2.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    ftype : str or int, optional
        Default is "all". Can also set to 1, 2, 3 or 4 to activate just one of the four functions.
    
    Returns
    -------
    float
        Function value at x.
    
    Notes
    -----
    Each of the four quadrants can be modeled effectively with a different approach 
    (rbf, linear, mars, trees).
    
    The function response is y = f1 + f2 + f3 + f4 where:
    - f1: An RBF centered at (0.75, 0.75) with variance of 0.13 and a correlation of -0.5
    - f2: The linear plane x_2 - 1.375*x_1
    - f3: A mars basis function with a knot at (0.35, 0.35)
    - f4: A scaled/shifted version of sharkfin, designed for tree based methods.
    """
    ind1 = x[0] > 0.5
    ind2 = x[1] > 0.5
    
    # Function 1: RBF
    mu = np.array(x[:2]) - np.array([0.75, 0.75])
    # Using identity matrix instead of correlated Sigma
    Sigma = np.eye(2)
    f1 = np.exp(-30 * mu @ np.linalg.inv(Sigma) @ mu)
    
    # Function 2: Linear
    f2 = x[1] - 1.375 * x[0]
    
    # Function 3: MARS
    def pos(xx):
        return (abs(xx) + xx) / 2
    
    a = pos(-(x[0] - 0.35))
    b = pos(-(x[1] - 0.35))
    f3 = 8.16 * a * b
    
    # Function 4: Tree-based
    fa = 0.33 * (x[0] > 0.75)
    fb = 0.27 * (x[0] > 0.75) * (x[1] > 0.25)
    fc = 0.09 * (x[0] > 0.75) * (x[1] > 0.25)
    fd = 0.10 * (x[0] > 0.85) * (x[1] > 0.25)
    fe = 0.45 * (x[0] > 0.50) * (x[1] < 0.50)
    f4 = fa + fb + fc + fd + fe
    
    # Combine functions based on ftype
    if ftype == "all":
        y = f1 + f2 + f3 + f4
    else:
        # Get the specified function
        if ftype == 1:
            y = f1
        elif ftype == 2:
            y = f2
        elif ftype == 3:
            y = f3
        elif ftype == 4:
            y = f4
        else:
            raise ValueError(f"Invalid ftype: {ftype}. Must be 'all', 1, 2, 3, or 4.")
    
    return y

# Register function with metadata
register_function(
    fname="foursquare",
    input_dim=2,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [0, 1],
        [0, 1]
    ])
)
