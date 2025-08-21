"""
Sharkfin Function.
"""

import itertools

import numpy as np

from ..utils import register_function


def sharkfin(x, scale01=True, steps=3, active_dim=None):
    """
    Sharkfin Function

    Dimensions default is 3, but can be any dimension. For inert variables,
    active_dim must be specified.

    Parameters
    ----------
    x : array_like
        Inputs of dimension (at least) 3.
    scale01 : bool, optional
        When True, inputs are expected to be given on unit interval. Default is True.
    steps : int, optional
        Default is 3. For larger values of steps, the function becomes more
        fractal-like and takes longer to run.
    active_dim : int, optional
        Only the first min(active_dim, len(x)) variables will be used.
        All other variables are inert. Default is len(x).

    Returns
    -------
    float
        Function value at x.

    Notes
    -----
    A non-smooth test function designed with trees in mind. As steps -> Inf,
    this function is nowhere-differentiable.

    References
    ----------
    Collins, Gavin, Devin Francom, and Kellin Rumsey. "Bayesian projection pursuit
    regression." arXiv preprint arXiv:2210.09181 (2022).
    """
    if active_dim is None:
        active_dim = len(x)

    k = steps
    p = min(active_dim, len(x))

    # Generate Fibonacci sequence
    fib = [0, 1]
    for j in range(k):
        fib.append(fib[j] + fib[j + 1])

    # Create the grid lists
    grid_lists = []
    for j in range(k):
        grid_lists.append([0, fib[j + 2], -fib[j + 2]])

    grid_lists.reverse()

    # Generate all combinations
    grid_combinations = list(itertools.product(*grid_lists))

    # Calculate sums for each combination
    f = [sum(combo) for combo in grid_combinations]
    f_max = max(f)

    # Define cutpoints
    cutpoints = np.linspace(0, 1, len(f) + 1)

    # Create F_x array
    F_x = np.zeros(len(x))

    # Loop through each active dimension and evaluate function
    for j in range(p):
        for cut in range(len(f)):
            # Convert to 1-based indexing in the condition to match R
            if x[j] > cutpoints[cut] and x[j] <= cutpoints[cut + 1]:
                # R uses f[cut] - Python needs to match this
                F_x[j] = f[cut] / f_max

    # Calculate mean
    y = np.mean(F_x)
    return y


# Register function with metadata
register_function(
    fname="sharkfin",
    input_dim=3,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([[0, 1], [0, 1], [0, 1]]),
)
