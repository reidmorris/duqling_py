"""
A Python interface that acts identically to the original Duqling package written in R.
"""

from typing import Optional, Callable
import numpy as np
import pandas as pd

from duqling_py.utils import quack
from duqling_py import functions

class Duqling:
    """
    Represents an interface that allows the R Duqling package to be callable in Python.
    """
    def __init__(self):
        """Initialize the duqling interface."""

    def quack(self,
              fname:         Optional[str]  = None,
              input_dim:     Optional[int]  = None,
              input_cat:     Optional[bool] = None,
              response_type: Optional[str]  = None,
              stochastic:    Optional[str]  = None,
              sorted:                 bool  = True
              ) -> pd.DataFrame:
        """
        Wrapper for the `duqling::quack(...)` function from the original Duqling repository.
        """
        args = locals()
        args.pop('self')
        return quack(**args)

    def duq(self, X:np.array, f:Callable|str, **kwargs) -> np.array:
        """
        Call functions from the duqling namespace on a batch of samples.

        Args:
            X:        An nxp matrix of inputs.
            f:        A function name or a function, usually from the duqling package.
            **kwargs: Additional kwargs pass to f.
        Returns:
            The output of the function f when called on the matrix of samples X.
        """
        if isinstance(f, str):
            f = getattr(functions, f)
        Y = np.apply_along_axis(lambda x: f(x, **kwargs), axis=1, arr=X).T
        # messy hard coded shape casting to match R's `apply` function
        if Y.ndim == 2 and Y.shape[0] == 1:
            return Y[0]
        elif Y.ndim == 3:
            return Y.reshape(-1, *Y.shape[2:])
        return Y