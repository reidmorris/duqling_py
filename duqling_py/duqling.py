"""
A Python interface that acts identically to the original Duqling package written in R.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from duqling_py import functions
from duqling_py.utils import quack


class Duqling:
    """
    Represents an interface that allows the R Duqling package to be callable in Python.
    """

    def __init__(self):
        """Initialize the duqling interface."""

    def quack(
        self,
        fname: Optional[str] = None,
        input_dim: Optional[int] = None,
        input_cat: Optional[bool] = None,
        response_type: Optional[str] = None,
        stochastic: Optional[str] = None,
        sorted: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve information about a function, or query for all functions that meet some criteria.
        """
        args = locals()
        args.pop("self")
        return quack(**args)

    def duq(self, X: np.array, f: Callable | str, **kwargs) -> np.array:
        """
        Call functions from the duqling namespace on a batch of samples.

        Args:
            X:        An nxp matrix of inputs.
            f:        A function name or a function, usually from the duqling package.
            **kwargs: Additional kwargs pass to f.
        Returns:
            The output of the function f when called on the matrix of samples X.
        """
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2D array-like, got ndim={X.ndim}")
        if isinstance(f, str):
            input_length = X.shape[1]
            expected_length = self.quack(f)["input_dim"]
            if input_length != expected_length:
                raise ValueError(
                    f"{f} expects samples of length {expected_length}, "
                    f"but received {input_length}. X.shape={X.shape}"
                )
            f = getattr(functions, f)
        Y = np.apply_along_axis(lambda x: f(x, **kwargs), axis=1, arr=X)
        Y = np.transpose(Y)
        # messy hard coded shape casting to match R's `apply` function
        if Y.ndim == 2 and Y.shape[0] == 1:
            return Y[0]
        elif Y.ndim == 3:
            return Y.reshape(-1, *Y.shape[2:])
        return Y
