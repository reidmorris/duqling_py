"""
A Python interface for the original Duqling package written in R.
"""

from typing import Optional, Callable
import numpy as np
import pandas as pd

import duqling_py.functions as functions
from duqling_r_interface import DuqlingRInterface

class DuqlingPyInterface:
    """
    Represents an interface that allows the R Duqling package to be callable in Python.
    """
    def __init__(self):
        """
        Initialize the duqling interface.
        """
        # Use the duqling R interface to retrieve function info (through the
        # quack method) to obtain the most up-to-date info from the original repo.
        self.duq_r = DuqlingRInterface()
        
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
        return self.duq_r.quack(*locals())

    def duq(self, x:np.array, f:Callable|str, **kwargs) -> np.array:
        """
        Call functions from the duqling namespace

        Args:
            x:        An nxp matrix of inputs.
            f:        A function name or a function, usually from the duqling package.
            **kwargs: Additional kwargs pass to f.
        Returns:
            The output of the function f when called on the matrix of samples x.
        """
        if isinstance(f, str):
            f = getattr(functions, f)
        return f(x, **kwargs)
