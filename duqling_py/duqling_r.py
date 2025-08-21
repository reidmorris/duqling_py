"""
A Python interface for the original Duqling package written in R.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.rinterface import rternalize
from rpy2.robjects import conversion, numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

conversion.set_conversion(
    conversion.get_conversion() + pandas2ri.converter + numpy2ri.converter
)


class DuqlingR:
    """
    Represents an interface that allows the R Duqling package to be callable in Python.
    """

    def __init__(self):
        """
        Initialize the duqling interface.
        """
        try:
            self.duqling = importr("duqling")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize duqling interface: {e}") from e

    def parse_r_args_str(self, **kwargs) -> str:
        """
        Parse function argument objects to their string representations in R.
        """
        args_str = []
        for key, value in kwargs.items():
            if isinstance(value, str):
                args_str.append(f'{key}="{value}"')
            elif isinstance(value, bool):
                args_str.append(f"{key}={str(value).upper()}")
            elif isinstance(value, (list, tuple, range)):
                vals = ",".join(map(str, value))
                args_str.append(f"{key}=c({vals})")
            elif callable(value):
                r_func = rternalize(value)
                args_str.append(f"{key}={r_func.r_repr()}")
            else:
                args_str.append(f"{key}={value}")
        return ", ".join(args_str)

    def get_function_info(self, function_name: str) -> dict:
        """
        Get detailed information about a specific function.

        Args:
            function_name: Name of the duqling function.
        Returns:
            Function information
        """
        try:
            result = self.duqling.quack(function_name)
            result = dict(zip(result.names(), result))
            # for some reason this doesn't get automatically parsed to np like the other objects
            result["input_cat"] = numpy2ri.rpy2py(result["input_cat"])[0]

            # cast values to expected types
            result["input_dim"] = int(result["input_dim"][0])
            result["input_cat"] = bool(result["input_cat"])
            result["response_type"] = str(result["response_type"][0])
            if "stochastic" in result:
                result["stochastic"] = str(result["stochastic"][0])

            # not necessary, but makes it look nicer
            return {str(k): result[k] for k in result}
        except Exception as e:
            raise ValueError(f"Function '{function_name}' not found: {e}") from e

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
        List available duqling functions with optional filtering.

        Args:
            fname: The name of a function in this package.
            input_dim: A vector specifying the input dimension of the function.
            input_cat: Logical, should functions with categorical inputs be included?
            response_type: One of ["all", "uni", "multi", "func"] specifying which
                           response type is requested.
            stochastic: Is function response stochastic?
            sorted: Should results be sorted (by input dimension and then alphabetically).

        Returns:
            Available functions and their properties
        """
        if fname:
            return self.get_function_info(fname)

        args = {k: v for k, v in locals().items() if v is not None}
        args.pop("self")
        args.pop("sorted")
        if args:
            args_str = self.parse_r_args_str(**args)
            call_str = f"duqling::quack({args_str})"
            result = robjects.r(call_str)
        else:
            result = robjects.r("duqling::quack()")
        return pd.DataFrame(result)

    def duq(self, X: np.array, f: Callable | str, **kwargs) -> np.array:
        """
        Call functions from the duqling namespace

        Args:
            X:        An nxp matrix of inputs.
            f:        A function name or a function, usually from the duqling package.
            scale01:  When TRUE, inputs are expected on the (0, 1) scale.
            **kwargs: Additional kwargs pass to f.
        Returns:
            The output of the function f when called on the matrix of samples X.
        """
        X_r = numpy2ri.py2rpy(np.atleast_2d(X))
        f_r = rternalize(f) if callable(f) else f
        kwargs = {k: rternalize(v) if callable(v) else v for k, v in kwargs.items()}
        apply_func = robjects.r["apply"]
        duq_func = self.duqling.duq
        result = apply_func(X_r, 1, duq_func, f_r, **kwargs)
        return result
