"""
May have to execute the following code in terminal first:
> R -e "devtools::install_github('knrumsey/duqling')"
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.rinterface import rternalize

# activate pandas and numpy conversion
pandas2ri.activate()
numpy2ri.activate()

class DuqlingInterface:
    """
    Represents an interface that allows the R Duqling package to be callable in Python.
    """
    def __init__(self, r_script_path:str="quack.R"):
        """
        Initialize the duqling interface.
        
        Args:
            r_script_path: Path to the R script containing the quack function
        """
        try:
            self.duqling = importr('duqling')

            script_path = Path(r_script_path).resolve()
            if not script_path.exists():
                raise FileNotFoundError(f"R script not found: {script_path}")

            robjects.r["source"](str(script_path))
            self.gen_func = robjects.globalenv["quack"]
        except Exception as e:
            raise RuntimeError(f"Failed to initialize duqling interface: {e}") from e

    def list_functions(self, **kwargs) -> pd.DataFrame:
        """
        List available duqling functions with optional filtering.
        
        Args:
            Filtering criteria (e.g., input_dim=5, stochastic="n")

        Returns:
            Available functions and their properties
        """
        if kwargs:
            args_str = []
            for key, value in kwargs.items():
                if isinstance(value, str):
                    args_str.append(f'{key}="{value}"')
                elif isinstance(value, bool):
                    args_str.append(f'{key}={str(value).upper()}')
                elif isinstance(value, (list, tuple)):
                    # Handle vectors like input_dim=c(2,3,4)
                    vals = ','.join(map(str, value))
                    args_str.append(f'{key}=c({vals})')
                else:
                    args_str.append(f'{key}={value}')

            call_str = f"duqling::quack({', '.join(args_str)})"
            result = robjects.r(call_str)
        else:
            result = robjects.r("duqling::quack()")

        return pandas2ri.rpy2py(result)

    def get_function_info(self, function_name:str) -> dict:
        """
        Get detailed information about a specific function.
        
        Args:
            function_name: Name of the duqling function
        
        Returns:
            Function information
        """
        try:
            result = self.duqling.quack(function_name)
            result = dict(result.items())
            # for some reason this doesn't get automatically parsed to np like the other objects
            result['input_cat'] = numpy2ri.rpy2py(result['input_cat'])
            # not necessary, but makes it look nicer
            return {str(k): result[k] for k in result}
        except Exception as e:
            raise ValueError(f"Function '{function_name}' not found: {e}") from e

    def generate_data(self, func_name:str, n_samples:int=None, X:Optional[np.array]=None,
                      seed:Optional[int]=42, **kwargs) -> Tuple[np.array, np.array]:
        """
        Generate data using a duqling function.
        
        Args:
            func_name: Name of the duqling function
            n_samples: Number of samples to generate
            X:         Input values to duqling function
            seed:      Random seed for reproducibility
        Kwargs:
            Additional arguments to pass to the duqling function
        
        Returns:
            (X, y)
        """
        try:
            X = robjects.NULL if X is None else X
            n_samples = robjects.NULL if n_samples is None else n_samples
            if kwargs:
                for key, value in kwargs.items():
                    if callable(value):
                        kwargs[key] = rternalize(value)
                result = self.gen_func(func_name, n_samples=n_samples, X=X, seed=seed, **kwargs)
            else:
                result = self.gen_func(func_name, n_samples=n_samples, X=X, seed=seed)

            X = np.array(result.rx2("X"))
            y = np.array(result.rx2("y"))

            return X, y

        except Exception as e:
            raise RuntimeError(f"Failed to generate data for '{func_name}': {e}") from e
