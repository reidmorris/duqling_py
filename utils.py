"""
Utility functions for the duqling package.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# dict to store function metadata
function_registry = {}

def register_function(
        fname: str,
        input_dim: int,
        input_cat: bool = False,
        response_type: str = "uni",
        stochastic: str = "n",
        input_range: Optional[np.ndarray] = None
        ) -> None:
    """Register a function's metadata in the central registry."""
    function_registry[fname] = {
        "input_dim": input_dim,
        "input_cat": input_cat,
        "response_type": response_type,
        "stochastic": stochastic,
        "input_range": input_range
    }

def quack(
        fname: Optional[str] = None,
        input_dim: Optional[List[int]] = None,
        input_cat: Optional[bool] = None,
        response_type: Optional[str] = None,
        stochastic: Optional[str] = None,
        sorted: bool = True,
        ) -> Union[Dict, List, pd.DataFrame]:
    """Query info about functions in the registry."""
    # single function case
    if fname is not None:
        if fname in function_registry:
            return function_registry[fname]
        else:
            raise ValueError(f"Function '{fname}' not found in registry.")
    
    filtered_registry = function_registry.copy()
    
    if input_dim is not None:
        input_dim = [input_dim] if isinstance(input_dim, (int, float)) else input_dim
        filtered_registry = {k: v for k, v in filtered_registry.items() 
                           if v["input_dim"] in input_dim}
    
    if input_cat is not None:
        filtered_registry = {k: v for k, v in filtered_registry.items() 
                           if v["input_cat"] == input_cat}
    
    if response_type is not None:
        filtered_registry = {k: v for k, v in filtered_registry.items() 
                           if v["response_type"] == response_type}
    
    if stochastic is not None:
        filtered_registry = {k: v for k, v in filtered_registry.items() 
                           if v["stochastic"] == stochastic}
    
    result_list = [{"fname": k, **v} for k, v in filtered_registry.items()]
    
    if sorted:
        result_list.sort(key=lambda x: (x["input_dim"], x["fname"]))

    return pd.DataFrame(result_list).drop('input_range', axis=1)
