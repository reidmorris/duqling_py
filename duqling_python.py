"""
Utility functions for the duqling package.
"""

from typing import Dict, List, Optional, Tuple, Union
import importlib
import numpy as np
import pandas as pd

# Registry to store function metadata
function_registry = {}

def lhs_array(n, d, seed=None):
    """Generate Latin Hypercube Sample array in [0,1]."""
    if seed is not None:
        np.random.seed(seed)
    
    samples = (np.arange(n)[:, None] + np.random.rand(n, d)) / n
    for i in range(d):
        samples[:, i] = np.random.permutation(samples[:, i])
    return samples

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
    as_dataframe: bool = False
) -> Union[Dict, List, pd.DataFrame]:
    """Get information about functions in the registry."""
    # Single function case
    if fname is not None:
        if fname in function_registry:
            return function_registry[fname]
        else:
            raise ValueError(f"Function '{fname}' not found in registry.")
    
    # Filter registry
    filtered_registry = function_registry.copy()
    
    if input_dim is not None:
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
    
    # Convert to list
    result_list = [{"fname": k, **v} for k, v in filtered_registry.items()]
    
    if sorted:
        result_list.sort(key=lambda x: (x["input_dim"], x["fname"]))
    
    return pd.DataFrame(result_list) if as_dataframe else result_list

def duq(fname: str, x: np.ndarray, scale01: bool = True, **kwargs) -> np.ndarray:
    """Call functions from registry."""
    if fname not in function_registry:
        raise ValueError(f"Function '{fname}' not found in registry.")
    
    # Import function
    try:
        import functions
        func = getattr(functions, fname)
    except (ImportError, AttributeError):
        try:
            import sys
            func = getattr(sys.modules[__name__], fname, None) or globals().get(fname)
        except:
            func = None
    
    if func is None:
        raise ValueError(f"Function '{fname}' implementation not found.")

    # Apply function
    try:
        result = func(x, scale01=scale01, **kwargs)
        return np.asarray(result)
    except:
        return np.array([func(row, scale01=scale01, **kwargs) for row in x])

def generate_data(fname: str, n_samples: int, X: Optional[np.ndarray] = None, 
                 seed: Optional[int] = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data using LHS or custom X. Mirrors R interface."""
    if seed is not None:
        np.random.seed(seed)
    
    if X is None:
        func_info = quack(fname)
        X = lhs_array(n_samples, func_info['input_dim'], seed=seed)
    
    y = duq(fname, X, **kwargs)
    return X, y
