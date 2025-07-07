"""
Reproducible Simulation Study (Test Functions) - Python Implementation
"""

import os
import sys
import time
import numpy as np
import importlib
import inspect
import pkgutil
from pathlib import Path
import pandas as pd
from functools import partial
import warnings

def rmsef(x, y):
    """Calculate root mean squared error"""
    return np.sqrt(np.mean((x - y)**2))

def crpsf(y, x, w=0):
    """Calculate Continuous Ranked Probability Score"""
    M = len(x)
    term1 = np.mean(np.abs(x - y))
    
    if M <= 6500:
        # Fastest way for small M
        term2 = np.sum(np.abs(np.subtract.outer(x, x)))
    else:
        # Faster for big M - equivalent to R's implementation
        indices = []
        for i in range(2, M + 1):
            indices.extend(list(range(1, i)))
        
        term2 = 2 * np.sum(np.abs(np.repeat(x[1:], np.arange(1, M)) - x[np.array(indices)-1]))
    
    res = term1 - (1 - w/M) * term2 / (2 * M * (M - 1))
    return res

def str2num(string):
    """Convert string to a numeric hash value"""
    return sum((i+1) * ord(c) for i, c in enumerate(string))

def transform_seed(seed, n, dt, NSR, fnum, rr):
    """Transform seed for reproducibility"""
    s1 = seed
    s2 = round(np.log(n))
    s3 = round(100 * (np.log(n) % 1))
    
    # Map design type to integer
    if dt == "LHS":
        s4 = 1
    elif dt == "grid":
        s4 = 2
    elif dt == "random":
        s4 = 3
    else:
        s4 = 0
    
    s5 = fnum
    s6 = round(100 * NSR)
    s7 = round(100 * ((100 * NSR) % 1))
    s8 = rr // 100
    s9 = rr % 100
    
    B = 101
    ss = 0
    for i, val in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9]):
        ss += B**i * val
    
    return ss % 100030001

def expand_grid(xx, k):
    """Create a grid of points (equivalent to R's expand.grid)"""
    import itertools
    points = list(itertools.product(*[xx for _ in range(k)]))
    return np.array(points)

def run_one_sim_case(rr, seed, fn, fnum, p, n, nsr, dsgn, n_test, conf_level, score, 
                     method_names, fit_func, pred_func, verbose):
    """Run one simulation case"""
    # Generate training data
    seed_t = transform_seed(seed, n, dsgn, nsr, fnum, rr)
    np.random.seed(seed_t)
    
    if dsgn == "LHS":
        try:
            from pyDOE import lhs
            if n <= 1200:
                # Generate maximin Latin Hypercube design
                X_train = lhs(p, samples=n, criterion='maximin')
                X_test = lhs(p, samples=n_test)
            else:
                # For larger samples, use random LHS
                X_train = lhs(p, samples=n)
                X_test = lhs(p, samples=n_test)
        except ImportError:
            warnings.warn("pyDOE not available, falling back to random sampling")
            X_train = np.random.uniform(0, 1, size=(n, p))
            X_test = np.random.uniform(0, 1, size=(n_test, p))
    elif dsgn == "random":
        X_train = np.random.uniform(0, 1, size=(n, p))
        X_test = np.random.uniform(0, 1, size=(n_test, p))
    else:  # grid
        ni = int(np.ceil(n**(1/p)))
        xx = np.linspace(0, 1, ni)
        X_train = expand_grid(xx, p)
        n = X_train.shape[0]  # Actual size might differ from requested
        
        nit = int(np.ceil(n_test**(1/p)))
        xxt = np.linspace(0, 1, nit)
        X_test = expand_grid(xxt, p)
    
    # Get the function to evaluate - assuming it's available through module import
    try:
        module_name, func_name = fn.split('.')
        module = importlib.import_module(f"duqling_py.functions.{module_name}")
        f = getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not import function {fn}: {e}")
    
    # Generate responses - using apply_along_axis for vectorization
    y_train = np.array([f(x, scale01=True) for x in X_train])
    
    # Add noise
    if np.var(y_train) == 0:
        noise_lvl = 1  # Cannot make sense of SNR when there is no signal
    else:
        noise_lvl = np.sqrt(np.var(y_train) * nsr)
    
    y_train = y_train + np.random.normal(0, noise_lvl, n)
    y_test = np.array([f(x, scale01=True) for x in X_test])  # no noise for testing data
    
    results = []
    
    # Ensure fit_func and pred_func are lists
    fit_funcs = fit_func if isinstance(fit_func, list) else [fit_func]
    pred_funcs = pred_func if isinstance(pred_func, list) else [pred_func]
    pred_funcs = pred_funcs + [None] * (len(fit_funcs) - len(pred_funcs))  # Pad with None
    
    # Fit models
    for ii, (fit_func_i, pred_func_i) in enumerate(zip(fit_funcs, pred_funcs)):
        method_name = method_names[ii] if method_names and ii < len(method_names) else f"method{ii}"
        
        result = {
            "method": method_name,
            "fname": fn,
            "input_dim": p,
            "n": n,
            "NSR": nsr,
            "design_type": dsgn,
            "rep": rr
        }
        
        # Time fitting and prediction
        if pred_func_i is None:
            start_time = time.time()
            preds = fit_func_i(X_train, y_train, X_test)
            total_time = time.time() - start_time
            
            result["t_tot"] = total_time
        else:
            start_time = time.time()
            fitted_object = fit_func_i(X_train, y_train)
            fit_time = time.time() - start_time
            
            start_time = time.time()
            preds = pred_func_i(fitted_object, X_test)
            pred_time = time.time() - start_time
            
            result["t_fit"] = fit_time
            result["t_pred"] = pred_time
            result["t_tot"] = fit_time + pred_time
        
        # Process predictions
        if not isinstance(preds, (list, np.ndarray)):
            raise ValueError("fit/pred functions must return a vector, matrix, or a list.")
        
        if isinstance(preds, np.ndarray):
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            
            # Calculate RMSE
            y_hat = np.mean(preds, axis=0)
            rmse = rmsef(y_test, y_hat)
            result["RMSE"] = rmse
            result["FVU"] = rmse**2 / np.var(y_test)
            
            # Calculate coverage and interval scores
            for conf in conf_level:
                alpha = 1 - conf
                # Calculate quantiles for each test point
                bounds = np.array([np.quantile(preds[:, i], [alpha/2, 1-alpha/2]) for i in range(preds.shape[1])]).T
                
                # Coverage
                cover = np.mean((y_test >= bounds[0]) & (y_test <= bounds[1]))
                result[f"COVER{conf:.7f}"] = cover
                
                # Interval Score
                term1 = bounds[1] - bounds[0]
                term2 = 2 * (bounds[0] - y_test) * (y_test < bounds[0]) / alpha
                term3 = 2 * (y_test - bounds[1]) * (y_test > bounds[1]) / alpha
                result[f"MIS{conf:.7f}"] = np.mean(term1 + term2 + term3)
            
            # Calculate CRPS if requested
            if score:
                if verbose:
                    print("Computing CRPS", end="")
                
                crps_values = [crpsf(y_test[i], preds[:, i]) for i in range(n_test)]
                result["CRPS"] = np.mean(crps_values)
                result["CRPS_min"] = np.min(crps_values)
                result["CRPS_Q1"] = np.percentile(crps_values, 25)
                result["CRPS_med"] = np.median(crps_values)
                result["CRPS_Q3"] = np.percentile(crps_values, 75)
                result["CRPS_max"] = np.max(crps_values)
                
                if verbose:
                    print("\nDone.")
        
        # If preds is a list with a special structure
        elif isinstance(preds, list):
            if 'samples' not in preds:
                raise ValueError("If pred/fit function returns a list, the samples field must be specified.")
            
            # Process predictions with samples, etc. (similar to the array case)
            # This would include handling preds$samples, preds$preds, preds$intervals
            
            # Calculate RMSE
            if 'preds' not in preds:
                preds['preds'] = np.mean(preds['samples'], axis=0)
                
            y_hat = preds['preds']
            rmse = rmsef(y_test, y_hat)
            result["RMSE"] = rmse
            result["FVU"] = rmse**2 / np.var(y_test)
            
            # Handle intervals and CRPS
        
        results.append(result)
    
    return results

def run_sim_study(fit_func, pred_func=None,
                 fnames=None,
                 conf_level=[0.8, 0.9, 0.95, 0.99],
                 score=True,
                 n_train=100,
                 n_test=1000,
                 NSR=0,
                 design_type="LHS",
                 replications=1,
                 seed=42,
                 method_names=None,
                 mc_cores=1,
                 verbose=True):
    """
    Reproducible Simulation Study (Test Functions)
    
    Parameters
    ----------
    fit_func : function or list of functions
        If pred_func is specified, fit_func should take two arguments: X_train and y_train,
        and return an object which will be passed to pred_func. If pred_func is not specified,
        then fit_func should take a third argument: X_test, and return predictive samples.
    pred_func : function or list of functions, optional
        A function taking two arguments: the object returned by fit_func and X_test.
        Should return a matrix of samples from the predictive distribution.
    fnames : list of str, optional
        A list of function names from the duqling package.
    conf_level : list of float, optional
        A list of confidence levels. Default is [0.8, 0.9, 0.95, 0.99].
    score : bool, optional
        Whether to compute CRPS. Default is True.
    n_train : int or list of int, optional
        The sample size(s) for each training set. Default is 100.
    n_test : int, optional
        The sample size for each testing set. Default is 1000.
    NSR : float or list of float, optional
        The noise to signal ratio. Default is 0.
    design_type : str or list of str, optional
        How should the training and testing designs be generated?
        Options are "LHS", "grid" and "random". Default is "LHS".
    replications : int, optional
        How many replications should be repeated for each data set? Default is 1.
    seed : int, optional
        Seed for random number generators. Default is 42.
    method_names : list of str, optional
        A list of method names, length equal to len(fit_func). If None, names will be
        generated automatically.
    mc_cores : int, optional
        Number of cores to use for parallelization. Default is 1.
    verbose : bool, optional
        Whether to report progress. Default is True.
    
    Returns
    -------
    DataFrame
        Results of the simulation study.
    
    Notes
    -----
    This function provides code to conduct a reproducible simulation study to compare emulators. 
    By reporting the parameters to the study, other authors can compare their results directly.
    
    The simplest (and recommended) approach is that the fit_func (or pred_func) should return 
    a matrix of posterior samples, with one column per test point (e.g., per row in X_test). 
    Any number of rows (predictive samples) is allowed. In this case, the mean of the samples 
    is used as a prediction and the Python np.quantile function is used to construct 
    confidence intervals.
    
    This default behavior can be changed by instead allowing fit_func (or pred_func) to return 
    a named dict with fields:
    - 'samples' (required): A matrix of samples
    - 'preds' (optional): A vector of predictions
    - 'intervals' (optional): A tuple of (lower_bounds, upper_bounds) arrays
    
    References
    ----------
    Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: 
    test functions and datasets." Simon Fraser University, Burnaby, BC, Canada, 
    accessed May 13 (2013): 2015.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> 
    >>> def my_fit(X, y):
    >>>     rf = RandomForestRegressor(n_estimators=100, random_state=42)
    >>>     rf.fit(X, y)
    >>>     return rf
    >>> 
    >>> def my_pred(model, X_test):
    >>>     # Return bootstrap predictions as samples
    >>>     n_samples = 100
    >>>     n_test = X_test.shape[0]
    >>>     samples = np.zeros((n_samples, n_test))
    >>>     
    >>>     for i in range(n_samples):
    >>>         # Bootstrap by sampling trees
    >>>         preds = np.zeros(n_test)
    >>>         for j in range(30):  # Sample 30 trees
    >>>             tree_idx = np.random.randint(0, 100)
    >>>             preds += model.estimators_[tree_idx].predict(X_test)
    >>>         samples[i, :] = preds / 30
    >>>     
    >>>     return samples
    >>> 
    >>> results = run_sim_study(my_fit, my_pred,
    >>>                         fnames=['friedman.friedman1', 'grlee.grlee1'],
    >>>                         n_train=50,
    >>>                         replications=2)
    """
    # Get function metadata if needed
    from duqling_py.utils import quack  # Import here to avoid circular imports
    
    # If method_names not provided, create default names
    if method_names is None:
        if isinstance(fit_func, list):
            method_names = [f"method{i}" for i in range(len(fit_func))]
        else:
            method_names = ["method0"]
    
    # Ensure n_train, NSR, design_type are lists
    if not isinstance(n_train, list):
        n_train = [n_train]
    if not isinstance(NSR, list):
        NSR = [NSR]
    if not isinstance(design_type, list):
        design_type = [design_type]
    
    # Initialize results
    all_results = []
    
    # Loop through all functions
    for ff, fn in enumerate(fnames):
        # Get function metadata
        try:
            fn_metadata = quack(fn)
            p = fn_metadata.get('input_dim')
            if p is None:
                raise ValueError(f"Could not get input dimension for function {fn}")
        except Exception as e:
            raise RuntimeError(f"Error getting metadata for function {fn}: {e}")
        
        fnum = str2num(fn)
        
        if verbose:
            print(f"Starting function {ff+1}/{len(fnames)}: {fn}")
        
        # Loop through all training sizes
        for n in n_train:
            if verbose:
                print(f"\t Running all combinations and replications for n = {n}")
            
            # Loop through all noise levels
            for nsr in NSR:
                # Loop through all design types
                for dsgn in design_type:
                    # Run all replications
                    if mc_cores == 1:
                        results = [run_one_sim_case(
                            rr, seed, fn, fnum, p, n, nsr, dsgn, n_test, 
                            conf_level, score, method_names, fit_func, pred_func, verbose
                        ) for rr in range(1, replications+1)]
                    else:
                        try:
                            from joblib import Parallel, delayed
                            results = Parallel(n_jobs=mc_cores)(
                                delayed(run_one_sim_case)(
                                    rr, seed, fn, fnum, p, n, nsr, dsgn, n_test,
                                    conf_level, score, method_names, fit_func, pred_func, verbose
                                ) for rr in range(1, replications+1)
                            )
                        except ImportError:
                            warnings.warn("joblib not available, falling back to sequential processing")
                            results = [run_one_sim_case(
                                rr, seed, fn, fnum, p, n, nsr, dsgn, n_test, 
                                conf_level, score, method_names, fit_func, pred_func, verbose
                            ) for rr in range(1, replications+1)]
                    
                    # Flatten results
                    flat_results = [item for sublist in results for item in sublist]
                    all_results.extend(flat_results)
    
    # Convert to DataFrame
    return pd.DataFrame(all_results)
