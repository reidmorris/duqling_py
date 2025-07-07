"""
Univariate version of `pollutant`.
"""

from __future__ import annotations
import numpy as np
from typing import Any
from .pollutant import pollutant  # same directory as above

from ..utils import register_function

def pollutant_uni(x,
                  /,
                  *,
                  scale01: bool = True,
                  space: float = 2.5,
                  time: float = 30.0) -> float:
    """
    Concentration at s = 2.5 m, t = 30 s (faithful to the R helper).
    """
    out = pollutant(x,
                    scale01=scale01,
                    space=np.asarray([space]),
                    time=np.asarray([time]))

    # out can be a scalar or a 1‑element array; this works for both
    return float(np.asarray(out).ravel()[0])


# Register function with metadata
register_function(
    fname="pollutant_uni",
    input_dim=4,
    input_cat=False,
    response_type="uni",
    stochastic="n",
    input_range=np.array([
        [7, 13],        # M
        [0.02, 0.12],   # D
        [0.01, 3],      # L
        [30.01, 30.295] # tau
    ])
)