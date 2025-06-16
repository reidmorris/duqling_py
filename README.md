# duqling

A Python package containing a large number of popular test functions for Uncertainty Quantification (UQ) research.

## Description

`duqling` is an R-to-Python translation of a collection of benchmark functions used in uncertainty quantification, computer experiments, and statistical emulation research. These functions facilitate reproducible UQ research. The package name "duqling" is a combination of "UQ" and "darling," suggesting a lovely set of UQ functions.

## Features

- Over 50 test functions widely used in UQ research
- Each function supports scaling inputs from the unit interval to native parameter ranges
- Functions include both deterministic and stochastic models
- Functions with univariate and multivariate outputs
- Documentation with references to original papers

## Installation

```bash
pip install duqling
```

## Usage

### Listing available functions

```python
import duqling
import numpy as np

# Get info about all functions
all_funcs = duqling.quack()

# Get only 2D functions
funcs_2d = duqling.quack(input_dim=[2])

# Get info about a specific function
borehole_info = duqling.quack("borehole")
```

### Basic function evaluation

```python
import duqling
import numpy as np

# Create a random input for the Borehole function
x = np.random.random(8)  # 8 parameters for borehole

# Evaluate the function (inputs scaled from [0,1])
y = duqling.borehole(x, scale01=True)

# Alternatively, using the duq utility function:
X = np.random.random((100, 8))  # 100 random inputs
Y = duqling.duq(X, "borehole")  # Evaluate all inputs
```

## Available Functions

The package includes many classic test functions:

- **borehole**: Simulates water flow through a borehole
- **friedman**: Classic benchmark with 5 inputs (only 5 active variables)
- **piston**: Simulates piston movement in a cylinder
- **circuit**: Models an output transformerless push-pull circuit
- **ishigami**: Widely used benchmark with strong nonlinearity
- **wingweight**: Models weight of an aircraft wing
- **cantilever**: Beam deflection and stress
- ... and many more

## Credits

This package is a Python translation of an R package. All functions and descriptions are based on existing definitions from the literature, with proper citations in the docstrings.
