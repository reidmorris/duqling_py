# PyDuqling

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This package is a Python implementation of the reproducable UQ research R package, [Duqling](https://github.com/knrumsey/duqling), by K. Rumsey et al.

## Description

This package includes two Python interfaces that should be used to interact with the library of test functions:
1. [Duqling](/tests/duqling.py): to interact with the Python-based Duqling package
2. [DuqlingR](/tests/duqling_r.py): to interact with the R-based Duqling package

Both of these interfaces implement the `quack` and `duq` functions as to match the behavior of the original R package. They can be used interchangeably, although the Python-based package offers a substantial speedup.

## Quick start

### Instantiate the Duqling interface
``` python
from duqling import Duqling
duqling = Duqling()
```

### Test Functions in the Library
A master list of all functions found in the `duqling` package can be
found with the command
``` python
duqling.quack()
```
##### **Output**:
|| fname                 |   input_dim | input_cat   | response_type   | stochastic   |
|---:|:----------------------|------------:|:------------|:----------------|:-------------|
|  0 | const_fn              |           1 | False       | uni             | n            |
|  1 | grlee1                |           1 | False       | uni             | n            |
|  2 | banana                |           2 | False       | uni             | n            |
|  3 | dms_additive          |           2 | False       | uni             | n            |
|  4 | dms_complicated       |           2 | False       | uni             | n            |
|  5 | dms_harmonic          |           2 | False       | uni             | n            |
|  6 | dms_radial            |           2 | False       | uni             | n            |
|  7 | dms_simple            |           2 | False       | uni             | n            |
|  8 | foursquare            |           2 | False       | uni             | n            |
|  9 | grlee2                |           2 | False       | uni             | n            |
| 10 | lim_non_polynomial    |           2 | False       | uni             | n            |
| 11 | lim_polynomial        |           2 | False       | uni             | n            |
| 12 | multivalley           |           2 | False       | uni             | n            |
| 13 | ripples               |           2 | False       | uni             | n            |
| 14 | simple_poly           |           2 | False       | uni             | n            |
| 15 | squiggle              |           2 | False       | uni             | n            |
| 16 | twin_galaxies         |           2 | False       | uni             | n            |
| 17 | Gfunction             |           3 | False       | uni             | n            |
| 18 | const_fn3             |           3 | False       | uni             | n            |
| 19 | cube3                 |           3 | False       | uni             | n            |
| 20 | cube3_rotate          |           3 | False       | uni             | n            |
| 21 | detpep_curve          |           3 | False       | uni             | n            |
| 22 | ishigami              |           3 | False       | uni             | n            |
| 23 | sharkfin              |           3 | False       | uni             | n            |
| 24 | simple_machine        |           3 | False       | func            | n            |
| 25 | vinet                 |           3 | False       | func            | n            |
| 26 | ocean_circ            |           4 | False       | uni             | y            |
| 27 | park4                 |           4 | False       | uni             | n            |
| 28 | park4_low_fidelity    |           4 | False       | uni             | n            |
| 29 | pollutant             |           4 | False       | func            | n            |
| 30 | pollutant_uni         |           4 | False       | uni             | n            |
| 31 | beam_deflection       |           5 | False       | func            | n            |
| 32 | cube5                 |           5 | False       | uni             | n            |
| 33 | friedman              |           5 | False       | uni             | n            |
| 34 | short_column          |           5 | False       | uni             | n            |
| 35 | simple_machine_cm     |           5 | False       | func            | n            |
| 36 | stochastic_piston     |           5 | False       | uni             | y            |
| 37 | Gfunction6            |           6 | False       | uni             | n            |
| 38 | cantilever_D          |           6 | False       | uni             | n            |
| 39 | cantilever_S          |           6 | False       | uni             | n            |
| 40 | circuit               |           6 | False       | uni             | n            |
| 41 | grlee6                |           6 | False       | uni             | n            |
| 42 | crater                |           7 | False       | uni             | n            |
| 43 | piston                |           7 | False       | uni             | n            |
| 44 | borehole              |           8 | False       | uni             | n            |
| 45 | borehole_low_fidelity |           8 | False       | uni             | n            |
| 46 | detpep8               |           8 | False       | uni             | n            |
| 47 | ebola                 |           8 | False       | uni             | n            |
| 48 | robot                 |           8 | False       | uni             | n            |
| 49 | dts_sirs              |           9 | False       | func            | y            |
| 50 | steel_column          |           9 | False       | uni             | n            |
| 51 | sulfur                |           9 | False       | uni             | n            |
| 52 | friedman10            |          10 | False       | uni             | n            |
| 53 | ignition              |          10 | False       | uni             | n            |
| 54 | wingweight            |          10 | False       | uni             | n            |
| 55 | Gfunction12           |          12 | False       | uni             | n            |
| 56 | const_fn15            |          15 | False       | uni             | n            |
| 57 | Gfunction18           |          18 | False       | uni             | n            |
| 58 | friedman20            |          20 | False       | uni             | n            |
| 59 | welch20               |          20 | False       | uni             | n            |
| 60 | onehundred            |         100 | False       | uni             | n            |'


<br>

### Function search using filtering criteria
A list of all functions meeting certain criterion can be found with the command
``` python
duqling.quack(input_dim=range(4,8), stochastic="n")
```
#### **Output:**
|    | fname              |   input_dim | input_cat   | response_type   | stochastic   |
|---:|:-------------------|------------:|:------------|:----------------|:-------------|
|  0 | park4              |           4 | False       | uni             | n            |
|  1 | park4_low_fidelity |           4 | False       | uni             | n            |
|  2 | pollutant          |           4 | False       | func            | n            |
|  3 | pollutant_uni      |           4 | False       | uni             | n            |
|  4 | beam_deflection    |           5 | False       | func            | n            |
|  5 | cube5              |           5 | False       | uni             | n            |
|  6 | friedman           |           5 | False       | uni             | n            |
|  7 | short_column       |           5 | False       | uni             | n            |
|  8 | simple_machine_cm  |           5 | False       | func            | n            |
|  9 | Gfunction6         |           6 | False       | uni             | n            |
| 10 | cantilever_D       |           6 | False       | uni             | n            |
| 11 | cantilever_S       |           6 | False       | uni             | n            |
| 12 | circuit            |           6 | False       | uni             | n            |
| 13 | grlee6             |           6 | False       | uni             | n            |
| 14 | crater             |           7 | False       | uni             | n            |
| 15 | piston             |           7 | False       | uni             | n            |

<br>

### Query function info
A detailed description of each function (the `borehole()` function, for example) can be found with the command
``` python
duqling.quack("borehole")
```
#### **Output**:
```python
 {'input_dim': 8,
  'input_cat': False,
  'response_type': 'uni',
  'stochastic': 'n',
  'input_range': array([[5.0000e-02, 1.5000e-01],
                        [1.0000e+02, 5.0000e+04],
                        [6.3070e+04, 1.1560e+05],
                        [9.9000e+02, 1.1100e+03],
                        [6.3100e+01, 1.1600e+02],
                        [7.0000e+02, 8.2000e+02],
                        [1.1200e+03, 1.6800e+03],
                        [9.8550e+03, 1.2045e+04]])}
```

<br>

### Call test functions

Use the `duq` method to call a function on a single input. This function can be the string name of a supported test function in this package, or a custom callable function.

For example,
```python
func_info = duqling.quack('borehole')
input_dim = func_info['input_dim']
x = np.random.rand(input_dim)
y = duqling.duq(x, 'borehole')
```

Or, equivalently,
```python
from duqling_py.functions import borehole

y = duqling.duq(x, borehole)
```

Use the `batch_duq` method to call a function on a batch of input samples. 

```python
NUM_SAMPLES = 10

func_info = duqling.quack('borehole')
input_dim = func_info['input_dim']
X = np.random.rand(NUM_SAMPLES, input_dim)
Y = duqling.batch_duq(X, 'borehole')
```