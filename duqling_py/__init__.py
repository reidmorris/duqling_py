"""
duqling - Python package for UQ test functions

This package contains a collection of test functions widely used in uncertainty 
quantification (UQ) and computer experiments. Each function includes the option
to scale inputs from [0,1] to their native parameter ranges.
"""

__version__ = "0.1.0"
__author__ = "Reid Morris; Triad National Security, LLC"
__license__ = "GPL-3"
__copyright__ = "Copyright 2024. Triad National Security, LLC. All rights reserved."

from .duqling import Duqling

from .duqling_r import DuqlingR

from .functions import (
    banana,
    beam_deflection,
    borehole, borehole_low_fidelity,
    cantilever_D, cantilever_S,
    circuit,
    const_fn, const_fn3, const_fn15,
    crater,
    cube3, cube3_rotate, cube5,
    detpep_curve, detpep8, welch20,
    dms_simple, dms_radial, dms_harmonic, dms_additive, dms_complicated,
    dts_sirs,
    ebola,
    foursquare,
    friedman, friedman10, friedman20,
    Gfunction, Gfunction6, Gfunction12, Gfunction18,
    grlee1, grlee2, grlee6,
    ignition,
    ishigami,
    lim_polynomial, lim_non_polynomial,
    multivalley,
    ocean_circ,
    onehundred, d_onehundred,
    park4, park4_low_fidelity,
    piston, stochastic_piston,
    pollutant, pollutant_uni,
    ripples,
    robot,
    sharkfin,
    short_column,
    simple_machine, simple_machine_cm,
    simple_poly,
    squiggle,
    steel_column,
    sulfur,
    twin_galaxies,
    vinet,
    wingweight
)
