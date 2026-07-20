# -*- coding: utf-8 -*-
"""
Researchpy Optimization Solvers Module

This module provides the optimization solvers used in ResearchPy for fitting statistical models. It includes numeric
estimation using ordinary least squares, and various optimization algorithms, such as Newton-Raphson and
SciPy's minimize function, as well as a tracker class for monitoring the optimization process.

Usage:
    >>> import researchpy.optimization.solvers
    >>> from researchpy.optimization.solvers import *

"""

from researchpy.optimization.solvers.closed_form import (
    ordinary_least_squares
)

from researchpy.optimization.solvers.iterative_algorithms import (
    scipy_minimize,
    newton_raphson
)


# Define what gets exported with "from reserachpy.optimization.solvers import *"
__all__ = [
    "ordinary_least_squares",
    "scipy_minimize",
    "newton_raphson",
]
