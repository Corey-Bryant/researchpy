# -*- coding: utf-8 -*-
"""
Researchpy Optimization Solvers Module

This module provides the optimization algorithms used in ResearchPy for fitting statistical models. It includes numeric
estimation using ordinary least squares, and various optimization algorithms, such as Newton-Raphson and
SciPy's minimize function, as well as a tracker class for monitoring the optimization process.

Usage:
    >>> import researchpy.optimize.algorithms
    >>> from researchpy.optimize.algorithms import *

"""

from researchpy.optimize.algorithms.methods import (
    IRLS,
)


# Define what gets exported with "from reserachpy.optimize.algorithms import *"
__all__ = [
    "IRLS",
]
