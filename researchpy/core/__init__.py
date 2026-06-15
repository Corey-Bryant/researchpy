# -*- coding: utf-8 -*-
"""
ResearchPy Code Module

This module provides the core classes for ResearchPy, including result containers, base model classes, and
helper/utility functions.

Usage:
    >>> import researchpy.core
    >>> from researchpy.core import *

"""

from .data_utils import as_array, validate_array
from .containerclasses import ModelFit, ModelEffects, CoefResults, FactorEffects, FitStatistics, ModelResults, TestResults, Term, ModelTerms, SolverOptions


# Define what gets exported with "from researchpy.core import *"
__all__ = [
    'as_array',
    'validate_array',
    #'CoreModel',
    #'GeneralModel',
    'ModelFit',
    'ModelEffects',
    'CoefResults',
    'FactorEffects',
    'FitStatistics',
    'ModelResults',
    'TestResults',
    'Term',
    'ModelTerms',
    'SolverOptions'
]