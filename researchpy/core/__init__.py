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
from researchpy.core.matrix_design import DesignMatrices, DesignMatrix, DMatrix
from researchpy.core.model import BaseModel
from .spec import ComputeSpec, TermSpec, resolve
from .matrix_engine import build_indicator_matrix, grouped_statistic, grouped_statistic_pivot


# Define what gets exported with "from researchpy.core import *"
__all__ = [
    'as_array',
    'validate_array',
    'DesignMatrices',
    'DesignMatrix',
    'DMatrix',
    'BaseModel',
    'ComputeSpec',
    'TermSpec',
    'resolve',
    'build_indicator_matrix',
    'grouped_statistic',
    'grouped_statistic_pivot',
]