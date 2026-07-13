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
from researchpy.core.matrix_design import DesignMatrices, DesignMatrix, ModelMatrix
from researchpy.core.model import BaseModel
from .syntax_engine import SyntaxSpec, TermSpec, resolve
from .matrix_engine import build_indicator_matrix, grouped_statistic, grouped_statistic_pivot


# Define what gets exported with "from researchpy.core import *"
__all__ = [
    'as_array',
    'validate_array',
    'DesignMatrices',
    'DesignMatrix',
    'ModelMatrix',
    'BaseModel',
    'SyntaxSpec',
    'TermSpec',
    'resolve',
    'build_indicator_matrix',
    'grouped_statistic',
    'grouped_statistic_pivot',
]