# -*- coding: utf-8 -*-
"""
ResearchPy Code Module

This module provides the core classes for ResearchPy, including result containers, helper/utility functions.

Usage:
    >>> import researchpy.core
    >>> from researchpy.core import *

"""

from researchpy.core.data_utils import as_array, validate_array
from researchpy.core.matrix_design import DesignMatrix
from researchpy.core.syntax_engine import SyntaxSpec, TermSpec, resolve
from researchpy.core.matrix_engine import build_indicator_matrix, grouped_statistic, grouped_statistic_pivot


# Define what gets exported with "from researchpy.core import *"
__all__ = [
    'as_array',
    'validate_array',
    'DesignMatrix',
    'SyntaxSpec',
    'TermSpec',
    'resolve',
    'build_indicator_matrix',
    'grouped_statistic',
    'grouped_statistic_pivot',
]