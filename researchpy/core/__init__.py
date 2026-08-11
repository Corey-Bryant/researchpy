# -*- coding: utf-8 -*-
"""
ResearchPy Code Module

This module provides the core classes for ResearchPy, including result containers, helper/utility functions.

Usage:
    >>> import researchpy.core
    >>> from researchpy.core import *

"""

from researchpy.core.data_utils import as_array, validate_array


# Define what gets exported with "from researchpy.core import *"
__all__ = [
    'as_array',
    'validate_array',
]