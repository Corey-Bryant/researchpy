# -*- coding: utf-8 -*-
"""
descriptive subpackage
====================

Provides modular univariate summary statistics for continuous data.

Modules:
    observation        - N, N missing, percent missing
    central_tendency   - Mean, median, mode, quartiles, percentiles, IQR
    dispersion         - Variance, SD, SE, range, coefficient of variation
    intervals          - Confidence intervals (t-based, future: bootstrap, proportion)
    shape              - Skewness, kurtosis
    categorical        - Value counts, proportions, cumulative proportions

Public API:
    summarize()        - Unified dispatcher for computing summary statistics
"""

from ._estimation_interval import _confidence_interval




__all__ = [
    "_confidence_interval",
]

