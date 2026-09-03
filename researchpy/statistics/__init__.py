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

from ._distribution import (
    _get_distribution,
    _estimate_confidence_interval,
    _compute_pvalue,
    _confidence_interval as confidence_interval,
)
#from ._hypothesis_test import compute_pvalue


__all__ = [
    "_get_distribution",
    "_estimate_confidence_interval",
    "_compute_pvalue",
    #"_confidence_interval",
    "confidence_interval",
]
