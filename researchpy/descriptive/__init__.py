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

from .observation import n_obs, n_missing, percent_missing
from .central_tendency import mean, median, mode, quartiles, percentile, iqr
from .dispersion import variance, standard_deviation, standard_error, value_range, coefficient_of_variation
from .intervals import confidence_interval
from .shape import skewness, kurtosis
from .categorical import n_unique, value_counts, proportions, cumulative_proportions, tabulate
from ._dispatcher import summarize
from .containers import SummaryResult


__all__ = [
    # Observation
    "n_obs",
    "n_missing",
    "percent_missing",
    # Central tendency
    "mean",
    "median",
    "mode",
    "quartiles",
    "percentile",
    "iqr",
    # Dispersion
    "variance",
    "standard_deviation",
    "standard_error",
    "value_range",
    "coefficient_of_variation",
    # Intervals
    "confidence_interval",
    # Shape
    "skewness",
    "kurtosis",
    # Categorical
    "n_unique",
    "value_counts",
    "proportions",
    "cumulative_proportions",
    "tabulate",
    # Result container
    "SummaryResult",
    # Dispatcher
    "summarize",
]

