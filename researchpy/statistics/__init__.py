# -*- coding: utf-8 -*-
"""
statistics subpackage
=====================

Provides modular univariate summary statistics built on the researchpy
engine (syntax + matrix + table).

Modules:
    _base              - DesignSpec design layer and missing-data helpers
    _distribution      - Distribution lookup, p-values, confidence intervals
    _hypothesis_test   - Hypothesis tests and effect sizes
    observation        - N, N missing, percent missing
    central_tendency   - Mean, median, mode, quartiles, percentiles, IQR
    dispersion         - Variance, SD, SE, range, coefficient of variation
    intervals          - Confidence intervals (t-based, future: bootstrap, proportion)
    shape              - Skewness, kurtosis
    categorical        - Tabulate, value counts, proportions, cumulative proportions

Public API:
    DesignSpec         - Normalized design specification consumed by all modules
    observations       - N, N Missing, Percent Missing, N Total
    n_obs              - Non-missing count
    n_missing          - Missing count
    percent_missing    - Percent missing

Notes
-----
``_base`` also defines scalar helpers named ``count_missing``,
``count_valid``, and ``percent_missing`` that operate on a single
array-like.  They are deliberately not re-exported here, because the
package-level ``percent_missing`` is the design-aware public function from
``observation``.  Import the helpers from ``researchpy.statistics._base``
when the scalar form is wanted.
"""

from ._base import DesignSpec
from ._distribution import (
    _get_distribution,
    _estimate_confidence_interval,
    _compute_pvalue,
    _confidence_interval,
)
from .observation import (
    observations,
    n_obs,
    n_missing,
    percent_missing,
)
#from ._hypothesis_test import compute_pvalue


__all__ = [
    "DesignSpec",
    "observations",
    "n_obs",
    "n_missing",
    "percent_missing",
    "_get_distribution",
    "_estimate_confidence_interval",
    "_compute_pvalue",
    #"_confidence_interval",
]
