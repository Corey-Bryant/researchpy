# -*- coding: utf-8 -*-
"""
Distribution shape measures: skewness and kurtosis.
"""

from typing import Union

import numpy
import pandas
import scipy.stats

from ..core.data_utils import validate_array




def skewness(data: Union[pandas.Series, numpy.ndarray, list]) -> float:
    """Compute the skewness of the distribution, ignoring NaN values.

    Uses the adjusted Fisher-Pearson standardized moment coefficient
    (scipy default).

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    float
        The skewness of the data distribution.
        - Negative: left-skewed (tail on the left)
        - Zero: symmetric
        - Positive: right-skewed (tail on the right)

    Notes
    -----
    Computed via scipy.stats.skew with nan_policy='omit'.

    Examples
    --------
    >>> skewness([1, 2, 3, 4, 100])  # Right-skewed
    2.0064...
    """
    arr = validate_array(data, dtype=float)

    return float(scipy.stats.skew(arr, nan_policy='omit'))


def kurtosis(
    data: Union[pandas.Series, numpy.ndarray, list],
    fisher: bool = False,
) -> float:
    """Compute the kurtosis of the distribution, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).
    fisher : bool, optional
        If False (default), Pearson's definition is used (normal ==> 3.0).
        If True, Fisher's definition is used (normal ==> 0.0), which is
        the excess kurtosis.

    Returns
    -------
    float
        The kurtosis of the data distribution.

    Notes
    -----
    Default uses Pearson's definition (fisher=False) for consistency with
    the existing researchpy API. A mesokurtic (normal) distribution has
    kurtosis = 3.0 under Pearson's definition.

    - Leptokurtic (heavy tails): kurtosis > 3 (Pearson)
    - Mesokurtic (normal-like): kurtosis ≈ 3 (Pearson)
    - Platykurtic (light tails): kurtosis < 3 (Pearson)

    Examples
    --------
    >>> kurtosis([1, 2, 3, 4, 5])  # Uniform-ish, platykurtic
    1.7
    """
    arr = validate_array(data, dtype=float)

    return float(scipy.stats.kurtosis(arr, fisher=fisher, nan_policy='omit'))

