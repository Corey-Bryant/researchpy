# -*- coding: utf-8 -*-
"""
Confidence interval computation.

Currently supports t-distribution based intervals for the mean.
Future: bootstrap CIs, proportion CIs, median CIs.
"""

from typing import Tuple, Union

import numpy
import pandas
import scipy.stats

from ..core.data_utils import validate_array
from .observation import n_obs
from .central_tendency import mean
from .dispersion import standard_error


def confidence_interval(data: Union[pandas.Series, numpy.ndarray, list], confidence_level: float = 0.95,
                        decimals: int = 4,) -> Tuple[float, float]:
    """Compute a t-distribution based confidence interval for the mean.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).
    confidence_level : float, optional
        The confidence level (between 0 and 1 exclusive). Default is 0.95.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    tuple of (float, float)
        A tuple of (lower_bound, upper_bound) for the confidence interval.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1 (exclusive).
        If fewer than 2 non-NaN observations are present.

    Notes
    -----
    Uses the t-distribution with n-1 degrees of freedom:
        CI = mean ± t_{α/2, n-1} * SE

    where SE = SD / sqrt(n) and α = 1 - confidence_level.

    References
    ----------
    .. [1] Devore, J.L. (2011). Probability and Statistics for Engineering
       and the Sciences, 8th ed. Cengage Learning.

    Examples
    --------
    >>> confidence_interval([1, 2, 3, 4, 5], confidence_level=0.95)
    (1.038, 4.962)
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(
            f"confidence_level must be between 0 and 1 (exclusive), got {confidence_level}. "
            f"For a 95% CI, use confidence_level=0.95."
        )

    arr = validate_array(data, dtype=float)
    n = n_obs(arr)

    if n < 2:
        raise ValueError(
            f"At least 2 non-NaN observations are required for a confidence interval, "
            f"got {n}. Cannot estimate variability from a single observation."
        )

    df = n - 1
    data_mean = mean(arr)
    se = standard_error(arr)

    lower, upper = scipy.stats.t.interval(confidence_level, df, loc=data_mean, scale=se)

    return (round(float(lower), decimals), round(float(upper), decimals))


