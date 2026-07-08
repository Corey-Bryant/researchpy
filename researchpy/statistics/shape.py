# -*- coding: utf-8 -*-
"""
Distribution shape measures: skewness and kurtosis.

Primary functions support 5 calling conventions:
    1. skewness(series_or_array)
    2. skewness("y ~ C(x)", df)
    3. skewness(["y", "k"], df)
    4. skewness(dv="y", by="x", data=df)
       skewness(dv="y", iv=["x","k"], data=df)
       skewness(dv="y", by="x", over="k", data=df)
    5. skewness("y ~ C(x):C(k)", df)
       skewness("y ~ C(x)*C(k)", df)
"""

from typing import Any, List, Optional, Union

import pandas
import scipy.stats

from ._compute import _route_computation


def skewness(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    decimals: int = 4,
) -> Union[float, pandas.DataFrame]:
    """Compute the skewness of the distribution, ignoring NaN values.

    Uses the adjusted Fisher-Pearson standardized moment coefficient
    (scipy default).

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute skewness directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
        Mutually exclusive with by/over.
    by : str or list of str, optional
        Row grouping variable(s) for cell computation.
    over : str or list of str, optional
        Column grouping variable(s) for pivot layout. Requires by.
    data : pd.DataFrame, optional
        Data source when using keyword arguments.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    float
        When single variable, no groups.
    pandas.DataFrame
        When multiple variables or grouped.

    Notes
    -----
    Computed via scipy.stats.skew with nan_policy='omit'.
    - Negative: left-skewed (tail on the left)
    - Zero: symmetric
    - Positive: right-skewed (tail on the right)

    Examples
    --------
    >>> skewness([1, 2, 3, 4, 100])  # Right-skewed
    2.0064...
    """
    def _skew_scalar(arr):
        return float(scipy.stats.skew(arr, nan_policy='omit'))

    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=_skew_scalar,
        matrix_stat=None,
        stat_label="Skewness",
        decimals=decimals,
    )


def kurtosis(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    fisher: bool = False,
    decimals: int = 4,
) -> Union[float, pandas.DataFrame]:
    """Compute the kurtosis of the distribution, ignoring NaN values.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        Data, formula, or column names.
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv, iv, by, over, data : keyword arguments
        See ``skewness()`` for full documentation.
    fisher : bool, optional
        If False (default), Pearson's definition is used (normal ==> 3.0).
        If True, Fisher's definition is used (normal ==> 0.0), which is
        the excess kurtosis.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    float or pandas.DataFrame

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
    def _kurtosis_scalar(arr):
        return float(scipy.stats.kurtosis(arr, fisher=fisher, nan_policy='omit'))

    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=_kurtosis_scalar,
        matrix_stat=None,
        stat_label="Kurtosis",
        decimals=decimals,
    )
