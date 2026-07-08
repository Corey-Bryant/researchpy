# -*- coding: utf-8 -*-
"""
Dispersion measures: variance, standard deviation, standard error, range,
coefficient of variation.

Primary functions support 5 calling conventions:
    1. variance(series_or_array)
    2. variance("y ~ C(x)", df)
    3. variance(["y", "k"], df)
    4. variance(dv="y", by="x", data=df)
       variance(dv="y", iv=["x","k"], data=df)
       variance(dv="y", by="x", over="k", data=df)
    5. variance("y ~ C(x):C(k)", df)
       variance("y ~ C(x)*C(k)", df)
"""

from typing import Any, List, Optional, Union

import numpy
import pandas
import scipy.stats

from ..core.data_utils import validate_array
from ._compute import _route_computation


def variance(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    ddof: int = 1,
    decimals: int = 4,
) -> Union[float, pandas.DataFrame]:
    """Compute the sample variance, ignoring NaN values.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        Data, formula, or column names.
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv, iv, by, over, data : keyword arguments
        See ``mean()`` for full documentation.
    ddof : int, optional
        Delta degrees of freedom. Default is 1 (sample variance).
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    float or pandas.DataFrame

    Examples
    --------
    >>> variance([1, 2, 3, 4, 5])
    2.5
    """
    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: numpy.nanvar(arr, ddof=ddof),
        matrix_stat="variance",
        stat_label="Variance",
        decimals=decimals,
    )


def standard_deviation(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    ddof: int = 1,
    decimals: int = 4,
) -> Union[float, pandas.DataFrame]:
    """Compute the sample standard deviation, ignoring NaN values.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        Data, formula, or column names.
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv, iv, by, over, data : keyword arguments
        See ``mean()`` for full documentation.
    ddof : int, optional
        Delta degrees of freedom. Default is 1 (sample SD).
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    float or pandas.DataFrame

    Examples
    --------
    >>> standard_deviation([1, 2, 3, 4, 5])
    1.5811
    """
    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: numpy.nanstd(arr, ddof=ddof),
        matrix_stat="sd",
        stat_label="SD",
        decimals=decimals,
    )


def standard_error(
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
    """Compute the standard error of the mean, ignoring NaN values.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        Data, formula, or column names.
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv, iv, by, over, data : keyword arguments
        See ``mean()`` for full documentation.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.

    Returns
    -------
    float or pandas.DataFrame

    Examples
    --------
    >>> standard_error([1, 2, 3, 4, 5])
    0.7071
    """
    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: float(scipy.stats.sem(arr, nan_policy='omit')),
        matrix_stat="se",
        stat_label="SE",
        decimals=decimals,
    )


def value_range(data: Union[pandas.Series, numpy.ndarray, list]) -> float:
    """Compute the range (max - min), ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    float
        The range of non-NaN values.

    Examples
    --------
    >>> value_range([1, 2, 3, 4, 5])
    4.0
    """
    arr = validate_array(data, dtype=float)
    return float(numpy.nanmax(arr) - numpy.nanmin(arr))


def coefficient_of_variation(data: Union[pandas.Series, numpy.ndarray, list], ddof: int = 1) -> float:
    """Compute the coefficient of variation (CV = SD / Mean * 100).

    Expressed as a percentage. Ignores NaN values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).
    ddof : int, optional
        Delta degrees of freedom for SD calculation. Default is 1.

    Returns
    -------
    float
        The coefficient of variation as a percentage.

    Raises
    ------
    ValueError
        If the mean is zero (CV is undefined).

    Examples
    --------
    >>> coefficient_of_variation([1, 2, 3, 4, 5])
    52.7046
    """
    arr = validate_array(data, dtype=float)
    data_mean = float(numpy.nanmean(arr))
    if data_mean == 0.0:
        raise ValueError(
            "Coefficient of variation is undefined when the mean is zero."
        )
    data_sd = float(numpy.nanstd(arr, ddof=ddof))

    return float(data_sd / data_mean * 100)
