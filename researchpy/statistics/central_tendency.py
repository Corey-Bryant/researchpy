# -*- coding: utf-8 -*-
"""
Central tendency measures: mean, median, mode, quartiles, percentiles, IQR.

All primary functions support 5 calling conventions:
    1. mean(series_or_array)
    2. mean("y ~ C(x)", df)
    3. mean(["y", "k"], df)
    4. mean(dv="y", by="x", data=df)
       mean(dv="y", iv=["x","k"], data=df)
       mean(dv="y", by="x", over="k", data=df)
    5. mean("y ~ C(x):C(k)", df)
       mean("y ~ C(x)*C(k)", df)
"""

from typing import Any, Dict, List, Optional, Union

import numpy
import pandas

from ..core.data_utils import validate_array
from ._compute import _route_computation, _grouped_via_iteration


def mean(arg1: Any = None, arg2: Any = None, /, *,
         dv: Optional[Union[str, List[str]]] = None,
         iv: Optional[Union[str, List[str]]] = None,
         by: Optional[Union[str, List[str]]] = None,
         over: Optional[Union[str, List[str]]] = None,
         data: Optional[pandas.DataFrame] = None,
         decimals: int = 4,) -> Union[float, pandas.DataFrame]:
    """Compute the arithmetic mean, ignoring NaN values.

    Supports multiple calling conventions for flexibility.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute mean directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
        Computes stat for each iv independently, stacks results.
        Mutually exclusive with by/over.
    by : str or list of str, optional
        Row grouping variable(s) for cell means.
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
        When multiple variables, marginal, cell, or pivot.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','b','b','c','c']})

    >>> mean(df['y'])
    3.5

    >>> mean("y ~ C(g)", df)
            Mean
    g
    a       1.5
    b       3.5
    c       5.5

    >>> mean(dv="y", by="g", data=df)
            Mean
    g
    a       1.5
    b       3.5
    c       5.5
    """
    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: numpy.nanmean(arr),
        matrix_stat="mean",
        stat_label="Mean",
        decimals=decimals,
    )


def median(
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
    """Compute the median, ignoring NaN values.

    Supports multiple calling conventions for flexibility.

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
    >>> median([1, 2, 3, 4, 5])
    3.0
    """
    def _median_grouped(data_df, dv_col, groups, dec, **kw):
        return _grouped_via_iteration(
            data_df, dv_col, groups, dec,
            scalar_func=numpy.nanmedian,
            stat_label="Median",
        )

    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: numpy.nanmedian(arr),
        matrix_stat=None,
        fallback_grouped_func=_median_grouped,
        stat_label="Median",
        decimals=decimals,
    )


def mode(data: Union[pandas.Series, numpy.ndarray, list]) -> Union[float, List[float]]:
    """Compute the mode (most frequent value), ignoring NaN values.

    If multiple modes exist, returns a list of all modal values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    float or list of float
        The mode value(s). Returns a single float if unimodal,
        or a list of floats if multimodal.

    Examples
    --------
    >>> mode([1, 2, 2, 3, 3])
    [2.0, 3.0]
    >>> mode([1, 2, 2, 3])
    2.0
    """
    if isinstance(data, pandas.Series):
        modes = data.mode().tolist()
    else:
        arr = validate_array(data, dtype=float)
        clean = arr[~numpy.isnan(arr)]
        unique, counts = numpy.unique(clean, return_counts=True)
        max_count = counts.max()
        modes = unique[counts == max_count].tolist()

    if len(modes) == 1:
        return float(modes[0])
    return [float(m) for m in modes]


def quartiles(data: Union[pandas.Series, numpy.ndarray, list]) -> Dict[str, float]:
    """Compute the first (Q1), second (Q2/median), and third (Q3) quartiles.

    Uses linear interpolation (numpy default).

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    dict
        Dictionary with keys 'Q1', 'Q2', 'Q3' mapped to float values.

    Examples
    --------
    >>> quartiles([1, 2, 3, 4, 5])
    {'Q1': 2.0, 'Q2': 3.0, 'Q3': 4.0}
    """
    arr = validate_array(data, dtype=float)
    q1, q2, q3 = numpy.nanpercentile(arr, [25, 50, 75])
    return {"Q1": float(q1), "Q2": float(q2), "Q3": float(q3)}


def percentile(
    data: Union[pandas.Series, numpy.ndarray, list],
    q: Union[float, List[float]] = 50.0,
) -> Union[float, Dict[str, float]]:
    """Compute one or more percentiles, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).
    q : float or list of float
        Percentile(s) to compute. Must be between 0 and 100 inclusive.

    Returns
    -------
    float or dict
        A single float if q is scalar, or a dict mapping
        'P{value}' -> computed percentile if q is a list.

    Raises
    ------
    ValueError
        If any percentile value is outside [0, 100].

    Examples
    --------
    >>> percentile([1, 2, 3, 4, 5], q=50)
    3.0
    >>> percentile([1, 2, 3, 4, 5], q=[10, 25, 50, 75, 90])
    {'P10': 1.4, 'P25': 2.0, 'P50': 3.0, 'P75': 4.0, 'P90': 4.6}
    """
    arr = validate_array(data, dtype=float)

    if isinstance(q, (int, float)):
        if q < 0 or q > 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {q}.")
        return float(numpy.nanpercentile(arr, q))
    else:
        for val in q:
            if val < 0 or val > 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {val}.")
        results = numpy.nanpercentile(arr, q)
        return {f"P{int(v) if v == int(v) else v}": float(r) for v, r in zip(q, results)}


def iqr(data: Union[pandas.Series, numpy.ndarray, list]) -> float:
    """Compute the interquartile range (Q3 - Q1), ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    float
        The interquartile range.

    Examples
    --------
    >>> iqr([1, 2, 3, 4, 5])
    2.0
    """
    arr = validate_array(data, dtype=float)
    q1, q3 = numpy.nanpercentile(arr, [25, 75])
    return float(q3 - q1)

