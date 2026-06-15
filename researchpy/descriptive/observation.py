# -*- coding: utf-8 -*-
"""
Observation metadata: counts, missingness.

Primary functions support 5 calling conventions:
    1. n_obs(series_or_array)
    2. n_obs("y ~ C(x)", df)
    3. n_obs(["y", "k"], df)
    4. n_obs(dv="y", by="x", data=df)
    5. n_obs("y ~ C(x):C(k)", df)
"""

from typing import Any, List, Optional, Union

import numpy
import pandas

from ..core.data_utils import validate_array
from ._compute import _route_computation


def n_obs(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    decimals: int = 0,
) -> Union[int, pandas.DataFrame]:
    """Count the number of non-missing (non-NaN) observations.

    Supports multiple calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        Data, formula, or column names.
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv, iv, by, data : keyword arguments
        See ``mean()`` for full documentation.
    decimals : int, optional
        Decimal places (default 0 since counts are integers).

    Returns
    -------
    int
        When single variable, no groups.
    pandas.DataFrame
        When grouped.

    Examples
    --------
    >>> import pandas as pd
    >>> n_obs(pd.Series([1, 2, None, 4]))
    3

    >>> n_obs("y ~ C(g)", df)
      Group   N
    0     a   2
    1     b   2
    """
    result = _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, data=data,
        scalar_func=lambda arr: int(numpy.count_nonzero(~numpy.isnan(arr))),
        matrix_stat="count",
        decimals=decimals,
    )
    # For ungrouped single-DV, return int instead of float
    if isinstance(result, float):
        return int(result)
    return result


def n_missing(data: Union[pandas.Series, numpy.ndarray, list]) -> int:
    """Count the number of missing (NaN) observations.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    int
        Number of NaN observations.

    Examples
    --------
    >>> import pandas as pd
    >>> n_missing(pd.Series([1, 2, None, 4]))
    1
    """
    arr = validate_array(data, dtype=float)
    return int(numpy.count_nonzero(numpy.isnan(arr)))


def percent_missing(data: Union[pandas.Series, numpy.ndarray, list]) -> float:
    """Calculate the percentage of missing (NaN) observations.

    Parameters
    ----------
    data : array_like
        Input data (Series, ndarray, or list).

    Returns
    -------
    float
        Percentage of missing values (0.0 to 100.0).

    Raises
    ------
    ValueError
        If data has zero total length.

    Examples
    --------
    >>> import pandas as pd
    >>> percent_missing(pd.Series([1, 2, None, 4]))
    25.0
    """
    arr = validate_array(data, dtype=float)
    total = len(arr)
    if total == 0:
        raise ValueError("Cannot compute percent missing on zero-length data.")
    missing_count = int(numpy.count_nonzero(numpy.isnan(arr)))
    return float(missing_count / total * 100)
