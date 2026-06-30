# -*- coding: utf-8 -*-
"""
Unified summarize() dispatcher.

Routes computation based on input type (Series, DataFrame, GroupBy) and
delegates to the individual statistic modules.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy
import pandas
 

#from ..containers.base import DesignInfo, EstimateResults
from ..containers.univariate import SummaryResult
from ..core.data_utils import validate_array
from .observation import n_obs, n_missing, percent_missing
from .central_tendency import mean, median, mode, quartiles, iqr
from .dispersion import variance, standard_deviation, standard_error, value_range, coefficient_of_variation
from .intervals import confidence_interval
from .shape import skewness, kurtosis


# ---------------------------------------------------------------------------
# Registry: maps user-facing stat names to computation callables.
# Each callable takes (arr, **kwargs) and returns a scalar or structured value.
# ---------------------------------------------------------------------------

_STAT_REGISTRY: Dict[str, Any] = {
    "N": lambda arr, **kw: n_obs(arr),
    "N Missing": lambda arr, **kw: n_missing(arr),
    "% Missing": lambda arr, **kw: percent_missing(arr),
    "Mean": lambda arr, **kw: mean(arr),
    "Median": lambda arr, **kw: median(arr),
    "Mode": lambda arr, **kw: mode(arr),
    "Variance": lambda arr, **kw: variance(arr),
    "SD": lambda arr, **kw: standard_deviation(arr),
    "SE": lambda arr, **kw: standard_error(arr),
    "CI": lambda arr, **kw: confidence_interval(
        arr,
        confidence_level=kw.get("ci_level", 0.95),
        decimals=kw.get("decimals", 4),
    ),
    "Min": lambda arr, **kw: float(numpy.nanmin(arr)),
    "Max": lambda arr, **kw: float(numpy.nanmax(arr)),
    "Range": lambda arr, **kw: value_range(arr),
    "IQR": lambda arr, **kw: iqr(arr),
    "Kurtosis": lambda arr, **kw: kurtosis(arr),
    "Skew": lambda arr, **kw: skewness(arr),
    "CV": lambda arr, **kw: coefficient_of_variation(arr),
    "Quartiles": lambda arr, **kw: quartiles(arr),
}

# Default stats when none are specified
_DEFAULT_STATS = ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"]



def _get_stat_label(stat_name: str, ci_level: float) -> str:
    """Convert internal stat name to display label."""
    if stat_name == "CI":
        return f"{int(ci_level * 100)}% Conf. Interval"
    return stat_name


def _compute_for_array(
    data: Any,
    stats: List[str],
    ci_level: float,
    decimals: int,
    name: Optional[str],
) -> SummaryResult:
    """Compute summary statistics for a single 1-D array-like input.

    Parameters
    ----------
    data : array_like
        The data to summarize.
    stats : list of str
        Statistics to compute. Names must be keys in _STAT_REGISTRY.
    ci_level : float
        Confidence level for CI computation.
    decimals : int
        Decimal places for rounding.
    name : str or None
        Variable name.

    Returns
    -------
    SummaryResult
        Container with computed statistics.
    """
    arr = validate_array(data, dtype=float)
    kwargs = {"ci_level": ci_level, "decimals": decimals}

    computed = {}
    for stat_name in stats:
        if stat_name not in _STAT_REGISTRY:
            raise ValueError(
                f"Unknown statistic '{stat_name}'. "
                f"Available options: {sorted(_STAT_REGISTRY.keys())}"
            )
        label = _get_stat_label(stat_name, ci_level)
        value = _STAT_REGISTRY[stat_name](arr, **kwargs)

        # Round scalar results
        if isinstance(value, float):
            value = round(value, decimals)
        elif isinstance(value, int):
            pass  # Keep ints as-is
        # Tuples/lists (like CI) and dicts (like quartiles) stay as-is

        computed[label] = value

    return SummaryResult(name=name, statistics=computed)


def _compute_for_dataframe(
    data: pandas.DataFrame,
    stats: List[str],
    ci_level: float,
    decimals: int,
) -> pandas.DataFrame:
    """Compute summary statistics for each column of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with numeric columns.
    stats : list of str
        Statistics to compute.
    ci_level : float
        Confidence level for CI computation.
    decimals : int
        Decimal places for rounding.

    Returns
    -------
    pandas.DataFrame
        One row per column, statistics as columns.
    """
    rows = []
    for col_name in data.columns:
        result = _compute_for_array(
            data[col_name], stats, ci_level, decimals, name=col_name
        )
        rows.append(result.to_dict())

    return pandas.DataFrame(rows)


def _compute_for_groupby(
    data: Union[pandas.core.groupby.SeriesGroupBy, pandas.core.groupby.DataFrameGroupBy],
    stats: List[str],
    ci_level: float,
    decimals: int,
) -> pandas.DataFrame:
    """Compute summary statistics for grouped data.

    Parameters
    ----------
    data : SeriesGroupBy or DataFrameGroupBy
        Grouped pandas object.
    stats : list of str
        Statistics to compute.
    ci_level : float
        Confidence level for CI computation.
    decimals : int
        Decimal places for rounding.

    Returns
    -------
    pandas.DataFrame
        Results with group index preserved.
    """
    kwargs = {"ci_level": ci_level, "decimals": decimals}
    rows = []

    for group_key, group_data in data:
        if isinstance(group_data, pandas.Series):
            arr = validate_array(group_data,)
            row = {"Group": group_key}

            for stat_name in stats:
                if stat_name not in _STAT_REGISTRY:
                    raise ValueError(
                        f"Unknown statistic '{stat_name}'. "
                        f"Available options: {sorted(_STAT_REGISTRY.keys())}"
                    )

                label = _get_stat_label(stat_name, ci_level)
                value = _STAT_REGISTRY[stat_name](arr, **kwargs)

                if isinstance(value, float):
                    value = round(value, decimals)

                row[label] = value
            rows.append(row)

        elif isinstance(group_data, pandas.DataFrame):
            for col_name in group_data.columns:
                arr = validate_array(group_data[col_name], dtype=float)
                row = {"Group": group_key, "Variable": col_name}
                for stat_name in stats:
                    if stat_name not in _STAT_REGISTRY:
                        raise ValueError(
                            f"Unknown statistic '{stat_name}'. "
                            f"Available options: {sorted(_STAT_REGISTRY.keys())}"
                        )
                    label = _get_stat_label(stat_name, ci_level)
                    value = _STAT_REGISTRY[stat_name](arr, **kwargs)
                    if isinstance(value, float):
                        value = round(value, decimals)
                    row[label] = value
                rows.append(row)

    return pandas.DataFrame(rows)


'''
class Estimate(DesignInfo, EstimateResults):

    def __init__(self, formula_like: Optional[str]=None,
                 data: Union[pandas.Series, pandas.DataFrame, numpy.ndarray, list]=None,
                 ci_level=0.95):
        ...
'''



def summarize(data: Any, name: Optional[str] = None, stats: Optional[List[str]] = None, ci_level: float = 0.95,
              decimals: int = 4, return_type: str = "Dataframe",) -> Union[pandas.DataFrame, dict]:
    """Compute univariate summary statistics for numeric data.

    A unified dispatcher that handles Series, DataFrames, GroupBy objects,
    and plain array-like inputs. Delegates computation to focused modules
    for central tendency, dispersion, intervals, and shape.

    Parameters
    ----------
    data : array_like, Series, DataFrame, or GroupBy
        The data to summarize.
    name : str, optional
        Override the variable name in output. Default is None (auto-detect).
    stats : list of str, optional
        Statistics to compute. Default is ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"].

        Available options:
            "N", "N Missing", "% Missing", "Mean", "Median", "Mode",
            "Variance", "SD", "SE", "CI", "Min", "Max", "Range",
            "IQR", "Kurtosis", "Skew", "CV", "Quartiles"
    ci_level : float, optional
        Confidence level for CI computation. Default is 0.95.
    decimals : int, optional
        Number of decimal places to round to. Default is 4.
    return_type : str, optional
        Output format: "Dataframe" (default) or "Dictionary".

    Returns
    -------
    pandas.DataFrame or dict
        Summary statistics in the requested format.

    Raises
    ------
    ValueError
        If return_type is not supported, or if an unknown stat name is given.
    TypeError
        If data cannot be converted to a numeric array.

    Examples
    --------
    >>> import pandas as pd
    >>> from researchpy.descriptive import summarize

    Single Series:
    >>> s = pd.Series([1, 2, 3, 4, 5], name="scores")
    >>> summarize(s)
        Name  N  Mean  Median  Variance      SD      SE  95% Conf. Interval
    0  scores  5   3.0     3.0       2.5  1.5811  0.7071        (1.038, 4.962)

    DataFrame:
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> summarize(df, stats=["N", "Mean", "SD"])
      Name  N  Mean      SD
    0    A  3   2.0  1.0000
    1    B  3   5.0  1.0000

    GroupBy:
    >>> df = pd.DataFrame({"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    >>> summarize(df.groupby("group")["val"], stats=["N", "Mean"])
      Group  N  Mean
    0     a  2   1.5
    1     b  2   3.5
    """
    # Validate return_type
    if return_type.upper() not in ("DATAFRAME", "DICTIONARY"):
        raise ValueError(
            f"Unsupported return_type '{return_type}'. "
            f"Use 'Dataframe' or 'Dictionary'."
        )

    # Default stats
    if stats is None:
        stats = list(_DEFAULT_STATS)

    # Dispatch based on data type
    if isinstance(data, (pandas.core.groupby.SeriesGroupBy, pandas.core.groupby.DataFrameGroupBy)):
        result_df = _compute_for_groupby(data, stats, ci_level, decimals)

        if return_type.upper() == "DICTIONARY":
            return result_df.to_dict(orient="list")

        return result_df

    elif isinstance(data, pandas.DataFrame):
        result_df = _compute_for_dataframe(data, stats, ci_level, decimals)

        if return_type.upper() == "DICTIONARY":
            return result_df.to_dict(orient="list")

        return result_df

    else:
        # Series, ndarray, list, or other array-like
        resolved_name = name
        if resolved_name is None:
            resolved_name = getattr(data, "name", None)

        result = _compute_for_array(data, stats, ci_level, decimals, name=resolved_name)

        if return_type.upper() == "DICTIONARY":
            return result.to_dict()

        return result.to_dataframe()


