# -*- coding: utf-8 -*-
"""
Unified summarize() dispatcher.

Routes computation based on input type (Series, DataFrame, GroupBy) and
delegates to the individual statistic modules.
"""

from typing import Any, Dict, List, Optional, Union

import numpy
import pandas

from ..containers.univariate import SummaryResult
from ..core.data_utils import validate_array
from .observation import n_obs, n_missing, percent_missing
from .central_tendency import mean, median, mode, quartiles, iqr
from .dispersion import variance, standard_deviation, standard_error, value_range, coefficient_of_variation
from .intervals import confidence_interval
from .shape import skewness, kurtosis


def _quartiles_as_dict(arr) -> dict:
    """Compute quartiles and return as dict for SummaryResult compatibility."""
    result_df = quartiles(arr)
    return {"Q1": result_df["Q1"].iloc[0], "Q2": result_df["Q2"].iloc[0], "Q3": result_df["Q3"].iloc[0]}


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
    "Quartiles": lambda arr, **kw: _quartiles_as_dict(arr),
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

    Uses actual group variable names in output columns instead of generic "Group".

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
        Results with actual group variable name(s) as columns.
    """
    kwargs = {"ci_level": ci_level, "decimals": decimals}
    rows = []

    # Extract group variable name(s) for output column headers
    group_keys = data.keys
    if isinstance(group_keys, list):
        group_names = group_keys
    else:
        group_names = [group_keys]

    for group_key, group_data in data:
        # Normalize group_key to tuple for uniform handling
        if not isinstance(group_key, tuple):
            group_key_tuple = (group_key,)
        else:
            group_key_tuple = group_key

        if isinstance(group_data, pandas.Series):
            arr = validate_array(group_data)
            row = {}
            for gname, gval in zip(group_names, group_key_tuple):
                row[gname] = gval

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
                row = {}
                for gname, gval in zip(group_names, group_key_tuple):
                    row[gname] = gval
                row["Variable"] = col_name
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


def _compute_marginal_from_spec(
    data: pandas.DataFrame,
    dv_cols: List[str],
    iv_vars: List[str],
    stats: List[str],
    ci_level: float,
    decimals: int,
) -> pandas.DataFrame:
    """Compute summary stats for DV(s) grouped by each iv independently, stack results.

    Equivalent to calling _compute_grouped_from_spec for each iv separately
    and concatenating the results. Output has a 'Factor' column indicating
    which iv variable the row belongs to.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_cols : list of str
        Dependent variable column names.
    iv_vars : list of str
        Independent variable column names (each computed separately).
    stats : list of str
        Statistics to compute.
    ci_level : float
        Confidence level.
    decimals : int
        Rounding decimal places.

    Returns
    -------
    pd.DataFrame
        Stacked results with columns: Factor, Level, [stats...].
        If multiple DVs: adds a Variable column.
    """
    kwargs = {"ci_level": ci_level, "decimals": decimals}
    frames = []

    for dv_col in dv_cols:
        for iv_var in iv_vars:
            grouped = data.groupby(iv_var)
            rows = []
            for group_key, group_data in grouped:
                arr = validate_array(group_data[dv_col], dtype=float)
                row = {"Factor": iv_var, "Level": group_key}

                if len(dv_cols) > 1:
                    row["Variable"] = dv_col

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
            frames.append(pandas.DataFrame(rows))

    return pandas.concat(frames, ignore_index=True)


'''
#class Estimate(DesignInfo, EstimateResults):
class Estimate(DesignInfo, EstimateResults):

    def __init__(self, formula_like: Optional[str]=None,
                 data: Union[pandas.Series, pandas.DataFrame, numpy.ndarray, list]=None,
                 ci_level=0.95):
        ...
'''



def estable(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    name: Optional[str] = None,
    stats: Optional[List[str]] = None,
    ci_level: float = 0.95,
    decimals: int = 4,
    return_type: str = "Dataframe",
) -> Union[pandas.DataFrame, dict]:
    """Compute univariate summary statistics for numeric data.

    A unified dispatcher that handles Series, DataFrames, GroupBy objects,
    formulas, and keyword-based calling conventions. Delegates computation
    to focused modules for central tendency, dispersion, intervals, and shape.

    Supports 5 calling conventions:
        1. estable(series_or_array)
        2. estable("y ~ C(x)", df)
        3. estable(["y", "k"], df)
        4. estable(dv="y", by="x", data=df)
           estable(dv="y", iv=["x","k"], data=df)
           estable(dv="y", by="x", over="k", data=df)
        5. estable(df.groupby("x")["y"])

    Parameters
    ----------
    arg1 : array_like, str, list of str, DataFrame, GroupBy, or None
        Data, formula, column names, or GroupBy object.
    arg2 : pd.DataFrame or None
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
        Mutually exclusive with by/over.
    by : str or list of str, optional
        Row grouping variable(s) for cell means.
    over : str or list of str, optional
        Column grouping variable(s) for pivot layout. Requires by.
    data : pd.DataFrame, optional
        Data source when using keyword arguments.
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

    Single Series:
    >>> s = pd.Series([1, 2, 3, 4, 5], name="scores")
    >>> estable(s)
        Name  N  Mean  Median  Variance      SD      SE  95% Conf. Interval
    0  scores  5   3.0     3.0       2.5  1.5811  0.7071        (1.038, 4.962)

    Formula:
    >>> df = pd.DataFrame({"y": [1,2,3,4], "g": ["a","a","b","b"]})
    >>> estable("y ~ C(g)", df, stats=["N", "Mean"])
       g  N  Mean
    0  a  2   1.5
    1  b  2   3.5

    Keywords:
    >>> estable(dv="y", by="g", data=df, stats=["N", "Mean"])
       g  N  Mean
    0  a  2   1.5
    1  b  2   3.5

    GroupBy:
    >>> estable(df.groupby("g")["y"], stats=["N", "Mean"])
       g  N  Mean
    0  a  2   1.5
    1  b  2   3.5
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

    # --- GroupBy objects: direct dispatch ---
    if isinstance(arg1, (pandas.core.groupby.SeriesGroupBy, pandas.core.groupby.DataFrameGroupBy)):
        result_df = _compute_for_groupby(arg1, stats, ci_level, decimals)

        if return_type.upper() == "DICTIONARY":
            return result_df.to_dict(orient="list")
        return result_df

    # --- DataFrame without formula/keywords: compute for each column ---
    if isinstance(arg1, pandas.DataFrame) and arg2 is None and dv is None:
        result_df = _compute_for_dataframe(arg1, stats, ci_level, decimals)

        if return_type.upper() == "DICTIONARY":
            return result_df.to_dict(orient="list")
        return result_df

    # --- Formula or keyword-based calling conventions ---
    # Check if we should resolve via the spec system
    use_spec = (
        isinstance(arg1, str) or  # formula
        (isinstance(arg1, list) and all(isinstance(x, str) for x in arg1)) or  # column list
        dv is not None  # keyword convention
    )

    if use_spec:
        from ..core.spec import resolve
        spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data)

        # Grouped computation
        if spec.by is not None or spec.iv is not None or spec.over is not None:

            if spec.iv is not None:
                # Marginal: compute for each iv independently, stack results
                result_df = _compute_marginal_from_spec(
                    spec.data, spec.dv, spec.iv, stats, ci_level, decimals
                )
            elif spec.over is not None:
                # Pivot: compute all cells, return flat with by+over columns
                group_vars = spec.by + spec.over
                result_df = _compute_grouped_from_spec(
                    spec.data, spec.dv, group_vars, stats, ci_level, decimals
                )
            else:
                # Cell (by only): grouped with actual variable names
                result_df = _compute_grouped_from_spec(
                    spec.data, spec.dv, spec.by, stats, ci_level, decimals
                )

            if return_type.upper() == "DICTIONARY":
                return result_df.to_dict(orient="list")
            return result_df
        else:
            # No grouping — compute for DV(s)
            if len(spec.dv) == 1:
                resolved_name = name if name is not None else spec.dv[0]
                result = _compute_for_array(
                    spec.data[spec.dv[0]], stats, ci_level, decimals, name=resolved_name
                )
                if return_type.upper() == "DICTIONARY":
                    return result.to_dict()
                return result.to_dataframe()
            else:
                result_df = _compute_for_dataframe(
                    spec.data[spec.dv], stats, ci_level, decimals
                )
                if return_type.upper() == "DICTIONARY":
                    return result_df.to_dict(orient="list")
                return result_df

    # --- Series, ndarray, list, or other array-like ---
    resolved_name = name
    if resolved_name is None:
        resolved_name = getattr(arg1, "name", None)

    result = _compute_for_array(arg1, stats, ci_level, decimals, name=resolved_name)

    if return_type.upper() == "DICTIONARY":
        return result.to_dict()

    return result.to_dataframe()


def _compute_grouped_from_spec(
    data: pandas.DataFrame,
    dv_cols: List[str],
    group_vars: List[str],
    stats: List[str],
    ci_level: float,
    decimals: int,
) -> pandas.DataFrame:
    """Compute summary stats for DV(s) grouped by specified variables.

    Uses actual group variable names in output.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_cols : list of str
        Dependent variable column names.
    group_vars : list of str
        Grouping variable column names.
    stats : list of str
        Statistics to compute.
    ci_level : float
        Confidence level.
    decimals : int
        Rounding decimal places.

    Returns
    -------
    pd.DataFrame
        Rows for each group × DV combination, with group variable columns.
    """
    kwargs = {"ci_level": ci_level, "decimals": decimals}
    rows = []

    grouped = data.groupby(group_vars)

    for group_key, group_data in grouped:
        # Normalize group_key to tuple
        if not isinstance(group_key, tuple):
            group_key_tuple = (group_key,)
        else:
            group_key_tuple = group_key

        for dv_col in dv_cols:
            if dv_col not in group_data.columns:
                continue

            arr = validate_array(group_data[dv_col], dtype=float)
            row = {}

            # Add group variable values
            for gname, gval in zip(group_vars, group_key_tuple):
                row[gname] = gval

            # Add DV name if multiple DVs
            if len(dv_cols) > 1:
                row["Variable"] = dv_col

            # Compute stats
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
