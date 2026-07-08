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
from ._compute import _route_computation


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
    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=lambda arr: numpy.nanmedian(arr),
        matrix_stat=None,
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


def quartiles(
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
) -> pandas.DataFrame:
    """Compute the first (Q1), second (Q2/median), and third (Q3) quartiles.

    Uses linear interpolation (numpy default). Returns a wide-format DataFrame
    with Q1, Q2, Q3 as columns. Supports all 5 calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute quartiles directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
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
    pandas.DataFrame
        Wide-format DataFrame with columns Q1, Q2, Q3.
        Ungrouped: single-row DataFrame.
        Grouped: one row per group with group columns + Q1, Q2, Q3.

    Examples
    --------
    >>> import pandas as pd
    >>> quartiles([1, 2, 3, 4, 5])
       Q1   Q2   Q3
    0  2.0  3.0  4.0

    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','a','b','b','b']})
    >>> quartiles("y ~ C(g)", df)
       g   Q1   Q2   Q3
    0  a  1.5  2.0  2.5
    1  b  4.5  5.0  5.5
    """
    from ..core.spec import resolve
    from ..core.matrix_engine import _build_cell_series

    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data)

    def _quartiles_for_array(arr: numpy.ndarray) -> Dict[str, float]:
        clean = arr[~numpy.isnan(arr)]
        if len(clean) == 0:
            return {"Q1": numpy.nan, "Q2": numpy.nan, "Q3": numpy.nan}
        q1, q2, q3 = numpy.nanpercentile(clean, [25, 50, 75])
        return {
            "Q1": round(float(q1), decimals),
            "Q2": round(float(q2), decimals),
            "Q3": round(float(q3), decimals),
        }

    # === No groups ===
    if spec.iv is None and spec.by is None and spec.over is None and spec.sub_specs is None:
        if len(spec.dv) == 1:
            arr = validate_array(spec.data[spec.dv[0]].to_numpy(), dtype=float)
            row = _quartiles_for_array(arr)
            return pandas.DataFrame([row])
        else:
            rows = []
            for col in spec.dv:
                arr = validate_array(spec.data[col].to_numpy(), dtype=float)
                row = _quartiles_for_array(arr)
                row["Variable"] = col
                rows.append(row)
            result = pandas.DataFrame(rows)
            # Reorder columns: Variable first
            cols = ["Variable"] + [c for c in result.columns if c != "Variable"]
            return result[cols]

    # === Grouped computation (by, iv, over, or sub_specs) ===
    # Determine grouping variables
    if spec.sub_specs is not None:
        # Mixed formula: compute for each sub-spec
        frames = []
        for ts in spec.sub_specs:
            sub_frames = []
            for dv_col in spec.dv:
                sub_result = _quartiles_grouped(
                    spec.data, dv_col, ts.variables, decimals, _build_cell_series
                )
                if len(ts.variables) == 1:
                    level_values = sub_result[ts.variables[0]].values
                else:
                    level_values = sub_result[ts.variables[0]].astype(str)
                    for var in ts.variables[1:]:
                        level_values = level_values + ":" + sub_result[var].astype(str)
                    level_values = level_values.values
                normalized = pandas.DataFrame({
                    "Level": level_values,
                    "Q1": sub_result["Q1"].values,
                    "Q2": sub_result["Q2"].values,
                    "Q3": sub_result["Q3"].values,
                })
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                sub_frames.append(normalized)
            sub_result_df = pandas.concat(sub_frames, ignore_index=True)
            sub_result_df.insert(0, "Term", ts.term_name)
            frames.append(sub_result_df)
        return pandas.concat(frames, ignore_index=True)

    if spec.iv is not None:
        # Marginal: compute for each iv independently
        frames = []
        for dv_col in spec.dv:
            for iv_var in spec.iv:
                result_df = _quartiles_grouped(
                    spec.data, dv_col, [iv_var], decimals, _build_cell_series
                )
                normalized = pandas.DataFrame({
                    "Factor": iv_var,
                    "Level": result_df[iv_var].values,
                    "Q1": result_df["Q1"].values,
                    "Q2": result_df["Q2"].values,
                    "Q3": result_df["Q3"].values,
                })
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                frames.append(normalized)
        return pandas.concat(frames, ignore_index=True)

    if spec.over is not None:
        # Pivot not directly applicable for wide quartile output;
        # compute flat with all groups
        all_groups = spec.by + spec.over
        frames = []
        for dv_col in spec.dv:
            result_df = _quartiles_grouped(
                spec.data, dv_col, all_groups, decimals, _build_cell_series
            )
            if len(spec.dv) > 1:
                result_df.insert(0, "Variable", dv_col)
            frames.append(result_df)
        if len(frames) == 1:
            return frames[0]
        return pandas.concat(frames, ignore_index=True)

    # Cell (by only)
    frames = []
    for dv_col in spec.dv:
        result_df = _quartiles_grouped(
            spec.data, dv_col, spec.by, decimals, _build_cell_series
        )
        if len(spec.dv) > 1:
            result_df.insert(0, "Variable", dv_col)
        frames.append(result_df)
    if len(frames) == 1:
        return frames[0]
    return pandas.concat(frames, ignore_index=True)


def _quartiles_grouped(
    data: pandas.DataFrame,
    dv_col: str,
    groups: List[str],
    decimals: int,
    _build_cell_series_func,
) -> pandas.DataFrame:
    """Compute Q1, Q2, Q3 for each group, returning wide-format DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_col : str
        Dependent variable column.
    groups : list of str
        Grouping columns.
    decimals : int
        Rounding decimal places.
    _build_cell_series_func : callable
        The _build_cell_series function from matrix_engine.

    Returns
    -------
    pd.DataFrame
        Columns: [group_vars..., Q1, Q2, Q3]
    """
    cell_series = _build_cell_series_func(data, groups)
    unique_cells = sorted(cell_series.unique())
    values = data[dv_col].to_numpy(dtype=float)

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = values[mask]
        clean = arr[~numpy.isnan(arr)]

        if len(clean) > 0:
            q1, q2, q3 = numpy.nanpercentile(clean, [25, 50, 75])
            q1, q2, q3 = round(float(q1), decimals), round(float(q2), decimals), round(float(q3), decimals)
        else:
            q1, q2, q3 = numpy.nan, numpy.nan, numpy.nan

        row = {}
        if len(groups) == 1:
            row[groups[0]] = cell
        else:
            parts = cell.split(":")
            for i, group_name in enumerate(groups):
                row[group_name] = parts[i]
        row["Q1"] = q1
        row["Q2"] = q2
        row["Q3"] = q3
        results.append(row)

    return pandas.DataFrame(results)


def percentile(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    q: Union[float, List[float]] = 50.0,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    over: Optional[Union[str, List[str]]] = None,
    data: Optional[pandas.DataFrame] = None,
    decimals: int = 4,
) -> pandas.DataFrame:
    """Compute one or more percentiles, ignoring NaN values.

    Returns a wide-format DataFrame with one column per requested percentile.
    Supports all 5 calling conventions.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute percentile directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    q : float or list of float
        Percentile(s) to compute. Must be between 0 and 100 inclusive.
        Default is 50.0 (median).
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
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
    pandas.DataFrame
        Wide-format DataFrame with one column per percentile (P10, P25, etc.).
        Ungrouped: single-row DataFrame.
        Grouped: one row per group with group columns + percentile columns.

    Raises
    ------
    ValueError
        If any percentile value is outside [0, 100].

    Examples
    --------
    >>> percentile([1, 2, 3, 4, 5], q=[25, 50, 75])
       P25  P50  P75
    0  2.0  3.0  4.0

    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','a','b','b','b']})
    >>> percentile("y ~ C(g)", df, q=[25, 50, 75])
       g  P25  P50  P75
    0  a  1.5  2.0  2.5
    1  b  4.5  5.0  5.5
    """
    from ..core.spec import resolve
    from ..core.matrix_engine import _build_cell_series

    # Normalize q to a list
    if isinstance(q, (int, float)):
        q_list = [q]
    else:
        q_list = list(q)

    # Validate percentile values
    for val in q_list:
        if val < 0 or val > 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {val}.")

    # Build column names
    p_col_names = [f"P{int(v) if v == int(v) else v}" for v in q_list]

    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data)

    def _percentiles_for_array(arr: numpy.ndarray) -> Dict[str, float]:
        clean = arr[~numpy.isnan(arr)]
        if len(clean) == 0:
            return {name: numpy.nan for name in p_col_names}
        results = numpy.nanpercentile(clean, q_list)
        if not hasattr(results, '__len__'):
            results = [results]
        return {name: round(float(r), decimals) for name, r in zip(p_col_names, results)}

    # === No groups ===
    if spec.iv is None and spec.by is None and spec.over is None and spec.sub_specs is None:
        if len(spec.dv) == 1:
            arr = validate_array(spec.data[spec.dv[0]].to_numpy(), dtype=float)
            row = _percentiles_for_array(arr)
            return pandas.DataFrame([row])
        else:
            rows = []
            for col in spec.dv:
                arr = validate_array(spec.data[col].to_numpy(), dtype=float)
                row = _percentiles_for_array(arr)
                row["Variable"] = col
                rows.append(row)
            result = pandas.DataFrame(rows)
            cols = ["Variable"] + [c for c in result.columns if c != "Variable"]
            return result[cols]

    # === Grouped ===
    if spec.iv is not None:
        frames = []
        for dv_col in spec.dv:
            for iv_var in spec.iv:
                result_df = _percentiles_grouped(
                    spec.data, dv_col, [iv_var], q_list, p_col_names, decimals, _build_cell_series
                )
                normalized = pandas.DataFrame({"Factor": iv_var, "Level": result_df[iv_var].values})
                for pc in p_col_names:
                    normalized[pc] = result_df[pc].values
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                frames.append(normalized)
        return pandas.concat(frames, ignore_index=True)

    if spec.over is not None:
        all_groups = spec.by + spec.over
        frames = []
        for dv_col in spec.dv:
            result_df = _percentiles_grouped(
                spec.data, dv_col, all_groups, q_list, p_col_names, decimals, _build_cell_series
            )
            if len(spec.dv) > 1:
                result_df.insert(0, "Variable", dv_col)
            frames.append(result_df)
        if len(frames) == 1:
            return frames[0]
        return pandas.concat(frames, ignore_index=True)

    # Cell (by only)
    frames = []
    for dv_col in spec.dv:
        result_df = _percentiles_grouped(
            spec.data, dv_col, spec.by, q_list, p_col_names, decimals, _build_cell_series
        )
        if len(spec.dv) > 1:
            result_df.insert(0, "Variable", dv_col)
        frames.append(result_df)
    if len(frames) == 1:
        return frames[0]
    return pandas.concat(frames, ignore_index=True)


def _percentiles_grouped(
    data: pandas.DataFrame,
    dv_col: str,
    groups: List[str],
    q_list: List[float],
    p_col_names: List[str],
    decimals: int,
    _build_cell_series_func,
) -> pandas.DataFrame:
    """Compute percentiles for each group, returning wide-format DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_col : str
        Dependent variable column.
    groups : list of str
        Grouping columns.
    q_list : list of float
        Percentile values to compute.
    p_col_names : list of str
        Column names for the percentiles (e.g., ['P25', 'P50', 'P75']).
    decimals : int
        Rounding decimal places.
    _build_cell_series_func : callable
        The _build_cell_series function from matrix_engine.

    Returns
    -------
    pd.DataFrame
        Columns: [group_vars..., P10, P25, ...]
    """
    cell_series = _build_cell_series_func(data, groups)
    unique_cells = sorted(cell_series.unique())
    values = data[dv_col].to_numpy(dtype=float)

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = values[mask]
        clean = arr[~numpy.isnan(arr)]

        row = {}
        if len(groups) == 1:
            row[groups[0]] = cell
        else:
            parts = cell.split(":")
            for i, group_name in enumerate(groups):
                row[group_name] = parts[i]

        if len(clean) > 0:
            pct_values = numpy.nanpercentile(clean, q_list)
            if not hasattr(pct_values, '__len__'):
                pct_values = [pct_values]
            for name, val in zip(p_col_names, pct_values):
                row[name] = round(float(val), decimals)
        else:
            for name in p_col_names:
                row[name] = numpy.nan

        results.append(row)

    return pandas.DataFrame(results)


def iqr(
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
    """Compute the interquartile range (Q3 - Q1), ignoring NaN values.

    Supports all 5 calling conventions. Returns a scalar when ungrouped
    with a single DV, or a DataFrame when grouped.

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute IQR directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
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
        When multiple variables, marginal, cell, or pivot.

    Examples
    --------
    >>> iqr([1, 2, 3, 4, 5])
    2.0

    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','a','b','b','b']})
    >>> iqr("y ~ C(g)", df)
          IQR
    g
    a     1.0
    b     1.0
    """
    def _iqr_scalar(arr):
        q1, q3 = numpy.nanpercentile(arr, [25, 75])
        return float(q3 - q1)

    return _route_computation(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over, data=data,
        scalar_func=_iqr_scalar,
        matrix_stat=None,
        stat_label="IQR",
        decimals=decimals,
    )

