# -*- coding: utf-8 -*-
"""
Comprehensive descriptive statistics: describe().

Provides a single function that computes a wide set of summary statistics
for numeric data, supporting all 5 calling conventions. Optimized to build
the indicator matrix once for linear stats and _build_cell_series once for
non-linear stats, avoiding redundant group-membership computation.
"""

from typing import Any, Dict, List, Optional, Union

import numpy
import pandas

from ..core.syntax_engine import resolve, SyntaxSpec
from ..core.data_utils import validate_array
from ..core.matrix_engine import (
    build_indicator_matrix,
    grouped_statistic,
    _build_cell_series,
)


def describe(
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
    """Compute comprehensive descriptive statistics for numeric data.

    Returns a wide-format DataFrame with columns:
    N, N Missing, Mean, Median, SD, Min, Q1, Q3, Max, IQR.

    Supports all 5 calling conventions. Builds the indicator matrix once
    for linear stats (N, Mean, SD) and _build_cell_series once for
    non-linear stats (Median, Min, Max, Q1, Q3, IQR).

    Parameters
    ----------
    arg1 : array_like, str, or list of str
        - Series/ndarray/list: compute descriptive stats directly
        - str: Patsy-style formula (e.g., "y ~ C(x)")
        - list of str: column names in a DataFrame
    arg2 : pd.DataFrame, optional
        DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Dependent variable column name(s).
    iv : str or list of str, optional
        Independent variable(s) for marginal computation.
        Computes stats for each iv independently, stacks results.
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
    pandas.DataFrame
        Wide-format DataFrame with descriptive statistics.
        - Ungrouped: one row (or one row per DV if multiple DVs)
        - Grouped: one row per group (with "Variable" column if multiple DVs)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'systolic': [120, 130, 125, 140, 135, 128],
    ...     'disease': ['a', 'a', 'b', 'b', 'c', 'c']
    ... })

    # Bare array
    >>> describe(df['systolic'])
       N  N Missing    Mean  Median      SD    Min     Q1     Q3    Max   IQR
    0  6          0  129.67   128.0  6.8896  120.0  125.5  134.0  140.0  8.75

    # Formula
    >>> describe("systolic ~ C(disease)", df)
      disease  N  N Missing   Mean  Median     SD    Min     Q1     Q3    Max   IQR
    0       a  2          0  125.0   125.0  7.071  120.0  122.5  127.5  130.0   5.0
    1       b  2          0  132.5   132.5  10.61  125.0  128.8  136.3  140.0   7.5
    2       c  2          0  131.5   131.5  4.950  128.0  129.8  133.3  135.0   3.5

    # Keywords
    >>> describe(dv="systolic", by="disease", data=df)
    # (same as formula output)

    # Multiple DVs
    >>> describe(dv=["systolic", "diastolic"], by="disease", data=df)
    # Adds "Variable" column
    """
    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data)

    # === No groups ===
    if spec.iv is None and spec.by is None and spec.over is None and spec.sub_specs is None:
        return _describe_ungrouped(spec, decimals)

    # === Mixed formula (sub_specs) ===
    if spec.sub_specs is not None:
        return _describe_mixed(spec, decimals)

    # === Marginal (iv) ===
    if spec.iv is not None:
        return _describe_marginal(spec, decimals)

    # === Pivot (by + over) — treat as flat grouped with all groups ===
    if spec.over is not None:
        groups = spec.by + spec.over
        return _describe_grouped(spec, groups, decimals)

    # === Cell (by only) ===
    return _describe_grouped(spec, spec.by, decimals)


def _describe_ungrouped(spec: SyntaxSpec, decimals: int) -> pandas.DataFrame:
    """Compute descriptive stats for ungrouped data."""
    rows = []
    for dv_col in spec.dv:
        arr = validate_array(spec.data[dv_col].to_numpy(), dtype=float)
        row = _compute_stats_for_array(arr, decimals)
        if len(spec.dv) > 1:
            row["Variable"] = dv_col
        rows.append(row)

    result = pandas.DataFrame(rows)
    if len(spec.dv) > 1:
        cols = ["Variable"] + [c for c in result.columns if c != "Variable"]
        result = result[cols]
    return result


def _describe_grouped(
    spec: SyntaxSpec,
    groups: List[str],
    decimals: int,
) -> pandas.DataFrame:
    """Compute descriptive stats for grouped data.

    Optimized: builds cell_series once and iterates for all non-linear stats.
    Uses grouped_statistic for linear stats (mean, sd, count).
    """
    frames = []
    for dv_col in spec.dv:
        result_df = _compute_grouped_describe(spec.data, dv_col, groups, decimals)
        if len(spec.dv) > 1:
            result_df.insert(0, "Variable", dv_col)
        frames.append(result_df)

    if len(frames) == 1:
        return frames[0]
    return pandas.concat(frames, ignore_index=True)


def _describe_marginal(spec: SyntaxSpec, decimals: int) -> pandas.DataFrame:
    """Compute descriptive stats for each iv variable independently, stack."""
    frames = []
    for dv_col in spec.dv:
        for iv_var in spec.iv:
            result_df = _compute_grouped_describe(spec.data, dv_col, [iv_var], decimals)
            # Normalize to Factor/Level structure
            normalized = pandas.DataFrame({
                "Factor": iv_var,
                "Level": result_df[iv_var].values,
            })
            # Add all stat columns
            stat_cols = [c for c in result_df.columns if c != iv_var]
            for col in stat_cols:
                normalized[col] = result_df[col].values

            if len(spec.dv) > 1:
                normalized.insert(0, "Variable", dv_col)
            frames.append(normalized)

    return pandas.concat(frames, ignore_index=True)


def _describe_mixed(spec: SyntaxSpec, decimals: int) -> pandas.DataFrame:
    """Compute descriptive stats for mixed formulas (sub_specs)."""
    frames = []
    for ts in spec.sub_specs:
        sub_frames = []
        for dv_col in spec.dv:
            result_df = _compute_grouped_describe(spec.data, dv_col, ts.variables, decimals)

            # Build Level column
            if len(ts.variables) == 1:
                level_values = result_df[ts.variables[0]].values
            else:
                level_values = result_df[ts.variables[0]].astype(str)
                for var in ts.variables[1:]:
                    level_values = level_values + ":" + result_df[var].astype(str)
                level_values = level_values.values

            # Normalize: Term, Level, stats...
            stat_cols = [c for c in result_df.columns if c not in ts.variables]
            normalized = pandas.DataFrame({"Level": level_values})
            for col in stat_cols:
                normalized[col] = result_df[col].values

            if len(spec.dv) > 1:
                normalized.insert(0, "Variable", dv_col)
            sub_frames.append(normalized)

        sub_result = pandas.concat(sub_frames, ignore_index=True)
        sub_result.insert(0, "Term", ts.term_name)
        frames.append(sub_result)

    return pandas.concat(frames, ignore_index=True)


def _compute_stats_for_array(arr: numpy.ndarray, decimals: int) -> Dict[str, Any]:
    """Compute all descriptive stats for a single 1-D array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (may contain NaN).
    decimals : int
        Rounding decimal places.

    Returns
    -------
    dict
        Statistic name → value.
    """
    n_total = len(arr)
    nan_mask = numpy.isnan(arr)
    n_missing = int(nan_mask.sum())
    clean = arr[~nan_mask]
    n = len(clean)

    if n == 0:
        return {
            "N": 0,
            "N Missing": n_missing,
            "Mean": numpy.nan,
            "Median": numpy.nan,
            "SD": numpy.nan,
            "Min": numpy.nan,
            "Q1": numpy.nan,
            "Q3": numpy.nan,
            "Max": numpy.nan,
            "IQR": numpy.nan,
        }

    mean_val = round(float(numpy.mean(clean)), decimals)
    median_val = round(float(numpy.median(clean)), decimals)
    sd_val = round(float(numpy.std(clean, ddof=1)), decimals) if n > 1 else numpy.nan
    min_val = round(float(numpy.min(clean)), decimals)
    max_val = round(float(numpy.max(clean)), decimals)
    q1_val = round(float(numpy.percentile(clean, 25)), decimals)
    q3_val = round(float(numpy.percentile(clean, 75)), decimals)
    iqr_val = round(float(q3_val - q1_val), decimals)

    return {
        "N": n,
        "N Missing": n_missing,
        "Mean": mean_val,
        "Median": median_val,
        "SD": sd_val,
        "Min": min_val,
        "Q1": q1_val,
        "Q3": q3_val,
        "Max": max_val,
        "IQR": iqr_val,
    }


def _compute_grouped_describe(
    data: pandas.DataFrame,
    dv_col: str,
    groups: List[str],
    decimals: int,
) -> pandas.DataFrame:
    """Compute all descriptive stats for each group.

    Optimized: builds _build_cell_series once and iterates for all stats
    per group in a single pass.

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

    Returns
    -------
    pd.DataFrame
        Columns: [group_vars..., N, N Missing, Mean, Median, SD, Min, Q1, Q3, Max, IQR]
    """
    cell_series = _build_cell_series(data, groups)
    unique_cells = sorted(cell_series.unique())
    values = data[dv_col].to_numpy(dtype=float)

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = values[mask]

        # Build group columns
        row: Dict[str, Any] = {}
        if len(groups) == 1:
            row[groups[0]] = cell
        else:
            parts = cell.split(":")
            for i, group_name in enumerate(groups):
                row[group_name] = parts[i]

        # Compute all stats for this group
        stats = _compute_stats_for_array(arr, decimals)
        row.update(stats)
        results.append(row)

    return pandas.DataFrame(results)

