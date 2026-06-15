# -*- coding: utf-8 -*-
"""
Shared computation routing for descriptive statistics.

Provides the ``_route_computation()`` helper that encapsulates the
resolve → route → compute → format pattern used by all descriptive
stat functions that support the 5 calling conventions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
import pandas

from ..core.spec import resolve, ComputeSpec
from ..core.data_utils import validate_array
from ..core.matrix_engine import build_indicator_matrix, grouped_statistic


def _route_computation(
    arg1: Any,
    arg2: Any,
    *,
    dv: Optional[Union[str, List[str]]],
    iv: Optional[Union[str, List[str]]],
    by: Optional[Union[str, List[str]]],
    data: Optional[pandas.DataFrame],
    scalar_func: Callable[[numpy.ndarray], float],
    matrix_stat: Optional[str] = None,
    fallback_grouped_func: Optional[Callable] = None,
    decimals: int = 4,
    weights: Optional[str] = None,
    **kwargs,
) -> Union[float, pandas.DataFrame]:
    """Route a descriptive stat computation through resolve → engine.

    This is the shared backbone for all descriptive functions that support
    the 5 calling conventions. It:
    1. Calls ``resolve()`` to normalize any input form into a ``ComputeSpec``
    2. Routes based on whether groups are present:
       - No groups, single DV → calls ``scalar_func(array)`` → returns float
       - No groups, multiple DVs → calls ``scalar_func`` per DV → returns DataFrame
       - With groups → uses matrix engine (``grouped_statistic``) or fallback

    Parameters
    ----------
    arg1, arg2 : Any
        Positional arguments from the calling function.
    dv, iv, by, data : keyword arguments forwarded to ``resolve()``.
    scalar_func : callable
        Pure computation function: takes a 1-D float numpy array, returns a float.
        Used for the ungrouped (fast) path.
    matrix_stat : str or None
        If the stat is directly supported by ``grouped_statistic()``
        (one of 'mean', 'sum', 'count', 'variance', 'sd', 'se'), pass
        the key here for the optimized matrix path.
    fallback_grouped_func : callable or None
        For stats NOT in the matrix engine (e.g., median, mode), provide
        a fallback that takes (data[dv_col], group_series) and returns
        a DataFrame. Used when ``matrix_stat`` is None.
    decimals : int
        Decimal places for rounding. Default is 4.
    weights : str or None
        Column name for observation weights (future use).
    **kwargs : dict
        Additional keyword arguments passed through to scalar_func.

    Returns
    -------
    float or pandas.DataFrame
        - float when single DV, no groups
        - DataFrame when multiple DVs or grouped
    """
    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, data=data, weights=weights)

    # === No groups ===
    if spec.groups is None:
        if len(spec.dv) == 1:
            # Single DV → scalar result
            arr = validate_array(spec.data[spec.dv[0]].to_numpy(), dtype=float)
            result = scalar_func(arr, **kwargs)
            return round(float(result), decimals)
        else:
            # Multiple DVs → DataFrame
            rows = []
            for col in spec.dv:
                arr = validate_array(spec.data[col].to_numpy(), dtype=float)
                value = scalar_func(arr, **kwargs)
                rows.append({"Variable": col, "Value": round(float(value), decimals)})
            return pandas.DataFrame(rows)

    # === With groups ===
    if matrix_stat is not None:
        # Optimized matrix path
        if len(spec.dv) == 1:
            result_df = grouped_statistic(
                spec.data, dv=spec.dv[0], groups=spec.groups,
                stat_func=matrix_stat, interactions=spec.interactions,
            )
            # Round the stat column
            stat_col = result_df.columns[-1]
            result_df[stat_col] = result_df[stat_col].round(decimals)
            return result_df
        else:
            # Multiple DVs: compute for each, stack results
            frames = []
            for col in spec.dv:
                df_result = grouped_statistic(
                    spec.data, dv=col, groups=spec.groups,
                    stat_func=matrix_stat, interactions=spec.interactions,
                )
                df_result.insert(0, "Variable", col)
                stat_col = df_result.columns[-1]
                df_result[stat_col] = df_result[stat_col].round(decimals)
                frames.append(df_result)
            return pandas.concat(frames, ignore_index=True)

    elif fallback_grouped_func is not None:
        # Fallback for stats not in matrix engine (e.g., median, mode)
        if len(spec.dv) == 1:
            return fallback_grouped_func(
                spec.data, spec.dv[0], spec.groups, spec.interactions, decimals, **kwargs
            )
        else:
            frames = []
            for col in spec.dv:
                df_result = fallback_grouped_func(
                    spec.data, col, spec.groups, spec.interactions, decimals, **kwargs
                )
                df_result.insert(0, "Variable", col)
                frames.append(df_result)
            return pandas.concat(frames, ignore_index=True)

    else:
        raise NotImplementedError(
            "Grouped computation not available for this statistic. "
            "Provide either matrix_stat or fallback_grouped_func."
        )


def _grouped_via_iteration(
    data: pandas.DataFrame,
    dv_col: str,
    groups: List[str],
    interactions: Optional[List[Tuple[str, ...]]],
    decimals: int,
    scalar_func: Callable[[numpy.ndarray], float],
    stat_label: str = "Value",
) -> pandas.DataFrame:
    """Compute a grouped statistic by iterating over groups.

    Used as a fallback for stats that don't have a matrix-arithmetic shortcut
    (e.g., median, mode, kurtosis, skewness).

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_col : str
        Dependent variable column name.
    groups : list of str
        Grouping columns.
    interactions : list of tuple or None
        Interaction specification.
    decimals : int
        Rounding decimal places.
    scalar_func : callable
        Function that takes a 1-D float array and returns a scalar.
    stat_label : str
        Column label for the statistic in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per group with columns: Group, stat_label.
    """
    # Build cell labels using the same logic as the matrix engine
    from ..core.matrix_engine import _build_cell_series

    group_vars = interactions[0] if interactions else groups
    cell_series = _build_cell_series(data, group_vars)
    unique_cells = sorted(cell_series.unique())

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = data.loc[mask, dv_col].to_numpy(dtype=float, na_value=numpy.nan)
        # Remove NaN for computation
        clean = arr[~numpy.isnan(arr)]
        if len(clean) > 0:
            value = round(float(scalar_func(clean)), decimals)
        else:
            value = numpy.nan
        results.append({"Group": cell, stat_label: value})

    return pandas.DataFrame(results)

