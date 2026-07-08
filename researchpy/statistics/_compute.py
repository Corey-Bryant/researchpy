# -*- coding: utf-8 -*-
"""
Shared computation routing for descriptive statistics.

Provides the ``_route_computation()`` helper that encapsulates the
resolve → route → compute → format pattern used by all descriptive
stat functions that support the 5 calling conventions.

Four output layouts:
    - **ungrouped**: single DV → scalar; multiple DVs → DataFrame
    - **marginal** (iv): compute stat for each iv independently, stack
    - **cell** (by): compute stat for each combination, MultiIndex rows
    - **pivot** (by + over): rows = by levels, columns = over levels
    - **mixed** (sub_specs): compute each term independently, stack with Term column
"""

from typing import Any, Callable, List, Optional, Union

import numpy
import pandas

from ..core.spec import resolve, ComputeSpec
from ..core.data_utils import validate_array
from ..core.matrix_engine import grouped_statistic, grouped_statistic_pivot, _build_cell_series


def _compute_grouped_flat(
    data: pandas.DataFrame,
    dv_col: str,
    groups: List[str],
    *,
    scalar_func: Callable[[numpy.ndarray], float],
    matrix_stat: Optional[str] = None,
    decimals: int = 4,
    stat_label: str = "Value",
) -> pandas.DataFrame:
    """Single entry point for grouped computation.

    Two paths based on stat type:
    - ``matrix_stat`` is not None: linear path (indicator matrix arithmetic
      via ``grouped_statistic``)
    - ``matrix_stat`` is None: non-linear path (``_build_cell_series`` for
      group membership + ``scalar_func`` iteration)

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv_col : str
        Dependent variable column name.
    groups : list of str
        Grouping column names.
    scalar_func : callable
        Function that takes a 1-D float array (NaN-free) and returns a scalar.
        Used only on the non-linear path.
    matrix_stat : str or None
        If not None, delegates to ``grouped_statistic()`` for the optimized
        matrix-arithmetic path. One of: 'mean', 'sum', 'count', 'variance',
        'sd', 'se'.
    decimals : int
        Rounding decimal places.
    stat_label : str
        Column label for the statistic in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per group with columns for each group variable + stat.
    """
    if matrix_stat is not None:
        # Linear path: matrix arithmetic
        result_df = grouped_statistic(data, dv=dv_col, groups=groups, stat_func=matrix_stat)
        stat_col = result_df.columns[-1]
        result_df[stat_col] = result_df[stat_col].round(decimals)
        return result_df

    # Non-linear path: iterate over groups using _build_cell_series
    cell_series = _build_cell_series(data, groups)
    unique_cells = sorted(cell_series.unique())
    values = data[dv_col].to_numpy(dtype=float)

    results = []
    for cell in unique_cells:
        mask = cell_series == cell
        arr = values[mask]
        # Remove NaN for computation
        clean = arr[~numpy.isnan(arr)]
        value = round(float(scalar_func(clean)), decimals) if len(clean) > 0 else numpy.nan

        row = {}
        if len(groups) == 1:
            row[groups[0]] = cell
        else:
            parts = cell.split(":")
            for i, group_name in enumerate(groups):
                row[group_name] = parts[i]
        row[stat_label] = value
        results.append(row)

    return pandas.DataFrame(results)


def _route_computation(arg1: Any,
                       arg2: Any,
                       *,
                       dv: Optional[Union[str, List[str]]],
                       iv: Optional[Union[str, List[str]]],
                       by: Optional[Union[str, List[str]]],
                       over: Optional[Union[str, List[str]]],
                       data: Optional[pandas.DataFrame],
                       scalar_func: Callable[[numpy.ndarray], float],
                       matrix_stat: Optional[str] = None,
                       decimals: int = 4,
                       weights: Optional[str] = None,
                       stat_label: str = "Value",
                       **kwargs: object,
                       ) -> Union[float, pandas.DataFrame]:
    """Route a descriptive stat computation through resolve → engine.

    This is the shared backbone for all descriptive functions that support
    the 5 calling conventions. It:
    1. Calls ``resolve()`` to normalize any input form into a ``ComputeSpec``
    2. Routes based on the spec layout:
       - No groups → scalar (single DV) or DataFrame (multiple DVs)
       - sub_specs → mixed formula: compute each term, stack with Term column
       - iv → marginal: compute for each iv independently, stack results
       - by → cell: grouped computation with actual group names
       - by + over → pivot: crossed pivot table

    Parameters
    ----------
    arg1, arg2 : Any
        Positional arguments from the calling function.
    dv, iv, by, over, data : keyword arguments forwarded to ``resolve()``.
    scalar_func : callable
        Pure computation function: takes a 1-D float numpy array, returns a float.
        Used for the ungrouped path and as the non-linear grouped path function.
    matrix_stat : str or None
        If the stat is directly supported by ``grouped_statistic()``
        (one of 'mean', 'sum', 'count', 'variance', 'sd', 'se'), pass
        the key here for the optimized matrix path.
    decimals : int
        Decimal places for rounding. Default is 4.
    weights : str or None
        Column name for observation weights (future use).
    stat_label : str
        Column label for the statistic in output. Default is "Value".
    **kwargs : dict
        Additional keyword arguments passed through to scalar_func.

    Returns
    -------
    float or pandas.DataFrame
        - float when single DV, no groups
        - DataFrame when multiple DVs, marginal, cell, pivot, or mixed
    """

    spec = resolve(arg1, arg2, dv=dv, iv=iv, by=by, over=over, data=data, weights=weights)

    # === Mixed formula (sub_specs) — compute each term independently, stack ===
    if spec.sub_specs is not None:
        return _compute_mixed(spec, scalar_func, matrix_stat, decimals, stat_label, **kwargs)

    # === No groups (no iv, no by, no over) ===
    if spec.iv is None and spec.by is None and spec.over is None:
        if len(spec.dv) == 1:
            # Single DV → scalar result
            arr = validate_array(spec.data[spec.dv[0]].to_numpy(), dtype=float)
            result = scalar_func(arr, **kwargs)
            return round(float(result), decimals)
        else:
            # Multiple DVs → DataFrame (one row per variable)
            rows = []
            for col in spec.dv:
                arr = validate_array(spec.data[col].to_numpy(), dtype=float)
                value = scalar_func(arr, **kwargs)
                rows.append({"Variable": col, stat_label: round(float(value), decimals)})
            return pandas.DataFrame(rows)

    # === Marginal (iv) — compute separately for each grouping variable, stack ===
    if spec.iv is not None:
        return _compute_marginal(spec, scalar_func, matrix_stat, decimals, stat_label, **kwargs)

    # === Pivot (by + over) ===
    if spec.over is not None:
        return _compute_pivot(spec, scalar_func, matrix_stat, decimals, stat_label, **kwargs)

    # === Cell (by only) — grouped with actual variable names ===
    return _compute_cell(spec, scalar_func, matrix_stat, decimals, stat_label, **kwargs)


def _compute_marginal(
    spec: ComputeSpec,
    scalar_func: Callable,
    matrix_stat: Optional[str],
    decimals: int,
    stat_label: str,
    **kwargs,
) -> pandas.DataFrame:
    """Compute stat for each iv variable independently, stack results.

    Equivalent to calling the stat function separately for each grouping
    variable and concatenating. Output has columns:
    [Factor, Level, stat_label] (+ Variable if multiple DVs).
    """
    frames = []
    for dv_col in spec.dv:
        for iv_var in spec.iv:
            result_df = _compute_grouped_flat(
                spec.data, dv_col, [iv_var],
                scalar_func=scalar_func,
                matrix_stat=matrix_stat,
                decimals=decimals,
                stat_label=stat_label,
            )

            # Normalize to common column structure: Factor, Level, stat
            stat_col_name = result_df.columns[-1]
            normalized = pandas.DataFrame({
                "Factor": iv_var,
                "Level": result_df[iv_var].values,
                stat_col_name: result_df[stat_col_name].values,
            })

            # Add DV column if multiple DVs
            if len(spec.dv) > 1:
                normalized.insert(0, "Variable", dv_col)

            frames.append(normalized)

    return pandas.concat(frames, ignore_index=True)


def _compute_cell(
    spec: ComputeSpec,
    scalar_func: Callable,
    matrix_stat: Optional[str],
    decimals: int,
    stat_label: str,
    **kwargs,
) -> pandas.DataFrame:
    """Compute stat for each cell (unique combination of by-variables).

    Output has one column per by-variable plus the stat column.
    Sets MultiIndex on by-variables.
    """
    frames = []
    for dv_col in spec.dv:
        result_df = _compute_grouped_flat(
            spec.data, dv_col, spec.by,
            scalar_func=scalar_func,
            matrix_stat=matrix_stat,
            decimals=decimals,
            stat_label=stat_label,
        )

        # Add DV column if multiple DVs
        if len(spec.dv) > 1:
            result_df.insert(0, "Variable", dv_col)

        frames.append(result_df)

    if len(frames) == 1:
        result = frames[0]
    else:
        result = pandas.concat(frames, ignore_index=True)

    # Set MultiIndex on by-variables for hierarchical display
    result = result.set_index(spec.by)

    return result


def _compute_pivot(
    spec: ComputeSpec,
    scalar_func: Callable,
    matrix_stat: Optional[str],
    decimals: int,
    stat_label: str,
    **kwargs,
) -> pandas.DataFrame:
    """Compute stat as a pivot table (by × over).

    Rows indexed by ``by`` levels, columns by ``over`` levels.
    """
    if matrix_stat is not None:
        if len(spec.dv) == 1:
            pivot = grouped_statistic_pivot(
                spec.data, dv=spec.dv[0], by=spec.by, over=spec.over, stat_func=matrix_stat
            )
            pivot = pivot.round(decimals)
            return pivot
        else:
            # Multiple DVs: stack as additional row index level
            frames = []
            for dv_col in spec.dv:
                pivot = grouped_statistic_pivot(
                    spec.data, dv=dv_col, by=spec.by, over=spec.over, stat_func=matrix_stat
                )
                pivot = pivot.round(decimals)
                # Add DV as an index level
                pivot.index = pandas.MultiIndex.from_arrays(
                    [[dv_col] * len(pivot)] + [pivot.index.get_level_values(i) for i in range(pivot.index.nlevels)],
                    names=["Variable"] + list(pivot.index.names),
                )
                frames.append(pivot)
            return pandas.concat(frames)
    else:
        # Non-linear path: compute flat then pivot manually
        all_groups = spec.by + spec.over
        frames = []
        for dv_col in spec.dv:
            result_df = _compute_grouped_flat(
                spec.data, dv_col, all_groups,
                scalar_func=scalar_func,
                matrix_stat=None,
                decimals=decimals,
                stat_label=stat_label,
            )
            frames.append(result_df)

        if len(frames) == 1:
            flat = frames[0]
        else:
            flat = pandas.concat(frames, ignore_index=True)

        # Determine stat column (last column that isn't a group variable)
        stat_col = [c for c in flat.columns if c not in all_groups and c != "Variable"][0]

        # Pivot
        if len(spec.by) == 1 and len(spec.over) == 1:
            pivot = flat.pivot(index=spec.by[0], columns=spec.over[0], values=stat_col)
        else:
            pivot = flat.pivot_table(index=spec.by, columns=spec.over, values=stat_col, aggfunc='first')

        pivot = pivot.round(decimals)
        return pivot


def _compute_mixed(
    spec: ComputeSpec,
    scalar_func: Callable,
    matrix_stat: Optional[str],
    decimals: int,
    stat_label: str,
    **kwargs,
) -> pandas.DataFrame:
    """Compute each term in a mixed formula independently and stack results.

    Each sub-spec is computed via the appropriate layout (marginal for main
    effects, cell for interactions), then tagged with a ``Term`` column and
    concatenated.

    Parameters
    ----------
    spec : ComputeSpec
        The resolved spec with sub_specs populated.
    scalar_func : callable
        Scalar computation function.
    matrix_stat : str or None
        Matrix engine stat key, if applicable.
    decimals : int
        Rounding decimal places.
    stat_label : str
        Column label for the statistic.

    Returns
    -------
    pd.DataFrame
        Stacked DataFrame with a ``Term`` column identifying each sub-result.
    """
    frames = []
    for ts in spec.sub_specs:
        if ts.layout == "iv":
            # Marginal: compute for this single variable
            sub_frames = []
            for dv_col in spec.dv:
                result_df = _compute_grouped_flat(
                    spec.data, dv_col, ts.variables,
                    scalar_func=scalar_func,
                    matrix_stat=matrix_stat,
                    decimals=decimals,
                    stat_label=stat_label,
                )
                # Normalize column names to Level + stat for consistency
                group_col = ts.variables[0]
                stat_col_name = result_df.columns[-1]
                normalized = pandas.DataFrame({
                    "Level": result_df[group_col].values,
                    stat_col_name: result_df[stat_col_name].values,
                })
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                sub_frames.append(normalized)
            sub_result = pandas.concat(sub_frames, ignore_index=True)
        else:
            # Cell (by): compute for each combination
            sub_frames = []
            for dv_col in spec.dv:
                result_df = _compute_grouped_flat(
                    spec.data, dv_col, ts.variables,
                    scalar_func=scalar_func,
                    matrix_stat=matrix_stat,
                    decimals=decimals,
                    stat_label=stat_label,
                )
                # Build Level column by joining group values
                if len(ts.variables) == 1:
                    level_values = result_df[ts.variables[0]].values
                else:
                    # Join multiple group columns into single Level string
                    level_values = result_df[ts.variables[0]].astype(str)
                    for var in ts.variables[1:]:
                        level_values = level_values + ":" + result_df[var].astype(str)
                    level_values = level_values.values

                stat_col_name = result_df.columns[-1]
                normalized = pandas.DataFrame({
                    "Level": level_values,
                    stat_col_name: result_df[stat_col_name].values,
                })
                if len(spec.dv) > 1:
                    normalized.insert(0, "Variable", dv_col)
                sub_frames.append(normalized)
            sub_result = pandas.concat(sub_frames, ignore_index=True)

        # Tag with Term column
        sub_result.insert(0, "Term", ts.term_name)
        frames.append(sub_result)

    return pandas.concat(frames, ignore_index=True)
