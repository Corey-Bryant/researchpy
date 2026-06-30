# -*- coding: utf-8 -*-
"""
Matrix computation engine.

Provides the indicator (one-hot) matrix builder for grouped descriptive
statistics, and shared matrix operations used across the package.

Two matrix pathways exist in researchpy:

1. **Indicator matrix** (this module) — full dummy coding, all levels present,
   no intercept, no contrast coding. Used by descriptive stats for grouped
   computations (e.g., group means via ``X.T @ values / X.sum(axis=0)``).

2. **Design matrix** (Patsy-backed, ``CoreModel``) — contrast-coded, intercept
   included, reference level dropped. Used by model fitting (OLS, logistic, etc.).

Both pathways share the ``Term`` / ``ModelTerms`` metadata infrastructure.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def build_indicator_matrix(data: pd.DataFrame,
                           groups: List[str],
                           ) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """Build a full indicator (one-hot) matrix for grouping structure.

    Creates a dense n × k matrix where each column corresponds to a unique
    cell (level or level-combination) and each row has exactly one 1.0
    indicating group membership.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the grouping columns.
    groups : list of str
        Column names to use as grouping factors.

    Returns
    -------
    indicator_matrix : np.ndarray
        Dense float64 array of shape (n_obs, n_cells).
        Each row has exactly one 1.0 at the column corresponding to
        the observation's cell membership.
    cell_labels : list of str
        Human-readable labels for each column (cell) in the indicator matrix.
        For single factors: the level values (e.g., ['a', 'b', 'c']).
        For interactions: joined level values (e.g., ['a:low', 'a:high', ...]).
    cell_components : list of list of str
        For each cell, the individual level values per group variable.
        E.g., for groups=['x','k']: [['a','low'], ['a','high'], ['b','low'], ...]
        For single group: [['a'], ['b'], ['c']]

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': ['a', 'b', 'a', 'b'], 'y': [1, 2, 3, 4]})
    >>> X, labels, components = build_indicator_matrix(df, groups=['x'])
    >>> labels
    ['a', 'b']
    >>> components
    [['a'], ['b']]
    """
    n = len(data)

    # Build combined cell series
    cell_series = _build_cell_series(data, groups)

    # Get unique cells (sorted for deterministic ordering)
    unique_cells = np.sort(cell_series.unique())
    cell_labels = unique_cells.tolist()
    n_cells = len(unique_cells)

    # Build the indicator matrix using integer codes
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    codes = cell_series.map(cell_to_idx).values.astype(int)

    indicator_matrix = np.zeros((n, n_cells), dtype=np.float64)
    indicator_matrix[np.arange(n), codes] = 1.0

    # Build cell components (split labels back into per-group values)
    if len(groups) == 1:
        cell_components = [[label] for label in cell_labels]
    else:
        cell_components = [label.split(":") for label in cell_labels]

    return indicator_matrix, cell_labels, cell_components


def _build_cell_series(data: pd.DataFrame, variables: Union[List[str], Tuple[str, ...]]) -> pd.Series:
    """Combine multiple grouping columns into a single cell-label Series.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    variables : list or tuple of str
        Column names to combine.

    Returns
    -------
    pd.Series
        Series of string cell labels. For single variables, just the
        string-cast values. For multiple variables, colon-joined.
    """
    if len(variables) == 1:
        return data[variables[0]].astype(str)
    else:
        # Join multiple columns with ":" separator
        parts = [data[v].astype(str) for v in variables]
        return parts[0].str.cat(parts[1:], sep=":")


def grouped_statistic(data: pd.DataFrame,
                      dv: str,
                      groups: List[str],
                      stat_func: str = "mean",
                      ) -> pd.DataFrame:
    """Compute a grouped statistic using indicator matrix arithmetic.

    This is the engine that powers grouped descriptive stats. Instead of
    pandas groupby, it uses matrix multiplication for computation:
        group_means = (X.T @ values) / X.sum(axis=0)

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv : str
        Dependent variable column name (must be numeric).
    groups : list of str
        Grouping variable column names.
    stat_func : str
        Statistic to compute. One of: "mean", "sum", "count", "variance", "sd", "se".

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per group variable, plus the statistic column.
        Uses actual group variable names (not generic "Group").

    Examples
    --------
    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','b','b','c','c']})
    >>> grouped_statistic(df, dv='y', groups=['g'], stat_func='mean')
       g  Mean
    0  a   1.5
    1  b   3.5
    2  c   5.5
    """
    X, cell_labels, cell_components = build_indicator_matrix(data, groups)
    values = data[dv].to_numpy(dtype=np.float64, na_value=np.nan)

    # Mask NaN values: zero them out in both X and values for correct arithmetic
    nan_mask = np.isnan(values)
    if nan_mask.any():
        values_clean = np.where(nan_mask, 0.0, values)
        # Zero out rows in X where value is NaN so they don't contribute
        X_clean = X.copy()
        X_clean[nan_mask] = 0.0
        counts = X_clean.sum(axis=0)
    else:
        values_clean = values
        X_clean = X
        counts = X.sum(axis=0)

    # Core matrix computations
    sums = X_clean.T @ values_clean

    if stat_func == "count":
        result_values = counts
    elif stat_func == "sum":
        result_values = sums
    elif stat_func == "mean":
        with np.errstate(divide='ignore', invalid='ignore'):
            result_values = np.where(counts > 0, sums / counts, np.nan)
    elif stat_func == "variance":
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
        obs_group_means = X_clean @ means
        sq_devs = np.where(nan_mask, 0.0, (values - obs_group_means) ** 2)
        ss = X_clean.T @ sq_devs
        with np.errstate(divide='ignore', invalid='ignore'):
            result_values = np.where(counts > 1, ss / (counts - 1), np.nan)
    elif stat_func == "sd":
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
        obs_group_means = X_clean @ means
        sq_devs = np.where(nan_mask, 0.0, (values - obs_group_means) ** 2)
        ss = X_clean.T @ sq_devs
        with np.errstate(divide='ignore', invalid='ignore'):
            variance = np.where(counts > 1, ss / (counts - 1), np.nan)
        result_values = np.sqrt(variance)
    elif stat_func == "se":
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
        obs_group_means = X_clean @ means
        sq_devs = np.where(nan_mask, 0.0, (values - obs_group_means) ** 2)
        ss = X_clean.T @ sq_devs
        with np.errstate(divide='ignore', invalid='ignore'):
            variance = np.where(counts > 1, ss / (counts - 1), np.nan)
            result_values = np.where(counts > 0, np.sqrt(variance) / np.sqrt(counts), np.nan)
    else:
        raise ValueError(
            f"Unknown stat_func '{stat_func}'. "
            f"Supported: 'mean', 'sum', 'count', 'variance', 'sd', 'se'."
        )

    # Format output with actual group variable names
    stat_label = {
        "mean": "Mean", "sum": "Sum", "count": "N",
        "variance": "Variance", "sd": "SD", "se": "SE",
    }[stat_func]

    # Cast counts to int for cleaner output
    if stat_func == "count":
        result_values = result_values.astype(int)

    # Build result DataFrame with separate columns per group variable
    result_dict = {}
    for i, group_name in enumerate(groups):
        result_dict[group_name] = [comp[i] for comp in cell_components]

    result_dict[stat_label] = result_values

    return pd.DataFrame(result_dict)


def grouped_statistic_pivot(data: pd.DataFrame,
                            dv: str,
                            by: List[str],
                            over: List[str],
                            stat_func: str = "mean",
                            ) -> pd.DataFrame:
    """Compute a grouped statistic and return as a pivot table.

    Rows are indexed by ``by`` variable levels, columns by ``over`` variable levels.

    Parameters
    ----------
    data : pd.DataFrame
        Source data.
    dv : str
        Dependent variable column name (must be numeric).
    by : list of str
        Row grouping variable(s).
    over : list of str
        Column grouping variable(s).
    stat_func : str
        Statistic to compute. One of: "mean", "sum", "count", "variance", "sd", "se".

    Returns
    -------
    pd.DataFrame
        Pivot table with row index from ``by`` levels and column index
        from ``over`` levels. Uses MultiIndex where appropriate.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'y': [1,2,3,4,5,6,7,8],
    ...     'row': ['a','a','b','b','a','a','b','b'],
    ...     'col': ['x','y','x','y','x','y','x','y']
    ... })
    >>> grouped_statistic_pivot(df, dv='y', by=['row'], over=['col'], stat_func='mean')
    col    x    y
    row
    a    3.0  4.0
    b    5.0  6.0
    """
    # Compute the flat grouped statistic with all grouping variables
    all_groups = by + over
    flat_df = grouped_statistic(data, dv=dv, groups=all_groups, stat_func=stat_func)

    # Get the stat column name
    stat_label = {
        "mean": "Mean", "sum": "Sum", "count": "N",
        "variance": "Variance", "sd": "SD", "se": "SE",
    }[stat_func]

    # Pivot: by-variables become the index, over-variables become the columns
    if len(by) == 1 and len(over) == 1:
        pivot = flat_df.pivot(index=by[0], columns=over[0], values=stat_label)
    elif len(by) > 1 and len(over) == 1:
        pivot = flat_df.pivot_table(index=by, columns=over[0], values=stat_label, aggfunc='first')
    elif len(by) == 1 and len(over) > 1:
        pivot = flat_df.pivot_table(index=by[0], columns=over, values=stat_label, aggfunc='first')
    else:
        pivot = flat_df.pivot_table(index=by, columns=over, values=stat_label, aggfunc='first')

    return pivot
