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


def build_indicator_matrix(
    data: pd.DataFrame,
    groups: List[str],
    interactions: Optional[List[Tuple[str, ...]]] = None,
) -> Tuple[np.ndarray, List[str]]:
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
    interactions : list of tuple of str, optional
        Interaction terms. If provided, builds cell indicators for each
        unique combination of the interacted variables.
        If None, builds a simple one-way indicator for each group crossed
        (i.e., treats groups as a single combined factor).

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

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': ['a', 'b', 'a', 'b'], 'y': [1, 2, 3, 4]})
    >>> X, labels = build_indicator_matrix(df, groups=['x'])
    >>> labels
    ['a', 'b']
    >>> X
    array([[1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.]])

    >>> df = pd.DataFrame({'g': ['a','a','b','b'], 'd': ['x','y','x','y'], 'v': [1,2,3,4]})
    >>> X, labels = build_indicator_matrix(df, groups=['g', 'd'], interactions=[('g', 'd')])
    >>> labels
    ['a:x', 'a:y', 'b:x', 'b:y']
    """
    n = len(data)

    if interactions:
        # Interaction: build cells from the combined levels of interacted variables
        # Use the first (and typically only) interaction tuple
        # For multiple interaction terms, we'd need to handle separately,
        # but for grouped descriptive stats there's usually one grouping structure.
        interaction_vars = interactions[0]
        cell_series = _build_cell_series(data, interaction_vars)
    else:
        # Simple grouping: cross all group variables into a single factor
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

    return indicator_matrix, cell_labels


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


def grouped_statistic(
    data: pd.DataFrame,
    dv: str,
    groups: List[str],
    stat_func: str = "mean",
    interactions: Optional[List[Tuple[str, ...]]] = None,
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
    interactions : list of tuple of str, optional
        Interaction terms for cell-level grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group/cell, columns for group labels and the statistic.

    Examples
    --------
    >>> df = pd.DataFrame({'y': [1,2,3,4,5,6], 'g': ['a','a','b','b','c','c']})
    >>> grouped_statistic(df, dv='y', groups=['g'], stat_func='mean')
      Group  Mean
    0     a   1.5
    1     b   3.5
    2     c   5.5
    """
    X, cell_labels = build_indicator_matrix(data, groups, interactions)
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
        # Avoid division by zero for empty groups
        with np.errstate(divide='ignore', invalid='ignore'):
            result_values = np.where(counts > 0, sums / counts, np.nan)
    elif stat_func == "variance":
        # Compute group means first
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
        # Expand means back to observation level: (n,) array of each obs's group mean
        obs_group_means = X_clean @ means
        # Squared deviations
        sq_devs = np.where(nan_mask, 0.0, (values - obs_group_means) ** 2)
        ss = X_clean.T @ sq_devs
        with np.errstate(divide='ignore', invalid='ignore'):
            result_values = np.where(counts > 1, ss / (counts - 1), np.nan)
    elif stat_func == "sd":
        # Standard deviation = sqrt(variance)
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
        obs_group_means = X_clean @ means
        sq_devs = np.where(nan_mask, 0.0, (values - obs_group_means) ** 2)
        ss = X_clean.T @ sq_devs
        with np.errstate(divide='ignore', invalid='ignore'):
            variance = np.where(counts > 1, ss / (counts - 1), np.nan)
        result_values = np.sqrt(variance)
    elif stat_func == "se":
        # Standard error = sd / sqrt(n)
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

    # Format output
    stat_label = {
        "mean": "Mean", "sum": "Sum", "count": "N",
        "variance": "Variance", "sd": "SD", "se": "SE",
    }[stat_func]

    # Cast counts to int for cleaner output
    if stat_func == "count":
        result_values = result_values.astype(int)

    result_df = pd.DataFrame({
        "Group": cell_labels,
        stat_label: result_values,
    })

    return result_df

