# -*- coding: utf-8 -*-
"""
ComputeSpec and resolve() — the universal input parsing layer.

Every researchpy function (descriptive, inferential, modeling) calls
``resolve()`` first to normalize any of the supported calling conventions
into a single ``ComputeSpec`` dataclass. The computation engine then
operates exclusively on the spec.

Supported calling conventions:
    1. mean(df['x'])                        → Series/array, no groups
    2. mean("y ~ C(x)", df)                 → formula string + DataFrame
    3. mean(["y", "k", "c"], df)            → column list + DataFrame
    4. mean(dv="y", by="x", data=df)        → explicit keywords (cell grouping)
       mean(dv="y", iv=["x","k"], data=df)  → explicit keywords (marginal)
       mean(dv="y", by="x", over="k", data=df) → pivot layout
    5. mean("y ~ C(x):C(k)", df)            → cell means (MultiIndex rows)
       mean("y ~ C(x)*C(k)", df)            → pivot layout

Formula operator semantics for descriptive stats:
    + : marginal (compute separately for each factor, stack results)
    : : cell (compute for each unique combination, MultiIndex rows)
    * : pivot (first factor → rows, second → columns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ComputeSpec:
    """Normalized specification for a computation.

    All calling conventions resolve to this single representation.
    The computation engine only ever sees a ``ComputeSpec``.

    Attributes
    ----------
    dv : list of str
        Dependent / target variable column name(s).
    iv : list of str or None
        Independent variables for marginal computation (each computed separately).
        Mutually exclusive with by/over.
    by : list of str or None
        Row grouping variable(s). Produces row index (MultiIndex if multiple).
    over : list of str or None
        Column grouping variable(s). Produces column index (pivot layout).
        Requires by to also be specified.
    data : pd.DataFrame
        The resolved data source containing all referenced columns.
    formula : str or None
        The original formula string, if one was provided.
    weights : str or None
        Column name for observation weights. None = unweighted.

    Examples
    --------
    >>> spec = resolve(pd.Series([1,2,3], name='x'))
    >>> spec.dv
    ['x']
    >>> spec.by is None
    True

    >>> spec = resolve("y ~ C(group)", df)
    >>> spec.dv
    ['y']
    >>> spec.by
    ['group']
    """

    dv: List[str] = field(default_factory=list)
    iv: Optional[List[str]] = None
    by: Optional[List[str]] = None
    over: Optional[List[str]] = None
    data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None
    weights: Optional[str] = None


def resolve(arg1: Any = None, arg2: Any = None, /, *,
            dv: Optional[Union[str, List[str]]] = None,
            iv: Optional[Union[str, List[str]]] = None,
            by: Optional[Union[str, List[str]]] = None,
            over: Optional[Union[str, List[str]]] = None,
            data: Optional[pd.DataFrame] = None,
            weights: Optional[str] = None,
            ) -> ComputeSpec:
    """Resolve any supported calling convention into a ComputeSpec.

    This is the universal input gate. Every researchpy function calls
    this first to normalize user input.

    Parameters
    ----------
    arg1 : various
        Positional argument 1. Can be:
        - pd.Series or np.ndarray or list/tuple → direct data
        - str → formula (requires arg2 or data= to be a DataFrame)
        - list of str → column names (requires arg2 or data=)
    arg2 : pd.DataFrame or None
        Positional argument 2. DataFrame when arg1 is a formula or column list.
    dv : str or list of str, optional
        Explicit dependent variable name(s).
    iv : str or list of str, optional
        Independent variables for marginal computation. Each variable is
        computed separately and results are stacked. Mutually exclusive
        with by/over.
    by : str or list of str, optional
        Row grouping variable(s) for cell means or pivot tables.
    over : str or list of str, optional
        Column grouping variable(s) for pivot layout. Requires by.
    data : pd.DataFrame, optional
        Explicit data source. Used when arg2 is not provided.
    weights : str, optional
        Column name for observation weights.

    Returns
    -------
    ComputeSpec
        Normalized computation specification.

    Raises
    ------
    ValueError
        If the input cannot be resolved, or if iv is used with by/over.
    TypeError
        If arg1 is an unsupported type.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [1,2,3], 'x': ['a','b','a']})

    # Convention 1: Series
    >>> spec = resolve(df['y'])
    >>> spec.dv, spec.by
    (['y'], None)

    # Convention 2: Formula
    >>> spec = resolve("y ~ C(x)", df)
    >>> spec.dv, spec.by
    (['y'], ['x'])

    # Convention 4: Keywords (marginal)
    >>> spec = resolve(dv="y", iv=["x", "k"], data=df)
    >>> spec.iv
    ['x', 'k']

    # Convention 4: Keywords (pivot)
    >>> spec = resolve(dv="y", by="x", over="k", data=df)
    >>> spec.by, spec.over
    (['x'], ['k'])
    """
    # Resolve data source (arg2 takes precedence, then keyword)
    resolved_data = arg2 if arg2 is not None else data

    # --- Normalize scalar strings to lists ---
    if isinstance(dv, str):
        dv = [dv]
    if isinstance(iv, str):
        iv = [iv]
    if isinstance(by, str):
        by = [by]
    if isinstance(over, str):
        over = [over]

    # --- Validate mutual exclusivity: iv XOR (by/over) ---
    if iv is not None and (by is not None or over is not None):
        raise ValueError(
            "Cannot use 'iv' together with 'by' or 'over'. "
            "'iv' produces marginal (stacked) results for each variable independently. "
            "'by'/'over' produce cell means or pivot tables. Use one approach or the other."
        )

    # --- Validate over requires by ---
    if over is not None and by is None:
        raise ValueError(
            "'over' requires 'by' to also be specified. "
            "'over' defines the column index and 'by' defines the row index of a pivot table."
        )

    # --- Convention 4: Explicit keywords (dv=, iv=/by=/over=, data=) ---
    if dv is not None:
        if resolved_data is None:
            raise ValueError(
                "When using keyword arguments (dv=, by=, iv=), 'data' must be provided."
            )

        _validate_columns(dv, resolved_data, "dv")
        if iv:
            _validate_columns(iv, resolved_data, "iv")
        if by:
            _validate_columns(by, resolved_data, "by")
        if over:
            _validate_columns(over, resolved_data, "over")

        return ComputeSpec(
            dv=dv,
            iv=iv,
            by=by,
            over=over,
            data=resolved_data,
            weights=weights,
        )

    # --- Convention 2: Formula string ---
    if isinstance(arg1, str):
        if resolved_data is None:
            raise ValueError(
                f"Formula '{arg1}' requires a DataFrame. "
                f"Pass it as the second positional argument or use data=."
            )

        spec = _parse_formula(arg1, resolved_data, weights)
        return spec

    # --- Convention 3: List of column names ---
    if isinstance(arg1, list) and all(isinstance(x, str) for x in arg1):
        if resolved_data is None:
            raise ValueError(
                "Column name list requires a DataFrame. "
                "Pass it as the second positional argument or use data=."
            )

        _validate_columns(arg1, resolved_data, "column names")

        return ComputeSpec(
            dv=arg1,
            iv=iv,
            by=by,
            over=over,
            data=resolved_data,
            weights=weights,
        )

    # --- Convention 1: Series, ndarray, list/tuple of values ---
    if isinstance(arg1, pd.Series):
        col_name = arg1.name if arg1.name is not None else "value"
        df = arg1.to_frame(name=col_name)
        return ComputeSpec(
            dv=[col_name],
            data=df,
            weights=weights,
        )

    if isinstance(arg1, (np.ndarray, list, tuple)):
        col_name = "value"
        df = pd.DataFrame({col_name: arg1})
        return ComputeSpec(
            dv=[col_name],
            data=df,
            weights=weights,
        )

    # --- GroupBy objects ---
    if isinstance(arg1, (pd.core.groupby.SeriesGroupBy, pd.core.groupby.DataFrameGroupBy)):
        return _resolve_groupby(arg1, weights)

    # --- Unsupported ---
    if arg1 is None and dv is None:
        raise ValueError(
            "No data provided. Pass data as a positional argument, "
            "or use keyword arguments (dv=, data=)."
        )

    raise TypeError(
        f"Cannot resolve input of type '{type(arg1).__name__}'. "
        f"Expected: pd.Series, np.ndarray, list, str (formula), "
        f"or list of str (column names)."
    )


def _resolve_groupby(groupby_obj: Any, weights: Optional[str] = None) -> ComputeSpec:
    """Resolve a pandas GroupBy object into a ComputeSpec.

    Extracts group key names and reconstructs the underlying data.

    Parameters
    ----------
    groupby_obj : SeriesGroupBy or DataFrameGroupBy
        The grouped pandas object.
    weights : str or None
        Column name for observation weights.

    Returns
    -------
    ComputeSpec
    """
    # Extract group variable name(s)
    group_keys = groupby_obj.keys
    if isinstance(group_keys, list):
        by_names = group_keys
    else:
        by_names = [group_keys]

    # Extract DV name(s)
    if isinstance(groupby_obj, pd.core.groupby.SeriesGroupBy):
        dv_name = groupby_obj.obj.name if groupby_obj.obj.name is not None else "value"
        dv_names = [dv_name]
        # Reconstruct full DataFrame with group columns + DV
        source_df = groupby_obj.obj.to_frame()
        # The group columns may not be in the Series frame, get from the grouper
        for key in by_names:
            if key not in source_df.columns:
                # Get group column from the original obj's index or grouper
                try:
                    source_df = groupby_obj.obj.reset_index()
                    break
                except Exception:
                    pass
    else:
        dv_names = [col for col in groupby_obj.obj.columns if col not in by_names]
        source_df = groupby_obj.obj

    # Ensure we have the full DataFrame with both group and DV columns
    if not all(col in source_df.columns for col in by_names):
        # Attempt to reconstruct from the groupby object
        source_df = groupby_obj.obj.copy()
        for key in by_names:
            if key not in source_df.columns:
                # Try to get from index
                if key in source_df.index.names:
                    source_df = source_df.reset_index()

    return ComputeSpec(
        dv=dv_names,
        by=by_names,
        data=source_df,
        weights=weights,
    )


def _parse_formula(formula: str, data: pd.DataFrame, weights: Optional[str] = None) -> ComputeSpec:
    """Parse a formula string into a ComputeSpec using Patsy's ModelDesc.

    Detects the formula operator pattern to determine layout:
    - Only single-factor terms (connected by +) → marginal (iv)
    - Only multi-factor interaction terms (:) → cell means (by)
    - Mix of main effects + interactions (*) → pivot (by + over)

    Parameters
    ----------
    formula : str
        Patsy-style formula, e.g., "y ~ C(x)", "y ~ C(x):C(k)", "y ~ C(x)*C(k)".
    data : pd.DataFrame
        Source DataFrame.
    weights : str or None
        Weight column name.

    Returns
    -------
    ComputeSpec

    Raises
    ------
    ValueError
        If the formula pattern is ambiguous or columns are not found.
    """
    from researchpy.containers.multivariable import ModelTerms

    mt = ModelTerms.from_formula(formula)

    dv_names = mt.dv if mt.dv else []
    _validate_columns(dv_names, data, "DV (left side of ~)")

    # Categorize RHS terms
    main_effect_terms = []   # single-factor terms (e.g., C(x))
    interaction_terms = []   # multi-factor terms (e.g., C(x):C(k))

    for term in mt.terms:
        if term.is_interaction:
            interaction_terms.append(term)
        else:
            main_effect_terms.append(term)

    # Determine layout based on pattern
    if len(interaction_terms) == 0 and len(main_effect_terms) > 0:
        # Pattern: "y ~ C(x) + C(k)" → marginal (iv)
        # OR single: "y ~ C(x)" → simple by (single variable)
        var_names = [t.name for t in main_effect_terms]
        _validate_columns(var_names, data, "RHS terms")

        if len(var_names) == 1:
            # Single grouping variable — use by (equivalent to iv for single var)
            return ComputeSpec(
                dv=dv_names,
                by=var_names,
                data=data,
                formula=formula,
                weights=weights,
            )
        else:
            # Multiple main effects with + → marginal
            return ComputeSpec(
                dv=dv_names,
                iv=var_names,
                data=data,
                formula=formula,
                weights=weights,
            )

    elif len(interaction_terms) > 0 and len(main_effect_terms) == 0:
        # Pattern: "y ~ C(x):C(k)" → cell means (by)
        # Collect all variables from interactions into by
        by_vars = []
        for term in interaction_terms:
            for var in term.name.split(":"):
                if var not in by_vars:
                    by_vars.append(var)

        _validate_columns(by_vars, data, "RHS interaction terms")

        return ComputeSpec(
            dv=dv_names,
            by=by_vars,
            data=data,
            formula=formula,
            weights=weights,
        )

    elif len(interaction_terms) > 0 and len(main_effect_terms) > 0:
        # Pattern: "y ~ C(x)*C(k)" → pivot (by + over)
        # * expands to main effects + interaction in Patsy
        # First factor of interaction → by, remaining → over

        # Get the interaction variable names
        # Use the first interaction term to determine by/over split
        first_interaction = interaction_terms[0]
        interaction_vars = first_interaction.name.split(":")

        by_var = [interaction_vars[0]]
        over_vars = interaction_vars[1:]

        # If there are multiple interaction terms sharing the same first factor,
        # collect all second factors into over
        # e.g., "y ~ C(x)*C(k) + C(x)*C(z)" → by=["x"], over=["k", "z"]
        for term in interaction_terms[1:]:
            term_vars = term.name.split(":")
            for v in term_vars[1:]:
                if v not in over_vars:
                    over_vars.append(v)

        all_vars = by_var + over_vars
        _validate_columns(all_vars, data, "RHS terms")

        return ComputeSpec(
            dv=dv_names,
            by=by_var,
            over=over_vars,
            data=data,
            formula=formula,
            weights=weights,
        )

    else:
        # No RHS terms — just compute for the DV(s) directly
        return ComputeSpec(
            dv=dv_names,
            data=data,
            formula=formula,
            weights=weights,
        )


def _validate_columns(columns: List[str], data: pd.DataFrame, label: str) -> None:
    """Check that all column names exist in the DataFrame.

    Parameters
    ----------
    columns : list of str
        Column names to validate.
    data : pd.DataFrame
        DataFrame to check against.
    label : str
        Label for the error message (e.g., "dv", "by").

    Raises
    ------
    ValueError
        If any column is not found in the DataFrame.
    """
    missing = [c for c in columns if c not in data.columns]
    if missing:
        available = list(data.columns)
        raise ValueError(
            f"Column(s) {missing} specified for {label} not found in DataFrame. "
            f"Available columns: {available}"
        )

