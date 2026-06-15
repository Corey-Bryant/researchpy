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
    4. mean(dv=["y"], by=["x"], data=df)    → explicit keywords
    5. mean("y ~ C(x):C(k)", df)            → formula with interaction
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
    groups : list of str or None
        Grouping / factor variable column name(s). None = no grouping.
    interactions : list of tuple of str or None
        Interaction terms as tuples of variable names.
        None = no interactions (main effects only or no groups at all).
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
    >>> spec.groups is None
    True

    >>> spec = resolve("y ~ C(group)", df)
    >>> spec.dv
    ['y']
    >>> spec.groups
    ['group']
    """

    dv: List[str] = field(default_factory=list)
    groups: Optional[List[str]] = None
    interactions: Optional[List[Tuple[str, ...]]] = None
    data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None
    weights: Optional[str] = None


def resolve(
    arg1: Any = None,
    arg2: Any = None,
    /,
    *,
    dv: Optional[Union[str, List[str]]] = None,
    iv: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
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
        Explicit independent variable name(s). Alias for ``by``.
    by : str or list of str, optional
        Explicit grouping variable name(s). Takes precedence over ``iv``
        if both are specified.
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
        If the input cannot be resolved (e.g., formula without data,
        column names not found in data).
    TypeError
        If arg1 is an unsupported type.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [1,2,3], 'x': ['a','b','a']})

    # Convention 1: Series
    >>> spec = resolve(df['y'])
    >>> spec.dv, spec.groups
    (['y'], None)

    # Convention 2: Formula
    >>> spec = resolve("y ~ C(x)", df)
    >>> spec.dv, spec.groups
    (['y'], ['x'])

    # Convention 3: Column list
    >>> spec = resolve(["y"], df)
    >>> spec.dv, spec.groups
    (['y'], None)

    # Convention 4: Keywords
    >>> spec = resolve(dv="y", by="x", data=df)
    >>> spec.dv, spec.groups
    (['y'], ['x'])
    """
    # Resolve data source (arg2 takes precedence, then keyword)
    resolved_data = arg2 if arg2 is not None else data

    # --- Convention 4: Explicit keywords (dv=, iv=/by=, data=) ---
    if dv is not None:
        if isinstance(dv, str):
            dv = [dv]

        # Resolve grouping: by= takes precedence over iv=
        groups = by if by is not None else iv
        if isinstance(groups, str):
            groups = [groups]

        if resolved_data is None:
            raise ValueError(
                "When using keyword arguments (dv=, by=), 'data' must be provided."
            )

        _validate_columns(dv, resolved_data, "dv")
        if groups:
            _validate_columns(groups, resolved_data, "by/iv")

        return ComputeSpec(
            dv=dv,
            groups=groups if groups else None,
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

        from researchpy.containers.multivariable import ModelTerms

        mt = ModelTerms.from_formula(arg1)

        dv_names = mt.dv if mt.dv else []
        group_names = []
        interaction_tuples = []

        for term in mt.terms:
            if term.is_interaction:
                # Interaction: collect all sub-variable names
                sub_vars = term.name.split(":")
                for v in sub_vars:
                    if v not in group_names:
                        group_names.append(v)
                interaction_tuples.append(tuple(sub_vars))
            else:
                # Simple term (could be factor or continuous grouping var)
                if term.name not in group_names:
                    group_names.append(term.name)

        _validate_columns(dv_names, resolved_data, "DV (left side of ~)")
        _validate_columns(group_names, resolved_data, "RHS terms")

        return ComputeSpec(
            dv=dv_names,
            groups=group_names if group_names else None,
            interactions=interaction_tuples if interaction_tuples else None,
            data=resolved_data,
            formula=arg1,
            weights=weights,
        )

    # --- Convention 3: List of column names ---
    if isinstance(arg1, list) and all(isinstance(x, str) for x in arg1):
        if resolved_data is None:
            raise ValueError(
                "Column name list requires a DataFrame. "
                "Pass it as the second positional argument or use data=."
            )

        _validate_columns(arg1, resolved_data, "column names")

        # Resolve grouping from keywords if provided
        groups = by if by is not None else iv
        if isinstance(groups, str):
            groups = [groups]
        if groups:
            _validate_columns(groups, resolved_data, "by/iv")

        return ComputeSpec(
            dv=arg1,
            groups=groups if groups else None,
            data=resolved_data,
            weights=weights,
        )

    # --- Convention 1: Series, ndarray, list/tuple of values ---
    if isinstance(arg1, pd.Series):
        col_name = arg1.name if arg1.name is not None else "value"
        df = arg1.to_frame(name=col_name)
        return ComputeSpec(
            dv=[col_name],
            groups=None,
            data=df,
            weights=weights,
        )

    if isinstance(arg1, (np.ndarray, list, tuple)):
        col_name = "value"
        df = pd.DataFrame({col_name: arg1})
        return ComputeSpec(
            dv=[col_name],
            groups=None,
            data=df,
            weights=weights,
        )

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


def _validate_columns(columns: List[str], data: pd.DataFrame, label: str) -> None:
    """Check that all column names exist in the DataFrame.

    Parameters
    ----------
    columns : list of str
        Column names to validate.
    data : pd.DataFrame
        DataFrame to check against.
    label : str
        Label for the error message (e.g., "dv", "by/iv").

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

