# -*- coding: utf-8 -*-
"""
SyntaxSpec and resolve() — the universal input parsing layer.

Every researchpy function (descriptive, inferential, modeling) calls
``SyntaxSpec.from_args()`` (or its subclass override) first to normalize
any of the supported calling conventions into a single ``SyntaxSpec``
dataclass. The computation engine then operates exclusively on the spec.

Subclasses (e.g., AnovaSpec, TTestSpec) override ``from_args()`` to add
domain-specific validation while inheriting the full parsing logic.

The module-level ``resolve()`` function is a backward-compatible wrapper.

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

Reverse mapping (to_formula):
    iv=["x","k"]         → "dv ~ C(x) + C(k)"
    by=["x"]             → "dv ~ C(x)"
    by=["x","k"]         → "dv ~ C(x):C(k)"
    by=["x"], over=["k"] → "dv ~ C(x)*C(k)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from researchpy.containers import CoreDataclass


@dataclass
class TermSpec:
    """One term extracted from a mixed formula.

    Used when a formula contains both main effects and interactions that
    don't form a pure star expansion (e.g., ``y ~ C(x) + C(k):C(z)``).
    Each term is computed independently and results are stacked with a
    ``Term`` column identifying the source.

    Attributes
    ----------
    term_name : str
        Cleaned term name (e.g., ``"x"`` or ``"k:z"``).
    term_raw : str
        Raw term string as written in the formula (e.g., ``"C(x)"`` or ``"C(k):C(z)"``).
    layout : str
        How this term should be computed: ``"iv"`` (marginal) or ``"by"`` (cell).
    variables : list of str
        The grouping variable column name(s) for this term.
    """

    term_name: str
    term_raw: str
    layout: str
    variables: List[str]


@dataclass
class SyntaxSpec(CoreDataclass):
    """Normalized specification for a computation.

    All calling conventions resolve to this single representation.
    The computation engine only ever sees a ``SyntaxSpec``.

    Subclasses (e.g., AnovaSpec, TTestSpec) can override ``from_args()``
    to add domain-specific validation while reusing the universal parsing
    logic via ``super().from_args(...)``.

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
    sub_specs : list of TermSpec or None
        For mixed formulas (e.g., ``y ~ C(x) + C(k):C(z)``), holds one
        TermSpec per formula term. When not None, the computation engine
        iterates each sub-spec independently and stacks results with a
        ``Term`` column. None for all non-mixed cases.

    Examples
    --------
    >>> spec = SyntaxSpec.from_args(pd.Series([1,2,3], name='x'))
    >>> spec.dv
    ['x']
    >>> spec.by is None
    True

    >>> spec = SyntaxSpec.from_args("y ~ C(group)", df)
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
    sub_specs: Optional[List[TermSpec]] = None


    def __post_init__(self):
        # Validate that dv is not empty
        if not self.dv:
            raise ValueError("SyntaxSpec must have at least one dependent variable (dv).")

        # Validate that iv and by/over are mutually exclusive
        if self.iv is not None and (self.by is not None or self.over is not None):
            raise ValueError(
                "Cannot use 'iv' together with 'by' or 'over'. "
                "'iv' produces marginal (stacked) results for each variable independently. "
                "'by'/'over' produce cell means or pivot tables. Use one approach or the other."
            )

        # Validate that over requires by
        if self.over is not None and self.by is None:
            raise ValueError(
                "'over' requires 'by' to also be specified. "
                "'over' defines the column index and 'by' defines the row index of a pivot table."
            )

        if not self.formula:
            self.formula = self._to_formula()

    # ------------------------------------------------------------------
    # Factory classmethod — the primary entry point for resolution
    # ------------------------------------------------------------------

    @classmethod
    def from_args(
        cls,
        arg1: Any = None,
        arg2: Any = None,
        /,
        *,
        dv: Optional[Union[str, List[str]]] = None,
        iv: Optional[Union[str, List[str]]] = None,
        by: Optional[Union[str, List[str]]] = None,
        over: Optional[Union[str, List[str]]] = None,
        data: Optional[pd.DataFrame] = None,
        weights: Optional[str] = None,
    ) -> "SyntaxSpec":
        """Resolve any supported calling convention into a SyntaxSpec.

        This is the universal input gate. Every researchpy function calls
        this (or its subclass override) first to normalize user input.

        Subclasses can override this method to inject domain-specific
        validation while delegating core parsing to ``super().from_args(...)``.

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
        SyntaxSpec (or subclass instance)
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
        >>> spec = SyntaxSpec.from_args(df['y'])
        >>> spec.dv, spec.by
        (['y'], None)

        # Convention 2: Formula
        >>> spec = SyntaxSpec.from_args("y ~ C(x)", df)
        >>> spec.dv, spec.by
        (['y'], ['x'])

        # Convention 4: Keywords (marginal)
        >>> spec = SyntaxSpec.from_args(dv="y", iv=["x", "k"], data=df)
        >>> spec.iv
        ['x', 'k']

        # Convention 4: Keywords (pivot)
        >>> spec = SyntaxSpec.from_args(dv="y", by="x", over="k", data=df)
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

            formula_like = cls._to_formula(dv=dv, iv=iv, by=by, over=over)

            return cls(
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

            spec = _parse_formula(arg1, resolved_data, weights, cls)
            return spec

        # --- Convention 3: List of column names ---
        if isinstance(arg1, list) and all(isinstance(x, str) for x in arg1):
            if resolved_data is None:
                raise ValueError(
                    "Column name list requires a DataFrame. "
                    "Pass it as the second positional argument or use data=."
                )

            _validate_columns(arg1, resolved_data, "column names")

            return cls(
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
            return cls(
                dv=[col_name],
                data=df,
                weights=weights,
            )

        if isinstance(arg1, (np.ndarray, list, tuple)):
            col_name = "value"
            df = pd.DataFrame({col_name: arg1})
            return cls(
                dv=[col_name],
                data=df,
                weights=weights,
            )

        # --- GroupBy objects ---
        if isinstance(arg1, (pd.core.groupby.SeriesGroupBy, pd.core.groupby.DataFrameGroupBy)):
            return _resolve_groupby(arg1, weights, cls)

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



    @classmethod
    def _to_formula(cls) -> str:
        """Generate a Wilkinson-style formula string from the spec parameters.

        Reconstructs the formula using the mapping:
            - iv (marginal)        → ``dv ~ C(x) + C(k)``
            - by (cell means)      → ``dv ~ C(x)`` or ``dv ~ C(x):C(k)``
            - by + over (pivot)    → ``dv ~ C(by)*C(over)``
            - no grouping          → ``dv`` (no RHS)

        When multiple DVs are present, only the first is used on the LHS
        (consistent with standard formula notation).

        Returns
        -------
        str
            The generated formula string.

        Raises
        ------
        ValueError
            If the spec has no DV columns to build a formula from.

        Examples
        --------
        >>> spec = SyntaxSpec(dv=['y'], by=['group'])
        >>> spec.to_formula()
        'y ~ C(group)'

        >>> spec = SyntaxSpec(dv=['y'], iv=['x', 'k'])
        >>> spec.to_formula()
        'y ~ C(x) + C(k)'

        >>> spec = SyntaxSpec(dv=['y'], by=['x'], over=['k'])
        >>> spec.to_formula()
        'y ~ C(x)*C(k)'

        >>> spec = SyntaxSpec(dv=['y'], by=['x', 'k'])
        >>> spec.to_formula()
        'y ~ C(x):C(k)'
        """
        if not self.dv:
            raise ValueError(
                "Cannot generate formula: no dependent variable (dv) specified."
            )

        # LHS: use first DV (standard formula convention)
        lhs = self.dv[0]

        # --- Build RHS based on which parameters are populated ---

        formula_like = f"{dv} ~ "
        rhs_terms = []

        if iv:
            for col in iv:
                if col != iv[-1]:
                    # formula_like += f"C({col}) + "
                    rhs_terms.append(f"C({col})")

        if by:
            by_terms = []
            for col in by: by_terms.append(f"C({col})")
            by_term = ":".join(by_terms)

            rhs_terms.append(by_term)

            # formula_like += f"{by_term} + "

        if over:
            over_terms = []
            for col in over: over_terms.append(f"C({col})")
            over_terms = "*".join(over_terms)

            rhs_terms.append("*".join(over_terms))

        rhs = " + ".join(rhs_terms)
        formula = formula_like + rhs





    # ------------------------------------------------------------------
    # Reverse mapping: spec → formula string
    # ------------------------------------------------------------------

    def to_formula(self) -> str:
        """Generate a Wilkinson-style formula string from the spec parameters.

        Reconstructs the formula using the mapping:
            - iv (marginal)        → ``dv ~ C(x) + C(k)``
            - by (cell means)      → ``dv ~ C(x)`` or ``dv ~ C(x):C(k)``
            - by + over (pivot)    → ``dv ~ C(by)*C(over)``
            - no grouping          → ``dv`` (no RHS)

        When multiple DVs are present, only the first is used on the LHS
        (consistent with standard formula notation).

        Returns
        -------
        str
            The generated formula string.

        Raises
        ------
        ValueError
            If the spec has no DV columns to build a formula from.

        Examples
        --------
        >>> spec = SyntaxSpec(dv=['y'], by=['group'])
        >>> spec.to_formula()
        'y ~ C(group)'

        >>> spec = SyntaxSpec(dv=['y'], iv=['x', 'k'])
        >>> spec.to_formula()
        'y ~ C(x) + C(k)'

        >>> spec = SyntaxSpec(dv=['y'], by=['x'], over=['k'])
        >>> spec.to_formula()
        'y ~ C(x)*C(k)'

        >>> spec = SyntaxSpec(dv=['y'], by=['x', 'k'])
        >>> spec.to_formula()
        'y ~ C(x):C(k)'
        """
        if not self.dv:
            raise ValueError(
                "Cannot generate formula: no dependent variable (dv) specified."
            )

        # LHS: use first DV (standard formula convention)
        lhs = self.dv[0]

        # --- Build RHS based on which parameters are populated ---

        # Pivot layout: by + over → star expansion
        if self.by is not None and self.over is not None:
            by_terms = [f"C({var})" for var in self.by]
            over_terms = [f"C({var})" for var in self.over]
            # Star connects all row and column factors
            all_terms = by_terms + over_terms
            rhs = "*".join(all_terms)
            return f"{lhs} ~ {rhs}"

        # Marginal: iv → plus-separated main effects
        if self.iv is not None:
            rhs_parts = [f"C({var})" for var in self.iv]
            rhs = " + ".join(rhs_parts)
            return f"{lhs} ~ {rhs}"

        # Cell means: by → colon-separated if multiple, single C() if one
        if self.by is not None:
            if len(self.by) == 1:
                rhs = f"C({self.by[0]})"
            else:
                rhs = ":".join(f"C({var})" for var in self.by)
            return f"{lhs} ~ {rhs}"

        # No grouping — formula is just the DV
        return lhs





def resolve(arg1: Any = None, arg2: Any = None, /, *,
            dv: Optional[Union[str, List[str]]] = None,
            iv: Optional[Union[str, List[str]]] = None,
            by: Optional[Union[str, List[str]]] = None,
            over: Optional[Union[str, List[str]]] = None,
            data: Optional[pd.DataFrame] = None,
            weights: Optional[str] = None,
            ) -> SyntaxSpec:
    """Resolve any supported calling convention into a SyntaxSpec.

    Backward-compatible module-level wrapper around
    :meth:`SyntaxSpec.from_args`. Prefer calling ``SyntaxSpec.from_args()``
    (or the subclass factory) directly in new code.

    Parameters
    ----------
    arg1 : various
        See :meth:`SyntaxSpec.from_args`.
    arg2 : pd.DataFrame or None
        See :meth:`SyntaxSpec.from_args`.
    dv, iv, by, over, data, weights
        See :meth:`SyntaxSpec.from_args`.

    Returns
    -------
    SyntaxSpec
    """
    return SyntaxSpec.from_args(
        arg1, arg2,
        dv=dv, iv=iv, by=by, over=over,
        data=data, weights=weights,
    )


def _resolve_groupby(
    groupby_obj: Any,
    weights: Optional[str] = None,
    cls: type = None,
) -> SyntaxSpec:
    """Resolve a pandas GroupBy object into a SyntaxSpec.

    Extracts group key names and reconstructs the underlying data as a
    DataFrame containing both the DV column(s) and group column(s).

    Parameters
    ----------
    groupby_obj : SeriesGroupBy or DataFrameGroupBy
        The grouped pandas object.
    weights : str or None
        Column name for observation weights.
    cls : type, optional
        The class to instantiate. Defaults to SyntaxSpec.

    Returns
    -------
    SyntaxSpec (or subclass)
    """
    if cls is None:
        cls = SyntaxSpec
    # Extract group variable name(s)
    group_keys = groupby_obj.keys
    if isinstance(group_keys, list):
        by_names = group_keys
    else:
        by_names = [group_keys]

    # Extract DV name(s) and build source DataFrame
    if isinstance(groupby_obj, pd.core.groupby.SeriesGroupBy):
        dv_name = groupby_obj.obj.name if groupby_obj.obj.name is not None else "value"
        dv_names = [dv_name]
        # Start with the DV Series as a DataFrame
        source_df = groupby_obj.obj.to_frame()
    else:
        dv_names = [col for col in groupby_obj.obj.columns if col not in by_names]
        source_df = groupby_obj.obj.copy()

    # Add missing group columns from the grouper internals
    for key in by_names:
        if key not in source_df.columns:
            group_values = _extract_group_column(groupby_obj, key)
            if group_values is not None:
                source_df[key] = group_values

    return cls(
        dv=dv_names,
        by=by_names,
        data=source_df,
        weights=weights,
    )


def _extract_group_column(groupby_obj: Any, key: str) -> Optional[Any]:
    """Extract a group column's values from a GroupBy object's internal grouper.

    Tries multiple pandas internal APIs for compatibility across versions.

    Parameters
    ----------
    groupby_obj : GroupBy
        The pandas GroupBy object.
    key : str
        The group column name to extract.

    Returns
    -------
    array-like or None
        The group column values, or None if extraction fails.
    """
    # Try pandas 3.x path: _grouper.groupings[].grouping_vector
    grouper = getattr(groupby_obj, '_grouper', None) or getattr(groupby_obj, 'grouper', None)
    if grouper is not None:
        groupings = getattr(grouper, 'groupings', None)
        if groupings is not None:
            for grouping in groupings:
                if getattr(grouping, 'name', None) == key:
                    # Try grouping_vector (pandas 3.x)
                    gv = getattr(grouping, 'grouping_vector', None)
                    if gv is not None:
                        return list(gv)
                    # Try obj.values (older pandas)
                    obj = getattr(grouping, 'obj', None)
                    if obj is not None:
                        if hasattr(obj, 'values'):
                            return obj.values
    return None


def _parse_formula(formula: str, data: pd.DataFrame,
                   weights: Optional[str] = None, cls: type = None, ) -> SyntaxSpec:
    """Parse a formula string into a SyntaxSpec using formulaic's parser.

    Detects the formula operator pattern to determine layout:
    - Single main effect term → cell (by)
    - Multiple main effect terms (connected by +) → marginal (iv)
    - Interaction-only term(s) (:) → cell means (by)
    - Star expansion (* → main effects + interaction) → pivot (by + over)
    - Mixed (main effects AND interactions, not pure star) → sub_specs

    Also validates that all RHS terms are wrapped in C() for descriptive
    statistics (grouping variables must be categorical).

    Parameters
    ----------
    formula : str
        Wilkinson-style formula, e.g., "y ~ C(x)", "y ~ C(x):C(k)", "y ~ C(x)*C(k)".
    data : pd.DataFrame
        Source DataFrame.
    weights : str or None
        Weight column name.
    cls : type, optional
        The class to instantiate. Defaults to SyntaxSpec.

    Returns
    -------
    SyntaxSpec (or subclass)

    Raises
    ------
    ValueError
        If the formula contains terms not wrapped in C(), or if columns
        are not found in the DataFrame.
    """
    if cls is None:
        cls = SyntaxSpec

    from researchpy.containers.multivariable import ModelTerms

    mt = ModelTerms.from_formula(formula)

    dv_names = mt.dv if mt.dv else []
    _validate_columns(dv_names, data, "DV (left side of ~)")

    # --- Validate all RHS terms are wrapped in C() ---
    for term in mt.terms:
        for part in term.term.split(":"):
            if "C(" not in part and part != "Intercept":
                raise ValueError(
                    f"Term '{part}' on the RHS is not wrapped in C(). "
                    f"For descriptive statistics, grouping variables must be categorical. "
                    f"Use: 'y ~ C({part})' to specify '{part}' as a grouping factor."
                )

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
            # Single grouping variable — use by
            return cls(
                dv=dv_names,
                by=var_names,
                data=data,
                formula=formula,
                weights=weights,
            )
        else:
            # Multiple main effects with + → marginal
            return cls(
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

        return cls(
            dv=dv_names,
            by=by_vars,
            data=data,
            formula=formula,
            weights=weights,
        )

    elif len(interaction_terms) > 0 and len(main_effect_terms) > 0:
        # Could be star expansion OR mixed formula.
        # Star expansion: main effects + interaction where the interaction
        # components exactly match the main effects.
        # e.g., "y ~ C(x)*C(k)" expands to C(x) + C(k) + C(x):C(k)

        if _is_star_expansion(main_effect_terms, interaction_terms):
            # Star expansion → pivot (by + over)
            first_interaction = interaction_terms[0]
            interaction_vars = first_interaction.name.split(":")

            by_var = [interaction_vars[0]]
            over_vars = interaction_vars[1:]

            # Collect additional over vars from further interaction terms
            for term in interaction_terms[1:]:
                term_vars = term.name.split(":")
                for v in term_vars[1:]:
                    if v not in over_vars:
                        over_vars.append(v)

            all_vars = by_var + over_vars
            _validate_columns(all_vars, data, "RHS terms")

            return cls(
                dv=dv_names,
                by=by_var,
                over=over_vars,
                data=data,
                formula=formula,
                weights=weights,
            )
        else:
            # Mixed formula → sub_specs
            # Each term becomes a TermSpec, computed independently and stacked
            sub_specs = _build_sub_specs(main_effect_terms, interaction_terms, data)

            return cls(
                dv=dv_names,
                sub_specs=sub_specs,
                data=data,
                formula=formula,
                weights=weights,
            )

    else:
        # No RHS terms — just compute for the DV(s) directly
        return cls(
            dv=dv_names,
            data=data,
            formula=formula,
            weights=weights,
        )


def _is_star_expansion(
    main_effect_terms: List[Any],
    interaction_terms: List[Any],
) -> bool:
    """Detect whether a set of main effects + interactions is a star expansion.

    A star expansion occurs when the main effect variable names exactly match
    the components of the interaction term(s). For example:
    - ``C(x) + C(k) + C(x):C(k)`` is a star expansion of ``C(x)*C(k)``
    - ``C(x) + C(k):C(z)`` is NOT a star expansion

    Parameters
    ----------
    main_effect_terms : list of Term
        The main effect terms from the formula RHS.
    interaction_terms : list of Term
        The interaction terms from the formula RHS.

    Returns
    -------
    bool
    """
    main_names = {t.name for t in main_effect_terms}

    # Check if all interaction components are present as main effects
    for term in interaction_terms:
        interaction_vars = set(term.name.split(":"))
        if not interaction_vars.issubset(main_names):
            return False

    # Check if all main effects appear in at least one interaction
    all_interaction_vars = set()
    for term in interaction_terms:
        all_interaction_vars.update(term.name.split(":"))

    if main_names != all_interaction_vars:
        return False

    return True


def _build_sub_specs(
    main_effect_terms: List[Any],
    interaction_terms: List[Any],
    data: pd.DataFrame,
) -> List[TermSpec]:
    """Build TermSpec list from a mixed formula's terms.

    Main effect terms get layout="iv" (marginal computation).
    Interaction terms get layout="by" (cell computation).

    Parameters
    ----------
    main_effect_terms : list of Term
        Main effect terms.
    interaction_terms : list of Term
        Interaction terms.
    data : pd.DataFrame
        Source data for column validation.

    Returns
    -------
    list of TermSpec
    """
    sub_specs = []

    for term in main_effect_terms:
        variables = [term.name]
        _validate_columns(variables, data, f"term '{term.term}'")
        sub_specs.append(TermSpec(
            term_name=term.name,
            term_raw=term.term,
            layout="iv",
            variables=variables,
        ))

    for term in interaction_terms:
        variables = term.name.split(":")
        _validate_columns(variables, data, f"term '{term.term}'")
        sub_specs.append(TermSpec(
            term_name=term.name,
            term_raw=term.term,
            layout="by",
            variables=variables,
        ))

    return sub_specs


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

