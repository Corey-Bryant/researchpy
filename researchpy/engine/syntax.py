"""
FormulaSpec — the universal input parsing layer.

Every researchpy function (descriptive, inferential, modeling) calls
``FormulaSpec.from_args()`` (or its subclass override) first to normalize
any of the supported calling conventions into a single ``FormulaSpec``
dataclass.  The computation engine then operates exclusively on the spec.

Subclasses (e.g., AnovaSpec, TTestSpec) override ``from_args()`` to add
domain-specific validation while inheriting the full parsing logic.

Supported calling conventions
-----------------------------
1. FormulaSpec(df[['y']])                          -> DataFrame/array, no groups
   FormulaSpec(df[['y', 'z']])                     -> DataFrame/array, no groups (compute for each column)
2. FormulaSpec(["y", "k", "c"], df)                -> column list + DataFrame
3. FormulaSpec(df['y'])                            -> Series/array, no groups
   FormulaSpec('y', df)                            -> Series/array, no groups
4. FormulaSpec("y ~ C(x)", df)                     -> formula string + DataFrame
   FormulaSpec("y ~ x", df)                        -> formula string + DataFrame
5. FormulaSpec(dv="y", by="x", data=df)            -> explicit keywords (cell grouping)
   FormulaSpec(dv="y", iv=["x","k"], data=df)      -> explicit keywords (marginal) (results stacked)
   FormulaSpec(dv="y", by="x", over="k", data=df)  -> pivot layout
6. FormulaSpec("y ~ C(x):C(k)", df)                -> cell means (MultiIndex rows)
   FormulaSpec("y ~ C(x)*C(k)", df)                -> pivot layout
   FormulaSpec("y ~ C(x) + C(k) + C(k):C(z)", data=df)  -> mixed (sub_specs)

Formula operator semantics for descriptive stats
-------------------------------------------------
+ : marginal (compute separately for each factor, stack results)
: : cell (compute for each unique combination, MultiIndex rows)
* : pivot (first factor → rows, second → columns)

Reverse mapping (to_formula)
-----------------------------
dv="y", iv=["x","k"]         → "y ~ C(x) + C(k)"
dv="y", by=["x"]             → "y ~ C(x)"
dv="y", by=["x","k"]         → "y ~ C(x):C(k)"
dv="y", by=["x"], over=["k"] → "y ~ C(x)*C(k)"

iv=["x","k"]         → "C(x) + C(k)"
by=["x"]             → "C(x)"
by=["x","k"]         → "C(x):C(k)"
by=["x"], over=["k"] → "C(x)*C(k)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from researchpy.containers import CoreDataclass, ModelTerms
from researchpy.engine.table import TableTermSpec


# ======================================================================
# Main dataclass
# ======================================================================

@dataclass
class FormulaSpec(CoreDataclass):
    """Normalized computation specification.

    All fields are populated by :meth:`from_args` or :meth:`from_formula`.
    Downstream engines (matrix, table, statistics) consume these fields
    without caring which calling convention was originally used.

    Attributes
    ----------
    DV : list of str
        Dependent variable column name(s).
    IV : list of str or None
        Independent variables for marginal (stacked) computation.
        Mutually exclusive with *by* / *over*.
    by : list of str or None
        Row grouping variable(s) for cell means or pivot tables.
    over : list of str or None
        Column grouping variable(s) for pivot layout.  Requires *by*.
    data : pd.DataFrame or None
        Source DataFrame.
    formula : str or None
        Canonical formula string (generated or user-supplied).
    weights : str or None
        Column name for observation weights.
    sub_specs : list of TableTermSpec or None
        Per-term layout specs for mixed formulas that contain both
        main-effect and interaction terms that don't form a pure star
        expansion.
    """

    DV: List[str] = field(default_factory=list)
    IV: Optional[List[str]] = None
    by: Optional[List[str]] = None
    over: Optional[List[str]] = None
    data: Optional[Any] = None          # pd.DataFrame at runtime
    formula: Optional[str] = None
    weights: Optional[str] = None
    sub_specs: Optional[List[TableTermSpec]] = None


    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        super().__post_init__()
        self.__name__ = "Researchpy.FormulaSpec"
        # --- Ensure formula always exists ---
        if self.formula is None:
            self.formula = _args_to_formula(dv=self.DV, iv=self.IV, by=self.by, over=self.over)


    # ==================================================================
    # Factory classmethod — the primary entry point
    # ==================================================================
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
        data: Optional[Any] = None,
        weights: Optional[str] = None,
    ) -> "FormulaSpec":
        """Resolve any supported calling convention into a *FormulaSpec*.

        This is the universal input gate.  Every researchpy function calls
        this (or its subclass override) first to normalize user input.

        Parameters
        ----------
        arg1 : various
            Positional argument 1.  Can be:

            - ``pd.DataFrame`` → compute for its columns (Convention 1)
            - ``pd.Series`` / ``np.ndarray`` / ``list`` / ``tuple`` →
              direct data (Convention 1/3)
            - ``str`` → formula string (Convention 4); requires *arg2* or
              *data* to be a DataFrame
            - ``list[str]`` → column names (Convention 2); requires *arg2*
              or *data*

        arg2 : pd.DataFrame or None
            Positional argument 2.  DataFrame when *arg1* is a formula or
            column list.
        dv, iv, by, over, data, weights
            Keyword-only arguments for Convention 5.

        Returns
        -------
        FormulaSpec (or subclass instance)

        Raises
        ------
        ValueError
            If the input cannot be resolved, or if iv is used with by/over.
        TypeError
            If *arg1* is an unsupported type.
        """
        # -- Resolve data source (positional takes precedence) --
        resolved_data = arg2 if arg2 is not None else data

        # -- Normalize scalars to lists --
        if isinstance(dv, str):
            dv = [dv]
        if isinstance(iv, str):
            iv = [iv]
        if isinstance(by, str):
            by = [by]
        if isinstance(over, str):
            over = [over]

        # -- Validate mutual exclusivity: iv XOR (by/over) --
        if iv is not None and (by is not None or over is not None):
            raise ValueError(
                "Cannot use 'iv' together with 'by' or 'over'. "
                "'iv' produces marginal (stacked) results for each variable independently. "
                "'by'/'over' produce cell means or pivot tables. Use one approach or the other."
            )

        # -- Validate over requires by --
        if over is not None and by is None:
            raise ValueError(
                "'over' requires 'by' to also be specified. "
                "'over' defines the column index and 'by' defines the row index of a pivot table."
            )

        # =============================================================
        # Convention 5: Explicit keywords (dv=, iv=/by=/over=, data=)
        # =============================================================
        if dv is not None:
            if resolved_data is None:
                raise ValueError(
                    "When using keyword arguments (dv=, iv=, by=, over=), "
                    "'data' must be provided."
                )
            _validate_columns(dv, resolved_data, "dv")
            if iv:
                _validate_columns(iv, resolved_data, "iv")
            if by:
                _validate_columns(by, resolved_data, "by")
            if over:
                _validate_columns(over, resolved_data, "over")

            formula_str = _args_to_formula(dv=dv, iv=iv, by=by, over=over)

            return cls(
                DV=dv,
                IV=iv,
                by=by,
                over=over,
                data=resolved_data,
                formula=formula_str,
                weights=weights,
            )

        # =============================================================
        # Convention 4: Formula string
        # =============================================================
        if isinstance(arg1, str):
            if resolved_data is None:
                raise ValueError(
                    f"Formula '{arg1}' requires a DataFrame. "
                    f"Pass it as the second positional argument or use data=."
                )

            return _parse_formula(arg1, resolved_data, weights=weights, cls=cls)

        # =============================================================
        # Convention 2: List of column names
        # =============================================================
        if isinstance(arg1, list) and all(isinstance(x, str) for x in arg1):
            if resolved_data is None:
                raise ValueError(
                    "Column name list requires a DataFrame. "
                    "Pass it as the second positional argument or use data=."
                )
            _validate_columns(arg1, resolved_data, "column names")

            return cls(
                DV=arg1,
                data=resolved_data,
                weights=weights,
            )

        # =============================================================
        # Convention 1: DataFrame (multi-column)
        # =============================================================
        if isinstance(arg1, pd.DataFrame):
            return cls(
                DV=list(arg1.columns),
                data=arg1,
                weights=weights,
            )

        # =============================================================
        # Convention 3: Series
        # =============================================================
        if isinstance(arg1, pd.Series):
            col_name = arg1.name if arg1.name is not None else "value"
            df = arg1.to_frame(name=col_name)
            return cls(
                DV=[col_name],
                data=df,
                weights=weights,
            )

        # =============================================================
        # Convention 1/3: ndarray, list, tuple (raw values)
        # =============================================================
        if isinstance(arg1, (np.ndarray, list, tuple)):
            arr = np.asarray(arg1)

            if arr.ndim == 1:
                df = pd.DataFrame({"value": arr})
                return cls(DV=["value"], data=df, weights=weights)

            else:
                col_names = [f"col_{i}" for i in range(arr.shape[1])]
                df = pd.DataFrame(arr, columns=col_names)
                return cls(DV=col_names, data=df, weights=weights)

        # =============================================================
        # Unsupported
        # =============================================================
        if arg1 is None and dv is None:
            raise ValueError(
                "No input provided.  Supply a DataFrame, Series, formula string, "
                "column list, or keyword arguments (dv=, data=)."
            )

        raise TypeError(
            f"Unsupported type for first argument: {type(arg1).__name__}. "
            f"Expected str, list, pd.Series, pd.DataFrame, np.ndarray, list, or tuple."
        )


# ======================================================================
# Formula construction: keywords → formula string
# ======================================================================
def _args_to_formula(
    *,
    dv: Optional[List[str]] = None,
    iv: Optional[List[str]] = None,
    by: Optional[List[str]] = None,
    over: Optional[List[str]] = None,
) -> Optional[str]:
    """Build a formula string from keyword spec parameters.

    Reverse mapping
    ---------------
    dv="y", iv=["x","k"]         → "y ~ C(x) + C(k)"
    dv="y", by=["x"]             → "y ~ C(x)"
    dv="y", by=["x","k"]         → "y ~ C(x):C(k)"
    dv="y", by=["x"], over=["k"] → "y ~ C(x)*C(k)"

    Parameters
    ----------
    dv : list of str or None
        Dependent variable name(s).
    iv, by, over : list of str or None
        Grouping parameters.

    Returns
    -------
    str or None
        Generated formula string, or None if no DV is provided.
    """
    if not dv:
        return None

    lhs = dv[0]
    rhs_parts: List[str] = []

    if iv:
        # Marginal: each IV as a separate main effect
        rhs_parts.extend(f"C({col})" for col in iv)

    if by:
        by_terms = [f"C({col})" for col in by]
        if over:
            # Star expansion: by * over
            over_terms = [f"C({col})" for col in over]
            rhs_parts.append("*".join(by_terms + over_terms))
        else:
            # Cell means: colon-joined
            rhs_parts.append(":".join(by_terms))

    if not rhs_parts:
        return lhs

    return f"{lhs} ~ {' + '.join(rhs_parts)}"


# ======================================================================
# Formula parsing: formula string → FormulaSpec
# ======================================================================
def _parse_formula(
    formula: str,
    data: Any,
    weights: Optional[str] = None,
    cls: Optional[type] = None,
) -> "FormulaSpec":
    """Parse a formula string into a *FormulaSpec* using formulaic's parser.

    Detects the formula operator pattern to determine layout:

    - Single main effect term → cell (*by*)
    - Multiple main effect terms (connected by ``+``) → marginal (*IV*)
    - Interaction-only term(s) (``:``) → cell means (*by*)
    - Star expansion (``*`` → main effects + interaction) → pivot (*by* + *over*)
    - Mixed (main effects AND interactions, not pure star) → *sub_specs*

    Parameters
    ----------
    formula : str
        Wilkinson-style formula, e.g., ``"y ~ C(x)"``,
        ``"y ~ C(x):C(k)"``, ``"y ~ C(x)*C(k)"``.
    data : pd.DataFrame
        Source DataFrame.
    weights : str or None
        Weight column name.
    cls : type, optional
        Class to instantiate.  Defaults to :class:`FormulaSpec`.

    Returns
    -------
    FormulaSpec (or subclass)
    """
    if cls is None:
        cls = FormulaSpec

    mt = ModelTerms.from_formula(formula)

    dv_names: List[str] = []            # Extract DV names from the LHS of the formula
    main_effect_terms: List[Any] = []   # Categorize RHS terms into main effects vs interactions
    interaction_terms: List[Any] = []   # Categorize RHS terms into main effects vs interactions
    all_rhs_vars: List[str] = []        # Validate that all RHS variable names exist in data


    # Extract DV names from the LHS of the formula
    if mt.lhs:
        dv_names.extend(v.name.strip() for v in mt.lhs)

        # Validate DV columns exist in data
        if dv_names:
            _validate_columns(dv_names, data, "formula LHS (dependent variable(s))")


    # Categorize RHS terms into main effects vs interactions
    if mt.rhs:
        for term in mt.rhs:
            if term.name == "1" or term.name.lower() == "intercept":
                continue

            if term.is_interaction:
                interaction_terms.append(term)
            else:
                main_effect_terms.append(term)

            for var in term.name.split(":"):
                if var not in all_rhs_vars:
                    all_rhs_vars.append(var)

        if all_rhs_vars:
            _validate_columns(all_rhs_vars, data, "formula RHS variable(s)")


    if (not mt.lhs and not mt.rhs) and mt.terms:
        all_terms = []

        for term in mt.terms:
            if term.name == "1" or term.name == "0" or term.name.lower() == "intercept":
                continue

            if term.is_interaction:
                interaction_terms.append(term)
            else:
                main_effect_terms.append(term)

            for var in term.name.split(":"):
                if var not in all_terms:
                    all_terms.append(var)

        if all_terms:
            _validate_columns(all_terms, data, "formula variable(s) (no RHS or LHS)")


    if weights:
        _validate_columns([weights], data, "formula weights")



    # =================================================================
    # Determine layout based on term pattern
    # =================================================================

    # --- No RHS terms: just DV(s) ---
    if not main_effect_terms and not interaction_terms:
        return cls(
            DV=dv_names,
            data=data,
            formula=formula,
            weights=weights,
        )

    # --- Only main effects ---
    if not interaction_terms and main_effect_terms:
        var_names = [t.name for t in main_effect_terms]

        if len(var_names) == 1:
            # Single grouping variable → by
            return cls(
                DV=dv_names,
                by=var_names,
                data=data,
                formula=formula,
                weights=weights,
            )
        else:
            # Multiple main effects with + → marginal (IV)
            return cls(
                DV=dv_names,
                IV=var_names,
                data=data,
                formula=formula,
                weights=weights,
            )

    # --- Only interactions ---
    if interaction_terms and not main_effect_terms:
        by_vars: List[str] = []
        for term in interaction_terms:
            for var in term.name.split(":"):
                if var not in by_vars:
                    by_vars.append(var)
        return cls(
            DV=dv_names,
            by=by_vars,
            data=data,
            formula=formula,
            weights=weights,
        )

    # --- Both main effects and interactions ---
    if _is_star_expansion(main_effect_terms, interaction_terms):
        # Star expansion → pivot (by + over)
        first_intx = interaction_terms[0]
        intx_vars = first_intx.name.split(":")

        by_var = [intx_vars[0]]
        over_vars = list(intx_vars[1:])

        # Gather additional over vars from further interactions
        for term in interaction_terms[1:]:
            for v in term.name.split(":")[1:]:
                if v not in over_vars:
                    over_vars.append(v)

        return cls(
            DV=dv_names,
            by=by_var,
            over=over_vars,
            data=data,
            formula=formula,
            weights=weights,
        )
    else:
        # Mixed formula → sub_specs
        sub_specs = _build_sub_specs(main_effect_terms, interaction_terms)
        return cls(
            DV=dv_names,
            sub_specs=sub_specs,
            data=data,
            formula=formula,
            weights=weights,
        )


# ======================================================================
# Formula analysis helpers
# ======================================================================
def _is_star_expansion(
    main_effect_terms: List[Any],
    interaction_terms: List[Any],
) -> bool:
    """Detect whether main effects + interactions form a star expansion.

    A star expansion occurs when the main effect variable names exactly
    match the components of the interaction term(s).

    Examples
    --------
    ``C(x) + C(k) + C(x):C(k)``  →  True  (star expansion of ``C(x)*C(k)``)
    ``C(x) + C(k):C(z)``         →  False
    """
    main_names = {t.name for t in main_effect_terms}

    # Check every interaction's components are present as main effects
    for term in interaction_terms:
        interaction_vars = set(term.name.split(":"))
        if not interaction_vars.issubset(main_names):
            return False

    # Check every main effect appears in at least one interaction
    all_intx_vars: set = set()
    for term in interaction_terms:
        all_intx_vars.update(term.name.split(":"))

    return main_names == all_intx_vars


def _build_sub_specs(
    main_effect_terms: List[Any],
    interaction_terms: List[Any],
) -> List[TableTermSpec]:
    """Build :class:`TableTermSpec` list from a mixed formula's terms.

    Main effect terms get ``layout="iv"`` (marginal computation).
    Interaction terms get ``layout="by"`` (cell computation).
    """
    sub_specs: List[TableTermSpec] = []

    for term in main_effect_terms:
        sub_specs.append(TableTermSpec(
            term=term.term,
            name=term.name,
            layout="iv",
            variables=[term.name],
        ))

    for term in interaction_terms:
        variables = term.name.split(":")
        sub_specs.append(TableTermSpec(
            term=term.term,
            name=term.name,
            layout="by",
            variables=variables,
        ))

    return sub_specs


# ======================================================================
# Validation helper
# ======================================================================

def _validate_columns(
    columns: List[str],
    data: Any,
    label: str,
) -> None:
    """Check that all column names exist in the DataFrame.

    Raises
    ------
    ValueError
        If any column is not found in *data*.
    """
    missing = [c for c in columns if c not in data.columns]
    if missing:
        available = list(data.columns)
        raise ValueError(
            f"Column(s) {missing} specified for {label} not found in DataFrame. "
            f"Available columns: {available}"
        )

