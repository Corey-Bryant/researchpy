"""
FormulaSyntax — the universal input parsing layer.

Every researchpy function (descriptive, inferential, modeling) calls
``FormulaSyntax.from_args()`` (or its subclass override) first to normalize
any of the supported calling conventions into a single ``FormulaSyntax``
dataclass. The computation engine then operates exclusively on the spec.

Subclasses (e.g., AnovaSpec, TTestSpec) override ``from_args()`` to add
domain-specific validation while inheriting the full parsing logic.

Supported calling conventions:
    1. SyntaxLayer(df[['y']])                          -> DataFrame/array, no groups
       SyntaxLayer(df[['y', 'z']])                     -> DataFrame/array, no groups (if array has c>1 then compute for each c)
    2. SyntaxLayer(["y", "k", "c"], df)                -> column list + DataFrame
    3. SyntaxLayer(df['y'])                            -> Series/array, no groups
       SyntaxLayer('y', df)                            -> Series/array, no groups
    4. SyntaxLayer("y ~ C(x)", df)                     -> formula string + DataFrame
       SyntaxLayer("y ~ x", df)                        -> formula string + DataFrame
    5. SyntaxLayer(dv="y", by="x", data=df)            -> explicit keywords (cell grouping)
       SyntaxLayer(dv="y", iv=["x","k"], data=df)      -> explicit keywords (marginal) (results stacked)
       SyntaxLayer(dv="y", by="x", over="k", data=df)  -> pivot layout
    6. SyntaxLayer("y ~ C(x):C(k)", df)                -> cell means (MultiIndex rows)
       SyntaxLayer("y ~ C(x)*C(k)", df)                -> pivot layout
       SyntaxLayer("y ~ C(x) + C(k) + C(k):C(z)", data=df)      -> explicit keywords (marginal) (results stacked)

Formula operator semantics for descriptive stats:
    + : marginal (compute separately for each factor, stack results)
    : : cell (compute for each unique combination, MultiIndex rows)
    * : pivot (first factor → rows, second → columns)

Reverse mapping (to_formula):
    dv="y", iv=["x","k"]         → "dv ~ C(x) + C(k)"
    dv="y", by=["x"]             → "dv ~ C(x)"
    dv="y", by=["x","k"]         → "dv ~ C(x):C(k)"
    dv="y", by=["x"], over=["k"] → "dv ~ C(x)*C(k)"

    iv=["x","k"]         → "C(x) + C(k)"
    by=["x"]             → "C(x)"
    by=["x","k"]         → "C(x):C(k)"
    by=["x"], over=["k"] → "C(x)*C(k)"
"""
from dataclasses import dataclass, field
from typing import Union, Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import itertools
import re


from researchpy.engine.table import TableTermSpec
from researchpy.containers import CoreDataclass, ModelTerms

from formulaic import Formula
from formulaic.parser import DefaultFormulaParser



@dataclass
class SyntaxLayer(CoreDataclass):

    DV: List[str] = field(default_factory=list)
    IV: Optional[List[str]] = None
    by: Optional[List[str]] = None
    over: Optional[List[str]] = None
    data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None
    weights: Optional[str] = None
    sub_specs: Optional[List[TableTermSpec]] = None


    def __post_init__(self):
        # Validate that IV and by/over are mutually exclusive
        if self.IV is not None and (self.by is not None or self.over is not None):
            raise ValueError(
                    "Cannot use 'IV' together with 'by' or 'over'. "
                    "'IV' produces marginal (stacked) results for each variable independently. "
                    "'by'/'over' produce cell means or pivot tables. Use one approach or the other."
            )



    # ------------------------------------------------------------------
    # Factory classmethod — the primary entry point for resolution
    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, arg1: Any = None, arg2: Any = None, /, *,
                  dv: Optional[Union[str, List[str]]] = None, iv: Optional[Union[str, List[str]]] = None,
                  by: Optional[Union[str, List[str]]] = None, over: Optional[Union[str, List[str]]] = None,
                  data: Optional[pd.DataFrame] = None,
                  weights: Optional[str] = None, ) -> "SyntaxLayer":
        """Resolve any supported calling convention into a SyntaxLayer.

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
        SyntaxLayer (or subclass instance)
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
        >>> spec = SyntaxLayer.from_args(df['y'])
        >>> spec.dv, spec.by
        (['y'], None)

        # Convention 2: Formula
        >>> spec = SyntaxLayer.from_args("y ~ C(x)", df)
        >>> spec.dv, spec.by
        (['y'], ['x'])

        # Convention 4: Keywords (marginal)
        >>> spec = SyntaxLayer.from_args(dv="y", iv=["x", "k"], data=df)
        >>> spec.iv
        ['x', 'k']

        # Convention 4: Keywords (pivot)
        >>> spec = SyntaxLayer.from_args(dv="y", by="x", over="k", data=df)
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
                    "When using keyword arguments (dv=, iv=, by=, over=), 'data' must be provided."
                )

            formula_like = cls._args_to_formula(dv=dv, iv=iv, by=by, over=over)

            return cls(
                    dv=dv,
                    iv=iv,
                    by=by,
                    over=over,
                    data=resolved_data,
                    formula=formula_like,
                    weights=weights,
            )


        # --- Convention 2: Formula string ---
        if isinstance(arg1, str):
            if resolved_data is None:
                raise ValueError(
                    f"Formula '{arg1}' requires a DataFrame. "
                    f"Pass it as the second positional argument or use data=."
                )

            #spec = _parse_formula(arg1, resolved_data, weights, cls)
            #return spec


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



    @staticmethod
    def from_formula(formula: str, data: object = {}, output: str = "numpy",
                     include_intercept: bool = True, ensure_full_rank: bool = True, **kwargs: object, ):

        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = Formula(formula,
                     _parser=DefaultFormulaParser(include_intercept=include_intercept),
                     **kwargs,
                     ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)

        # Extract what we need
        DV = mm.lhs  # Maintains a formuliac.model_spec.ModelSpec
        IV = mm.rhs  # Maintains a formuliac.model_spec.ModelSpec
        model_terms = {"dv": ModelTerms.from_model_spec(mm.lhs.model_spec),
                       "iv": ModelTerms.from_model_spec(mm.rhs.model_spec)
                       }

        return DV, IV, model_terms


    @staticmethod
    def _args_to_formula(dv: Optional[List[str]] = None, iv: Optional[List[str]] = None, by: Optional[List[str]] = None,
                         over: Optional[List[str]] = None, ) -> Optional[str]:
        """Build a formula string from keyword spec parameters.

        Used internally by ``from_args()`` to generate a formula for specs
        created via keyword arguments (Convention 4).

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
        rhs = []

        if iv:
            rhs.extend(f"C({col})" for col in iv)

        if by:
            by_parts = [f"C({col})" for col in by]
            if over:
                over_parts = [f"C({col})" for col in over]
                # Star expansion: by * over
                all_parts = by_parts + over_parts
                rhs.append("*".join(all_parts))
            else:
                # Cell means: colon-joined
                rhs.append(":".join(by_parts))

        if not rhs:
            return lhs

        return f"{lhs} ~ {' + '.join(rhs)}"





# _parse_formula shouldn't force as_factor (i.e. C(term)), it should just parse the formulaic.Formula as is
#  Create 2 new methods: (1) that converts all rhs variables to continuous (i.e. checks/strips "C()" from around term
#                        (2) that converts all rhs variables to categorical (i.e. wraps "C()" around term if not already wrapped)
def _parse_formula(formula: str, data: pd.DataFrame, weights: Optional[str] = None, cls: type = None,) -> SyntaxLayer:
    """Parse a formula string into a SyntaxLayer using formulaic's parser.

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
        The class to instantiate. Defaults to SyntaxLayer.

    Returns
    -------
    SyntaxLayer (or subclass)

    Raises
    ------
    ValueError
        If the formula contains terms not wrapped in C(), or if columns
        are not found in the DataFrame.
    """
    if cls is None:
        cls = SyntaxLayer

    from researchpy.containers.multivariable import ModelTerms

    mt = ModelTerms.from_formula(formula)

    dv_names = mt.dv if mt.dv else []
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



def _is_star_expansion(main_effect_terms: List[Any], interaction_terms: List[Any], ) -> bool:
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


def _build_sub_specs(main_effect_terms: List[Any], interaction_terms: List[Any], data: pd.DataFrame,) -> List[TableTermSpec]:
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
        sub_specs.append(TableTermSpec(
                term=term.term,
                name=term.name,
                layout="iv",
                variables=variables,)
        )


    for term in interaction_terms:
        variables = term.name.split(":")
        sub_specs.append(TableTermSpec(
                term=term.term,
                name=term.name,
                layout="by",
                variables=variables,)
        )

    return sub_specs