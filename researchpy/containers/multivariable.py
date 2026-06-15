# -*- coding: utf-8 -*-
"""
Result Containers

Standardized dataclass containers for model and test results returned by
researchpy model classes and postestimation utilities.

``ModelResults`` is returned by model ``.results()`` methods (e.g., Regress,
Anova, LogisticRegression).  ``TestResults`` is returned by postestimation
test ``.results()`` methods (e.g., LikelihoodRatioTest, future Wald/contrast
tests).

Both support iteration for tuple-style unpacking::

    # ModelResults
    stats, table, coefs, details = model.results()

    # TestResults
    name, stats, details = lr_test.results()

Or attribute access::

    result = model.results()
    result.fit_statistics
    result.coefficients
"""

from __future__ import annotations


from .base import CoreDataclass
from dataclasses import dataclass, field, fields
from typing import Union, Any, Dict, Optional

import numpy as np
import pandas as pd
import re




@dataclass
class ModelFit(CoreDataclass):
    """

    Standardized container for model fit information.

    The following attributes are defined:
    - ``formula``: The model formula as a string (e.g., "y ~ x1 + x2").
    - ``family``: The model family (e.g., "gaussian", "binomial").
    - ``link``: The link function used in the model (e.g., "identity", "logit").
    - ``n``: The number of observations used to fit the model.
    - ``k``: The number of predictors (including intercept) in the model.
    - ``ci_level``: The confidence interval level used for coefficient estimates (default is 0.95).
    - ``dv``: A list of dependent variable names (optional).
    - ``iv``: A list of independent variable names (optional).
    - ``model_display_name``: A user-friendly name for the model (optional).
    - ``model``: The internal name of the model (optional).

    """

    formula: Optional[str] = None
    family: Optional[str] = None
    model: Optional[str] = None
    model_display_name: Optional[str] = None
    link: Optional[str] = None
    solver_method: Optional[str] = None
    ci_level: Optional[float] = 0.95
    dv_term_names: Optional[list] = None
    iv_term_names: Optional[list] = None
    additional_stats: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        self.__name__ = "Researchpy.ModelFit"



@dataclass
class SolverOptions(CoreDataclass):
    """
    Standardized container for optimization/solver parameters.

    Used by ``GeneralModel`` and its subclasses (``LogisticRegression``,
    ``PoissonRegression``, etc.) to configure the iterative solver.

    Parameters
    ----------
    method : str
        High-level estimation method. One of ``"ols"``, ``"mle"``, ``"irls"``.
        Default is ``"ols"``.
    algorithm : str or None
        Specific optimization algorithm to use (e.g., ``"BFGS"``,
        ``"newton-raphson"``, ``"L-BFGS-B"``). Default is ``None``, meaning
        the model class will choose a sensible default.
    obj_function : str
        Objective function type. One of ``"numeric"``, ``"log-likelihood"``.
        Default is ``"numeric"``.
    tol : float
        Convergence tolerance for the optimizer. Default is ``1e-7``.
    max_iter : int
        Maximum number of iterations allowed. Default is ``300``.
    display : bool
        Whether to print iteration/convergence information during fitting.
        Default is ``True``.
    regularization : str or None
        Type of regularization to apply. One of ``None``, ``"l1"``, ``"l2"``.
        Default is ``None`` (no regularization).
    alpha : float
        Regularization strength (penalty weight). Only used when
        ``regularization`` is not ``None``. Default is ``0.0``.

    Examples
    --------
    >>> from researchpy.core.containerclasses import SolverOptions
    >>> opts = SolverOptions(method="mle", algorithm="BFGS", max_iter=1000)
    >>> opts.algorithm
    'BFGS'

    Users may also pass a plain dictionary to model constructors; the
    constructor will unpack it into a ``SolverOptions`` instance:

    >>> model = LogisticRegression("y ~ x", data=df,
    ...                            solver_options={"algorithm": "BFGS", "max_iter": 500})
    """

    method: str = "ols"
    algorithm: Optional[str] = None
    obj_function: str = "numeric"
    tol: float = 1e-7
    max_iter: int = 300
    display: bool = True
    regularization: Optional[str] = None
    alpha: float = 0.0

    def __post_init__(self):
        self.__name__ = "Researchpy.SolverOptions"

    @classmethod
    def from_dict(cls, d: dict) -> "SolverOptions":
        """Create a SolverOptions instance from a dictionary, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary of solver option key-value pairs. Keys that do not
            correspond to a ``SolverOptions`` field are silently ignored.

        Returns
        -------
        SolverOptions
            A new instance with values from *d* merged over the defaults.
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def with_overrides(self, overrides: dict) -> "SolverOptions":
        """Return a new SolverOptions with selected fields replaced.

        Creates a copy of this instance with any valid keys in *overrides*
        applied on top. Unknown keys are silently ignored.

        Parameters
        ----------
        overrides : dict
            Dictionary of field names to new values.

        Returns
        -------
        SolverOptions
            A new instance with the overrides applied.
        """
        from containers import replace as _replace
        valid_keys = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid_keys}
        return _replace(self, **filtered)

    def to_scipy_options(self) -> dict:
        """Return a dict suitable for ``scipy.optimize.minimize(options=...)``.

        Maps ``max_iter`` → ``maxiter`` and ``tol`` → ``gtol`` as expected by
        scipy's gradient-based optimizers.

        Returns
        -------
        dict
            Options dictionary for scipy.optimize.minimize.
        """
        return {
            "maxiter": self.max_iter,
            "gtol": self.tol,
        }


@dataclass
class FitStatistics(CoreDataclass):

    n: Optional[float] = None
    k: Optional[float] = None

    test_stat_name: Optional[str] = None
    df_model: Optional[float] = None
    df_residual: Optional[float] = None

    test_stat: Optional[float] = None
    test_pval: Optional[float] = None

    r_squared: Optional[float] = None
    r_squared_adj: Optional[float] = None
    r_squared_pseudo: Optional[float] = None
    root_mse: Optional[float] = None

    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    additional_stats: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        self.__name__ = "Researchpy.FitStatistics"




@dataclass
class ModelEffects(CoreDataclass):
    """

    Standardized container for model effects and test statistics.

    The following attributes are defined:
    - ``ss_total``: Total sum of squares (optional).
    - ``ss_model``: Model sum of squares (optional).
    - ``ss_residual``: Residual sum of squares (optional).
    - ``df_model``: Degrees of freedom for the model (optional).
    - ``df_residual``: Degrees of freedom for the residuals (optional).
    - ``df_total``: Total degrees of freedom (optional).
    - ``msr``: Mean square for the model (optional).
        - (LinearModel) Calculated as ``ss_model / df_model``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``mse``: Mean square for the residuals (optional).
        - (LinearModel) Calculated as ``ss_residual / df_residual``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``mst``: Mean square total (optional).
        - (LinearModel) Calculated as ``ss_total / df_total``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``root_mse``: Root mean square error (optional).
        - (LinearModel) Calculated as ``sqrt(mse)``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``test_stat``: Test statistic for the overall model fit (optional).
        - (LinearModel) F-statistic calculated as ``msr / mse``.
        - (MLEModel) Likelihood ratio chi-square statistic.
    - ``test_pval``: P-value for the overall model fit (optional).
        - (LinearModel) P-value for the F-test.
        - (MLEModel) P-value for the likelihood ratio test.
    - ``r_squared``: R-squared value (optional).
        - (LinearModel) Calculated as ``ss_model / ss_total``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``r_squared_adj``: Adjusted R-squared value (optional).
        - (LinearModel) Calculated as ``1 - (df_total / df_residual) * (ss_residual / ss_total)``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``eta_squared``: Eta-squared effect size (optional).
        - (LinearModel) Calculated as ``ss_model / ss_total``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``epsilon_squared``: Epsilon-squared effect size (optional).
        - (LinearModel) Calculated as ``(df_model * (msr - mse)) / ss_total``.
        - (MLEModel) Not applicable, set to ``None``.
    - ``omega_squared``: Omega-squared effect size (optional).
        - (LinearModel) Calculated as ``(df_model * (msr - mse)) / (ss_total + mse)``.
        - (MLEModel) Not applicable, set to ``None``.

    """

    ss_total: Optional[float] = None
    ss_model: Optional[float] = None
    ss_residual: Optional[float] = None
    df_model: Optional[float] = None
    df_residual: Optional[float] = None
    df_total: Optional[float] = None
    msr: Optional[float] = None
    mse: Optional[float] = None
    mst: Optional[float] = None
    root_mse: Optional[float] = None
    test_stat: Optional[float] = None
    test_pval: Optional[float] = None
    r_squared: Optional[float] = None
    r_squared_adj: Optional[float] = None
    eta_squared: Optional[float] = None
    epsilon_squared: Optional[float] = None
    omega_squared: Optional[float] = None

    def __post_init__(self):
        self.__name__ = "Researchpy.ModelEffects"



@dataclass
class FactorEffects(CoreDataclass):
    """
    Standardized container for factor effects.
    """

    ss_type: Optional[Union[str, int]] = None
    source: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    ss: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    df: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    ms: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    test_stat: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    test_pval: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    eta_squared: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    epsilon_squared: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    omega_squared: Optional[Union[np.ndarray, list]] = field(default_factory=list)


    def __post_init__(self):
        self.__name__ = "Researchpy.FactorEffects"



@dataclass
class CoefResults(CoreDataclass):

    term: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    betas: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    std_error: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    test_stat_name: Optional[str] = None
    test_stat: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    test_pval: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    conf_int_lower: Optional[Union[np.ndarray, list]] = field(default_factory=list)
    conf_int_upper: Optional[Union[np.ndarray, list]] = field(default_factory=list)

    def __post_init__(self):
        self.__name__ = "Researchpy.CoefResults"





@dataclass
class ModelResults(CoreDataclass):
    """
    Standardized container for model results.

    Returned by ``Regress.results()``, ``Anova.results()``,
    ``LogisticRegression.results()``, and future model classes.

    Parameters
    ----------
    model_name : str
        Display name of the model (e.g., "Linear Regression (OLS)",
        "Analysis of Variance", "Logistic Regression").
    fit_statistics : DataFrame or dict
        Model fit statistics such as N, R², Root MSE, log-likelihood, etc.
    model_table : DataFrame, dict, or None
        Model-level summary table. For OLS this is the SS/df/MS/F ANOVA
        decomposition; for ANOVA it is the full ANOVA table with factor
        rows and effect sizes. ``None`` for MLE-based models that lack
        a sum-of-squares decomposition (e.g., logistic, Poisson).
    coefficients : DataFrame, dict, or None
        Coefficient / parameter estimate table with columns for estimate,
        standard error, test statistic, p-value, and confidence interval.
        ``None`` when the model's primary output is not a coefficient table
        (e.g., ANOVA where the main table is ``model_table``).
    details : dict or None
        Any additional model-specific information (e.g., type of standard
        errors, odds ratios, convergence info for MLE models). Default is
        ``None``.

    Examples
    --------
    Tuple unpacking (all 5 fields):

    >>> result = model.results()
    >>> name, stats, table, coefs, details = result

    Attribute access:

    >>> result.fit_statistics
    >>> result.coefficients
    """

    model_name: str
    fit_statistics: Union[pd.DataFrame, dict] = None
    model_table: Optional[Union[pd.DataFrame, dict]] = None
    coefficients: Optional[Union[pd.DataFrame, dict]] = None
    details: Optional[dict] = None

    def __post_init__(self):
        self.__name__ = "Researchpy.ModelResults"

    @staticmethod
    def as_dataframe(attribute: str, data: Union[pd.DataFrame, dict, None]) -> Optional[pd.DataFrame]:
        """Convert a ModelResults attribute stored as a dictionary to a DataFrame.

        Each attribute uses a different dictionary-to-DataFrame conversion:

        - ``fit_statistics``: orient="index", single column named "Model Fit"
        - ``model_table``: orient="columns"
        - ``coefficients``: orient="columns"
        - ``details``: orient="index", single column named "Details"

        Parameters
        ----------
        attribute : str
            Name of the ModelResults attribute. Must be one of
            "fit_statistics", "model_table", "coefficients", or "details".
        data : DataFrame, dict, or None
            The value of the attribute to convert.

        Returns
        -------
        pd.DataFrame or None
            The converted DataFrame, or None if *data* is None.

        Raises
        ------
        ValueError
            If *attribute* is not a recognised ModelResults field.
        """
        if data is None:
            return None

        if isinstance(data, pd.DataFrame):
            return data

        conversions = {
            "fit_statistics": lambda d: pd.DataFrame.from_dict(d, orient="index", columns=["Model Fit"]).reset_index(drop=True),
            "model_table": lambda d: pd.DataFrame.from_dict(d, orient="columns"),
            "coefficients": lambda d: pd.DataFrame.from_dict(d, orient="columns"),
            "details": lambda d: pd.DataFrame.from_dict(d, orient="index", columns=["Details"]),
        }

        if attribute not in conversions:
            raise ValueError(
                f"Unknown attribute '{attribute}'. Must be one of: {list(conversions.keys())}"
            )

        return conversions[attribute](data)






@dataclass
class TestResults(CoreDataclass):
    """
    Standardized container for postestimation test results.

    Returned by ``LikelihoodRatioTest.results()`` and future postestimation
    test classes (Wald test, contrast tables, etc.).

    Parameters
    ----------
    test_name : str
        Display name of the test (e.g., "Likelihood Ratio Test").
    statistics : DataFrame or dict
        Test statistics including test statistic value, degrees of freedom,
        and p-value.
    details : dict or None
        Any additional test-specific information (e.g., null model
        coefficients, convergence info). Default is ``None``.

    Examples
    --------
    Tuple unpacking:

    >>> name, stats, details = lr_test.results()

    Attribute access:

    >>> result = lr_test.results()
    >>> result.statistics
    """

    test_name: str
    statistics: Union[pd.DataFrame, dict]
    details: Optional[dict] = None

    def __post_init__(self):
        self.__name__ = "Researchpy.TestResults"




@dataclass
class Term:
    """
    Dataclass for storing information about a single regression model term.

    Attributes
    ----------
    term : str
        The original Patsy term name (e.g., ``"C(drug, Treatment(2))"``).
    name : str or None
        The cleaned term name (e.g., ``"drug"``).  Computed automatically.
    is_factor : bool or list[bool] or None
        Whether the term is categorical.  For interaction terms this is a
        ``list[bool]``, one flag per sub-term.
    is_interaction : bool or None
        Whether the term is an interaction (contains ``":"``).
    columns : list[str]
        Original Patsy column names that belong to this term
        (e.g., ``["C(drug, Treatment(2))[T.1]", ...]``).
    columns_cleaned : list[str]
        Cleaned column names (e.g., ``["1", "3", "4"]``).
    levels : list[str] or list[list[str]] or None
        All category levels (including reference) for factor terms.
        For non-interaction factors: ``["1", "2", "3", "4"]``.
        For interaction terms: list-per-sub-term, ``None`` for continuous
        sub-terms (e.g., ``[["1","2","3","4"], None]``).
        ``None`` for continuous-only terms and Intercept.
    reference : str or list or None
        The reference category for factor terms.
        For non-interaction factors: ``"2"``.
        For interaction terms: list-per-sub-term, ``None`` for continuous
        sub-terms (e.g., ``["2", None]``).
        ``None`` for continuous-only terms and Intercept.

    Examples
    --------
    >>> t = Term("C(drug, Treatment(2))")
    >>> t.name           # "drug"
    >>> t.is_factor      # True

    >>> t = Term("C(drug):disease",
    ...          columns=["C(drug)[T.1]:disease", "C(drug)[T.3]:disease"])
    >>> t.columns_cleaned  # ["1:disease", "3:disease"]
    >>> t.is_factor        # [True, False]
    """

    term: str
    name: Optional[str] = None
    is_interaction: Optional[Union[bool, list]] = None
    is_factor: Optional[Union[bool, list]] = None
    columns: Optional[list] = field(default_factory=list)
    columns_cleaned: Optional[list] = field(default_factory=list)
    levels: Optional[Union[list, None]] = None
    reference: Optional[Union[str, list, None]] = None

    def __post_init__(self):
        self._parts = self.term.split(":")
        self.name = self._clean_term()
        self.is_interaction = self._resolve_is_interaction()
        self.is_factor = self._resolve_is_factor()
        # If columns were provided but not yet cleaned, clean them
        if self.columns and not self.columns_cleaned:
            self.columns_cleaned = [self._clean_column(c) for c in self.columns]

    # ------------------------------------------------------------------ #
    #  Resolve helpers                                                     #
    # ------------------------------------------------------------------ #
    def _resolve_is_interaction(self) -> bool:
        """True when the term contains two or more sub-terms joined by ':'."""
        return len(self._parts) > 1

    def _resolve_is_factor(self) -> Union[bool, list]:
        """Per-sub-term factor check.

        Returns
        -------
        bool
            For simple (non-interaction) terms – ``True`` if the term is
            wrapped in ``C()``.
        list[bool]
            For interaction terms – one flag per sub-term, in order.
        """
        flags = ["C(" in p for p in self._parts]
        return flags if self.is_interaction else flags[0]

    # ------------------------------------------------------------------ #
    #  Cleaning helpers                                                    #
    # ------------------------------------------------------------------ #
    def _clean_term(self) -> str:
        """Clean the Patsy term name.

        ``"C(drug, Treatment(2))"``  →  ``"drug"``
        ``"C(drug):disease"``        →  ``"drug:disease"``
        """
        factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')
        cleaned_factors = [
            ''.join(re.findall(factor_pattern, f)) if "C(" in f else f
            for f in self._parts
        ]
        return ":".join(cleaned_factors)

    @staticmethod
    def _clean_column(column: str) -> str:
        """Clean a single Patsy column name.

        Extracts the level value from bracket notation and strips ``C(…)``
        wrappers.

        ``"C(drug, Treatment(2))[T.3]"``              →  ``"3"``
        ``"C(drug, Treatment(2))[T.1]:disease"``       →  ``"1:disease"``
        ``"disease"``                                  →  ``"disease"``
        ``"Intercept"``                                →  ``"Intercept"``
        """
        level_pattern = re.compile(r'(?<=\[..)(.*?)(?=\])')

        parts = column.split(":")
        cleaned = []
        for part in parts:
            if "C(" in part:
                match = re.findall(level_pattern, part)
                cleaned.append(match[0] if match else part)
            else:
                cleaned.append(part)
        return ":".join(cleaned)


@dataclass
class ModelTerms:
    """
    Container for all terms in a regression model.

    Holds a list of :class:`Term` objects and provides convenient mapping
    properties for translating between original Patsy names and cleaned
    display names.

    Parameters
    ----------
    terms : list[Term]
        List of ``Term`` objects, one per model term.
    dv : list[str] or None
        Dependent variable name(s) extracted from the formula LHS.
        ``None`` when built from ``from_design_info()`` (DV info lives
        separately in that pathway).

    Examples
    --------
    Build from a Patsy design matrix::

        mt = ModelTerms.from_design_info(IV.design_info)
        mt.term_map      # {"C(drug)": "drug", "disease": "disease", …}
        mt.column_map    # {"C(drug)[T.1]": "1", "disease": "disease", …}
        mt["C(drug)"]    # Term object for drug
        mt[0]            # first Term (usually Intercept)

    Build from a formula string (lightweight, no data required)::

        mt = ModelTerms.from_formula("y ~ C(x) + C(a):C(b) + z")
        mt.dv            # ['y']
        mt.terms         # [Term('C(x)'), Term('C(a):C(b)'), Term('z')]
        mt[0].name       # 'x'
        mt[1].is_interaction  # True
    """

    terms: list = field(default_factory=list)   # list[Term]
    dv: Optional[list] = None                   # list[str] — dependent variable names

    def __post_init__(self):
        self.__name__ = "Researchpy.ModelTerms"

    # ------------------------------------------------------------------ #
    #  Factories                                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_formula(cls, formula: str, include_intercept: bool = False) -> "ModelTerms":
        """Parse a formula string into ``ModelTerms`` using Patsy's formula parser.

        This is a lightweight parse that extracts structural information
        (DV names, term names, interaction flags, factor flags) WITHOUT
        requiring data and WITHOUT building a design matrix.

        Parameters
        ----------
        formula : str
            Patsy-style formula string (e.g., ``"y ~ C(x) + C(a):C(b)"``).
        include_intercept : bool, optional
            Whether to include the implicit intercept term. Default is False
            (intercept is excluded since descriptive stats don't use it).

        Returns
        -------
        ModelTerms
            A ``ModelTerms`` instance with ``dv`` populated and ``Term``
            objects for each RHS term. Note: ``columns``, ``levels``, and
            ``reference`` will be empty/None since no data is available.

        Raises
        ------
        ValueError
            If the formula does not contain '~'.

        Examples
        --------
        >>> mt = ModelTerms.from_formula("y ~ C(gender) + C(drug):C(dose)")
        >>> mt.dv
        ['y']
        >>> mt[0].name
        'gender'
        >>> mt[0].is_factor
        True
        >>> mt[1].name
        'drug:dose'
        >>> mt[1].is_interaction
        True
        >>> mt[1].is_factor
        [True, True]
        """
        import patsy

        if "~" not in formula:
            raise ValueError(
                f"Formula must contain '~' separating dependent and independent "
                f"variables. Got: '{formula}'. Example: 'y ~ C(x)'."
            )

        desc = patsy.ModelDesc.from_formula(formula)

        # --- Extract DV names from LHS ---
        dv_names = []
        for lhs_term in desc.lhs_termlist:
            for factor in lhs_term.factors:
                dv_names.append(factor.code)

        # --- Build Term objects from RHS ---
        terms = []
        for rhs_term in desc.rhs_termlist:
            # Skip intercept (empty factors list) unless requested
            if not rhs_term.factors:
                if include_intercept:
                    terms.append(Term(term="Intercept"))
                continue

            # Build the term string in Patsy notation: "C(x):C(k)" or "z"
            term_str = ":".join(f.code for f in rhs_term.factors)

            # Term.__post_init__ handles: name, is_interaction, is_factor
            term_obj = Term(term=term_str)
            terms.append(term_obj)

        return cls(terms=terms, dv=dv_names)
    @classmethod
    def from_design_info(cls, design_info) -> "ModelTerms":
        """Build ``ModelTerms`` from a Patsy ``DesignInfo`` object.

        Uses ``design_info.term_name_slices`` to reliably map each
        column name to its parent term, and ``design_info.factor_infos``
        to determine all category levels and reference categories for
        factor terms.

        Parameters
        ----------
        design_info : patsy.DesignInfo
            Typically ``IV.design_info`` from a Patsy design matrix.

        Returns
        -------
        ModelTerms
        """
        term_names = design_info.term_names
        column_names = list(design_info.column_names)
        term_slices = design_info.term_name_slices

        # Map factor expression code → all categorical levels (as strings)
        factor_cats = {}
        for factor, finfo in design_info.factor_infos.items():
            if finfo.type == "categorical" and finfo.categories is not None:
                factor_cats[factor.code] = [str(c) for c in finfo.categories]

        terms = []
        for patsy_term, t_name in zip(design_info.terms, term_names):
            slc = term_slices[t_name]
            t_columns = column_names[slc]

            # Build the Term (columns_cleaned is computed in __post_init__)
            term_obj = Term(term=t_name, columns=t_columns)

            # Determine levels and reference for factor terms
            cat_factors = [f for f in patsy_term.factors
                           if f.code in factor_cats]

            if cat_factors:
                sub_parts = t_name.split(":")
                is_intx = len(sub_parts) > 1

                # Build a quick lookup: factor_code → categories
                code_to_cats = {f.code: factor_cats[f.code]
                                for f in cat_factors}

                part_levels = []
                part_refs = []

                for i, sub in enumerate(sub_parts):
                    if sub in code_to_cats:
                        all_cats = code_to_cats[sub]

                        # Extract the levels that actually appear in
                        # columns_cleaned at position i
                        appearing = set()
                        for cc in term_obj.columns_cleaned:
                            cc_parts = cc.split(":")
                            if i < len(cc_parts):
                                appearing.add(cc_parts[i])

                        # Reference = categories NOT in the appearing set
                        ref = [c for c in all_cats if c not in appearing]

                        part_levels.append(all_cats)
                        part_refs.append(
                            ref[0] if len(ref) == 1
                            else (ref if ref else None)
                        )
                    else:
                        # Continuous sub-term in an interaction
                        part_levels.append(None)
                        part_refs.append(None)

                # Flatten for non-interaction terms
                if is_intx:
                    term_obj.levels = part_levels
                    term_obj.reference = part_refs
                else:
                    term_obj.levels = part_levels[0]
                    term_obj.reference = part_refs[0]

            terms.append(term_obj)

        return cls(terms=terms)

    # ------------------------------------------------------------------ #
    #  Mapping properties                                                  #
    # ------------------------------------------------------------------ #
    @property
    def term_map(self) -> dict:
        """Original Patsy term name → cleaned term name.

        Example: ``{"C(drug, Treatment(2))": "drug", "disease": "disease"}``
        """
        return {t.term: t.name for t in self.terms}

    @property
    def column_map(self) -> dict:
        """Original Patsy column name → cleaned column name.

        Example: ``{"C(drug, Treatment(2))[T.3]": "3", "disease": "disease"}``
        """
        mapping = {}
        for t in self.terms:
            for orig, clean in zip(t.columns, t.columns_cleaned):
                mapping[orig] = clean
        return mapping

    # ------------------------------------------------------------------ #
    #  Container protocol                                                  #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key):
        """Look up a Term by integer index, original term name, or cleaned name."""
        if isinstance(key, int):
            return self.terms[key]
        for t in self.terms:
            if t.term == key or t.name == key:
                return t
        raise KeyError(f"Term '{key}' not found")

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        lines = [f"ModelTerms({len(self.terms)} terms)"]
        for t in self.terms:
            lines.append(f"  {t.term!r} → {t.name!r}  "
                         f"(factor={t.is_factor}, intx={t.is_interaction}, "
                         f"cols={len(t.columns)})")
        return "\n".join(lines)
