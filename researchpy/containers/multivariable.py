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

import numpy as np
import pandas as pd
import re

from dataclasses import dataclass, field, fields
from typing import Union, Any, Dict, Optional, TYPE_CHECKING

import formulaic

from itertools import product


if TYPE_CHECKING:
    from researchpy.models.families import Family
from researchpy.containers.base import CoreDataclass




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
class ModelDesignSpec(CoreDataclass):
    """

    Standardized container for formulaic model output + Researchpy model fit information.

    The following attributes are defined:
    - ``formula``: The model formula as a string (e.g., "y ~ x1 + x2").
    - ``family``: The distribution family as a ``Family`` instance (e.g., ``GaussianFamily()``, ``BinomialFamily()``).
    - ``link``: The link function used in the model (e.g., "identity", "logit").
    - ``n``: The number of observations used to fit the model.
    - ``k``: The number of predictors (including intercept) in the model.
    - ``ci_level``: The confidence interval level used for coefficient estimates (default is 0.95).
    - ``dv``: A list of dependent variable names (optional).
    - ``iv``: A list of independent variable names (optional).
    - ``model_display_name``: A user-friendly name for the model (optional).
    - ``model``: The internal name of the model (optional).

    """

    # Model specification attributes
    formula: Optional[str] = None
    model_terms: Optional[ModelTerms] = None
    DV: Optional[np.ndarray] = None
    IV: Optional[np.ndarray] = None
    dv_term_names: Optional[list] = None
    iv_term_names: Optional[list] = None
    ci_level: Optional[float] = 0.95
    # The distribution family and link function
    family: Optional[Family] = None
    model: Optional[str] = None
    model_display_name: Optional[str] = None
    link: Optional[str] = None
    # Attributes for model fit and estimation
    solver_options: Optional[SolverOptions] = None
    solver_method: Optional[str] = None
    # Additional information
    additional_stats: Optional[dict] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        self.__name__ = "Researchpy.ModelDesignSpec"


    @classmethod
    def from_formulaic(cls, formula: str, data: object, output: str = "numpy",
                       include_intercept: bool = True, ensure_full_rank: bool = True, **kwargs, ) -> "ModelDesignSpec":
        """Build from a formulaic model_matrix call."""
        if not include_intercept:
            formula = formula + " + 0"

        mm = formulaic.Formula(formula,
                               _parser=formulaic.parser.DefaultFormulaParser(include_intercept=include_intercept),
                               **kwargs,
                               ).get_model_matrix(data, output=output, ensure_full_rank=ensure_full_rank, **kwargs)

        # Extract what we need
        dv_term_names = (
            list(mm.lhs.model_spec.column_names)
            if hasattr(mm.lhs.model_spec, "column_names")
            else [str(t) for t in mm.lhs.model_spec.terms]
        )
        iv_term_names = (
            list(mm.rhs.model_spec.column_names)
            if hasattr(mm.rhs.model_spec, "column_names")
            else [str(t) for t in mm.rhs.model_spec.terms]
        )

        model_terms = ModelTerms.from_model_spec(mm.rhs.model_spec)

        return cls(DV=mm.lhs,
                   IV=mm.rhs,
                   model_terms=model_terms,
                   formula=formula,
                   dv_term_names=dv_term_names,
                   iv_term_names=iv_term_names,
                   metadata={
                                "include_intercept": include_intercept,
                                "ensure_full_rank" : ensure_full_rank,
                                "output"           : output,
                            } | kwargs,
                   )




@dataclass
class SolverOptions(CoreDataclass):
    """
    Standardized container for optimization/solver parameters.

    Used by ``BaseModel`` and is required for all of its subclasses (``LinearModel``, ``Anova``, ``GeneralizedLinearModel``,
    ``LogisticRegression``, ``PoissonRegression``, etc.) to configure the solver (analytical or iterative).

    Parameters
    ----------
    estimation_method : str
        Solver method (a.k.a. estimator or estimation principal), current supported options are ``"ols"`` or ``"mle"``.
        If user specifies an option outside what is currently supported, the user must also provide the algorithm to
        use that is able to be used with scipy.optimize.minimize;
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        for passing custom method.
    obj_function : str
        Objective function type. One of ``"numeric"``, ``"log-likelihood"``.
        Default is ``"numeric"``.
    algorithm : str or None
        Specific optimization algorithm to use (e.g., ``"BFGS"``, ``"newton-raphson"``, ``"L-BFGS-B"``). Default is
        ``None``, meaning the model class will choose a sensible default.
    tol : float
        Convergence tolerance for the coefficient vector. Default is ``1e-6`` unless coefficient estimators programmed with mle then == 1e-4.
    tolerance : float
        Convergence tolerance for the coefficient vector. Default is ``1e-6`` unless coefficient estimators programmed with mle then == 1e-4.
    logtolerance : float
        Convergence tolerance for the log likelihood. Default is ``1e-7`` unless log likelhood estimators programmed with mle then == 0
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
    >>> from researchpy.containers.multivariable import SolverOptions
    >>> opts = SolverOptions(estimation_method="mle", algorithm="BFGS", max_iter=1000)
    >>> opts.algorithm
    'BFGS'

    Users may also pass a plain dictionary to model constructors; the
    constructor will unpack it into a ``SolverOptions`` instance:

    >>> model = LogisticRegression("y ~ x", data=df,
    ...                            SolverOptions={"algorithm": "BFGS", "max_iter": 500})
    """

    estimation_method: str
    obj_function: str

    # Irrelevant for OLS
    algorithm: Optional[str] = None       # Defaults to "IRLS" if estimation_method == "mle", otherwise defaults to "numeric" for OLS
    tol: float = 1e-6               # The default in Stata 19, unless coefficient estimators programmed with ml then == 1e-4
    tolerance: float = 1e-6         # The default in Stata 19, unless coefficient estimators programmed with ml then == 1e-4
    logtolerance: float = 1e-7      # The default in Stata 19, unless log likelhood estimators programmed with ml then == 0
    max_iter: int = 300
    regularization: Optional[str] = None
    alpha: float = 0.0
    display: bool = True

    def __post_init__(self):
        self.__name__ = "Researchpy.SolverOptions"

        # -- Setting default solver algorithm and objective_function options based on estimation method provided -- #
        if self.estimation_method.lower() in ['maximum_likelihood', 'maximum likelihood estimation', 'mle']:
            if self.algorithm is None: self.algorithm = 'IRLS'
            if self.obj_function is None: self.obj_function = 'log-likelihood'
            self.tol = 1e-4
            self.tolerance = 1e-4
            self.logtolerance = 0.0

        elif self.estimation_method.lower() in ['ols', 'ordinary_least_squares', 'numeric', 'analytic']:
            if self.algorithm is None: self.algorithm = 'numeric'
            if self.obj_function is None: self.obj_function = 'ssr'

        else:
            if not hasattr(self, "estimation_method"):
                raise NotImplementedError(f"Must have a self.SolverOptions.estimation_method attribute.")


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
        from dataclasses import replace as _replace
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


    def __dict__(self) -> dict:
        """Return a dict representation of the FitStatistics.

        This merges the dataclass fields with any keys in `additional_stats`.
        Keys in `additional_stats` take precedence when collisions occur.
        """
        # Gather all dataclass fields and their current values
        result = {f.name: getattr(self, f.name) for f in fields(self)}

        # Merge additional_stats (if present), allowing it to override named fields
        add = self.additional_stats or {}

        if not isinstance(add, dict):
            try:
                add = dict(add)
            except Exception:
                add = {}

        result.update(add)

        return result





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
    statistics: dict
    details: Optional[dict] = None

    def __post_init__(self):
        self.__name__ = "Researchpy.TestResults"


    def __dict__(self) -> dict:
        """Return a dict representation of the TestResults object.

        This merges the dataclass fields with any keys in `additional_stats`.
        Keys in `additional_stats` take precedence when collisions occur.
        """
        # Gather all dataclass fields and their current values
        result = {f.name: getattr(self, f.name) for f in fields(self)}

        # Merge additional_stats (if present), allowing it to override named fields
        add = self.additional_stats if hasattr(self, "additional_stats") else {}
        if not isinstance(add, dict):
            try:
                add = dict(add)

            except Exception:
                add = {}

        result.update(add)

        return result


    def to_dataframe(self, details_as_col: bool = False) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame (single row).

        Parameters
        ----------
        details_as_col : bool, default False
            Whether to include the `details` attribute as a column in the DataFrame.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with statistic names as columns.
        """
        stats = {"Test Name": self.test_name}
        stats.update(self.statistics)

        if hasattr(self, "details") and details_as_col:
            stats['Details'] = self.details

        return pd.DataFrame([stats])



@dataclass
class Term(CoreDataclass):
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

    _term: Any
    term: str
    name: Optional[str] = None
    is_interaction: Optional[Union[bool, list]] = None
    is_factor: Optional[Union[bool, list]] = None
    columns: Optional[list] = field(default_factory=list)
    columns_cleaned: Optional[list] = field(default_factory=list)
    levels: Optional[Union[list, None]] = None
    reference: Optional[Union[str, list, None]] = None
    #term_map: Optional[dict] = field(default_factory=dict)
    #term_column_map: Optional[dict] = field(default_factory=dict)

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


    def _include_formulaic_reference_column(self):
        return [f"{self.term}[T.{lvl}]" for lvl in self.levels]

    def _reference_as_formulaic_column(self):
        return f"{self.term}[T.{self.reference}]"


    # ---
    # Map helpers
    # ---
    def _term_map(self) -> dict:
        return {self.term: self.name}

    def _column_map(self) -> dict:
        """
        Original Formulaic/Patsy column name → cleaned column name.
        Example: ``{"C(drug, Treatment(2))[T.3]": "3", "disease": "disease"}``

        Note: These are the names of the columns in the design matrix, while the term names are the names of the model_terms in the model formula.
        """
        mapping = {}
        for col in self.columns:
            mapping[col] = self._clean_column(col)
        return mapping

    def _term_level_map(self, pretty_format=True, include_base=True) -> dict:

        term_as = "term"
        levels_as = "columns"
        if pretty_format:
            term_as = "name"
            if include_base:
                levels_as = "levels"
            else:
                levels_as = "columns_cleaned"

        mapping = {}
        if str(self.term) == 1 or str(self.term.lower()) == "intercept":
            mapping[getattr(self, term_as)] = getattr(self, "columns")

        elif self.is_interaction:
            if not include_base:
                mapping[getattr(self, term_as)] = getattr(self, levels_as)
            else:
                interaction_terms_levels = []
                for ix, sub_term in enumerate(self.term.split(":")):
                    if self.is_factor[ix]:
                        all_levels = []
                        if include_base and not pretty_format:
                            all_levels = [f"{sub_term}[T.{lvl}]" for lvl in self.levels[ix]]
                            interaction_terms_levels.append(all_levels)
                        else:
                            interaction_terms_levels.append(self.levels[ix])
                    else:
                        interaction_terms_levels.append([sub_term])

                full_level_combinations = list(product(*interaction_terms_levels))
                full_level_combinations = [":".join(level) for level in full_level_combinations]
                mapping[getattr(self, term_as)] = full_level_combinations

        else:
            if self.is_factor:
                if include_base and not pretty_format:
                    mapping[getattr(self, term_as)] = self._include_formulaic_reference_column()
                else:
                    mapping[getattr(self, term_as)] = getattr(self, levels_as)
            else:
                mapping[getattr(self, term_as)] = getattr(self, levels_as)

        return mapping



@dataclass
class ModelTerms(CoreDataclass):
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
    #dv: Optional[list] = None                   # list[str] — dependent variable names


    def __post_init__(self):
        self.__name__ = "Researchpy.ModelTerms"

    # ------------------------------------------------------------------ #
    #  Factories                                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_formula(cls, formula: str, include_intercept: bool = False) -> "ModelTerms":
        """Parse a formula string into ``ModelTerms`` using formulaic's formula parser.

        This is a lightweight parse that extracts structural information
        (DV names, term names, interaction flags, factor flags) WITHOUT
        requiring data and WITHOUT building a design matrix.

        Parameters
        ----------
        formula : str
            Wilkinson-style formula string (e.g., ``"y ~ C(x) + C(a):C(b)"``).
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
        if "~" not in formula:
            raise ValueError(
                f"Formula must contain '~' separating dependent and independent "
                f"variables. Got: '{formula}'. Example: 'y ~ C(x)'."
            )

        parsed = formulaic.Formula(formula)

        # --- Extract DV names from LHS ---
        dv_names = []
        for lhs_term in parsed.lhs:
            for factor in lhs_term.factors:
                factor_str = str(factor)
                if factor_str != "1":
                    dv_names.append(factor_str)

        # --- Build Term objects from RHS ---
        terms = []
        for rhs_term in parsed.rhs:
            # Build the term string from factors
            factors_strs = [str(f) for f in rhs_term.factors]

            # Skip intercept (factor is "1") unless requested
            if factors_strs == ["1"]:
                if include_intercept:
                    terms.append(Term(_term=1, term="Intercept"))
                continue

            # Build the term string: "C(x):C(k)" or "z"
            term_str = ":".join(factors_strs)

            # Term.__post_init__ handles: name, is_interaction, is_factor
            term_obj = Term(_term=term_str, term=term_str)
            terms.append(term_obj)

        return cls(terms=terms, dv=dv_names)


    @classmethod
    def from_model_spec(cls, model_spec) -> "ModelTerms":
        """Build ``ModelTerms`` from a formulaic ``ModelSpec`` object.

        Uses ``model_spec.structure`` to reliably map each column name to
        its parent term, and ``model_spec.encoder_state`` to determine all
        category levels and reference categories for factor terms.

        Parameters
        ----------
        model_spec : formulaic.ModelSpec
            Typically ``mm.rhs.model_spec`` from a formulaic ModelMatrix.

        Returns
        -------
        ModelTerms
        """
        structure = model_spec.structure
        encoder_state = model_spec.encoder_state

        # Build a lookup: factor expression string → (kind, categories)
        # encoder_state maps factor_expr → (Kind, state_dict)
        factor_cats = {}
        for factor_expr, (kind, state) in encoder_state.items():
            factor_str = str(factor_expr)
            if hasattr(kind, 'value') and kind.value == 'categorical':
                categories = state.get('categories', None)
                if categories is not None:
                    factor_cats[factor_str] = [str(c) for c in categories]

        terms = []
        for encoded_term in structure:
            term_str = str(encoded_term.term)
            t_columns = list(encoded_term.columns)

            # Skip intercept — represented as term "1" with column "Intercept"
            if term_str == "1" or term_str == "Intercept":
                term_obj = Term(_term=encoded_term.term, term="Intercept", columns=t_columns)
                terms.append(term_obj)
                continue

            # Build the Term (columns_cleaned is computed in __post_init__)
            term_obj = Term(_term=encoded_term.term, term=term_str, columns=t_columns)

            # Determine levels and reference for factor terms
            # Get sub-parts of the term (for interactions like "C(group):C(drug)")
            sub_parts = term_str.split(":")

            # Identify which sub-parts are categorical
            cat_sub_parts = [p for p in sub_parts if p in factor_cats]

            if cat_sub_parts:
                is_intx = len(sub_parts) > 1

                part_levels = []
                part_refs = []

                for i, sub in enumerate(sub_parts):
                    if sub in factor_cats:
                        all_cats = factor_cats[sub]

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


    @classmethod
    def from_design_info(cls, design_info) -> "ModelTerms":
        """Build ``ModelTerms`` from a formulaic ``ModelSpec`` object.

        .. deprecated::
            Use ``from_model_spec()`` instead. This method is provided
            for backward compatibility during the patsy→formulaic migration.

        Parameters
        ----------
        design_info : formulaic.ModelSpec
            A formulaic ModelSpec object (named ``design_info`` for
            backward compatibility with code that previously passed
            patsy DesignInfo objects).

        Returns
        -------
        ModelTerms
        """
        return cls.from_model_spec(design_info)


    # ------------------------------------------------------------------ #
    #  Mapping properties                                                  #
    # ------------------------------------------------------------------ #
    @property
    def term_map(self) -> dict:
        """
        Original Patsy term name → cleaned term name.
        Example: ``{"C(drug, Treatment(2))": "drug", "disease": "disease"}``

        Note: These are the names of the terms in the model formula, while the column names are the names of the columns in the design matrix.
        """
        return {t.term: t.name for t in self.terms}
        #return {t._term_map() for t in self.terms}

    @property
    def columns(self) -> list:
        """
        List of all column names in the design matrix, in order.
        Example: ``["Intercept", "C(drug, Treatment(2))[T.3]", "C(drug, Treatment(2))[T.3]:disease"]``

        """
        cols = []
        for t in self.model_terms.values():
            cols.extend(t.columns)
        return cols

    @property
    def column_map(self) -> dict:
        """
        Original Formulaic/Patsy column name → cleaned column name.
        Example: ``{"C(drug, Treatment(2))[T.3]": "3", "disease": "disease"}``

        Note: These are the names of the columns in the design matrix, while the term names are the names of the model_terms in the model formula.
        """
        mapping = {}
        for t in self.terms:
            for orig, clean in zip(t.columns, t.columns_cleaned):
                mapping[orig] = clean
        return mapping


    def term_level_map(self, pretty_format=True, include_base=True) -> dict:
        term_as = "term"
        levels_as = "columns"
        if pretty_format:
            term_as = "name"
            if include_base:
                levels_as = "levels"
            else:
                levels_as = "columns_cleaned"

        mapping = {}
        for t in self.terms:
            if str(t.term) == "1" or str(t.term.lower()) == "intercept":
                mapping[getattr(t, term_as)] = getattr(t, "columns")

            elif t.is_interaction:
                interaction_terms_levels = []

                for sub_term in t.term.split(":"):
                    if self[sub_term].is_factor:
                        if not pretty_format and include_base:
                            sub_term_reference = self[sub_term]._reference_as_formulaic_column()
                            interaction_terms_levels.append([sub_term_reference] + getattr(self[sub_term], levels_as))

                        else:
                            interaction_terms_levels.append(getattr(self[sub_term], levels_as))

                    else:
                        interaction_terms_levels.append([sub_term])

                full_level_combinations = list(product(*interaction_terms_levels))
                full_level_combinations = [":".join(level) for level in full_level_combinations]
                mapping[getattr(t, term_as)] = full_level_combinations

            else:
                if t.is_factor:
                    if include_base and not pretty_format:
                        mapping[getattr(t, term_as)] = [t._reference_as_formulaic_column()] + getattr(t, levels_as)
                    else:
                        mapping[getattr(t, term_as)] = getattr(t, levels_as)
                else:
                    mapping[getattr(t, term_as)] = getattr(t, levels_as)

        return mapping


    # --------
    # Cleaning helpers
    # --------
    def as_formulaic_column(term):
        return f"{term.term}[T.{term.reference}]"

    @staticmethod
    def clean_term(term) -> str:
        """Clean the Patsy term name.

        ``"C(drug, Treatment(2))"``  →  ``"drug"``
        ``"C(drug):disease"``        →  ``"drug:disease"``
        """
        factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')
        cleaned_factors = [
            ''.join(re.findall(factor_pattern, f)) if "C(" in f else f
            for f in term.split(":")
        ]
        return ":".join(cleaned_factors)


    @staticmethod
    def clean_column(column: str) -> str:
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



    # ------------------------------------------------------------------ #
    #  Container protocol                                                  #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key):
        """
        Look up a Term by integer index, original term name, or cleaned name.

        """
        if isinstance(key, int):
            return self.terms[key]

        for t in self.terms:
            if t.term == key or t.name == key:
                return t

        raise KeyError(f"Term '{key}' not found")

    def __len__(self):
        return len(self.terms)

    def to_list(self) -> list:
        return [t.term for t in self.model_terms.values()]

    def info(self):
        lines = [f"ModelTerms({len(self.terms)} terms)"]
        for t in self.terms:
            lines.append(f"  {t.term!r} → {t.name!r}  "
                         f"(factor={t.is_factor}, intx={t.is_interaction}, "
                         f"cols={len(t.columns)})"
                         )
        return "\n".join(lines)

    def __repr__(self):
        return self.info()


