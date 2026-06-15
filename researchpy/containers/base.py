# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd




@dataclass
class CoreDataclass:
    """Generic base dataclass providing common utility methods.

    All methods reference only ``self`` and dynamically inspect fields,
    so subclasses inherit them without needing to override.
    """

    def __post_init__(self):
        self.__name__ = "Researchpy.CoreDataclass"


    def to_dict(self, drop_none=True):
        dct = {}
        for f in fields(self):
            if drop_none and getattr(self, f.name) is not None:
                dct[f.name] = getattr(self, f.name)

        return dct


    def _get_summary(self, skip_raw_arrays=False):
        """Print a human-readable summary of all DataFrame/dict fields."""
        printed = []

        for f in fields(self):
            val = getattr(self, f.name)

            if val is None:
                continue

            if isinstance(val, pd.DataFrame):
                printed.append(val.to_string(index=False))

            elif isinstance(val, (list, np.ndarray)):
                if not skip_raw_arrays:
                    try:
                        val = pd.Series(val)
                        printed.append(val.to_string(index=False))

                    except:
                        try:
                            for x in val: printed.append(f"{x}")
                        except:
                            print(f"{f.name} could not be printed as a DataFrame or Series, and is not a simple list/array. Skipping.")
                            continue
                else:
                    continue # skip raw arrays

            else:
                printed.append(f"{f.name}: {val}")

        print("\n".join(printed))


    def info(self):
        class_name = type(self).__name__
        parts = [f"Class({class_name})"]

        for f in fields(self):
            val = getattr(self, f.name)

            if val is None:
                parts.append(f"  {f.name}=None")

            elif isinstance(val, pd.DataFrame):
                parts.append(f"  {f.name}=pd.DataFrame({val.shape[0]}x{val.shape[1]})")

            elif isinstance(val, dict):
                parts.append(f"  {f.name}=dict({len(val)} keys)")

            elif isinstance(val, (list, np.ndarray)):
                if isinstance(val, list):
                    length = len(val)
                    parts.append(f"  {f.name}={type(val).__name__}(len={length})")
                else:
                    parts.append(f"  {f.name}=np.ndarray(shape={val.shape})")

            else:
                parts.append(f"  {f.name}={val!r}")

        return "\n\n".join(parts) + "\n)"


    def __iter__(self):
        """Yield fields in order for tuple-style unpacking."""
        for f in fields(self):
            yield getattr(self, f.name)


    def __repr__(self):
        return self.info()





class DesignInfo():
    """
    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.
    """

    @property
    def CI_LEVEL(self):
        return self._CI_LEVEL
    @CI_LEVEL.setter
    def CI_LEVEL(self, conf_level):
        self._CI_LEVEL = float(conf_level)

    @property
    def CI_LEVEL(self):
        return self._ci_level
    @CI_LEVEL.setter
    def CI_LEVEL(self, conf_level):
        self._CI_LEVEL = float(conf_level)

    @property
    def conf_level(self):
        return self._CI_LEVEL
    @conf_level.setter
    def conf_level(self, conf_level):
        self._CI_LEVEL = float(conf_level)

    @property
    def obj_function(self):
        return self._obj_function
    @obj_function.setter
    def obj_function(self, obj_function):
        self._obj_function = obj_function

    def __init__(self, formula_like: Optional[str]=None,
                 data: Union[pd.Series, pd.DataFrame, np.ndarray, list]=None,
                 ci_level=0.95,
                 matrix_type=1,
                 family="gaussian", link="normal",
                 solver_options=None, table_decimals=None):

        self.__name__ = "Researchpy.model.DesignInfo"

        self._beta_type = "coef"

        if data is None:
            data = {}

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        # Build a SolverOptions dataclass instance.
        # Subclasses (LinearModel, GeneralModel) should resolve their own defaults
        # and pass a fully-formed SolverOptions instance. If None or dict arrives
        # here, we fall back to the SolverOptions dataclass defaults.
        if isinstance(solver_options, SolverOptions):
            self.solver_options = solver_options
        elif isinstance(solver_options, dict):
            self.solver_options = SolverOptions.from_dict(solver_options)
        else:
            self.solver_options = SolverOptions()

        self.obj_function = self.solver_options.obj_function

        self.ci_level = ci_level

        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        if not hasattr(self, "_test_stat_name"):
            self._test_stat_name = "t" if family == "gaussian" else "z"
        self._family = family
        self._link = link
        self._CI_LEVEL = ci_level

        # Initialize an optimization tracker instance for this model. This tracker can be used by optimization
        # algorithms to store and monitor the optimization process.
        self._OptimizationTracker = OptimizationTracker()

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        # New dataclass-based term/column mapping
        self._model_terms = ModelTerms.from_design_info(self._IV_design_info)

        # Will be refractoring to use containers to clean up codebase and make it more modular. This ModelFit
        # dataclass will store the model design information and fit parameters that are common across different
        # regression models. By centralizing this information in a dataclass, it allows for cleaner code and easier
        # maintenance, as well as providing a standardized way to access model fit information across different model types.
        self.ModelFit = ModelFit(
            formula = formula_like,
            family = family,
            link = link,
            solver_method = self.solver_options.method,
            ci_level = ci_level,
            dv_term_names = self.DV.design_info.term_names,
            iv_term_names = list(self._model_terms.column_map.keys())
        )

        self.FitStatistics = FitStatistics(
            n = self.n,
            k = self.k,
            test_stat_name = "F"
        )
        self.ModelEffects = ModelEffects()

        self.CoefResults = CoefResults()
        self.CoefResults.term = self._model_terms.column_map.keys()
        #self.CoefResults.test_stat_name = "t" if self.ModelFit.family == "gaussian" else "z"

        ## Creating variable table information
        if not hasattr(self, "regression_table_info"):
            self.regression_table_info = {
                self.DV_name: [],
                "Coef.": [],
                "Std. Err.": [],
                f"{self._test_stat_name}": [],
                "p-value": [],
                f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []
            }


        # Checking to see if the `self._table_decimals` attribute is defined. If it's not then create it.
        # This is used to specify the number of decimal places to round to for different statistics in the summary
        # table. By defining it in the base class, it allows subclasses to override or update the decimal settings as
        # needed without having to redefine the entire dictionary.
        if not hasattr(self, "_table_decimals"):
            self._table_decimals = {
                "Coef.": 2, "Std. Err.": 3, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                        'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4
            }

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals










@dataclass
class EstimateResults(CoreDataclass):
    """Container for results of a statistical estimation.

    Attributes
    ----------
    name : str or None
        The variable name.
    statistics : dict
        Ordered mapping of statistic name -> computed value.

    Examples
    --------
    >>> result = EstimateResults(name="age", statistics={"N": 100, "Mean": 35.4})
    >>> result.to_dataframe()
       Name    N  Mean
    0   age  100  35.4
    """

    test_name: str
    dv_name: str
    iv_name: Optional[Union[str, List[tuple, str], np.ndarray, dict]] = None
    statistics: Dict[str, Any] = field(default_factory=dict)




    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result to a pandas DataFrame (single row).

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with statistic names as columns.
        """
        row = {}
        if self.name is not None:
            row["Name"] = self.name

        row.update(self.statistics)

        return pd.DataFrame([row])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with statistic names as keys and computed values as values.
        """
        result = {}
        if self.name is not None:
            result["Name"] = self.name

        result.update(self.statistics)

        return result



@dataclass
class VariableInfo(CoreDataclass):
    """Descriptive profile for a single variable.

    Attributes
    ----------
    name : str
        Variable (column) name.
    dtype : str
        String representation of the variable's dtype.
    category : str
        Classification of the variable: 'numeric', 'categorical', or 'datetime'.
    n_total : int
        Total number of observations (including missing).
    n_valid : int
        Number of non-missing observations.
    n_missing : int
        Number of missing observations.
    percent_missing : float
        Percentage of observations that are missing.
    n_unique : int
        Number of unique non-null values.
    numeric_summary : SummaryResult or None
        Summary statistics for numeric variables (N, Mean, SD, etc.).
        None for non-numeric variables.
    frequency_table : pd.DataFrame or None
        Value counts table for categorical variables.
        None for non-categorical variables.
    date_range : tuple or None
        (min_date, max_date) for datetime variables.
        None for non-datetime variables.

    Examples
    --------
    >>> info = VariableInfo(name="age", dtype="float64", category="numeric",
    ...                     n_total=100, n_valid=95, n_missing=5,
    ...                     percent_missing=5.0, n_unique=45)
    """

    name: str = ""
    dtype: str = ""
    category: str = ""  # 'numeric', 'categorical', 'datetime'
    n_total: int = 0
    n_valid: int = 0
    n_missing: int = 0
    percent_missing: float = 0.0
    n_unique: int = 0
    numeric_summary: Optional[Any] = None  # SummaryResult instance
    frequency_table: Optional[pd.DataFrame] = None
    date_range: Optional[Tuple[Any, Any]] = None

    def __post_init__(self):
        self.__name__ = "Researchpy.VariableInfo"


@dataclass
class CodeBook(CoreDataclass):
    """Container for detailed data characteristics and variable summaries.

    Provides a structured, inspectable representation of a dataset's metadata
    and per-variable descriptive statistics. Replaces the print-only `codebook()`
    function with a proper data object.

    Attributes
    ----------
    name : str or None
        Name of the dataset (from DataFrame or Series name).
    input_type : str or None
        Type of the input data ('Series' or 'DataFrame').
    shape : tuple or None
        Shape of the input data (n_rows,) or (n_rows, n_cols).
    n_variables : int
        Number of variables (columns) in the dataset.
    variables : list of VariableInfo
        Per-variable descriptive profiles.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [25, 30, 35], "gender": ["M", "F", "M"]})
    >>> cb = CodeBook.from_data(df)
    >>> cb.n_variables
    2
    >>> cb.variables[0].name
    'age'
    """

    name: Optional[str] = None
    input_type: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    n_variables: int = 0
    variables: List[VariableInfo] = field(default_factory=list)

    def __post_init__(self):
        self.__name__ = "Researchpy.CodeBook"

    @classmethod
    def from_data(
            cls, data: Union[pd.Series, pd.DataFrame],
            numeric_stats: Optional[List[str]] = None,
            ci_level: float = 0.95, decimals: int = 4,
    ) -> "CodeBook":
        """Create a CodeBook from a pandas Series or DataFrame.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data to profile.
        numeric_stats : list of str, optional
            Statistics to compute for numeric variables.
            Default is ["N", "Mean", "SD", "Min", "Max", "Median", "CI"].
        ci_level : float, optional
            Confidence level for interval computation. Default is 0.95.
        decimals : int, optional
            Decimal places for rounding. Default is 4.

        Returns
        -------
        CodeBook
            A populated CodeBook instance.

        Raises
        ------
        TypeError
            If data is not a pandas Series or DataFrame.
        """
        from ..descriptive import (
            standard_error, confidence_interval, skewness, kurtosis,
            percentile, mode,
        )
        from ..descriptive.categorical import proportions
        from .univariate import SummaryResult

        if numeric_stats is None:
            numeric_stats = ["N", "Mean", "SD", "Min", "Max", "Median", "CI"]

        # --- Resolve input metadata ---
        if isinstance(data, pd.Series):
            input_type = "Series"
            dataset_name = data.name if data.name is not None else None
            shape = (len(data),)
            columns = [data]

        elif isinstance(data, pd.DataFrame):
            input_type = "DataFrame"
            dataset_name = data.columns.name if data.columns.name else None
            shape = data.shape
            columns = [data[col] for col in data.columns]

        else:
            raise TypeError(
                f"CodeBook.from_data() expects a pandas Series or DataFrame, "
                f"got {type(data).__name__}."
            )

        # --- Build VariableInfo for each column ---
        variable_infos: List[VariableInfo] = []

        for col_series in columns:
            col_name = str(col_series.name) if col_series.name is not None else ""
            col_dtype = str(col_series.dtype)
            total = col_series.size
            valid = int(col_series.count())
            missing = total - valid
            pct_missing = round(float(missing / total * 100), decimals) if total > 0 else 0.0
            unique = int(col_series.nunique())

            # Classify variable type
            if "int" in col_dtype or "float" in col_dtype:
                var_category = "numeric"
            elif "datetime" in col_dtype or "timedelta" in col_dtype:
                var_category = "datetime"
            else:
                var_category = "categorical"

            var_info = VariableInfo(
                name=col_name,
                dtype=col_dtype,
                category=var_category,
                n_total=total,
                n_valid=valid,
                n_missing=missing,
                percent_missing=pct_missing,
                n_unique=unique,
            )

            # --- Compute type-specific summaries ---
            if var_category == "numeric":
                stats_dict: Dict[str, Any] = {}
                arr = col_series.to_numpy(dtype=float, na_value=np.nan)

                if "N" in numeric_stats:
                    stats_dict["N"] = valid
                if "Mean" in numeric_stats:
                    stats_dict["Mean"] = round(float(np.nanmean(arr)), decimals)
                if "Median" in numeric_stats:
                    stats_dict["Median"] = round(float(np.nanmedian(arr)), decimals)
                if "Variance" in numeric_stats:
                    stats_dict["Variance"] = round(float(np.nanvar(arr, ddof=1)), decimals)
                if "SD" in numeric_stats:
                    stats_dict["SD"] = round(float(np.nanstd(arr, ddof=1)), decimals)
                if "SE" in numeric_stats:
                    stats_dict["SE"] = round(float(standard_error(arr)), decimals)
                if "Min" in numeric_stats:
                    stats_dict["Min"] = round(float(np.nanmin(arr)), decimals)
                if "Max" in numeric_stats:
                    stats_dict["Max"] = round(float(np.nanmax(arr)), decimals)
                if "Range" in numeric_stats:
                    stats_dict["Range"] = round(float(np.nanmax(arr) - np.nanmin(arr)), decimals)
                if "CI" in numeric_stats:
                    if valid >= 2:
                        ci = confidence_interval(arr, confidence_level=ci_level, decimals=decimals)
                        ci_label = f"{int(ci_level * 100)}% Conf. Interval"
                        stats_dict[ci_label] = ci
                if "Mode" in numeric_stats:
                    stats_dict["Mode"] = mode(arr)
                if "Kurtosis" in numeric_stats:
                    stats_dict["Kurtosis"] = round(float(kurtosis(arr)), decimals)
                if "Skew" in numeric_stats:
                    stats_dict["Skew"] = round(float(skewness(arr)), decimals)
                if "Percentiles" in numeric_stats:
                    stats_dict["Percentiles"] = percentile(arr, q=[10, 25, 50, 75, 90])

                var_info.numeric_summary = SummaryResult(
                    name=col_name,
                    statistics=stats_dict,
                )

            elif var_category == "categorical":
                var_info.frequency_table = proportions(col_series)

            elif var_category == "datetime":
                var_info.date_range = (col_series.min(), col_series.max())

            variable_infos.append(var_info)

        return cls(
            name=dataset_name,
            input_type=input_type,
            shape=shape,
            n_variables=len(variable_infos),
            variables=variable_infos,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the codebook to a summary DataFrame.

        Returns a DataFrame with one row per variable showing key metadata.

        Returns
        -------
        pd.DataFrame
            Summary table with columns: Variable, Type, Dtype, N Total,
            N Missing, % Missing, N Unique.
        """
        rows = []
        for v in self.variables:
            rows.append({
                "Variable": v.name,
                "Type": v.category,
                "Dtype": v.dtype,
                "N Total": v.n_total,
                "N Valid": v.n_valid,
                "N Missing": v.n_missing,
                "% Missing": v.percent_missing,
                "N Unique": v.n_unique,
            })
        return pd.DataFrame(rows)

    def display(self) -> None:
        """Print a human-readable codebook report, similar to the legacy codebook() output.

        Prints metadata header followed by per-variable detail blocks.
        """
        # Header
        print(f"{'=' * 60}")
        print(f"  CodeBook Summary")
        print(f"{'=' * 60}")
        if self.name:
            print(f"  Dataset: {self.name}")
        print(f"  Input Type: {self.input_type}")
        print(f"  Shape: {self.shape}")
        print(f"  N Variables: {self.n_variables}")
        print(f"{'=' * 60}")
        print()

        for var in self.variables:
            print(f"{'-' * 60}")
            print(f"  Variable: {var.name}    Data Type: {var.dtype}")
            print(f"{'-' * 60}")
            print(f"  Number of Obs.: {var.n_total}")
            print(f"  Number of missing obs.: {var.n_missing}")
            print(f"  Percent missing: {var.percent_missing}")
            print(f"  Number of unique values: {var.n_unique}")
            print()

            if var.category == "numeric" and var.numeric_summary is not None:
                stats = var.numeric_summary.statistics
                for key, val in stats.items():
                    if isinstance(val, tuple):
                        print(f"  {key}: [{val[0]}, {val[1]}]")
                    elif isinstance(val, dict):
                        for pk, pv in val.items():
                            print(f"  {pk}: {pv}")
                    else:
                        print(f"  {key}: {val}")

            elif var.category == "categorical" and var.frequency_table is not None:
                print("  Value Counts:")
                print(f"  {var.frequency_table.to_string(index=False)}")

            elif var.category == "datetime" and var.date_range is not None:
                print(f"  Range: [{var.date_range[0]}, {var.date_range[1]}]")

            print()

    def __str__(self) -> str:
        """String representation showing the summary table."""
        return self.to_dataframe().to_string(index=False)
