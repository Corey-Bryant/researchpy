from researchpy.models.generalized.general_model import GeneralizedLinearModel
from researchpy.containers import ModelResults, SolverOptions
from researchpy.utility import *

from researchpy.models.postestimation.predict import predict


class LogisticRegression(GeneralizedLinearModel):
    """
    Logistic regression for binary outcomes using Maximum Likelihood Estimation (MLE).

    This class provides logistic regression analysis with support for:
    - Multiple optimization algorithms (BFGS, Newton-Raphson)
    - L1 and L2 regularization
    - Odds ratio reporting
    - Confidence intervals
    - Model fit statistics (LR Chi-squared, etc.)

    Parameters
    ----------
    formula : str
        A string representing a valid Patsy formula (e.g., "outcome ~ predictor1 + predictor2").
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    solver_method : str, optional
        Method for fitting the model. Default is "mle".
    solver_options : dict or SolverOptions, optional
        Options for the solver including:
        - algorithm: "BFGS" (default) or "newton-raphson"
        - tol: Convergence tolerance (default 1e-7)
        - max_iter: Maximum iterations (default 1000)
        - display: Whether to print iteration info (default True)
        - regularization: None, "l1", or "l2"
        - alpha: Regularization strength (default 0.0)
    initial_betas : ndarray, optional
        Initial coefficient values.
    initial_betas_method : str, optional
        Method to initialize betas. Default is "ols".

    Examples
    --------
    >>> import researchpy as rp
    >>> model = rp.LogisticRegression("outcome ~ age + treatment",data=df)
    >>> model.results()

    See Also
    --------
    Logistic : Alias for LogisticRegression
    """

    def __init__(self, formula, data=None, conf_level=0.95,
                 report_as="or", display_summary=True,
                 solver_options=None, table_decimals=None,
                 initial_betas=None, initial_betas_method="ols"):

        if data is None: data = {}
        self._test_stat_name = "z"

        #-------------------------------------------------#
        # -- Build a SolverOptions dataclass instance. -- #
        #-------------------------------------------------#
        # Subclasses (LinearModel, GeneralizedLinearModel) should resolve their own defaults and pass a fully-formed SolverOptions instance.
        # If None or dict arrives here, we fall back to the SolverOptions dataclass defaults.
        self.SolverOptions = SolverOptions(
                estimation_method="mle",
                algorithm="IRLS",
                obj_function="log-likelihood",
                tol=1e-7,
                tolerance=1e-4,
                logtolerance=0,
                max_iter=300,
                display=True,
                regularization=None,
                alpha=0.0
        )
        if isinstance(solver_options, SolverOptions):
            self.SolverOptions = self.SolverOptions.with_overrides(solver_options.to_dict())
        elif isinstance(solver_options, dict):
            self.SolverOptions = self.SolverOptions.with_overrides(solver_options)


        super().__init__(formula, data, conf_level=conf_level, family="binomial", link="logit",
                         solver_options=self.SolverOptions, table_decimals=table_decimals)

        self.__name__ = "Researchpy.LogisticRegression"

        # Initializing betas
        self._GeneralizedLinearModel__initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

        # Fit the model
        self.fit()

        # Compute standard errors and statistics
        self._compute_statistics()

        # Build ModelResults (results() sets self.ModelResults internally)
        self.results(report_as=report_as, return_type="Dataframe", pretty_format=True)

        # Display the model results summary
        if display_summary:
            self.summary()


    def _get_from_child(self, **kwargs):
        """
        Return LogisticRegression-specific data requested by the parent class.

        Parameters
        ----------
        key : str
            Identifier for the requested data. Supported keys:
            - ``"default_transform"`` : ``np.exp`` (odds ratios)
            - ``"test_stat_name"`` : ``"z"``
            - ``"additional_fit_stats"`` : extra logistic-specific fit stats
            - ``"report_options"`` : default reporting configuration
        **kwargs
            Additional context from the parent.

        Returns
        -------
        object
            The requested value, or ``None`` for unrecognized keys.
        """
        key = kwargs.get("key", "additional_fit_stats")

        if key == "default_transform":
            return np.exp

        if key == "test_stat_name":
            return "z"

        if key == "additional_fit_stats":
            return {}

        if key == "model_table":
            return {"": [self._get_model_display_name(),
                         f"Log likelihood = {self.FitStatistics.log_likelihood:.4f}"]}

        if key == "df_model_table":
            return pd.DataFrame.from_dict({"": [self.ModelResults.model_name,
                                                f"Log likelihood = {self.FitStatistics.log_likelihood:.4f}"]})
        if key == "report_options":
            return {"report_as": "or", "beta_type": "odds ratio"}

        #return super()._get_from_child(key, **kwargs)





    def predict(self, estimate="y", trans=None, decimals=4, **kwargs):

        #return super().predict(self, estimate=estimate, trans=expit)
        return predict(self, estimate=estimate, trans=trans, decimals=decimals)



    def classification_table(self, threshold: float = 0.5,
                             return_type: str = "dataframe") -> tuple:
        """
        Generate a classification table and diagnostic statistics.

        Produces a confusion matrix cross-tabulating predicted vs. actual
        classes together with sensitivity, specificity, predictive values,
        false-positive/negative rates, and overall correct classification.

        Parameters
        ----------
        threshold : float, optional
            Probability cutoff for classifying an observation as positive.
            An observation is classified "+" if Pr(D) >= threshold.
            Default is 0.5.
        return_type : str, optional
            ``"dataframe"`` (default) returns pandas DataFrames.
            ``"dict"`` or ``"dictionary"`` returns plain dictionaries.

        Returns
        -------
        tuple
            ``(confusion_matrix, statistics)`` where:

            - **confusion_matrix** : pd.DataFrame or dict
              Rows are Classified ["+", "-", "Total"], columns are
              ``"{dv_name}={value}"`` for each observed DV level plus "Total".
            - **statistics** : pd.DataFrame or dict
              Diagnostic statistics reported as percentages (0–100).

        Notes
        -----
        Classification rule:
            Classified + if predicted Pr(D) >= *threshold*
            True D defined as DV != 0

        Formulas:
            - Sensitivity Pr(+|1) = TP / (TP + FN) * 100
            - Specificity Pr(-|0) = TN / (TN + FP) * 100
            - Positive predictive value Pr(1|+) = TP / (TP + FP) * 100
            - Negative predictive value Pr(0|-) = TN / (TN + FN) * 100
            - False + rate for true 0 Pr(+|0) = FP / (FP + TN) * 100
            - False - rate for true 1 Pr(-|1) = FN / (FN + TP) * 100
            - False + rate for classified + Pr(0|+) = FP / (TP + FP) * 100
            - False - rate for classified - Pr(1|-)  = FN / (TN + FN) * 100
            - Correctly classified = (TP + TN) / N * 100

        Where:
            TP = True Positives  (actual 1, classified +)
            TN = True Negatives  (actual 0, classified -)
            FP = False Positives (actual 0, classified +)
            FN = False Negatives (actual 1, classified -)

        Examples
        --------
        >>> model = LogisticRegression("outcome ~ age + treatment",data=df)
        >>> table, stats = model.classification_table()
        >>> print(table)
        >>> print(stats)
        """
        from scipy.special import expit as _expit

        # --- Predicted probabilities and classification ---
        linear_pred = self.IV @ self.CoefResults.betas
        predicted_probs = _expit(linear_pred).flatten()
        predicted_class = (predicted_probs >= threshold).astype(int)

        # --- Actual classes (True D defined as DV != 0) ---
        actual_class = (self.DV.flatten() != 0).astype(int)

        # --- Confusion matrix counts ---
        # TP: actual=1, predicted=1
        # FP: actual=0, predicted=1
        # FN: actual=1, predicted=0
        # TN: actual=0, predicted=0
        tp = int(np.sum((actual_class == 1) & (predicted_class == 1)))
        fp = int(np.sum((actual_class == 0) & (predicted_class == 1)))
        fn = int(np.sum((actual_class == 1) & (predicted_class == 0)))
        tn = int(np.sum((actual_class == 0) & (predicted_class == 0)))
        n = tp + fp + fn + tn

        # --- Build confusion matrix table ---
        dv_name = self.ModelDesignSpec.dv_term_names[0]
        col_positive = f"{dv_name}=1"
        col_negative = f"{dv_name}=0"

        confusion_data = {
            "Classified": ["+", "-", "Total"],
            col_positive: [tp, fn, tp + fn],
            col_negative: [fp, tn, fp + tn],
            "Total": [tp + fp, fn + tn, n],
        }

        # --- Compute diagnostic statistics (as percentages) ---
        # Guard against division by zero
        def _safe_pct(numerator: int, denominator: int) -> float:
            if denominator == 0:
                return float("nan")
            return round((numerator / denominator) * 100.0, 2)

        stats_data = {
            "Statistic": [
                "                   Sensitivity Pr(+|1)",
                "                   Specificity Pr(-|0)",
                "     Positive predictive value Pr(1|+)",
                "     Negative predictive value Pr(0|-)",
                "       False + rate for true 0 Pr(+|0)",
                "       False - rate for true 1 Pr(-|1)",
                " False + rate for classified + Pr(0|+)",
                " False - rate for classified - Pr(1|-)",
                "Correctly classified",
            ],

            "Percent": [
                _safe_pct(tp, tp + fn),        # Sensitivity: TP / (TP + FN)
                _safe_pct(tn, tn + fp),        # Specificity: TN / (TN + FP)
                _safe_pct(tp, tp + fp),        # PPV: TP / (TP + FP)
                _safe_pct(tn, tn + fn),        # NPV: TN / (TN + FN)
                _safe_pct(fp, fp + tn),        # False+ for true ~D: FP / (FP + TN)
                _safe_pct(fn, fn + tp),        # False- for true D: FN / (FN + TP)
                _safe_pct(fp, tp + fp),        # False+ for classified+: FP / (TP + FP)
                _safe_pct(fn, tn + fn),        # False- for classified-: FN / (TN + FN)
                _safe_pct(tp + tn, n),         # Correctly classified: (TP + TN) / N
            ],
        }


        # --- Return based on return_type ---
        if return_type.lower() in ["dict", "dictionary"]:
            return confusion_data, stats_data

        confusion_df = pd.DataFrame(confusion_data).set_index("Classified")
        stats_df = pd.DataFrame(stats_data)

        return confusion_df, stats_df


    def results(self, report_as="or", return_type="Dataframe", pretty_format=True,
                table_decimals=None, **kwargs):
        """
        Return the logistic regression results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        report_as : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw log-odds.
        return_type : str, optional
            ``"Dataframe"`` (default) or ``"Dictionary"``.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Logistic Regression"``
            - ``fit_statistics``: Combined fit statistics (DataFrame or dict)
            - ``model_table``: ``None`` (MLE-based model, no SS decomposition)
            - ``coefficients``: Coefficient / odds ratio table (DataFrame or dict)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, model_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.coefficients
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Determine the coefficient transform based on report_as
        if report_as.lower() in ["or", "odds ratio", "odds_ratio"]:
            transform = np.exp
            self._beta_type = "odds ratio"
        else:
            transform = None
            self._beta_type = "coef"


        # Use the new GeneralizedLinearModel flow to build ModelResults
        mr = self._get_ModelResults(return_type=return_type,
                                    pretty_format=pretty_format,
                                    table_decimals=self._table_decimals,
                                    coef_transform=transform)


        # Returning the classification table as the "model_table" component of ModelResults for logistic regression
        classification_table, classification_stats = self.classification_table(return_type="Dictionary")
        self.ModelResults.details = {"Classification table": classification_table,
                                     "Classification stats": classification_stats}


        # Rename "Coef." column to "Odds Ratio" if reporting odds ratios
        if self._beta_type == "odds ratio":
            if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe"]:
                coef_df = self.ModelResults.as_dataframe("coefficients", mr.coefficients)

                if "Coef." in coef_df.columns:
                    coef_df = coef_df.rename(columns={"Coef.": "Odds Ratio"})

                self.ModelResults.coefficients = coef_df

            elif isinstance(mr.coefficients, dict) and "Coef." in mr.coefficients:
                self.ModelResults.coefficients = {
                    ('Odds Ratio' if k == 'Coef.' else k): v
                    for k, v in mr.coefficients.items()
                }


        if return_type.lower() in ["dataframe", "df", "pandas.dataframe", "pd.dataframe", ]:
            return (self.ModelResults.as_dataframe("fit_statistics", mr.fit_statistics),
                    self.ModelResults.as_dataframe("model_table", mr.model_table),
                    self.ModelResults.as_dataframe("coefficients", mr.coefficients) )

        else:
            return mr.fit_statistics, mr.model_table, mr.coefficients





# Convenience alias
Logistic = LogisticRegression
