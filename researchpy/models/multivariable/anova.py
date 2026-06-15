# -*- coding: utf-8 -*-
"""
Analysis of Variance (ANOVA)

This module provides the Anova class for conducting analysis of variance
using sum of squares types I, II, and III.
"""

import re

import numpy as np
import scipy.stats
import patsy
import pandas as pd
from pandas import DataFrame

from researchpy.models.linear_model import LinearModel
from researchpy.core.containerclasses import ModelResults, FactorEffects
from researchpy.utility import as_numeric


class Anova(LinearModel):
    """
    Analysis of Variance (ANOVA) for factorial designs.

    This class extends OLS to provide ANOVA functionality including:
    - Type I (Sequential) sum of squares
    - Type II (Hierarchical/Partially sequential) sum of squares
    - Type III (Marginal/Orthogonal) sum of squares
    - Partial effect size measures (Eta squared, Epsilon squared, Omega squared)

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "y ~ C(factor1) + C(factor2)").
        See https://patsy.readthedocs.io/en/latest/ for formula syntax.
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    sum_of_squares : int, optional
        Type of sum of squares to calculate. Options are:
        - 1 or "I": Sequential sum of squares
        - 2 or "II": Hierarchical/Partially sequential sum of squares
        - 3 or "III": Marginal/Orthogonal sum of squares (default)

    Attributes
    ----------
    factor_effects : dict
        Dictionary containing factor-level results including:
        - 'Source': Factor names
        - 'Sum of Squares': Sum of squares for each factor
        - 'Degrees of Freedom': Degrees of freedom for each factor
        - 'Mean Squares': Mean squares for each factor
        - 'F value': F-statistics for each factor
        - 'p-value': p-values for F-tests
        - 'Eta squared': Partial eta squared effect sizes
        - 'Epsilon squared': Partial epsilon squared effect sizes
        - 'Omega squared': Partial omega squared effect sizes

    Notes
    -----
    **Type I (Sequential) Sum of Squares:**
    Calculated by measuring the reduction in the model's overall sum of square
    error by subtracting the sum of square error for each term from the sum of
    square error from the model before.

    **Type II (Hierarchical) Sum of Squares:**
    The reduction in residual error due to adding the term to the model after
    all other terms except those that contain it.

    **Type III (Marginal) Sum of Squares:**
    Gives the sum of squares that would be obtained for each factor if it were
    to be entered into the model last. Best for models with interactions.

    Examples
    --------
    >>> import researchpy as rp
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C']
    ... })
    >>> model = rp.Anova("y ~ C(group)", data=df)
    >>> model.results()

    See Also
    --------
    OLS : Ordinary Least Squares regression (parent class)
    """

    # ------------------------------------------------------------------ #
    #                        Helper Methods                               #
    # ------------------------------------------------------------------ #




    @staticmethod
    def _compute_sse_from_design(x, y):
        """
        Compute the error sum of squares (SSE) given a design matrix and
        dependent variable vector.

        Computes: H = X(X'X)^{-1}X', ŷ = Hy, e = y - ŷ, SSE = e'e

        Parameters
        ----------
        x : ndarray
            Design matrix (n × p).
        y : ndarray
            Dependent variable vector (n × 1).

        Returns
        -------
        float
            The error (residual) sum of squares.
        """
        try:
            h = x @ np.linalg.inv(x.T @ x) @ x.T
        except np.linalg.LinAlgError:
            h = x @ np.linalg.pinv(x.T @ x) @ x.T

        predicted_y = h @ y
        residuals = y - predicted_y
        return residuals.T @ residuals


    def _compute_factor_stats(self, ss_factor, df_factor):
        """
        Compute mean square, F-statistic, p-value, and partial effect size
        measures for a single factor.

        All computations use the model-level MSE and SS values stored in
        ``self. sModelEffects``.

        Parameters
        ----------
        ss_factor : float or ndarray
            Sum of squares attributable to this factor.
        df_factor : int or float
            Degrees of freedom for this factor.

        Returns
        -------
        dict
            Keys: 'msr_f', 'f_value', 'f_p_value', 'eta_sq', 'epsilon_sq', 'omega_sq'
        """
        me = self.ModelEffects
        mse = me.mse
        ss_residual = me.ss_residual
        ss_total = me.ss_total
        df_residual = me.df_residual

        # Mean Square - Factor
        msr_f = ss_factor * (1 / df_factor)

        # F-statistic and p-value
        f_value = msr_f / mse
        f_p_value = scipy.stats.f.sf(f_value, df_factor, df_residual)

        # Partial Effect Size Measures
        eta_sq = ss_factor / (ss_factor + ss_residual)

        epsilon_sq = (df_factor * (msr_f - mse)) / \
            (ss_factor + ss_total)

        omega_sq = (df_factor * (msr_f - mse)) / \
            ((df_factor * msr_f) + (self.n - df_factor) * mse)

        return {
            "msr_f": msr_f,
            "f_value": f_value,
            "f_p_value": f_p_value,
            "eta_sq": eta_sq,
            "epsilon_sq": epsilon_sq,
            "omega_sq": omega_sq,
        }

    @classmethod
    def _append_factor_effects2(cls, factor_effects, source, ss_factor,
                               df_factor, stats):
        """
        Append a single factor's results to the factor_effects accumulator dict.

        Uses ``as_numeric()`` for safe numeric conversion.

        Parameters
        ----------
        factor_effects : dict
            The accumulator dictionary being built up during ANOVA computation.
        source : str
            Term name for this factor.
        ss_factor : float or ndarray
            Sum of squares for this factor.
        df_factor : float or int
            Degrees of freedom for this factor.
        stats : dict
            Output from ``_compute_factor_stats()``.
        """
        factor_effects["Source"].append(source)
        factor_effects["Sum of Squares"].append(as_numeric(ss_factor))
        factor_effects["Degrees of Freedom"].append(as_numeric(df_factor))
        factor_effects["Mean Squares"].append(as_numeric(stats["msr_f"]))
        factor_effects["F value"].append(as_numeric(stats["f_value"]))
        factor_effects["p-value"].append(as_numeric(stats["f_p_value"]))
        factor_effects["Eta squared"].append(as_numeric(stats["eta_sq"]))
        factor_effects["Epsilon squared"].append(as_numeric(stats["epsilon_sq"]))
        factor_effects["Omega squared"].append(as_numeric(stats["omega_sq"]))


    def _append_factor_effects(self, source, ss_factor, df_factor, stats):
        """
        Append a single factor's results to the factor_effects accumulator dict.

        Uses ``as_numeric()`` for safe numeric conversion.

        Parameters
        ----------
        factor_effects : dict
            The accumulator dictionary being built up during ANOVA computation.
        source : str
            Term name for this factor.
        ss_factor : float or ndarray
            Sum of squares for this factor.
        df_factor : float or int
            Degrees of freedom for this factor.
        stats : dict
            Output from ``_compute_factor_stats()``.
        """
        self.FactorEffects.source.append(source)
        self.FactorEffects.ss.append(as_numeric(ss_factor))
        self.FactorEffects.df.append(as_numeric(df_factor))
        self.FactorEffects.ms.append(as_numeric(stats["msr_f"]))
        self.FactorEffects.test_stat.append(as_numeric(stats["f_value"]))
        self.FactorEffects.test_pval.append(as_numeric(stats["f_p_value"]))
        self.FactorEffects.eta_squared.append(as_numeric(stats["eta_sq"]))
        self.FactorEffects.epsilon_squared.append(as_numeric(stats["epsilon_sq"]))
        self.FactorEffects.omega_squared.append(as_numeric(stats["omega_sq"]))


    @staticmethod
    def _build_type3_terms(term_names):
        """
        Convert patsy Treatment-coded term names to Sum-coded equivalents
        for Type III SS computation.

        Handles both main effects and interaction terms, preserving any
        user-specified reference categories by replacing Treatment(...) with Sum.

        Parameters
        ----------
        term_names : list of str
            Original term names from the design info (e.g., from
            ``self._IV_design_info.term_names``).

        Returns
        -------
        list of str
            Term names rewritten for Sum coding.
        """
        reference_pattern = re.compile(r'(?<=,|\s)(Treatment\(.*\))(?=\))')

        terms_sum = []
        for term in term_names:
            if "Treatment" in term:
                split_terms = term.split(":")

                if len(split_terms) == 1:
                    terms_sum.append(
                        re.sub(reference_pattern, 'Sum', split_terms[0]))
                else:
                    interaction_terms = []
                    for intterm in split_terms:
                        if "Treatment" in intterm:
                            interaction_terms.append(
                                re.sub(reference_pattern, 'Sum', intterm))
                        else:
                            interaction_terms.append(
                                intterm.replace(")", ", Sum)"))
                    terms_sum.append(':'.join(interaction_terms))
            else:
                terms_sum.append(term.replace(")", ", Sum)"))

        return terms_sum


    @staticmethod
    def _new_factor_effects_dict():
        """Return a fresh factor_effects accumulator dictionary."""
        '''
        return {
            "Source": [],
            "Sum of Squares": [],
            "Degrees of Freedom": [],
            "Mean Squares": [],
            "F value": [],
            "p-value": [],
            "Eta squared": [],
            "Epsilon squared": [],
            "Omega squared": [],
        }
        '''
        return FactorEffects()


    @staticmethod
    def _new_factor_effects():
        """Return a fresh factor_effects accumulator dictionary."""
        return FactorEffects()

    # ------------------------------------------------------------------ #
    #                        Constructor                                  #
    # ------------------------------------------------------------------ #

    def __init__(self, formula_like, data=None, sum_of_squares=3, conf_level=0.95, display_summary=True,
                 solver_options=None, table_decimals=None):

        if data is None:
            data = {}

        self._test_stat_name = "t"
        self._CI_LEVEL = conf_level

        super().__init__(formula_like, data, conf_level=conf_level, solver_options=solver_options,
                         table_decimals=table_decimals)

        self.__name__ = "Researchpy.ANOVA"
        self.ModelFit.model_type = self.__name__
        self.ModelFit.model_display_name = self._get_model_display_name()

        # Validate sum_of_squares parameter
        valid_ss_types = [1, 2, 3, "I", "II", "III"]
        if sum_of_squares not in valid_ss_types:
            raise ValueError(
                f"sum_of_squares must be one of {valid_ss_types}, "
                f"got {sum_of_squares!r}"
            )

        # Store for potential re-use / inspection
        self.FactorEffects.ss_type = sum_of_squares

        # Compute factor-level ANOVA effects
        self._compute_factor_effects(data)

        # Build ModelResults (results() sets self.ModelResults internally)
        self.results(return_type="Dataframe", na_rep='', pretty_format=True, table_decimals=table_decimals)

        # Display the model results summary
        if display_summary: self.summary()


    def _compute_factor_effects(self, data):
        """
        Compute per-factor ANOVA statistics based on the requested SS type.

        Parameters
        ----------
        data : DataFrame or dict
            The data used for building design matrices.
        sum_of_squares : int or str
            Type of sum of squares (1/"I", 2/"II", 3/"III").

        Returns
        -------
        dict
            The factor_effects dictionary with all per-factor statistics.
        """
        #factor_effects = self._new_factor_effects_dict()

        if self.FactorEffects.ss_type in ["I", 1]:
            self._compute_ss_type1(data)

        elif self.FactorEffects.ss_type in ["II", 2]:
            self._compute_ss_type2(data)

        elif self.FactorEffects.ss_type in ["III", 3]:
            self._compute_ss_type3(data)

        return self.FactorEffects


    def _compute_ss_type1(self, data):
        """
        Compute Type I (Sequential) sum of squares.

        Type I SS is calculated by measuring the reduction in the model's
        overall sum of square error by subtracting the SSE for each term
        from the SSE of the model before it. The difference is the factor's
        sum of squares.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        previous_sse = self.ModelEffects.ss_total
        terms_to_include = []

        for term in self._IV_design_info.term_names:
            if term.strip().upper() == "INTERCEPT":
                terms_to_include.append(term)
                continue

            if term not in terms_to_include:
                terms_to_include.append(term)

            # Build design matrix for terms included so far
            design_info = self._IV_design_info.subset(terms_to_include)
            x = np.asarray(patsy.build_design_matrices([design_info], data))[0]

            # Compute SSE for this cumulative model
            current_sse = self._compute_sse_from_design(x, self.DV)

            # Factor SS = reduction in SSE from adding this term
            ss_factor = previous_sse - current_sse

            # Degrees of freedom for this factor
            term_subset = self._IV_design_info.subset(term)
            term_design = np.asarray(
                patsy.build_design_matrices([term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(term, ss_factor, df_factor, stats)

            # Update previous SSE for next iteration
            previous_sse = current_sse


    def _compute_ss_type2(self, data):
        """
        Compute Type II (Hierarchical/Partially Sequential) sum of squares.

        Type II SS is the reduction in residual error due to adding the term
        to the model after all other terms except those that contain it.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        # Use complete cases for Type II (consistent with original implementation)
        complete_data = data[~data.isna().any(axis=1)]

        for current_term in self._IV_design_info.term_names:
            if current_term.strip().upper() == "INTERCEPT":
                continue

            # Build reduced model: all terms except those that contain current_term
            terms_in_model = list(self._IV_design_info.term_names)
            for term in self._IV_design_info.term_names:
                if term.strip().upper() == "INTERCEPT":
                    continue
                if current_term in term:
                    terms_in_model.remove(term)

            # Build full comparison model: reduced + current_term
            compare_model = list(terms_in_model)
            compare_model.append(current_term)

            # Build design matrices
            design_reduced = self._IV_design_info.subset(terms_in_model)
            x_reduced = np.asarray(patsy.build_design_matrices(
                [design_reduced], complete_data))[0]

            design_full = self._IV_design_info.subset(compare_model)
            x_full = np.asarray(patsy.build_design_matrices(
                [design_full], complete_data))[0]

            # Compute SSE for both models
            sse_reduced = self._compute_sse_from_design(x_reduced, self.DV)
            sse_full = self._compute_sse_from_design(x_full, self.DV)

            # Factor SS = SSE(reduced) - SSE(full)
            ss_factor = sse_reduced - sse_full

            # Degrees of freedom for this factor
            current_term_subset = self._IV_design_info.subset(current_term)
            current_term_design = np.asarray(
                patsy.build_design_matrices([current_term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(current_term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(current_term, ss_factor, df_factor, stats)


    def _compute_ss_type3(self, data):
        """
        Compute Type III (Marginal/Orthogonal) sum of squares.

        Type III SS gives the sum of squares that would be obtained for each
        factor if it were entered into the model last. This requires re-fitting
        the model using Sum coding (rather than Treatment/dummy coding) to
        ensure orthogonality.

        Best for models with interactions, where each main effect can be
        interpreted independently.

        Parameters
        ----------
        factor_effects : dict
            Accumulator dictionary to append results into.
        data : DataFrame or dict
            Data for building design matrices.
        """
        # Convert Treatment-coded terms to Sum-coded equivalents
        the_terms_3 = self._build_type3_terms(self._IV_design_info.term_names)

        # Re-fit the model using Sum coding
        full_model_formula = (
            self._DV_design_info.term_names[0]
            + " ~ "
            + " + ".join(the_terms_3[1:])
        )
        y, x_full = patsy.dmatrices(full_model_formula, data, eval_env=1)

        # For each term, compute SS by comparing full model to model-without-term
        for current_term in the_terms_3:
            if current_term.strip().upper() == "INTERCEPT":
                continue

            # Build the Sum-coded terms list for the reduced model (all terms minus current)
            terms_in_model = list(the_terms_3)
            terms_in_model.remove(current_term)

            # Build design matrix for reduced model
            design_reduced = x_full.design_info.subset(terms_in_model)
            x_reduced = np.asarray(patsy.build_design_matrices(
                [design_reduced], data))[0]

            # Compute SSE for reduced model
            sse_reduced = self._compute_sse_from_design(x_reduced, self.DV)

            # Factor SS = SSE(reduced) - SSE(full model)
            ss_factor = sse_reduced - self.ModelEffects.ss_residual

            # Degrees of freedom for this factor
            current_term_subset = x_full.design_info.subset(current_term)
            current_term_design = np.asarray(
                patsy.build_design_matrices([current_term_subset], data))[0]
            df_factor = np.linalg.matrix_rank(current_term_design) - 1

            # Compute F, p-value, and effect sizes
            stats = self._compute_factor_stats(ss_factor, df_factor)

            # Store results
            self._append_factor_effects(current_term, ss_factor, df_factor, stats)


    def results(self, include_test_stat_p: object = True, include_effect_sizes: object = True, factor_effects: object = True,
                return_type: object = "Dataframe", na_rep: object = '', pretty_format: object = True, table_decimals: object = None, *args: object) -> ModelResults:
        """
        Return the ANOVA results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        return_type : str, optional
            ``"Dataframe"`` (default) or ``"Dictionary"``.
        decimals : int, optional
            Number of decimal places for rounding. Default is 4.
        pretty_format : bool, optional
            If True (default), empty-string placeholders are used for cells
            that don't apply (e.g., F-value for Residual row) — suitable for
            display. If False, ``np.nan`` is used instead — suitable for
            programmatic consumption.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Analysis of Variance"``
            - ``fit_statistics``: Descriptive fit statistics (DataFrame or dict)
            - ``model_table``: Full ANOVA table with factor rows, effect sizes (DataFrame or dict)
            - ``coefficients``: ``None`` (ANOVA's primary output is the model_table)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, anova_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.model_table
        """

        return self._get_results(include_test_stat_p=include_test_stat_p,
                                 include_effect_sizes=include_effect_sizes,
                                 factor_effects=factor_effects,
                                 return_type=return_type,
                                 pretty_format=pretty_format,
                                 table_decimals=table_decimals)



    def regression_table(self, return_type: object = "Dataframe", *args: object) -> DataFrame | None:
        """

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        args : object

        Returns
        -------
        If return_type is "Dataframe": self.ModelResults.as_dataframe("coefficients", self.ModelResults.coefficients)
        If return_type is "Dictionary": self.ModelResults.coefficients
        """
        if return_type == "Dataframe":
            if isinstance(self.ModelResults.coefficients, pd.DataFrame):
                return self.ModelResults.coefficients

            elif isinstance(self.ModelResults.coefficients, dict):
                return self.ModelResults.as_dataframe("coefficients", self.ModelResults.coefficients)

        return None


    # -------------------------------------------------------------------- #
    #               Summary Overrides for ANOVA                            #
    # -------------------------------------------------------------------- #
    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Left header for ANOVA: just the model display name.

        Overrides OLS's mini-ANOVA table because the full ANOVA table
        is already shown in the body section.
        """
        #return [self._get_model_display_name()] + [f"Number of obs = {self.n}"]
        return [self._get_model_display_name()]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Right header for ANOVA: fit statistics from the descriptives DataFrame.

        Anova's descriptives_df is *index-oriented* (stat names as index,
        values in column 0), unlike OLS's transposed single-row format.
        """
        #if descriptives_df is not None:
        #    desc_lines = descriptives_df.to_string(header=False).split("\n")
        #    return [desc_lines[0]] + self._splice_f_stat_lines() + desc_lines[1:]

        #return super()._summary_header_right(width)
        return [f"Number of obs = {self.n:>8}"]


    def _summary_header_anova(self, width=78, model_summary_df=None):
        """
        Build the left side of the OLS summary header with an ANOVA source table.

        Shows model name followed by a Source/SS/df/MS table with Model, Residual,
        and Total rows.

        Parameters
        ----------
        width : int
            Total character width of the output.
        model_summary_df : DataFrame or None
            Model summary DataFrame from ``self.results()``.  When None the
            ANOVA mini-table is skipped (used by Anova subclass).

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """

        #if model_summary_df is None: return [self._get_model_display_name()]

        if model_summary_df is None:
            if self.ModelResults.model_table is None:
                return [self._get_model_display_name()]
            else:
                if not isinstance(self.ModelResults.model_table, pd.DataFrame):
                    table = self.ModelResults.as_dataframe("model_table", self.ModelResults.model_table)
                else:
                    table = self.ModelResults.model_table.copy()
        else:
            if not isinstance(self.ModelResults.model_table, pd.DataFrame):
                table = pd.DataFrame.from_dict(model_summary_df)
            else:
                table = model_summary_df.copy()


        model_display = self.ModelFit.model_display_name


        lines = []
        sep = "-" * width
        lines.append(sep)
        lines.append(model_display)
        lines.append(table.to_string(index=False, na_rep="", justify="right"))
        lines.append(sep)

        return lines


    def _summary_coef_table(self, df_results, width=78, table_decimals=None):
        """
        Build the ANOVA table as the summary body using
        ``DataFrame.to_string()``.

        Overrides the coefficient table that OLS / CoreModel would normally
        show.  Displays Model, individual factor rows, Residual, and Total
        with SS, df, MS, F, p-value, Eta², Epsilon², Omega².

        Parameters
        ----------
        df_results : DataFrame or None
            The ANOVA results DataFrame from ``self.results()``.
        width : int
            Total character width of the output.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        str
            Formatted ANOVA table string.
        """
        # ---- Resolve decimal places -------------------------------------
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # ---- Prepare the DataFrame for display -------------------------
        table = self.ModelResults.as_dataframe("model_table", self.ModelResults.model_table)


        # ---- Build the output string -----------------------------------
        sep = "-" * width

        table_str = table.to_string(
            index=False,
            na_rep="",
            justify="right",
        )

        lines = [
            sep,
            table_str,
            sep,
            "Note: Effect size values for factors are partial.",
        ]

        return "\n".join(lines)


# Convenience aliases for users who prefer different naming conventions
ANOVA = Anova