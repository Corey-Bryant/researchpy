# -*- coding: utf-8 -*-
"""
Analysis of Variance (ANOVA)

This module provides the anova class for conducting analysis of variance using the ordinary least squares method.

DEPRECATION NOTICE:
    The `anova` class in this module is maintained for backward compatibility.
    It will be deprecated in a future version. New code should use:

        from researchpy.models import CoreModel, GeneralModel
        from researchpy.models.multivariable import LogisticRegression
        # or use aliases
        from researchpy.models.multivariable import Logistic

    The new modular structure provides better organization and more functionality.
"""

# %% Importing libraries
import warnings
import numpy
import scipy.stats
import patsy
import pandas

from researchpy.summary import summarize
from researchpy.ols import ols
from researchpy.utility import *


# %% Creating anova class which
class anova(ols):
    """

    Parameters
    ----------
    formula_like: string
        A string which represents a valid Patsy formula; https://patsy.readthedocs.io/en/latest/

    data : array_like
        Array like data object.

    sum_of_squares : integer
        Integer value to indicate which type of sum of squares should be calculated.
        Sum of squares types 1, 2, and 3 are supported.

    Returns
    -------
    Anova object with assessible methods and stored class data. The class data
    which is stored is the following:


        self.model_data: dictionary object
            The following data is stored with the dictionary key ("Key"):
                J matrix ('J')
                Identify matrix ('I')
                Hat matrix ('H')
                Coeffeicients ('betas')
                Total Sum of Squares ('sum_of_square_total')
                Model Sum of Squares ('sum_of_square_model')
                Residual Sum of Squares ('sum_of_square_residual')
                Model Degrees of Freedom ('degrees_of_freedom_model')
                Residual Degrees of Freedom ('degrees_of_freedom_residual')
                Total Degrees of Freedom ('degrees_of_freedom_total')
                Model Mean Squares ('msr')
                Error Mean Squares ('mse')
                Total Mean Squares ('mst')
                Root Mean Square Error ('root_mse')
                Model F-value ('f_value_model')
                Model p-value ('f_p_value_model')
                R-sqaured ('r squared')
                Adjusted R-squared ('r squared adj.')
                Eta squared ('Eta squared')
                Omega squared ('Omega squared')

    """

    def __init__(self, formula_like, data={}, sum_of_squares=3):
        super().__init__(formula_like, data)
        warnings.warn(
                "The 'anova' class in researchpy.anova is deprecated and will be removed in a future version. "
                "Please use 'Anova' from researchpy.models.multivariable instead: "
                "from researchpy.models.multivariable import Anova",
                DeprecationWarning,
                stacklevel=2
        )

        self.__name__ = "researchpy.anova"

        ###########

        factor_effects = {"Source": [],
                          "Sum of Squares": [],
                          "Degrees of Freedom": [],
                          "Mean Squares": [],
                          "F value": [],
                          "p-value": [],
                          "Eta squared": [],
                          "Epsilon squared" : [],
                          "Omega squared": []}

        ###########################################################
        #               CODUCTING THE TEST                        #
        ###########################################################
        if sum_of_squares in ["I", 1]:

            """
            Type I sum of squares is also known as sequential sum of squares and is calculated by
                measuring the reduction in the model's overall sum of square error by subtracting
                the sum of square error for each term from the sum of square error from the model
                before. The difference between the two is the sum of square factor effect for the
                current term.
            """

            previous_sum_of_squares_error = self.model_data["sum_of_square_total"]
            terms_to_include = []

            for term in self._IV_design_info.term_names:
                if term.strip().upper() == "INTERCEPT":
                    terms_to_include.append(term)
                    continue

                else:
                    if term not in terms_to_include:
                        terms_to_include.append(term)

                    # Getting design matrix set up
                    design_info2 = self._IV_design_info.subset(
                        terms_to_include)
                    x = numpy.asarray(patsy.build_design_matrices(
                        [design_info2], data))[0]

                    # Hat matrix
                    h = x @ numpy.linalg.inv(x.T @ x) @ x.T

                    # Predicted y values
                    predicted_y = h @ self.DV

                    # Calculation of residuals (error)
                    residuals = self.DV - predicted_y

                    ### Error sum of squares (SSE)
                    sum_of_square_residual = residuals.T @ residuals

                    # Calculating the Factor Effect Sum of Squares(SSR, and SSE)
                    sum_of_square_factor = previous_sum_of_squares_error - sum_of_square_residual

                    ### Degrees of freedom - Factor
                    term_subset = self._IV_design_info.subset(term)
                    term_design = numpy.asarray(
                        patsy.build_design_matrices([term_subset], data))[0]

                    degrees_of_freedom_factor = numpy.linalg.matrix_rank(
                        term_design) - 1

                    ### Mean Square - Factor (MSR-F)
                    msr_f = sum_of_square_factor * \
                        (1 / degrees_of_freedom_factor)

                    ### F-values
                    # Model
                    f_value_model = msr_f / self.model_data["mse"]
                    f_p_value_model = scipy.stats.f.sf(
                        f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                    ### Partial Effect Size Measures
                    eta_squared_partial = sum_of_square_factor / \
                        (sum_of_square_factor
                         + self.model_data["sum_of_square_residual"])
                        
                    epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                        (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                    omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                        ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])

                    
                    # Updating items
                    factor_effects["Source"].append(term)
                    factor_effects["Sum of Squares"].append(
                        float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(
                        float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    
                    factor_effects["Omega squared"].append(float(omega_squared_partial))
                    #factor_effects["r squared"].append("")
                    #factor_effects["r squared adj."].append("")

                    # Setting new Sum of Square Residual
                    previous_sum_of_squares_error = sum_of_square_residual

                self.factor_effects = factor_effects

        elif sum_of_squares in ["II", 2]:
            """
            Type II sum of square is also known as hierarchical or
                partially sequential. This type of sum of square is
                the reduction in residual error due to adding the term
                to the model after all other terms except those
                that contain it.
            """

            for current_term in self._IV_design_info.term_names:
                terms_in_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue

                else:
                    for term in self._IV_design_info.term_names:
                        if term.strip().upper() == "INTERCEPT":
                            continue

                        elif current_term in term:
                            terms_in_model.remove(term)

                        else:
                            continue

                    if current_term.strip().upper() == "INTERCEPT":
                        continue

                    else:
                        compare_model = [item for item in terms_in_model]
                        compare_model.append(current_term)

                # Getting design matrix set up
                design_info2 = self._IV_design_info.subset(terms_in_model)
                x = numpy.asarray(patsy.build_design_matrices(
                    [design_info2], data[~data.isna().any(axis= 1)]))[0]

                design_info3 = self._IV_design_info.subset(compare_model)
                x_c = numpy.asarray(
                    patsy.build_design_matrices([design_info3], data[~data.isna().any(axis= 1)]))[0]




                # Hat matrix
                h = x @ numpy.linalg.inv(x.T @ x) @ x.T
                h_c = x_c @ numpy.linalg.inv(x_c.T @ x_c) @ x_c.T

                # Predicted y values
                predicted_y = h @ self.DV
                predicted_y_c = h_c @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y
                residuals_c = self.DV - predicted_y_c

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals
                sum_of_square_residual_c = residuals_c.T @ residuals_c

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual - sum_of_square_residual_c

                ### Degrees of freedom - Factor
                current_term_subset = self._IV_design_info.subset(current_term)
                current_term_design = numpy.asarray(
                    patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = numpy.linalg.matrix_rank(
                    current_term_design) - 1

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)

                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(
                    f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                ### Partial Effect Size Measures
                eta_squared_partial = sum_of_square_factor / \
                    (sum_of_square_factor
                     + self.model_data["sum_of_square_residual"])
                    
                epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])

                
                # Updating items
                try:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    factor_effects["Omega squared"].append(float(omega_squared_partial))

                except:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float((sum_of_square_factor).item()))
                    factor_effects["Degrees of Freedom"].append(float((degrees_of_freedom_factor).item()))
                    factor_effects["Mean Squares"].append(float((msr_f).item()))
                    factor_effects["F value"].append(float((f_value_model).item()))
                    factor_effects["p-value"].append(float((f_p_value_model).item()))

                    factor_effects["Eta squared"].append(float((eta_squared_partial).item()))
                    factor_effects["Epsilon squared"].append(float((epsilon_squared_partial).item()))
                    factor_effects["Omega squared"].append(float((omega_squared_partial).item()))

            self.factor_effects = factor_effects

        elif sum_of_squares in ["III", 3]:

            """
            Type III sum of square is also known as marginal or orthogonal. This
                calculation of sum of squares give the sum of squares that would be
                obtained for each factor if it were to be entered into the model last.

            Best for if using an interaction (each main effect is able to be interpereted
                although not worthwhile if the interaction is significant)
            """

            ## This handles specification of reference category
            reference_pattern = re.compile(r'(?<=,|\s)(Treatment\(.*\))(?=\))')

            the_terms_3 = []
            for term in self._IV_design_info.term_names:

                if "Treatment" in term:
                    split_terms = term.split(":")

                    if len(split_terms) == 1:
                        the_terms_3.append(
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

                        the_terms_3.append(':'.join(interaction_terms))

                else:
                    the_terms_3.append(term.replace(")", ", Sum)"))


            # Re-fitting the model using the correct specifications
            full_model = self._DV_design_info.term_names[0] + \
                " ~ " + " + ".join(the_terms_3[1:])
            y, x_full = patsy.dmatrices(full_model, data, eval_env=1)
            
                    
            terms_design_info = []
            for term in x_full.design_info.term_names[1:]:
                terms_design_info.append(x_full.design_info.subset(term))


            for current_term in the_terms_3:

                terms_in_model = []
                for term in self._IV_design_info.term_names:
                    if "Treatment" in term:
                        split_terms = term.split(":")

                        if len(split_terms) == 1:
                            terms_in_model.append(
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

                            terms_in_model.append(':'.join(interaction_terms))

                    else:
                        terms_in_model.append(term.replace(")", ", Sum)"))

                compare_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue
                else:
                    terms_in_model.remove(current_term)                

                
                # Building Design Matrices
                design_info2 = x_full.design_info.subset(terms_in_model)
                
                x = numpy.asarray(patsy.build_design_matrices(
                    [design_info2], data))[0]
                

                # Hat matrix
                h = x @ numpy.linalg.inv(x.T @ x) @ x.T

                # Predicted y values
                predicted_y = h @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual - \
                    self.model_data["sum_of_square_residual"]

                ### Degrees of freedom - Factor
                current_term_subset = x_full.design_info.subset(current_term)
                current_term_design = numpy.asarray(
                    patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = numpy.linalg.matrix_rank(
                    current_term_design) - 1
                

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)
                


                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(
                    f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                ### Partial Effect Size Measures
                eta_squared_partial = sum_of_square_factor / \
                    (sum_of_square_factor
                     + self.model_data["sum_of_square_residual"])
                    
                epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])
                

                
                # Updating items
                try:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    factor_effects["Omega squared"].append(float(omega_squared_partial))

                except:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float((sum_of_square_factor).item()))
                    factor_effects["Degrees of Freedom"].append(float((degrees_of_freedom_factor).item()))
                    factor_effects["Mean Squares"].append(float((msr_f).item()))
                    factor_effects["F value"].append(float((f_value_model).item()))
                    factor_effects["p-value"].append(float((f_p_value_model).item()))

                    factor_effects["Eta squared"].append(float((eta_squared_partial).item()))
                    factor_effects["Epsilon squared"].append(float((epsilon_squared_partial).item()))
                    factor_effects["Omega squared"].append(float((omega_squared_partial).item()))

                self.factor_effects = factor_effects
                

    def results(self, return_type="Dataframe", decimals=4, pretty_format=True):

        if pretty_format == True:

            descriptives = {

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals),
                    "R-squared = ": round(self.model_data["r squared"], decimals),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source": ["Model", ''],
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals), ''],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals), ''],
                    "Mean Squares": [round(self.model_data["msr"], decimals), ''],
                    "F value": [round(self.model_data["f_value_model"], decimals), ''],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals), ''],
                    "Eta squared": [round(self.model_data["Eta squared"], decimals), ''],
                    "Epsilon squared": [round(self.model_data["Epsilon squared"], decimals), ''],
                    "Omega squared": [round(self.model_data["Omega squared"], decimals), '']

                    }

            factors = self.factor_effects.copy()
            factors["Source"] = [patsy_term_cleaner(
                term) for term in self._IV_design_info.term_names[1:]]
            rounder(factors["Sum of Squares"], decimals=decimals)
            rounder(factors["Degrees of Freedom"], decimals=decimals)
            rounder(factors["Mean Squares"], decimals=decimals)
            rounder(factors["F value"], decimals=decimals)
            rounder(factors["p-value"], decimals=decimals)
            rounder(factors["Eta squared"], decimals=decimals)
            rounder(factors["Epsilon squared"], decimals=decimals)
            rounder(factors["Omega squared"], decimals=decimals)

            bottom = {

                    "Source": ['', "Residual", "Total"],
                    "Sum of Squares": ['', round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom": ['', round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares": ['', round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value": ['', '', ''],
                    "p-value": ['', '', ''],
                    "Eta squared": ['', '', ''],
                    "Epsilon squared" : ['', '', ''],
                    "Omega squared": ['', '', '']

                    }

            results = {

                    "Source": top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares": top["Sum of Squares"] + factors["Sum of Squares"] + bottom["Sum of Squares"],
                    "Degrees of Freedom": top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares": top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value": top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value": top["p-value"] + factors["p-value"] + bottom["p-value"],
                    "Eta squared": top["Eta squared"] + factors["Eta squared"] + bottom["Eta squared"],
                    "Epsilon squared": top["Epsilon squared"] + factors["Epsilon squared"] + bottom["Epsilon squared"],
                    "Omega squared": top["Omega squared"] + factors["Omega squared"] + bottom["Omega squared"]

                    }

        else:

            descriptives = {

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals),
                    "R-squared = ": round(self.model_data["r squared"], decimals),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source": ["Model"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals)],
                    "Mean Squares": [round(self.model_data["msr"], decimals)],
                    "F value": [round(self.model_data["f_value_model"], decimals)],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals)]

                    }

            factors = self.factor_effects.copy()
            patsy_term_cleaner(factors["Source"])
            rounder(factors["Sum of Squares"], decimals=decimals)
            rounder(factors["Degrees of Freedom"], decimals=decimals)
            rounder(factors["Mean Squares"], decimals=decimals)
            rounder(factors["F value"], decimals=decimals)
            rounder(factors["p-value"], decimals=decimals)

            bottom = {

                    "Source": ["Residual", "Total"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares": [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value": [numpy.nan, numpy.nan],
                    "p-value": [numpy.nan, numpy.nan]

                    }

            results = {

                    "Source": top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares": top["Sum of Squares"] + factors["Sum of Squares"] + bottom["Sum of Squares"],
                    "Degrees of Freedom": top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares": top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value": top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value": top["p-value"] + factors["p-value"] + bottom["p-value"]

                    }

        if return_type == "Dataframe":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (pandas.DataFrame.from_dict(descriptives, orient="index"), pandas.DataFrame.from_dict(results))

        elif return_type == "Dictionary":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (descriptives, results)

        else:

            print(
                "Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

    def regression_table(self, return_type="Dataframe", decimals=4, pretty_format=True, conf_level=0.95):

        return super().results(return_type=return_type, decimals=decimals, pretty_format=pretty_format, conf_level=conf_level)[2]

    def predict(self, estimate=None):
        return super().predict(estimate=estimate)
