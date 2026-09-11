import numpy
import scipy.stats
import patsy
import pandas
import re

from .summary import summarize
from .model import model
from .utility import *




class difference_test(object):
    """

    A method that conducts various difference tests and effect size measures which
    will be returned as a Pandas DataFrame (default) or a Python dictionary object. Two
    objects will be returned for all tests; first object is descriptive statistic
    information, and the second object is the statistical testing information as
    well as any effect sizes that were specified.

    Available difference tests:

        [] Independent sampes t-test;
            > equal_variances = True, independent_samples = True
        [] Paired samples t-test;
            > equal_variances = True, independent_samples = False
        [] Welch's t-test;
            > equal_variances = False, independent_samples = True
        [] Wilcoxon signed-rank test.
            > equal_variances = False, independent_samples = False

    Effect size measures are passed in the -conduct()- step; available effect
    size options are: "Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2",
    "r", or "all".

    Can be conducted in one step, or two steps:

        One step
        --------
        difference_test(formula_like, data).conduct(return_type = "Dataframe", effect_size = None)

        Two step
        --------
        model = difference_test(formula_like, data)
        model.conduct(return_type = "Dataframe", effect_size = None)

    """
    def __init__(self, formula_like, data = {}, conf_level = 0.95,
                 equal_variances = True, independent_samples = True,
                 wilcox_parameters = {"zero_method" : "pratt", "correction" : False, "mode" : "auto"},
                 welch_dof = "satterthwaite", **keywords):

        if wilcox_parameters["zero_method"] not in ["pratt", "wilcox"]:
            return print(" ",
                         "Only 'pratt' and 'wilcox' methods are supported."
                         " ",
                         sep = "\n"*2)

        # Determining which test to conduct
        if equal_variances == True and independent_samples == True:
            name = "Independent samples t-test"
        if equal_variances == True and independent_samples == False:
            name = "Paired samples t-test"
        if equal_variances == False and independent_samples == True:
            name = "Welch's t-test"
        if equal_variances == False and independent_samples == False:
            name = "Wilcoxon signed-rank test"
            parameters = {"zero_method" : "pratt", "correction" : False, "mode" : "auto"}
            parameters.update(wilcox_parameters)

        self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        # Checking number of groups in IV, if > 2 stop function
        if len(self.IV.design_info.column_names) > 2:
            return print(" ",
                         "ERROR",
                         "The independent variables has more than 2 groups.",
                         "This method is not appropriate for this many groups.",
                         " ",
                         sep = "\n"*2)

        # Cleaning up category names from Patsy output
        categories = [re.findall(r"\[(.*)\]", name)[0] for name in self.IV.design_info.column_names]

        if name == "Wilcoxon signed-rank test":
            self.parameters = {"Test name" : name,
                               "Formula" : formula_like,
                               "Conf. Level": conf_level,
                               "Categories" : categories,
                               "Equal variances" : equal_variances,
                               "Independent samples" : independent_samples,
                               "Wilcox parameters" : parameters}
        else:
            self.parameters = {"Test name" : name,
                               "Formula" : formula_like,
                               "Conf. Level": conf_level,
                               "Categories" : categories,
                               "Equal variances" : equal_variances,
                               "Independent samples" : independent_samples,
                               "Welch DoF" : welch_dof}


    def conduct(self, return_type = "Dataframe", effect_size = None, decimals = 4):
        """
        Parameters
        ----------
        return_type : string, optional
            What data structure to return, available options are "Dataframe" or "Dictionary". The default is "Dataframe".
        effect_size : string or list, optional
            What effect size measures should be calculated and returned apart of the results table; the default is None.

            Options are: "Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2", "r", or "all" - "all" will calculated all the effect sizes.
            Note that only Rank-Biserial r will be calculated for the Wilcoxon signed-rank test.

        decimals : Integer, optional
            How many decimal places should the returned data be rounded to; the default is 4.

        Returns
        -------
        Summary table : Pandas Dataframe or Dictionary
            Contains the summary statistic information for the test.
        Testing results table : Pandas Dataframe or Dictionary
            Containts the statistical testing information as well as any effect size measures.

        """
        # -- Parameter check
        if return_type.upper() not in ["DATAFRAME", "DICTIONARY"]:
            return print(" ",
                         "Not a supported return type. Only 'Dataframe' and 'Dictionary' are supported at this time.",
                         " ",
                         sep = "\n"*2)

        if effect_size is not None:
            if type(effect_size) == str and effect_size != "all":
                effect_size = list(effect_size)
            elif type(effect_size) == str and effect_size == "all":
                effect_size = ["Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2", "r"]

            for es in effect_size:
                if es  not in [None, "Cohen's D", "Hedge's G", "Glass's delta1", "Glass's delta2", "r", "all"]:
                    return print(" ",
                                 "Not a supported effect size. Either enter None or one of the following: 'Cohen's D', 'Hedge's G', 'Glass's delta1', 'Glass's delta2', 'r', and 'all'.",
                                 " ",
                                 sep = "\n"*2)


        # -- Splitting into seperate arrays and getting descriptive statistics
        group1, group2 = numpy.hsplit(self.IV, 2)
        group1 = self.DV[group1 == 1]
        group2 = self.DV[group2 == 1]

        # Getting the summary table ready - part 1
        group1_info = summarize(group1, stats = ["N", "Mean", "SD", "SE", "Variance", "CI"], name = self.parameters["Categories"][0], ci_level = self.parameters["Conf. Level"], decimals = 64, return_type = "Dictionary")
        group2_info = summarize(group2, stats = ["N", "Mean", "SD", "SE", "Variance", "CI"], name = self.parameters["Categories"][1], ci_level = self.parameters["Conf. Level"], decimals = 64, return_type = "Dictionary")

        combined = summarize(self.DV,
                             stats = ["N", "Mean", "SD", "SE", "Variance", "CI"],
                             name = "combined",
                             ci_level = self.parameters["Conf. Level"],
                             decimals = 64,
                             return_type = "Dictionary")

        diff = {}
        diff["Name"] = "diff"
        diff["Mean"]  = group1_info["Mean"] - group2_info["Mean"]

        # Performing the statistical test
        if self.parameters["Test name"] == "Independent samples t-test":
            stat, pval = scipy.stats.ttest_ind(group1, group2, nan_policy = 'omit')
            stat_name = "t"

            dof = group1_info["N"] + group2_info["N"] - 2

            var_pooled = ( (group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"] ) / (group1_info["N"] + group2_info["N"] - 2)

            se_pooled = numpy.sqrt( (var_pooled / group1_info["N"])  +  (var_pooled / group2_info["N"])    )

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se_pooled)
            diff["SE"] = float(se_pooled)

        if self.parameters["Test name"] == "Paired samples t-test":
            stat, pval = scipy.stats.ttest_rel(group1, group2, nan_policy = 'omit')
            stat_name = "t"

            dof = group1_info["N"] - 1

            difference = group1 - group2

            se = float(scipy.stats.sem(difference, nan_policy= 'omit'))

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se)

            diff["SE"] = se
            diff["SD"] = float(difference.std(ddof = 1))

        if self.parameters["Test name"] == "Welch's t-test":
            stat, pval = scipy.stats.ttest_ind(group1, group2, equal_var = False, nan_policy = 'omit')
            stat_name = "t"

            se_unpooled = numpy.sqrt( group1_info["Variance"] / group1_info["N"] +  group2_info["Variance"] / group2_info["N"]  )

            if self.parameters["Welch DoF"] == "satterthwaite":
                ## Satterthwaite (1946) Degrees of Freedom ##
                dof = ((group1_info["Variance"]/group1_info["N"]) + (group2_info["Variance"]/group2_info["N"]))**2 / ((group1_info["Variance"]/group1_info["N"])**2 / (group1_info["N"]-1) + (group2_info["Variance"]/group2_info["N"])**2 / (group2_info["N"]-1))

            elif self.parameters["Welch DoF"] == "welch":
                ## Welch (1947) Degrees of Freedom ##
                dof = -2 + (((group1_info["Variance"]/group1_info["N"]) + (group2_info["Variance"]/group2_info["N"]))**2 / ((group1_info["Variance"]/group1_info["N"])**2 / (group1_info["N"]+1) + (group2_info["Variance"]/group2_info["N"])**2 / (group2_info["N"]+1)))

                pval = 2 * min((1 - scipy.stats.t.cdf(stat, dof)), scipy.stats.t.cdf(stat, dof))

            ci_lower_diff, ci_upper_diff = scipy.stats.t.interval(self.parameters["Conf. Level"],
                                                                  dof,
                                                                  loc = diff["Mean"],
                                                                  scale = se_unpooled)
            diff["SE"] = float(se_unpooled)

        if self.parameters["Test name"] == "Wilcoxon signed-rank test":
            difference = group1 - group2
            difference = numpy.reshape(difference, (difference.shape[0], ))

            if self.parameters["Wilcox parameters"]['zero_method'] == 'pratt':
                difference_abs = numpy.abs(difference)

                total_n = difference.shape[0]
                positive_n = difference[difference > 0].shape[0]
                negative_n = difference[difference < 0].shape[0]
                zero_n = difference[difference == 0].shape[0]

            elif self.parameters["Wilcox parameters"]['zero_method'] == 'wilcox':
                difference = difference[difference != 0]
                difference_abs = numpy.abs(difference)

                # Calculating Signed Information
                total_n = difference.shape[0]
                positive_n = difference[difference > 0].shape[0]
                negative_n = difference[difference < 0].shape[0]
                zero_n = difference[difference == 0].shape[0]

            elif self.parameters["Wilcox parameters"]['zero_method'] == 'zsplit':
                # Includes zero-differences in the ranking process and split the zero tank between positive and negative ones
                print("This method is not currently supported, please enter either 'wilcox' or 'pratt'.")

            # Ranking the absolute difference |d|
            ranked = scipy.stats.rankdata(difference_abs)
            sign = numpy.where(difference < 0, -1, 1)
            ranked_sign = (sign * ranked)

            # Descriptive Information #
            total_sum_ranks = ranked.sum()
            positive_sum_ranks = ranked[difference > 0].sum()
            negative_sum_ranks = ranked[difference < 0].sum()
            zero_sum_ranks = ranked[difference == 0].sum()

            ## Dropping the Rank of the Zeros
            sign2 = numpy.where(difference == 0, 0, sign)
            ranked2 = sign2 * ranked
            ranked2 = numpy.where(difference == 0, 0, ranked2)

            # Expected T
            T = (sign * ranked_sign).sum()

            # Observered T
            T_obs = (sign2 * ranked2).sum()

            # Expected T+ and T-
            exp_positive = T_obs / 2
            exp_negative = T_obs / 2
            exp_zero = T - T_obs

            var_adj_T = (ranked2 * ranked2).sum()
            e_T_pos = total_n  * (total_n  + 1) / 4
            var_adj_T_pos = (1/4) * var_adj_T

            var_unadj_T_pos = ((total_n * (total_n + 1)) * (2 * total_n + 1)) /24
            var_zero_adj_T_pos = -1 * ((zero_n * (zero_n + 1)) * (2 * zero_n + 1)) /24
            var_ties_adj = var_adj_T_pos - var_unadj_T_pos - var_zero_adj_T_pos

            z = (positive_sum_ranks - exp_positive) / numpy.sqrt(var_adj_T_pos)

            t_val, p_val = scipy.stats.wilcoxon(difference,
                                                zero_method = self.parameters["Wilcox parameters"]['zero_method'],
                                                correction = self.parameters["Wilcox parameters"]["correction"],
                                                mode = self.parameters["Wilcox parameters"]["mode"])

            # -- Effect sizes
            # Pearson r = z / square_root(N)
            pr = z / numpy.sqrt(total_n)
            pbr = t_val / total_sum_ranks

            # -- Descriptive table
            descriptives = {"sign" : ["positive", "negative", "zero", "all"],
                            "obs" : [positive_n, negative_n, zero_n, total_n],
                            "sum ranks" : [positive_sum_ranks, negative_sum_ranks, zero_sum_ranks, total_sum_ranks],
                            "expected" : [exp_positive, exp_negative, exp_zero, T]}

            # -- Variance table
            variance = {"unadjusted variance" : var_unadj_T_pos,
                        "adjustment for ties" : var_ties_adj,
                        "adjustment for zeros" : var_zero_adj_T_pos,
                        "adjusted variance" : var_adj_T_pos}

            # -- Results table
            results = {"z" : z,
                       "w" : t_val,
                       "pval" : p_val}

        if self.parameters["Test name"] != "Wilcoxon signed-rank test":
            # P value tails
            pval_lt = scipy.stats.t.cdf(stat, dof)
            pval_rt = 1 - scipy.stats.t.cdf(stat, dof)

        # Creating testing information table
        if self.parameters["Test name"] == "Wilcoxon signed-rank test":
            ...

        else:
            if self.parameters["Test name"] == "Independent samples t-test":
                test_name = "Independent samples t-test with equal variances"
            elif self.parameters["Test name"] == "Welch's t-test":
                test_name = "Independent samples t-test with unequal variance"
            else:
                test_name = self.parameters["Test name"]

            dof_type = "Degrees of freedom ="
            if self.parameters["Test name"] == "Welch's t-test" :
                if self.parameters["Welch DoF"] == "satterthwaite":
                    dof_type = "Satterthwaite's Degrees of freedom ="
                else:
                    dof_type = "Welch's Degrees of freedom ="


            # -- Result table
            result_table = {test_name : [f"Difference ({self.parameters['Categories'][0]} - {self.parameters['Categories'][1]})",
                                                            dof_type,
                                                            f"{stat_name} =",
                                                            "Two sided test p-value =",
                                                            "Difference < 0 p-value =",
                                                            "Difference > 0 p-value ="
                                                            ],
                            "Results" : [float(diff["Mean"]),
                                         float(dof),
                                         float(stat),
                                         float(pval),
                                         float(pval_lt),
                                         float(pval_rt),

                                         ]}

        # Creating effect size table
        if effect_size is not None:
            if self.parameters["Test name"] == "Wilcoxon signed-rank test":
                if effect_size != "r":
                    print(" ",
                          f"Rank-Biserial r  and Pearson r will be calulcated for the {self.parameters['Test name']}.",
                          " ",
                          sep = "\n"*2)

                results["Rank-Biserial r"] = (descriptives["sum ranks"][0] / descriptives["sum ranks"][-1]) - (descriptives["sum ranks"][1] / descriptives["sum ranks"][-1])
                results["Pearson r"] = results["z"] / numpy.sqrt(descriptives["obs"][-1])

            else:
                for es in effect_size:
                    if es == "Cohen's D":
                        if self.parameters["Test name"] == "Paired samples t-test":
                            # -- DECIDE IF YOU WANT TO SUPPORT THIS VERSION - USED IN RESEARCHPY TTEST()
                            # Cohen's Dz (within-subjects design)
                            #d = stat / numpy.sqrt(group1_info["N"])
                            #result_table[self.parameters["Test name"]].append("Cohen's Dz")
                            #result_table["Results"].append(float(d))

                            # Cohen's Dav (within-subjects design)
                            d = (group1_info["Mean"] - group2_info["Mean"]) / ((group1_info["SD"] + group2_info["SD"]) / 2)
                            result_table[test_name].append("Cohen's Dav")
                            result_table["Results"].append(float(d))

                        else:
                            # Cohen's d (between-subjects desgn)
                            d = (group1_info['Mean'] - group2_info["Mean"]) / numpy.sqrt((((group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"]) / (group1_info["N"] + group2_info["N"] - 2)))

                            result_table[test_name].append("Cohen's Ds")
                            result_table["Results"].append(float(d))

                    if es == "Hedge's G":
                        if self.parameters["Test name"] == "Paired samples t-test":
                            # Cohen's Dz (within-subjects design)
                            #d = stat / numpy.sqrt(group1_info["N"])

                            # Cohen's Dav (within-subjects design)
                            d = (group1_info["Mean"] - group2_info["Mean"]) / ((group1_info["SD"] + group2_info["SD"]) / 2)
                            g = d * (1 - (3 / ((4*(group1_info["N"] + group2_info["N"])) - 9)))

                            result_table[test_name].append("Hedge's Gav")
                            result_table["Results"].append(float(g))

                        else:
                            # Cohen's d (between-subjects desgn)
                            d = (group1_info['Mean'] - group2_info["Mean"]) / numpy.sqrt((((group1_info["N"] - 1)*group1_info["Variance"] + (group2_info["N"] - 1)*group2_info["Variance"]) / (group1_info["N"] + group2_info["N"] - 2)))
                            g = d * (1 - (3 / ((4*(group1_info["N"] + group2_info["N"])) - 9)))

                            result_table[test_name].append("Hedge's G")
                            result_table["Results"].append(float(g))

                    if es == "Glass's delta1":
                        d1 = (group1_info["Mean"] - group2_info["Mean"]) / group1_info["SD"]

                        result_table[test_name].append("Glass's delta1")
                        result_table["Results"].append(float(d1))

                    if es == "Glass's delta2":
                        d2 = (group1_info["Mean"] - group2_info["Mean"]) / group2_info["SD"]

                        result_table[test_name].append("Glass's delta2")
                        result_table["Results"].append(float(d2))

                    if es == "r":
                        r = stat / numpy.sqrt(stat**2 + dof)

                        result_table[test_name].append("Point-Biserial r")
                        result_table["Results"].append(float(r))

        # -- Getting the summary table ready - part 2
        if self.parameters["Test name"] != "Wilcoxon signed-rank test":
            diff[f"{int(self.parameters['Conf. Level'] * 100)}% Conf. Interval"] = [ci_lower_diff, ci_upper_diff]

            # P value tails
            #pval_lt = scipy.stats.t.cdf(stat, dof, loc = diff["Mean"], scale = diff["SE"])
            #pval_rt = 1 - scipy.stats.t.cdf(stat, dof, loc = diff["Mean"], scale = diff["SE"])

            group1_table = pandas.DataFrame.from_dict(group1_info, orient = 'index').T
            group2_table = pandas.DataFrame.from_dict(group2_info, orient = 'index').T
            combined_table = pandas.DataFrame.from_dict(combined, orient = 'index').T
            diff_table = pandas.DataFrame.from_dict(diff, orient = 'index').T

            summary_table = pandas.concat([group1_table, group2_table, combined_table, diff_table], ignore_index = True)
            summary_table.replace(numpy.nan, ' ', inplace = True)

            # Rounding the summary table
            summary_table = summary_table.round(decimals)

            result_table = pandas.DataFrame(result_table)
            result_table = result_table.round(decimals)

            # Rounding the confidence interval values
            if self.parameters["Test name"] != "Wilcoxon signed-rank test":
                for row in summary_table[f"{int(self.parameters['Conf. Level'] * 100)}% Conf. Interval"]:
                    idx = 0
                    for value in row:
                        row[idx] = round(value, 4)
                        idx +=1

            # Returning the information
            if return_type == "Dataframe":
                if self.parameters["Test name"] == "Wilcoxon signed-rank test":
                    return summary_table.iloc[:-2, :], result_table
                elif self.parameters["Test name"] == "Paired samples t-test":
                    return summary_table.drop(2), result_table
                else:
                    return summary_table, result_table

            if return_type == "Dictionary":
                if self.parameters["Test name"] == "Wilcoxon signed-rank test":
                    return summary_table.iloc[:-2, :].to_dict(), result_table.to_dict()
                elif self.parameters["Test name"] == "Paired samples t-test":
                    return summary_table.drop(2).to_dict(), result_table.to_dict()
                else:
                    return summary_table.to_dict(), result_table.to_dict()

        else:
            if return_type == "Dataframe":
                descriptives = pandas.DataFrame.from_dict(descriptives)
                variance = pandas.DataFrame.from_dict(variance, orient = 'index').T
                results = pandas.DataFrame.from_dict(results, orient = 'index').T

            return descriptives, variance, results
