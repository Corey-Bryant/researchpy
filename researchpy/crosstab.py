# -*- coding: utf-8 -*-
"""
Created on Thursday, August  2, 2018
@author: Corey Bryant

Last updated on Wednesday, March 4, 2026
Updated by @author: Corey Bryant
"""

import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats import contingency_tables

def crosstab(group1, group2, prop=None, test=False, margins=True,
             correction=False, cramer_correction=False, exact=False, expected_freqs=False):

    if not isinstance(group1, pd.Series) or not isinstance(group2, pd.Series):
        return "Operation only supports Pandas Series"

    else:
        ## Creating the contingency table ##
        contingency_table = pd.crosstab(group1, group2, margins = True)

        # --- Shape and size of contigency table ---
        num_row = contingency_table.shape[0] - 1
        num_col = contingency_table.shape[1] - 1
        n = contingency_table.iloc[-1, -1]

        # --- Creating a multi-index for columns and index ---
        multiindex_columns = pd.MultiIndex.from_product([[f"{contingency_table.columns.name}"], contingency_table.columns])
        multiindex_index = pd.MultiIndex.from_product([[f"{contingency_table.index.name}"], contingency_table.index])
        multiindex_columns_names = ['', '']
        multiindex_index_names = ['', '']

        ## Setting main crosstabulation table ##
        if prop:
            if prop == 'row':
                ct = round(contingency_table.div(contingency_table.iloc[:,-1], axis=0).mul(100, axis=0), 2)
            elif prop == 'col':
                ct = round(contingency_table.div(contingency_table.iloc[-1, :], axis=1).mul(100, axis=1), 2)
            elif prop == 'cell':
                ct = round(contingency_table.div(contingency_table.iloc[-1,-1], axis=0).mul(100, axis=1), 2)
        else:
            ct = contingency_table

        ct.columns = multiindex_columns
        ct.index = multiindex_index
        ct.columns.names = multiindex_columns_names
        ct.index.names = multiindex_index_names

        if not margins: ct = ct.iloc[:-1, :-1]

        if test:
            lambda_ = None

            if test.lower() == "fisher":
                if (num_row, num_col) != (2, 2):
                    raise ValueError("Fisher's exact test is only valid for 2x2 contingency tables.")

                test_name = "Fisher's exact test"
                report_name = "Odds ratio"

            elif test.lower() in ["chi-square", "chi2"]:
                test_name = "Chi-square test of independence"
                report_name = "Pearson Chi-square"

            elif test.lower() in ["g-test", "gtest", "g2"]:
                test_name = "G-test"
                report_name = "Log-likelihood ratio"
                lambda_ = "log-likelihood"

            elif test.lower() == "mcnemar":
                test_name = "McNemar"
                report_name = "McNemar's Chi-square"


            if test_name == "McNemar":
                results = contingency_tables.mcnemar(contingency_table.iloc[:-1, :-1],
                                                     exact=exact,
                                                     correction=correction)
                chi2 = results.statistic
                p = results.pvalue
                dof = 1
            else:
                chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table.iloc[:-1, :-1],
                                                                      correction=correction,
                                                                      lambda_=lambda_)
                if test_name == "Fisher's exact test":
                    ods2, p2 = scipy.stats.fisher_exact(contingency_table.iloc[:-1, :-1])
                    odsl, pl = scipy.stats.fisher_exact(contingency_table.iloc[:-1, :-1], 'less')
                    odsg, pg = scipy.stats.fisher_exact(contingency_table.iloc[:-1, :-1], 'greater')


                expected = pd.DataFrame(expected,
                                        index=multiindex_index[:-1],
                                        columns=multiindex_columns[:-1]
                                        )

                expected.columns.names = multiindex_columns_names
                expected.index.names = multiindex_index_names

            # --- Effect Size Calculations
            if contingency_table.iloc[:-1, :-1].size == 4:
                # --- Cramer's phi = square_root(chi_square / N)
                es_name = "Cramer's phi"
                es = np.sqrt(chi2 / n)

            elif contingency_table.iloc[:-1, :-1].size > 4:
                # --- Cramer's V = square_root(chi_square / min(c-1, r-1))
                es_name = "Cramer's V"
                min_dim = min((num_row - 1), (num_col - 1))

                if cramer_correction == True:
                    phi_corrected = (chi2 / n) - ((num_row - 1) * (num_col - 1) / (n - 1))
                    phi_corrected = max(0, phi_corrected)

                    es = np.sqrt(phi_corrected / min_dim)
                else:
                    es = np.sqrt(chi2 / (n * min_dim))

            ## Creating the results table ##
            if test_name == "Fisher's exact test":
                if (num_row, num_col) == (2, 2):
                    results = {f"{test_name}": [f"{report_name} = ",
                                                "2 sided p-value = ",
                                                "Left tail p-value = ",
                                                "Right tail p-value = ",
                                                f"{es_name} = "],
                               "results"     : [round(ods2, 4),
                                                round(p2, 4),
                                                round(pl, 4),
                                                round(pg, 4),
                                                round(es, 4)]
                               }
                else:
                    results = {f"{test_name}": [f"{report_name} = ",
                                                "p-value = ",
                                                f"{es_name} = "],
                               "results"     : [round(ods2, 4),
                                                round(p2, 4),
                                                round(es, 4)]
                               }

            else:
                results = {f"{test_name}": [f"{report_name} ({round(dof, 1)}) = ",
                                            "p-value = ",
                                            f"{es_name} = "],
                           "results"     : [round(chi2, 4),
                                            round(p, 4),
                                            round(es, 4)]}

            results_table = pd.DataFrame.from_dict(results)

            ## Returning DataFrame objects ##
            if expected_freqs:
                if test_name == "McNemar":
                    print("Expected frequency table is not returned. There is no expected frequency assumption for this test.")
                return ct, results_table, expected

            else:
                return ct, results_table

        ## Returning DataFrame objects ##
        if expected_freqs:
            print("Expected frequency table is not returned. There is no test specified with expected frequency assumption.")
        return ct