import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats import contingency_tables

def crosstab(group1, group2, prop=None, test=False, margins=True,
             correction=None, cramer_correction=None, exact=False, expected_freqs=False):

    if not isinstance(group1, pd.Series) or not isinstance(group2, pd.Series):
        return "Operation only supports Pandas Series"

    else:
        ## Creating the crosstab table ##
        crosstab = pd.crosstab(group1, group2)
        crosstab2 = pd.crosstab(group1, group2, margins = True)
        num_row = crosstab2.shape[0] - 1
        num_col = crosstab2.shape[1] - 1
        n = crosstab2.iloc[-1, -1]


        ## Creating percentage tables ##
        # Row
        crosstab_perrow = round(crosstab2.div(crosstab2.iloc[:,-1], axis=0).mul(100, axis=0), 2)
        # Column
        crosstab_percol = round(crosstab2.div(crosstab2.iloc[-1,:], axis=1).mul(100, axis=1), 2)
        # Cell
        crosstab_percell = round(crosstab2.div(crosstab2.iloc[-1,-1], axis=0).mul(100, axis=1), 2)


        ## Chi-square and effect size results ##
        if correction == None or correction == False:
            correction = False

        elif correction == True:
            correction = True

        chi2, p, dof, expected = scipy.stats.chi2_contingency(crosstab,
                                                              correction=correction)
        test_val = chi2

        if test == "chi-square":
            test_name = "Chi-square test"
            test_val = chi2
            p = p
            dof = dof
            expected = expected
        elif test == "g-test":
            test_name = "G-test"
            test_val, p, dof, expected = scipy.stats.chi2_contingency(crosstab,
                                                                      correction=correction,
                                                                      lambda_="log-likelihood")
        elif test == "mcnemar":
            test_name = "McNemar"
            results = contingency_tables.mcnemar(crosstab, exact=exact, correction=correction)
            test_val = results.statistic
            p = results.pvalue
            dof = 1
        elif test == "fisher":
            test_name = "Fisher's exact test"
            test_val = chi2
            ods2, p2 = scipy.stats.fisher_exact(crosstab)
            odsl, pl = scipy.stats.fisher_exact(crosstab, 'less')
            odsg, pg = scipy.stats.fisher_exact(crosstab, 'greater')



        ## Effect size measures ##
        if test:
            # Cramer's phi
            # phi = square_root(chi_square / N)
            # Where N = total sample size
            phi = np.sqrt(test_val / n)

            # Cramer's V
            # V = square_root(chi_square / min(c-1, r-1))
            min_dim = min((num_row - 1), (num_col - 1))

            if min_dim == 0:
                V = np.nan
            elif cramer_correction == True:
                phi_corrected = (test_val / n) - ((num_row - 1) * (num_col - 1) / (n - 1))
                phi_corrected = max(0, phi_corrected)

                row_corrected = num_row - np.square(num_row - 1) / (n - 1)
                col_corrected = num_col - np.square(num_col - 1) / (n - 1)

                V = np.sqrt(phi_corrected / min_dim)

            else:
                V = np.sqrt(test_val / (n * min_dim))




        ## Setting main crosstabulation table ##
        if margins == True and prop == None:
            ct = crosstab2
        elif margins == False and prop == None:
            ct = crosstab
        elif prop == 'row':
            ct = crosstab_perrow
        elif prop == 'col':
            ct = crosstab_percol
        elif prop == 'cell':
            ct = crosstab_percell

        expected = pd.DataFrame(expected, index=crosstab.index,
                                columns=pd.MultiIndex.from_product([[f"{crosstab.columns.name}"], crosstab.columns]))

        ct.columns = pd.MultiIndex.from_product([[f"{ct.columns.name}"], ct.columns])



        ## Creating the results table ##
        if test != False:
            if test == "fisher":
                results = {f"{test_name}": [],
                           "results": []}

                results[f"{test_name}"].append(f"Odds ratio = ")
                results["results"].append(round(ods2, 4))

                results[f"{test_name}"].append(f"2 sided p-value = ")
                results["results"].append(round(p2, 4))

                results[f"{test_name}"].append(f"Left tail p-value = ")
                results["results"].append(round(pl, 4))

                results[f"{test_name}"].append(f"Right tail p-value = ")
                results["results"].append(round(pg, 4))

                results[f"{test_name}"].append(f"Cramer's phi = ")
                results["results"].append(round(phi, 4))

                table = pd.DataFrame.from_dict(results)

            elif test != "fisher":
                results = {f"{test_name}": [],
                           "results": []}

                if test == "chi-square":
                    results[f"{test_name}"].append(f"Pearson Chi-square ({round(dof, 1)}) = ")
                    results["results"].append(round(test_val, 4))
                elif test == "g-test":
                    results[f"{test_name}"].append(f"Log-likelihood ratio ({round(dof, 1)}) = ")
                    results["results"].append(round(test_val, 4))
                elif test == "mcnemar":
                    results[f"{test_name}"].append(f"McNemar's Chi-square ({round(dof, 1)}) = ")
                    results["results"].append(round(test_val, 4))

                results[f"{test_name}"].append("p-value = ")
                results["results"].append(round(p, 4))

                if crosstab.size == 4:
                    results[f"{test_name}"].append("Cramer's phi = ")
                    results["results"].append(round(phi, 4))
                elif crosstab.size > 4:
                    results[f"{test_name}"].append("Cramer's V = ")
                    results["results"].append(round(V, 4))

                table = pd.DataFrame.from_dict(results)


        ## Returning DataFrame objects ##
        if expected_freqs == False:
            if test == False:
                return ct

            elif test != False:
                return ct, table

        elif expected_freqs == True:
            if test == "mcnemar":
              print("Expected frequencies not appropriate for this test, remove argument and try again.")

            elif test == False:
                return ct, expected

            elif test != False:
                return ct, table, expected
