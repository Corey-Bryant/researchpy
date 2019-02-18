# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:36:11 2018

@author: CoreySSD
"""

import pandas
import numpy
import scipy.stats
from statsmodels.stats import contingency_tables

def crosstab(group1, group2, prop= None, test = False, margins= True,
             correction = None, exact = False, expected_freqs= False):
    
    if type(group1) != pandas.core.series.Series or type(group2) != pandas.core.series.Series:
        return "Operation only supports Pandas Series"
    
    else:
        ## Creating the crosstab table ##
        crosstab = pandas.crosstab(group1, group2)
        crosstab2 = pandas.crosstab(group1, group2, margins = True)
        
        
        
        ## Creating percentage tables ##
        # Row
        crosstab_perrow = round(crosstab2.div(crosstab2.iloc[:,-1], axis=0).mul(100, axis=0), 2)
        # Column
        crosstab_percol = round(crosstab2.div(crosstab2.iloc[-1,:], axis=1).mul(100, axis=1), 2)
        # Cell
        crosstab_percell = round(crosstab2.div(crosstab2.iloc[-1,-1], axis=0).mul(100, axis=1), 2)
        
        
        
        ## Chi-square and effect size results ##
        if correction == None:
            correction = False
        elif correction == True:
            correction = True
        
        chi2, p, dof, expected = scipy.stats.chi2_contingency(crosstab,
                                                              correction = correction)
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
                                                              correction = correction,
                                                              lambda_ = "log-likelihood")
        elif test == "mcnemar":
            test_name = "McNemar"
            results = contingency_tables.mcnemar(crosstab, exact= exact, correction = correction)
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
        
        # Cramer's phi
        # phi = square_root(chi_square / N)
        # Where N = total sample size
        phi = numpy.sqrt(test_val / crosstab2.iloc[-1,-1])
        
        # Cramer's V
        # V = square_root(chi_square / min(c-1, r-1))
        V = numpy.sqrt(test_val / (crosstab2.iloc[-1,-1] * min((len(crosstab.index) - 1), (len(crosstab.columns) - 1))))
        
        
        
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
            
        expected = pandas.DataFrame(expected, index= crosstab.index, 
                             columns= pandas.MultiIndex.from_product([[f"{crosstab.columns.name}"], 
                             crosstab.columns]))
            
        ct.columns = pandas.MultiIndex.from_product([[f"{ct.columns.name}"], 
                                                     ct.columns])
        
        
        
        ## Creating the results table ##
        if test != False:
            if test == "fisher":
                table = pandas.DataFrame(numpy.zeros(shape= (5,2)), 
                         columns = [f'{test_name}', 'results'])
                
                table.iloc[0,0] = f"Odds ratio = "
                table.iloc[0,1] = round(ods2, 4)
                
                table.iloc[1,0] = f"2 sided p-value = "
                table.iloc[1,1] = round(p2, 4)
                
                table.iloc[2,0] = f"Left tail p-value = "
                table.iloc[2,1] = round(pl, 4)
                
                table.iloc[3,0] = f"Right tail p-value = "
                table.iloc[3,1] = round(pg, 4)
                
                table.iloc[4,0] = f"Cramer's phi = "
                table.iloc[4,1] = round(phi, 4)
                
            elif test != "fisher":
                table = pandas.DataFrame(numpy.zeros(shape= (3,2)), 
                         columns = [f'{test_name}', 'results'])
        
                if test == "chi-square":
                    table.iloc[0,0] = f"Pearson Chi-square ({dof: .1f}) = "
                elif test == "g-test":
                    table.iloc[0,0] = f"Log-likelihood ratio ({dof: .1f}) = "
                elif test == "mcnemar":
                    table.iloc[0,0] = f"McNemar's Chi-square ({dof: .1f}) = "
                table.iloc[0,1] = round(test_val, 4)
        
                table.iloc[1,0] = f"p-value = "
                table.iloc[1,1] = round(p, 4)
        
                if crosstab.size == 4:
                    table.iloc[2,0] = f"Cramer's phi = "
                    table.iloc[2,1] = round(phi, 4)
                elif crosstab.size > 4:
                    table.iloc[2,0] = f"Cramer's V = "
                    table.iloc[2,1] = round(V, 4)
            
        
        
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
