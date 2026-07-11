# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:18:47 2018

@author: bryantcm
"""

import pandas
import numpy
import scipy.stats
import itertools




def corr_case(dataframe, method="pearson"):
    df = dataframe.dropna(how='any')._get_numeric_data()
    dfcols = pandas.DataFrame(columns=df.columns)

    # Getting r an p value dataframes ready
    r_vals = dfcols.transpose().join(dfcols, how='outer')
    p_vals = dfcols.transpose().join(dfcols, how='outer')
    length = str(len(df))

    # Setting test
    if method in (None, "pearson"):
        test = scipy.stats.pearsonr
        test_name = "Pearson"

    elif method == "spearman":
        test = scipy.stats.spearmanr
        test_name = "Spearman Rank"

    elif method == "kendall":
        test = scipy.stats.kendalltau
        test_name = "Kendall's Tau-b"

    else:
        raise ValueError("Unknown method: " + method)

    # Rounding values for the r and p value dataframes
    for r in df.columns:
        for c in df.columns:
            r_vals.loc[r, c] = round(test(df[r], df[c])[0], 4)
       
    for r in df.columns:
        for c in df.columns:
            p_vals.loc[r, c] = format(test(df[r], df[c])[1], '.4f')
            
            
    # Getting the testing information dataframe ready
    info = pandas.DataFrame(numpy.zeros(shape= (1,1)),
                            columns = [f"{test_name} correlation test using list-wise deletion"],
                            dtype=object)
    info.iloc[0,0] = f"Total observations used = {length}"


    return info, r_vals, p_vals




def corr_pair(dataframe, method="pearson"):
    df = dataframe

    correlations = {}
    pvalues = {}
    length = {}
    columns = df.columns.tolist()

    # Setting test
    if method in (None, "pearson"):
        test = scipy.stats.pearsonr
        test_name = "Pearson"

    elif method == "spearman":
        test = scipy.stats.spearmanr
        test_name = "Spearman Rank"

    elif method == "kendall":
        test = scipy.stats.kendalltau
        test_name = "Kendall's Tau-b"

    else:
        raise ValueError("Unknown method: " + method)

    # Iterrating through the Pandas series and performing the correlation
    # analysis
    for col1, col2 in itertools.combinations(columns, 2):
        sub = df[[col1, col2]].dropna(how="any")
        correlations[col1 + " " + "&" + " " + col2] = format(test(sub.loc[:, col1], sub.loc[:, col2])[0], '.4f')
        pvalues[col1 + " " + "&" + " " + col2] = format(test(sub.loc[:, col1], sub.loc[:, col2])[1], '.4f')
        length[col1 + " " + "&" + " " + col2] = len(df[[col1, col2]].dropna(how="any"))

    corrs = pandas.DataFrame.from_dict(correlations, orient="index")
    corrs.columns = ["r value"]

    pvals = pandas.DataFrame.from_dict(pvalues, orient="index")
    pvals.columns = ["p-value"]

    l = pandas.DataFrame.from_dict(length, orient="index")
    l.columns = ["N"]

    results = corrs.join([pvals, l])

    return results

