# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:28:20 2018

@author: bryantcm


To do:
    - Add categorical data summary support

"""

import pandas
import numpy
import scipy.stats
from .basic_stats import *

## summary_cont() provides descriptive statistics for continuous data
import pandas
import numpy
import scipy
import scipy.stats

def summary_cont(group1, conf = 0.95, decimals = 4):



    conf_level = f"{round(conf * 100)}%"

    if type(group1) == pandas.core.series.Series:

        #### PUTTING THE INFORMATION INTO A DATAFRAME #####
        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE', f'{conf_level} Conf.', 'Interval'])
        # 'Variable' holds a string name; np.zeros scaffold makes it float64, so
        # cast to object before assigning (modern pandas raises otherwise).
        table['Variable'] = table['Variable'].astype(object)

        # Setting up the first column (Variable names)
        table.iloc[0,0] = group1.name

        # Setting up the second column (Number of observations)
        table.iloc[0,1] = group1.count()

        # Setting up the third column (Mean)
        table.iloc[0,2] = numpy.mean(group1)

        # Setting up the fourth column (Standard Deviation (SD))
        table.iloc[0,3] = numpy.std(group1, ddof= 1)

        # Setting up the fith column (Standard Error (SE))
        table.iloc[0,4] = scipy.stats.sem(group1, nan_policy= 'omit')

        # Setting up the sixth and seventh column (95% CI)
        table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(conf,
                                          group1.count() - 1,
                                          loc= numpy.mean(group1), scale= scipy.stats.sem(group1, nan_policy= 'omit'))

    elif type(group1) == pandas.core.frame.DataFrame:

        table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])
        # 'Variable' holds a string name; np.zeros scaffold makes it float64, so
        # cast to object before assigning (modern pandas raises otherwise).
        table['Variable'] = table['Variable'].astype(object)

        count = 0
        for ix, df_col in group1.items():
            count = count + 1
            if count == 0:
                table = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])
                # 'Variable' holds a string name; np.zeros scaffold makes it float64, so
                # cast to object before assigning (modern pandas raises otherwise).
                table_a['Variable'] = table_a['Variable'].astype(object)

                # Setting up the first column (Variable names)
                table.iloc[0,0] = ix

                # Setting up the second column (Number of observations)
                table.iloc[0,1] = df_col.count()

                # Setting up the third column (Mean)
                table.iloc[0,2] = numpy.mean(df_col)

                # Setting up the fourth column (Standard Deviation (SD))
                table.iloc[0,3] = numpy.std(df_col, ddof= 1)

                # Setting up the fith column (Standard Error (SE))
                table.iloc[0,4] = scipy.stats.sem(df_col, nan_policy= 'omit')

                # Setting up the sixth and seventh column (95% CI)
                table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(conf,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))
            else:
                table_a = pandas.DataFrame(numpy.zeros(shape= (1,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    f'{conf_level} Conf.', 'Interval'])
                # 'Variable' holds a string name; np.zeros scaffold makes it float64, so
                # cast to object before assigning (modern pandas raises otherwise).
                table_a['Variable'] = table_a['Variable'].astype(object)

                # Setting up the first column (Variable names)
                table_a.iloc[0,0] = ix

                # Setting up the second column (Number of observations)
                table_a.iloc[0,1] = df_col.count()

                # Setting up the third column (Mean)
                table_a.iloc[0,2] = numpy.mean(df_col)

                # Setting up the fourth column (Standard Deviation (SD))
                table_a.iloc[0,3] = numpy.std(df_col, ddof= 1)

                # Setting up the fith column (Standard Error (SE))
                table_a.iloc[0,4] = scipy.stats.sem(df_col, nan_policy= 'omit')

                # Setting up the sixth and seventh column (95% CI)
                table_a.iloc[0,5], table_a.iloc[0,6] = scipy.stats.t.interval(conf,
                                          df_col.count() - 1,
                                          loc= numpy.mean(df_col),
                                          scale= scipy.stats.sem(df_col, nan_policy= 'omit'))

            table = pandas.concat([table, table_a], ignore_index= "true")

        table.drop(0, inplace= True)
        table.reset_index(inplace= True, drop= True)


    elif type(group1) == pandas.core.groupby.SeriesGroupBy:
        ## Validated with R
        cnt = group1.count()
        cnt.rename("N", inplace= True)
        mean = group1.mean()
        mean.rename("Mean", inplace= True)
        std = group1.std(ddof= 1)
        std.rename("SD", inplace= True)
        se = group1.sem()
        se.rename("SE", inplace= True)

        # 95% CI
        l_ci, u_ci = scipy.stats.t.interval(conf,
                                          cnt - 1,
                                          loc= group1.mean(),
                                          scale= group1.sem())

        table = pandas.concat([cnt, mean, std, se,
                               pandas.Series(l_ci, index = cnt.index),
                               pandas.Series(u_ci, index = cnt.index)],
                              axis= 'columns')

        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD',
                                'sem': 'SE', 0 : f'{conf_level} Conf.', 1 : "Interval" },
                                inplace= True)


    elif type(group1) == pandas.core.groupby.DataFrameGroupBy :
        # There has to be a better way to get the lower and upper CI limits
        # in the groupby table. Until then, this works :/
        def l_ci(x):
            l_ci, _ = scipy.stats.t.interval(conf,
                                                x.count() - 1,
                                                loc= x.mean(),
                                                scale= x.sem())


            return l_ci

        def u_ci(x):
            _, u_ci = scipy.stats.t.interval(conf,
                                                x.count() - 1,
                                                loc= x.mean(),
                                                scale= x.sem())


            return u_ci

        table = group1.agg(['count', numpy.mean, numpy.std,
                            pandas.DataFrame.sem, l_ci, u_ci])

        table.rename(columns = {'count': 'N', 'mean': 'Mean', 'std': 'SD',
                                'sem': 'SE', "l_ci" : f'{conf_level} Conf.', "u_ci" : "Interval"}, inplace= True)

    else:
        return "This method only works with a Pandas Series, Dataframe, or Groupby object"

    print("\n")
    return table.round(decimals)








## summary_cat provides descriptives information for categorical data. It can
##  also handle numeric data since it's just counts and percents

def summary_cat(group1, ascending= False):

    if type(group1) == pandas.core.series.Series:
        table = group1.value_counts()
        table.rename("Count", inplace= True)
        table = pandas.DataFrame(table)

        table['Percent'] = ((table['Count']/table['Count'].sum()) * 100).round(2)

        if ascending == False:
            table.sort_values(by= 'Count', ascending= False, inplace= True)
        else:
            table.sort_values(by= 'Count', ascending= True, inplace= True)

        index_name = group1.name
        table['Variable'] = ""
        table.iloc[0,2] = index_name

        table.reset_index(inplace= True)
        table = table.rename(columns= {index_name: 'Outcome'})
        table = table[['Variable', 'Outcome', 'Count', 'Percent']]

    elif type(group1) == pandas.core.frame.DataFrame:

        count = 0

        for ix, df_col in group1.items():

            count = count + 1

            if count == 1:

                table = df_col.value_counts()
                table.rename("Count", inplace= True)
                table = pandas.DataFrame(table)

                table['Percent'] = ((table['Count']/table['Count'].sum()) * 100).round(2)

                if ascending == False:
                    table.sort_values(by= 'Count', ascending= False, inplace= True)
                else:
                    table.sort_values(by= 'Count', ascending= True, inplace= True)

                table['Variable'] = ""
                table.iloc[0,2] = ix

                index_name = table.index.name
                table.reset_index(inplace= True)
                table = table.rename(columns= {index_name: 'Outcome'})
                table = table[['Variable', 'Outcome', 'Count', 'Percent']]

            else:

                table_c = df_col.value_counts()
                table_c.rename("Count", inplace= True)
                table_c = pandas.DataFrame(table_c)

                table_c['Percent'] = ((table_c['Count']/table_c['Count'].sum()) * 100).round(2)

                if ascending == False:
                    table_c.sort_values(by= 'Count', ascending= False, inplace= True)
                else:
                    table_c.sort_values(by= 'Count', ascending= True, inplace= True)

                table_c['Variable'] = ""
                table_c.iloc[0,2] = ix

                index_name_c = table_c.index.name
                table_c.reset_index(inplace= True)
                table_c = table_c.rename(columns= {index_name_c: 'Outcome'})
                table_c = table_c[['Variable', 'Outcome', 'Count', 'Percent']]

                table = pandas.concat([table, table_c], ignore_index= "true")

    else:
        print("This method can only be used with Pandas Series or DataFrames")

    return table



def codebook(data):
    """
    This function returns descriptive information about the variables at hand.
    Accepts Pandas Series or Pandas DataFrame objects.
    """



    if type(data) == pandas.core.series.Series:
        """

        How to provide summary information for Series objects

        """



        if "int" in str(data.dtype) or "float" in str(data.dtype):
            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Range: [{data.min()}, {data.max()}] \n",
                  f"Mean: {round(data.mean(), 2)} \n",
                  f"Standard Deviation: {round(data.std(), 2)} \n",
                  f"Mode: {data.mode()[0]} \n",
                  f"10th Percentile: {data.quantile(.10, interpolation= 'linear')} \n",
                  f"25th Percentile: {data.quantile(.25, interpolation= 'linear')} \n",
                  f"50th Percentile: {data.quantile(.50, interpolation= 'linear')} \n",
                  f"75th Percentile: {data.quantile(.75, interpolation= 'linear')} \n",
                  f"90th Percentile: {data.quantile(.90, interpolation= 'linear')} \n",)

            print("\n" * 3)



        elif "object" in str(data.dtype) or "category" == data.dtype:
            tab = dict(data.value_counts())
            tab = dict(sorted(tab.items()))
            tab = {"Values" : list(tab.keys()), "Frequency" : list(tab.values())}
            tab = pandas.DataFrame(tab)


            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Data Values and Counts: \n \n",
                  tab.to_string(index = False))

            print("\n" * 3)



        elif "datetime" in str(data.dtype):
            print(f"Variable: {data.name}    Data Type: {data.dtype}", "\n")

            print(f" Number of Obs.: {data.size} \n",
                  f"Number of missing obs.: {data.size - data.count()} \n",
                  f"Percent missing: {((data.size - data.count()) / data.size * 100).round(2)} \n",
                  f"Number of unique values: {data.nunique()} \n")

            print(f" Range: [{data.min()}, {data.max()}]")

            print("\n" * 3)




        else:
            print(f"type(data) is not supported at this time.")

            print("\n" * 3)





    elif type(data) == pandas.core.frame.DataFrame:
        """

        How to provide summary information for DataFrame objects

        """

        for col in data.columns:

            if "int" in str(data[col].dtype) or "float" in str(data[col].dtype):
                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Range: [{data[col].min()}, {data[col].max()}] \n",
                      f"Mean: {round(data[col].mean(), 2)} \n",
                      f"Standard Deviation: {round(data[col].std(), 2)} \n",
                      f"Mode: {data[col].mode()[0]} \n",
                      f"10th Percentile: {data[col].quantile(.10, interpolation= 'linear')} \n",
                      f"25th Percentile: {data[col].quantile(.25, interpolation= 'linear')} \n",
                      f"50th Percentile: {data[col].quantile(.50, interpolation= 'linear')} \n",
                      f"75th Percentile: {data[col].quantile(.75, interpolation= 'linear')} \n",
                      f"90th Percentile: {data[col].quantile(.90, interpolation= 'linear')} \n")

                print("\n" * 3)



            elif "object" in str(data[col].dtype) or "category" == data[col].dtype:
                tab = dict(data[col].value_counts())
                tab = dict(sorted(tab.items()))
                tab = {"Values" : list(tab.keys()), "Frequency" : list(tab.values())}
                tab = pandas.DataFrame(tab)


                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Data Values and Counts: \n \n",
                      tab.to_string(index = False))

                print("\n" * 3)



            elif "datetime" in str(data[col].dtype):
                print(f"Variable: {data[col].name}    Data Type: {data[col].dtype}", "\n")

                print(f" Number of Obs.: {data[col].size} \n",
                      f"Number of missing obs.: {data[col].size - data[col].count()} \n",
                      f"Percent missing: {((data[col].size - data[col].count()) / data[col].size * 100).round(2)} \n",
                      f"Number of unique values: {data[col].nunique()} \n")

                print(f" Range: [{data[col].min()}, {data[col].max()}]")

                print("\n" * 3)




            else:
                print(f"{data.dtype} is not supported at this time.")

                print("\n" * 3)





    else:
        print(f"Current data type, {type(data)}, is not supported. Currently, only Pandas Series and DataFrame are supported.")





def summarize(data = {}, name = None, stats = [], ci_level = 0.95, decimals = 4, return_type = "Dataframe"):
    """

    Parameters
    ----------
    data : array_like
        Array like data object.
    name : String, optional
        The name of the variable returned if the name of the column is not desired. The default is None, i.e. name of variable.
    stats : List, optional
        The statistics to be calculated; the default is ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"].

        Supported options are: ["N", "Mean", "Median", "Variance", "SD", "SE", "CI", 'Min', 'Max', 'Range', "Kurtosis", "Skew"]
    ci_level : Float, optional
        The confidence level to be calculated. The default is 0.95.
    decimals : Integer, optional
        The number of decimal places to be rounded to. The default is 4.
    return_type : String, optional
        The data structure to be returne; the default is "Dataframe".

        Available options are:
            "Dataframe" which will return a Pandas Dataframe.
            "Dictionary" which will return a dictionary.

    Returns
    -------
    Pandas Dataframe or dictionary depending on what is specified.

    """
    # Parameter check
    if return_type.upper() not in ["DATAFRAME", "DICTIONARY"]:
        return print(" ",
                     "Not a supported return type. Only 'Dataframe' and 'Dictionary' are supported at this time.",
                     " ",
                     sep = "\n"*2)






    if len(stats) == 0:
        stats = ["N", "Mean", "Median", "Variance", "SD", "SE", "CI"]
        stats_to_conduct = [count, numpy.nanmean, numpy.nanmean, nanvar, nanstd, nansem, confidence_interval]


    results = {}

    if name is not None:
        results["Name"] = name

    if name is None:
        try:
            results["Name"] = data.name
        except:
            ...

    flag = "nongroupby"

    # Calculating the requested information #
    if type(data) == pandas.core.groupby.generic.DataFrameGroupBy or type(data) == pandas.core.groupby.generic.SeriesGroupBy:

        flag = "groupby"
        stats_to_conduct = []

        idx = -1
        for test in stats:
            idx += 1

            if "N" == test: stats_to_conduct.append(count)
            if "Mean" == test: stats_to_conduct.append(numpy.nanmean)
            if "Median" == test: stats_to_conduct.append(numpy.nanmedian)
            if "Variance" == test: stats_to_conduct.append(nanvar)
            if "SD" == test: stats_to_conduct.append(nanstd)
            if "SE" == test: stats_to_conduct.append(nansem)
            if "CI" == test:
                #stats_to_conduct.append(lambda x: [l_ci(x, alpha = ci_level, decimals = decimals), u_ci(x, alpha = ci_level, decimals = decimals)])
                stats_to_conduct.append(confidence_interval)
                stats[idx] = f"{int(ci_level * 100)}% Conf. Interval"
            if "Min" == test: stats_to_conduct.append(numpy.nanmin)
            if "Max" == test: stats_to_conduct.append(numpy.nanmax)
            if "Range" == test: stats_to_conduct.append(value_range)
            if "Kurtosis" == test: stats_to_conduct.append(kurtosis)
            if "Skew" == test: stats_to_conduct.append(skew)


        results = round(data.agg(stats_to_conduct), decimals)

        if type(data) == pandas.core.groupby.generic.DataFrameGroupBy:
            ## Need to clean up the lmbda_0 name
            col = [(x[0], f"{int(ci_level * 100)}% Conf. Interval") if x[1] == '<lambda_0>' else x for x in results.columns.tolist()]
            results.columns = pandas.MultiIndex.from_tuples(col)

            if return_type == "Dictionary":
                results = results.to_dict()
        else:
            col = [f"{int(ci_level * 100)}% Conf. Interval" if x == '<lambda_0>' else x for x in results.columns.tolist()]
            results.columns = col




    elif type(data) == pandas.core.frame.DataFrame:

        results["Name"] = data.columns.tolist()

        idx = -1
        for test in stats:
            idx += 1

            if "N" == test:
                results[test] = round(data.apply(count), decimals)

            if "Mean" == test:
                results[test] = round(data.apply(numpy.nanmean), decimals)

            if "Median" == test:
                results[test] = round(data.apply(numpy.nanmedian), decimals)

            if "Variance" == test:
                results[test] = round(data.apply(nanvar), decimals)

            if "SD" == test:
                results[test] = round(data.apply(nanstd), decimals)

            if "SE" == test:
                results[test] = round(data.apply(nansem), decimals)

            if "CI" == test:
                results[f"{int(ci_level * 100)}% Conf. Interval"] = data.apply(lambda x: confidence_interval(x, alpha = ci_level, decimals = decimals))
                stats[idx] = f"{int(ci_level * 100)}% Conf. Interval"

            if "Min" == test:
                results[test] = round(data.apply(numpy.nanmin), decimals)

            if "Max" == test:
                results[test] = round(data.apply(numpy.nanmax), decimals)

            if "Range" == test:
                results[test] = round(data.apply(value_range), decimals)

            if "Kurtosis" == test:
                results[test] = round(data.apply(kurtosis), decimals)

            if "Skew" == test:
                results[test] = round(data.apply(skew), decimals)





    else:

        if "N" in stats:
            try:
                results["N"] = count(data)
            except:
                results["N"] = data.apply(lambda x: numpy.count_nonzero(~x.apply(numpy.isnan)))

        if "Mean" in stats:
            try:
                results["Mean"] = round(numpy.nanmean(data), decimals)
            except:
                try:
                    results["Mean"] = round(data.apply(numpy.nanmean), decimals)
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Mean"] = float(numpy.nanmean(data))

        if "Median" in stats:
            try:
                results["Median"] = float(numpy.nanmedian(data))
            except:
                results["Median"] = float(data.apply(numpy.nanmedian))

        if "Variance" in stats:
            try:
                results["Variance"] = round(nanvar(data), decimals)
            except:
                try:
                    results["Variance"] = data.apply(lambda x: round(nanvar(x), decimals))
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Variance"] = float(nanvar(data))

        if "SD" in stats:
            try:
                results["SD"] = round(nanstd(data), decimals)
            except:
                try:
                    results["SD"] = data.apply(lambda x: round(nanstd(x), decimals))
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["SD"] = float(nanstd(data))

        if "SE" in stats:
            try:
                results["SE"] = round(nansem(data), decimals)
            except:
                try:
                    results["SE"] = data.apply(lambda x: nansem(x))
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["SE"] = float(nansem(data)[0])

        if "CI" in stats:
            try:
                ci_lower, ci_upper = scipy.stats.t.interval(ci_level,
                                                            count(data) - 1,
                                                            loc = numpy.nanmean(data),
                                                            scale = nansem(data))

                results[f"{int(ci_level * 100)}% Conf. Interval"] = [round(ci_lower, decimals), round(ci_upper, decimals)]
            except:
                try:
                    # Used on patsy.design_info.DesignMatrix objects
                    ci_lower, ci_upper = scipy.stats.t.interval(ci_level,
                                                                count(data) - 1,
                                                                loc = numpy.nanmean(data),
                                                                scale = nansem(data))

                    results[f"{int(ci_level * 100)}% Conf. Interval"] = [round(float(ci_lower[0]), decimals), round(float(ci_upper[0]), decimals)]
                except:
                    try:
                        ci_intervals = data.apply(lambda x: list(scipy.stats.t.interval(ci_level,
                                                                                        count(x) - 1,
                                                                                        loc = numpy.nanmean(x),
                                                                                        scale = nansem(x))))

                        ci_intervals = ci_intervals.to_dict()
                        for lst in ci_intervals.values():
                            idx = 0
                            for value in lst:
                                lst[idx] = round(value, decimals)
                                idx += 1


                        results[f"{int(ci_level * 100)}% Conf. Interval"] = ci_intervals

                    except:
                        ci_intervals = data.apply(lambda x: confidence_interval(x, alpha = ci_level, decimals = decimals))
                        print(ci_intervals)

                        results[f"{int(ci_level * 100)}% Conf. Interval"] = ci_intervals

        if "Min" in stats:
            try:
                results["Min"] = round(numpy.nanmin(data), decimals)
            except:
                try:
                    results["Min"] = round(data.apply(numpy.nanmin), decimals)
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Min"] = float(numpy.nanmin(data))

        if "Max" in stats:
            try:
                results["Max"] = round(numpy.nanmax(data), decimals)
            except:
                try:
                    results["Max"] = round(data.apply(numpy.nanmax), decimals)
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Max"] = float(numpy.nanmax(data))

        if "Range" in stats:
            try:
                results["Range"] = round(numpy.nanmax(data) - numpy.nanmin(data), decimals)
            except:
                try:
                    results["Range"] = round(data.apply(lambda x: numpy.nanmax(x) - numpy.nanmin(x)), decimals)
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Range"] = float(numpy.nanmax(data) - numpy.nanmin(data))

        if "Kurtosis" in stats:
            # This computes kurtosis using Pearson's definition
            try:
                results["Kurtosis"] = round(kurtosis(data), decimals)
            except:
                try:
                    results["Kurtosis"] = round(data.apply(lambda x: kurtosis(x)), decimals)

                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Kurtosis"] = float(kurtosis(data))

        if "Skew" in stats:
            try:
                results["Skew"] = round(skew(data), decimals)
            except:
                try:
                    results["Skew"] = round(data.apply(lambda x: skew(x)), decimals)
                except:
                    # Used on patsy.design_info.DesignMatrix objects
                    results["Skew"] = float(skew(data))





    if return_type == "Dataframe":
        try:
            results = pandas.DataFrame.from_dict(results, orient='index').T
            return results
        except:
            try:
                results = results.reset_index()
                results.rename(columns = {"count" : "N",
                                      "nanmean" : "Mean",
                                      "nanmedian" : "Median",
                                      "nanvar" : "Variance",
                                      "nanstd" : "SD",
                                      "nansem" : "SE",
                                      "confidence_interval" : f"{int(ci_level * 100)}% Conf. Interval",
                                      "nanmin" : "Min",
                                      "nanmax" : "Max",
                                      "ptp" : "Range",
                                      "kurtosis" : "Kurtosis",
                                      "skew" : "Skew"}, inplace = True)
            except:
                results = pandas.DataFrame.from_dict(results)



            return results

    elif return_type == "Dictionary":

        if flag == "groupby":

            results.rename(columns = {"count" : "N",
                                      "nanmean" : "Mean",
                                      "nanmedian" : "Median",
                                      "nanvar" : "Variance",
                                      "nanstd" : "SD",
                                      "nansem" : "SE",
                                      "confidence_interval" : f"{int(ci_level * 100)}% Conf. Interval",
                                      "nanmin" : "Min",
                                      "nanmax" : "Max",
                                      "ptp" : "Range",
                                      "kurtosis" : "Kurtosis",
                                      "skew" : "Skew"}, inplace = True)


            return dict(results)

        else:
            return results
