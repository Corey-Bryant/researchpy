# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:18:39 2018
Updated on December 23, 2018

@author: Corey Bryant

"""

import pandas
import numpy
import scipy.stats



def ttest(group1, group2, group1_name= None, group2_name= None,
           equal_variances= True, paired= False,
           wilcox_parameters = {"zero_method" : "pratt", "correction" : False, "mode" : "auto"},
           welch_dof = "satterthwaite"):

    # Joining groups for table and calculating group mean difference
    groups = pandas.concat([group1, group2], ignore_index=True)
    groups_diff = numpy.mean(group1) - numpy.mean(group2)



    # Deciding which test to use
    if equal_variances == True and paired == False:
        test = "Independent t-test"
        t_val, p_val = scipy.stats.ttest_ind(group1, group2, nan_policy= 'omit')
        dof = group1.count() + group2.count() - 2

        # Less than or greater than 0 p_vals
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)




        # Effect sizes
        # Cohen's d
        # Calculated using t*square root(1/group1_n + 1/group2_n)
        d = (group1.mean() - group2.mean()) / numpy.sqrt(((group1.count() - 1) * group1.std()**2 + (group2.count() - 1) * group2.std()**2) / (group1.count() + group2.count() - 2))

        # Common language effect size
        # Calculated z = |group1_mean - group2_mean| / square root (group1_variance + group2_variance)
        cles = 0 - abs(group1.mean() - group2.mean()) / numpy.sqrt(group1.var() + group2.var())

        # Hedge's G
        #Calculated as g = d * (1 - 3 / 4(group1_n + group2_n) - 9)
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))

        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()

        # Pearson r
        #r = numpy.sqrt(t_val**2 / ((t_val**2) + dof))

        # Point-Biserial r -- should I add?
        r = t_val / numpy.sqrt(t_val**2 + dof)


    elif equal_variances == False and paired == False:

        ## This p-value is the Welch-Satterthwaite p-value ##
        t_val, p_val = scipy.stats.ttest_ind(group1, group2, equal_var = False,
                                                 nan_policy= 'omit')

        ## Determining which version of the Welch's t-test to use ##
        if welch_dof == "satterthwaite":

            test = "Satterthwaite t-test"

            ## Satterthwaite (1946) Degrees of Freedom ##
            dof = ((group1.var()/group1.count()) + (group2.var()/group2.count()))**2 / ((group1.var()/group1.count())**2 / (group1.count()-1) + (group2.var()/group2.count())**2 / (group2.count()-1))



        elif welch_dof == "welch":
            test = "Welch's t-test"

            ## Welch (1947) Degrees of Freedom ##
            dof = -2 + (((group1.var()/group1.count()) + (group2.var()/group2.count()))**2 / ((group1.var()/group1.count())**2 / (group1.count()+1) + (group2.var()/group2.count())**2 / (group2.count()+1)))

            p_val = 2 * min((1 - scipy.stats.t.cdf(t_val, dof)), scipy.stats.t.cdf(t_val, dof))



        # Less than or greater than 0 p_vals
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)


        #if t_val > 0:
        #    temp = rt_p_val
        #    rt_p_val = lt_p_val
        #    lt_p_val = temp



        # Effect size
        # Cohen's d
        d = (group1.mean() - group2.mean()) / numpy.sqrt(((group1.count() - 1) * group1.std()**2 + (group2.count() - 1) * group2.std()**2) / (group1.count() + group2.count() - 2))

        # Hedge's G
        #Calculated as g = d * (1 - 3 / 4(group1_n + group2_n) - 9)
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))

        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()

        # Pearson r
        #r = numpy.sqrt(t_val**2 / ((t_val**2) + dof))

        # Point-Biserial r -- should I add?
        r = t_val / numpy.sqrt(t_val**2 + dof)



    elif equal_variances == True and paired == True:
        group1 = group1[(group1.notnull()) & (group2.notnull())]
        group2 = group2[(group1.notnull()) & (group2.notnull())]

        groups = pandas.concat([group1, group2], ignore_index=True)
        groups_diff = numpy.mean(group1) - numpy.mean(group2)
        diff = group1 - group2

        test = "Paired samples t-test"
        t_val, p_val = scipy.stats.ttest_rel(group1, group2)
        dof = group1.count() - 1

        # Less than or greater than 0 p_vals
        lt_p_val = scipy.stats.t.cdf(t_val, dof)
        rt_p_val = 1 - scipy.stats.t.cdf(t_val, dof)


        if t_val > 0:
            temp = rt_p_val
            rt_p_val = lt_p_val
            lt_p_val = temp

        # Effect sizes

        # Cohen's d (1988)
        d = (group1.mean() - group2.mean()) / diff.std()

        # Cohen's d_av
        d_av = (group1.mean() - group2.mean()) / ((group1.std() + group2.std()) / 2)

        # Hedge's G
        g = d * (1 - (3 / ((4*(group1.count() + group2.count())) - 9)))

        # Glass's delta
        delta = (group1.mean() - group2.mean()) / group1.std()

        # Pearson r
        #r = numpy.sqrt(t_val**2 / ((t_val**2) + dof))

        # Point-Biserial r
        r = t_val / numpy.sqrt(t_val**2 + dof)




    elif equal_variances == False and paired == True:

        test = "Wilcoxon signed-rank test"

        difference = group1 - group2

        parameters = {"zero_method" : "pratt", "correction" : False, "mode" : "auto"}
        parameters.update(wilcox_parameters)



        if parameters["zero_method"] == 'pratt':

            difference_abs = numpy.abs(difference)

            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]

        elif parameters["zero_method"] == 'wilcox':

            difference = difference[difference != 0]
            difference_abs = numpy.abs(difference)

            # Calculating Signed Information
            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]

        elif parameters["zero_method"] == 'zsplit':
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



        t_val, p_val = scipy.stats.wilcoxon(group1, group2,
                                            zero_method = parameters["zero_method"],
                                            correction = parameters["correction"],
                                            mode = parameters["mode"])


        ## Effect size
        ##  Pearson r = z / square_root(N)
        pr = z / numpy.sqrt(total_n)

        # Rank-Biserial r
        pbr = (positive_sum_ranks - negative_sum_ranks) / total_sum_ranks





    #### PUTTING THE INFORMATION INTO A DATAFRAME #####
    table = pandas.DataFrame(numpy.zeros(shape= (3,7)),
                         columns = ['Variable', 'N', 'Mean', 'SD', 'SE',
                                    '95% Conf.', 'Interval'])


    # Setting up the first column (Variable names)
    if group1_name != None:
        group1_name = group1_name
    else:
        group1_name = group1.name

    if group2_name != None:
        group2_name = group2_name
    else:
        group2_name = group2.name

    table.iloc[0,0] = group1_name
    table.iloc[1,0] = group2_name

    if test == "Paired samples t-test":
        table.iloc[2,0] = 'diff'
    else:
        table.iloc[2,0] = 'combined'


    # Setting up the second column (Number of observations)
    table.iloc[0,1] = group1.count()
    table.iloc[1,1] = group2.count()

    if test == "Paired samples t-test":
        table.iloc[2,1] = diff.count()
    else:
        table.iloc[2,1] = groups.count()


    # Setting up the third column (Mean)
    table.iloc[0,2] = numpy.mean(group1)
    table.iloc[1,2] = numpy.mean(group2)

    if test == "Paired samples t-test":
        table.iloc[2,2] = numpy.mean(diff)
    else:
        table.iloc[2,2] = numpy.mean(groups)


    # Setting up the fourth column (Standard Deviation (SD))
    table.iloc[0,3] = numpy.std(group1, ddof= 1)
    table.iloc[1,3] = numpy.std(group2, ddof= 1)

    if test == "Paired samples t-test":
        table.iloc[2,3] = numpy.std(diff, ddof= 1)
    else:
        table.iloc[2,3] = numpy.std(groups, ddof= 1)


    # Setting up the fith column (Standard Error (SE))
    table.iloc[0,4] = scipy.stats.sem(group1, nan_policy= 'omit')
    table.iloc[1,4] = scipy.stats.sem(group2, nan_policy= 'omit')

    if test == "Paired samples t-test":
        table.iloc[2,4] = scipy.stats.sem(diff, nan_policy= 'omit')
    else:
        table.iloc[2,4] = scipy.stats.sem(groups, nan_policy= 'omit')


    # Setting up the sixth and seventh column (95% CI)
    table.iloc[0,5], table.iloc[0,6] = scipy.stats.t.interval(0.95,
                                          group1.count() - 1,
                                          loc= numpy.mean(group1),
                                          scale= scipy.stats.sem(group1, nan_policy= 'omit'))
    table.iloc[1,5], table.iloc[1,6] = scipy.stats.t.interval(0.95,
                                          group2.count() - 1,
                                          loc= numpy.mean(group2),
                                          scale= scipy.stats.sem(group2, nan_policy= 'omit'))

    if test == "Paired samples t-test":
        table.iloc[2,5], table.iloc[2,6] = scipy.stats.t.interval(0.95,
                                          diff.count() - 1,
                                          loc= numpy.mean(diff),
                                          scale= scipy.stats.sem(diff, nan_policy= 'omit'))
    else:
        table.iloc[2,5], table.iloc[2,6] = scipy.stats.t.interval(0.95,
                                          groups.count() - 1,
                                          loc= numpy.mean(groups),
                                          scale= scipy.stats.sem(groups, nan_policy= 'omit'))


    if equal_variances == False and paired == True:

        # Descriptive Information Regarding Ranked-Signs #

        # table = pandas.DataFrame(numpy.zeros(shape= (5,4)),
        #                 columns = ['sign', 'obs', 'sum ranks', 'expected'])
        descriptives = {"sign" : ["positive", "negative", "zero", "all"],
                        "obs" : [positive_n, negative_n, zero_n, total_n],
                        "sum ranks" : [positive_sum_ranks, negative_sum_ranks, zero_sum_ranks, total_sum_ranks],
                        "expected" : [exp_positive, exp_negative, exp_zero, T]}

        table1 = pandas.DataFrame.from_dict(descriptives)


        # Testing Results #
        table2 = pandas.DataFrame(numpy.zeros(shape= (7,2)),
                         columns = ['Wilcoxon signed-rank test', 'results'])

        table2.iloc[0,0] = f"Mean for {group1_name} = "
        table2.iloc[0,1] = numpy.mean(group1)

        table2.iloc[1,0] = f"Mean for {group2_name} = "
        table2.iloc[1,1] = numpy.mean(group2)

        table2.iloc[2,0] = f"W value = "
        table2.iloc[2,1] = round(t_val, 4)

        table2.iloc[3,0] = f"Z value = "
        table2.iloc[3,1] = round(z, 4)

        table2.iloc[4,0] = f"p value = "
        table2.iloc[4,1] = round(p_val, 4)

        table2.iloc[5,0] = f"Rank-Biserial r = "
        table2.iloc[5,1] = round(pbr, 4)

        table2.iloc[6,0] = f"Pearson r = "
        table2.iloc[6,1] = round(pr, 4)

        return table1, table2

    elif equal_variances == True and paired == True:
        table2 = pandas.DataFrame(numpy.zeros(shape= (11,2)),
                         columns = ['Test', 'results'])

        table2.rename(columns= {'Test': f'{test}'}, inplace= True)

        table2.iloc[0,0] = f"Difference ({group1_name} - {group2_name}) = "
        table2.iloc[0,1] = round(groups_diff, 4)

        table2.iloc[1,0] = "Degrees of freedom = "
        table2.iloc[1,1] = round(dof, 4)

        table2.iloc[2,0] = "t = "
        table2.iloc[2,1] = round(t_val, 4)

        table2.iloc[3,0] = "Two side test p value = "
        table2.iloc[3,1] = round(p_val, 4)

        table2.iloc[4,0] = f"Difference < 0 p value = "
        table2.iloc[4,1] = round(lt_p_val, 4)

        table2.iloc[5,0] = f"Difference > 0 p value = "
        table2.iloc[5,1] = round(rt_p_val, 4)

        table2.iloc[6,0] = f"Cohen's d = "
        table2.iloc[6,1] = round(d, 4)

        table2.iloc[7,0] = f"Cohen's d_av = "
        table2.iloc[7,1] = round(d_av, 4)

        table2.iloc[8,0] = f"Hedge's g = "
        table2.iloc[8,1] = round(g, 4)

        table2.iloc[9,0] = f"Glass's delta1 = "
        table2.iloc[9,1] = round(delta, 4)

        table2.iloc[10,0] = f"Point-Biserial r = "
        table2.iloc[10,1] = round(r, 4)

        return table, table2

    else:
        table2 = pandas.DataFrame(numpy.zeros(shape= (10,2)),
                         columns = ['Test', 'results'])

        table2.rename(columns= {'Test': f'{test}'}, inplace= True)

        table2.iloc[0,0] = f"Difference ({group1_name} - {group2_name}) = "
        table2.iloc[0,1] = round(groups_diff, 4)

        table2.iloc[1,0] = "Degrees of freedom = "
        table2.iloc[1,1] = round(dof, 4)

        table2.iloc[2,0] = "t = "
        table2.iloc[2,1] = round(t_val, 4)

        table2.iloc[3,0] = "Two side test p value = "
        table2.iloc[3,1] = round(p_val, 4)

        table2.iloc[4,0] = f"Difference < 0 p value = "
        table2.iloc[4,1] = round(lt_p_val, 4)

        table2.iloc[5,0] = f"Difference > 0 p value = "
        table2.iloc[5,1] = round(rt_p_val, 4)

        table2.iloc[6,0] = f"Cohen's d = "
        table2.iloc[6,1] = round(d, 4)

        table2.iloc[7,0] = f"Hedge's g = "
        table2.iloc[7,1] = round(g, 4)

        table2.iloc[8,0] = f"Glass's delta1 = "
        table2.iloc[8,1] = round(delta, 4)

        table2.iloc[9,0] = f"Point-Biserial r = "
        table2.iloc[9,1] = round(r, 4)

        return table, table2
