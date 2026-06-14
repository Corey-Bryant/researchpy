import pandas as pd
import scipy.stats
import numpy as np
import re
import itertools


def rounder(lst, decimals=4):
    """
        Iterates through a list and returns the rounded number.
    """

    idx = 0
    for item in lst:
        lst[idx] = round(item, decimals)
        idx += 1


def patsy_column_cleaner(factor):
    """

    Parameters
    ----------
    factor : A list of factors and levels which is provided by Patsy's design_info.column_names object.


    Returns
    -------
    String


    Description
    -----------
    This function returns a level name as a string.

    Ex.
    column_names = ['Intercept',
                    'C(drug, Treatment(2))[T.1]',
                    'C(drug, Treatment(2))[T.3]',
                    'C(drug, Treatment(2))[T.4]',
                    'disease',
                    'C(drug, Treatment(2))[T.1]:disease',
                    'C(drug, Treatment(2))[T.3]:disease',
                    'C(drug, Treatment(2))[T.4]:disease']


    [IN]  [patsy_column_cleaner(level) for level in column_names]
    [OUT] ['Intercept', '1', '3', '4', 'disease', '1:disease', '3:disease', '4:disease']


    """

    treatment_reference_pattern = re.compile(r'(?<=Treatment\()(.*?)(?=\))')
    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')
    level_pattern = re.compile(r'(?<=\[..)(.*?)(?=\])')

    factor = factor.split(":")

    # This renames non-interaction term names
    if len(factor) == 1:
        # This renames categorical variables
        if factor[0].startswith("C("):

            var_name = re.findall(level_pattern, factor[0])
            return var_name[0]

        else:
            # This keeps non-categorical variables names the same
            return factor[0]
    else:

        var_name = []

        for level in factor:
            if "C(" in level:

                var_name.append(''.join(re.findall(level_pattern, level)))
            else:
                var_name.append(level)

        return ':'.join(var_name)


def patsy_term_cleaner(factor):
    """


    Parameters
    ----------
    factor : A list of factors and levels which is provided by Patsy's design_info.term_names object.

    Returns
    -------
    String


    Description
    -----------
    This function returns a level name as a string.

    Ex.
    term_names = ['Intercept',
                  'C(drug, Treatment(2))',
                  'disease',
                  'C(drug, Treatment(2)):disease']


    [IN]  [patsy_term_cleaner(term) for term in term_names]
    [OUT] ['Intercept', 'drug', 'disease', 'drug:disease']

    """

    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=,|\))')

    factor = factor.split(":")

    # This renames non-interaction term names
    if len(factor) == 1:
        # This renames categorical variables
        if factor[0].startswith("C("):

            var_name = ''.join(re.findall(factor_pattern, factor[0]))
            return var_name

        else:
            # This keeps non-categorical variables names the same
            return factor[0]
    else:

        var_name = []

        for level in factor:
            if "C(" in level:

                var_name.append(''.join(re.findall(factor_pattern, level)))
            else:
                var_name.append(level)

        return ':'.join(var_name)


def variable_information(term_names, column_names, data):

    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=\)|,)')

    patsy_factor_information = {}
    my_factor_information = {}

    for factor in term_names:

        if factor == "Intercept" or "C(" not in factor:

            my_factor_information[factor] = factor
            patsy_factor_information[factor] = factor

        else:

            factor_split = factor.split(":")

            if len(factor_split) == 1:

                variable = (re.findall(factor_pattern, factor_split[0]))[0]
                variable_levels = list(np.unique(
                    data[variable][~data[variable].isnull()]))
                variable_levels = [str(level) for level in variable_levels]

                my_factor_information[variable] = variable_levels
                patsy_factor_information[factor_split[0]] = variable

            else:

                interaction_terms = []
                interaction_terms_levels = []

                for intfact in factor_split:

                    # Checking if it's a categorical factor
                    if intfact.startswith("C("):

                        variable = (re.findall(factor_pattern, intfact))[0]
                        variable_levels = list(np.unique(data[variable]))
                        variable_levels = [str(level)
                                           for level in variable_levels]

                        interaction_terms.append(variable)
                        interaction_terms_levels.append(variable_levels)

                    # This is for non-categorical factors in the model
                    else:
                        interaction_terms.append(intfact)
                        interaction_terms_levels.append([intfact])

                # Creating all combinations of interaction terms
                interaction_terms_var_names_interactions = list(
                    itertools.product(*interaction_terms_levels))
                interaction_terms_var_names_interactions = [
                    ":".join(level) for level in interaction_terms_var_names_interactions]

                my_factor_information[':'.join(
                    interaction_terms)] = interaction_terms_var_names_interactions
                patsy_factor_information[factor] = ':'.join(interaction_terms)

    ## Creating left hand of regression output ##
    mapping = {}
    for column_name in column_names:
        mapping[column_name] = patsy_column_cleaner(column_name)

    return patsy_factor_information, mapping, my_factor_information





def base_table(high_level_term_info, mapping_info, info_terms, reg_table):

    dv = list(reg_table)[0]

    # Creating the first table #
    terms = (pd.DataFrame.from_dict(
        high_level_term_info, orient="index")).reset_index()
    terms.columns = ["term", "term_cleaned"]
    terms["intx"] = [1 if ":" in t else 0 for t in list(
        high_level_term_info.keys())]
    terms["factor"] = [1 if "C(" in t else 0 for t in list(
        high_level_term_info.keys())]

    # Creating the second table #
    term_levels = {"term_cleaned": [],
                   "term_level_cleaned": []}

    for key in info_terms.keys():

        count = 1

        if key == 'Intercept' or terms[terms.term_cleaned == key].factor.item() == 0:
            term_levels["term_cleaned"].append(key)
            term_levels["term_level_cleaned"].append(info_terms[key])

        else:
            for value in info_terms[key]:

                term_levels["term_cleaned"].append(key)

                if count == 1:

                    term_levels["term_cleaned"].append(key)
                    term_levels["term_level_cleaned"].append(key)
                    term_levels["term_level_cleaned"].append(value)

                    count += 1
                else:
                    term_levels["term_level_cleaned"].append(value)

    # Creating the third table #
    current_terms = (pd.DataFrame.from_dict(
        mapping_info, orient="index")).reset_index()
    current_terms.columns = [dv, "term_level_cleaned"]
    current_terms["term_cleaned"] = [
        patsy_term_cleaner(key) for key in mapping_info.keys()]

    # Joining the tables together #
    table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                     how="left", on="term_cleaned")

    table = pd.merge(table, current_terms,
                     how="left", on=["term_cleaned", "term_level_cleaned"])

    table = pd.merge(table, pd.DataFrame.from_dict(reg_table),
                     how="left", on=dv)

    # Cleaning up final table #
    table[dv] = table["term_level_cleaned"]

    for idx in table.index:
        if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(info_terms.keys())[1:]:
            table.iloc[idx, 6] = "(reference)"
            table.iloc[idx, 7:] = ""
        else:
            if table.iloc[idx][dv] in list(info_terms.keys())[1:] and pd.isnull(table.iloc[idx, 6]):
                table.iloc[idx, 6:] = ""

    table = table[(table.intx == 0) | ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

    return table.iloc[:, 5:]


