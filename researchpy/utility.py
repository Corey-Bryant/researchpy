import pandas as pd
import scipy.stats
import numpy as np


def rounder(lst, decimals=4, in_place=True):
    """
        Iterates through a list and returns the rounded number.
    """
    if in_place:
        idx = 0
        for item in lst:
            lst[idx] = round(item, decimals)
            idx += 1
    else:
        return [round(item, decimals) for item in lst]


def return_numeric(value):
    """
    Ensures that a value is numeric, and if not, returns original value.
    """

    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return value.item()
    else:
        return value


def rp_round(value, decimals=4):
    """
    Ensures that a value is numeric, and if not, returns original value.
    """

    if isinstance(value, np.ndarray):
        # Extract scalar from array (handles 0-d and 1-element arrays)
        val = value.flat[0] if value.size == 1 else float(value.ravel()[0])
        return round(float(val), decimals)

    if isinstance(value, (np.floating, np.integer)):
        return round(float(value), decimals)

    else:
        try:
            return round(float(value), decimals)
        except:
            return value


def as_numeric(value):
    """
    Ensures that a value is numeric, and if not, returns original value.
    """

    if isinstance(value, np.ndarray):
        # Extract scalar from array (handles 0-d and 1-element arrays)
        val = value.flat[0] if value.size == 1 else value.ravel()[0]
        return float(val)

    if isinstance(value, (np.floating, np.integer)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def patsy_column_cleaner(factor):
    """Thin backward-compatible wrapper.

    Delegates to :func:`researchpy.core.syntax_engine.clean_column_name`.
    New code should import ``clean_column_name`` directly.
    """
    from researchpy.core.syntax_engine import clean_column_name
    return clean_column_name(factor)


def patsy_term_cleaner(factor):
    """Thin backward-compatible wrapper.

    Delegates to :func:`researchpy.core.syntax_engine.clean_term_name`.
    New code should import ``clean_term_name`` directly.
    """
    from researchpy.core.syntax_engine import clean_term_name
    return clean_term_name(factor)



def variable_information(term_names, column_names, data):
    """Thin backward-compatible wrapper.

    Delegates to :func:`researchpy.core.syntax_engine.variable_information`.
    New code should import from ``researchpy.core.syntax_engine`` directly.
    """
    from researchpy.core.syntax_engine import variable_information as _variable_information
    return _variable_information(term_names, column_names, data)





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
    current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in mapping_info.keys()]

    # Joining the tables together #
    table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                     how="left", on="term_cleaned")

    table = pd.merge(table, current_terms,
                     how="left", on=["term_cleaned", "term_level_cleaned"])

    table = pd.merge(table, pd.DataFrame.from_dict(reg_table),
                     how="left", on=dv)
    #table = pandas.merge(table, pd.DataFrame.from_dict(reg_table).astype(object), how="left", on=dv)  # From dev3.7.1

    # Cleaning up final table #
    table[dv] = table["term_level_cleaned"]

    for idx in table.index:
        if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(info_terms.keys())[1:]:
            table.iloc[idx, 6] = "(reference)"
            table.iloc[idx, 7:] = np.nan            # used to be = ""
        else:
            if table.iloc[idx][dv] in list(info_terms.keys())[1:] and pd.isnull(table.iloc[idx, 6]):
                table.iloc[idx, 6:] = np.nan        # used to be = ""

    table = table[(table.intx == 0) | ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

    return table.iloc[:, 5:]
