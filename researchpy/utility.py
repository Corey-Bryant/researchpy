from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
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



def _infer_numeric_from_string(value: str, decimal_format: str = '.') -> tuple[float | int | str, bool]:
    """
    Attempt to infer the best numeric type for a string value.

    Parameters
    ----------
    value : str
        The string value to infer the best numeric type for.
    decimal_format : str, optional
        The character used for decimal points in the string. Default is '.'.

    Returns
    -------
    tuple[converted_value, success]
        converted_value is int, float, or the original string if conversion failed.
        success is True if numeric conversion succeeded.
    """
    # Strip whitespace
    s = value.strip()

    # Honor a non-standard decimal separator (e.g. ',' for many European
    # locales). When one is configured, a literal '.' is NOT a valid decimal
    # separator, so inference must fail; the locale separator is normalized to
    # '.' for parsing.
    if decimal_format != '.':
        if '.' in s:
            return value, False
        s = s.replace(decimal_format, '.')

    # Check if it looks like an integer (no decimal point or exponent notation)
    has_decimal = '.' in s
    has_exponent = 'e' in s.lower()

    if not has_decimal and not has_exponent:
        # Try int first
        try:
            return int(s), True
        except (ValueError, TypeError):
            pass

    # Try float
    try:
        return float(s), True
    except (ValueError, TypeError):
        pass

    # Conversion failed, return original
    return value, False


def as_numeric(value: object,
               numeric_dtype: Union[float | int | any] = float,
               errors: str = "raise",
               ndigits: Optional[int] = None,
               ) -> Union[float | int | any]:
    """Converts a value to Python built-in numeric, and if not possible, returns original value if 'errors=raise' (default).

    Parameters
    ----------
    value : object
        The value to convert to numeric.
    numeric_dtype : Union[float | int | any], optional
        The desired numeric type. float is the default. If any, the function will attempt to infer
        the best numeric type (int preferred for whole numbers, float otherwise).
    errors : str, optional
        How to handle errors during conversion. Options are:
        - 'raise': Raise a ValueError if conversion fails (default).
        - 'ignore': Return the original value if conversion fails.
        - 'fallback': Attempt to convert to float if the specified numeric_dtype fails.
    ndigits : int, optional
        Number of decimal places to round the result to (default is None, no rounding).
        Equivalent to the ``ndigits`` parameter in Python's built-in ``round()``.
        Rounding is only applied to successfully converted numeric values;
        values returned as-is via ``errors='ignore'`` are not rounded.

    Returns
    -------
    float, int, or original value
        The converted numeric value, or the original value if conversion fails
        and errors='ignore'.

    Examples
    --------
    >>> as_numeric(5)
    5
    >>> as_numeric("5")
    5.0
    >>> as_numeric("five", errors='ignore')
    'five'
    >>> as_numeric(np.array([5]))
    5.0
    >>> as_numeric(np.array([[5]]))
    5.0
    >>> as_numeric(3.14159, ndigits=2)
    3.14
    >>> as_numeric(np.float64(3.14159), ndigits=2)
    3.14
    >>> as_numeric("5.6789", ndigits=1)
    5.7
    >>> as_numeric('5',any)
    5
    >>> as_numeric('5.5',any)
    5.5
    >>> as_numeric('taco',any)
    'taco'

    """

    def _to_numeric(scalar,
                    _numeric_dtype: Union[float | int | any] = float,
                    _ndigits: int | None = None,
                    ) -> Union[float, int, object]:
        """
        Helper method to apply desired numeric type conversion, optional rounding,
        and handle errors.
        """
        if _numeric_dtype == any:
            # Infer the best numeric type
            if isinstance(scalar, str):
                result, success = _infer_numeric_from_string(scalar)
                if not success:
                    raise ValueError(f"Cannot convert {scalar} to numeric.")

            elif isinstance(scalar, (np.floating, np.integer)):
                result = scalar.item()

            elif isinstance(scalar, np.ndarray):
                result = scalar.item() if scalar.size == 1 else scalar.ravel()[0].item()

            else:
                # Existing Python numeric or other type
                result = float(scalar) if not isinstance(scalar, (int, float)) else scalar

        else:
            result = _numeric_dtype(scalar)

        if _ndigits is not None:
            result = round(result, _ndigits)

        return result




    if isinstance(value, np.ndarray):
        # Extract scalar from array (handles 0-d and 1-element arrays)
        val = value.flat[0] if value.size == 1 else value.ravel()[0]

        if numeric_dtype == any:
            # Special handling for arrays with any dtype
            if value.size == 1:
                result = val.item()
            else:
                #result = val
                try:
                    result = float(val.item())
                except (ValueError, TypeError):
                    result = val

            if ndigits is not None and isinstance(result, (int, float, np.number)):
                result = round(result, ndigits)

            return result

        try:
            return _to_numeric(val, numeric_dtype, ndigits)
        except (TypeError, ValueError):
            if errors == "fallback" and numeric_dtype != float:
                return _to_numeric(val, float, ndigits) # Wgat
            elif errors == "ignore":
                return value
            else:
                raise ValueError(f"Cannot convert {val} to {numeric_dtype.__name__}.")

    elif isinstance(value, (np.floating, np.integer)):
        try:
            return _to_numeric(value, numeric_dtype, ndigits)
        except (TypeError, ValueError):
            if errors == "fallback" and numeric_dtype != float:
                return _to_numeric(value, float, ndigits)
            elif errors == "raise":
                raise ValueError(f"Cannot convert {value} to {numeric_dtype.__name__}.")
            else:
                return value

    else:
        if numeric_dtype == any:
            # Use inference for non-numpy types. Pass ``any`` (not ``float``) so already-int values
            # are preserved instead of being widened.
            result, success = _infer_numeric_from_string(str(value)) if isinstance(value, str) else (
                _to_numeric(value, any), True
                )

            if not success:
                if errors == "raise":
                    raise ValueError(f"Cannot convert {value} to numeric.")
                else:
                    return value

            if ndigits is not None:
                result = round(result, ndigits)
            return result
        else:
            try:
                return _to_numeric(value, numeric_dtype, ndigits)
            except (TypeError, ValueError):
                if errors == "fallback" and numeric_dtype != float:
                    return _to_numeric(value, float, ndigits)
                elif errors == "raise":
                    raise ValueError(f"Cannot convert {value} to {numeric_dtype.__name__}.")
                else:
                    return value







#-------------------------------------------------------------------#
# -- These are deprecated and will be removed in future versions -- #
#-------------------------------------------------------------------#
def patsy_column_cleaner(factor):
    """Thin backward-compatible wrapper.

    Delegates to :meth:`researchpy.containers.multivariable.Term.clean_column_name`.
    New code should use ``Term.clean_column_name`` directly.
    """
    from researchpy.containers.multivariable import Term
    return Term.clean_column_name(factor)


def patsy_term_cleaner(factor):
    """Thin backward-compatible wrapper.

    Delegates to :meth:`researchpy.containers.multivariable.Term.clean_term_name`.
    New code should use ``Term.clean_term_name`` directly.
    """
    from researchpy.containers.multivariable import Term
    return Term.clean_term_name(factor)


def variable_information(term_names: List[str], column_names: List[str],
                         data: pd.DataFrame,) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Extract factor/variable metadata from formula term and column names.

    Builds three mappings used by model output formatting:
    1. High-level term info: raw term → cleaned variable name
    2. Column mapping: raw column name → cleaned level name
    3. Detailed factor info: variable name → list of unique levels

    Parameters
    ----------
    term_names : list of str
        Formula term names (e.g., from ``model_spec.terms`` or
        ``design_info.term_names``).
    column_names : list of str
        Column names from the design matrix (e.g., from
        ``model_spec.column_names`` or ``design_info.column_names``).
    data : pd.DataFrame
        Source data used to determine unique factor levels.

    Returns
    -------
    high_level_term_info : dict
        Maps raw term string → cleaned variable name string.
    mapping : dict
        Maps raw column name → cleaned level name.
    factor_info : dict
        Maps cleaned variable name → list of unique level strings
        (or the variable name itself for continuous variables).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'drug': [1,2,3,1], 'disease': [0,1,0,1]})
    >>> terms = ['Intercept', 'C(drug)']
    >>> cols = ['Intercept', 'C(drug)[T.2]', 'C(drug)[T.3]']
    >>> high, mapping, info = variable_information(terms, cols, df)
    >>> high['C(drug)']
    'drug'
    >>> info['drug']
    ['1', '2', '3']
    """
    import itertools
    import re
    from researchpy.containers.multivariable import Term

    factor_pattern = re.compile(r'(?<=C\()(.*?)(?=\)|,)')

    high_level_term_info: Dict[str, str] = {}
    factor_info: Dict[str, Any] = {}

    for factor in term_names:
        if factor == "Intercept" or "C(" not in factor:
            factor_info[factor] = factor
            high_level_term_info[factor] = factor
        else:
            factor_split = factor.split(":")

            if len(factor_split) == 1:
                variable = (re.findall(factor_pattern, factor_split[0]))[0]
                variable_levels = list(np.unique(data[variable][~data[variable].isnull()]))
                variable_levels = [str(level) for level in variable_levels]

                factor_info[variable] = variable_levels
                high_level_term_info[factor_split[0]] = variable

            else:
                interaction_terms = []
                interaction_terms_levels = []

                for intfact in factor_split:
                    if intfact.startswith("C("):
                        variable = (re.findall(factor_pattern, intfact))[0]
                        variable_levels = list(np.unique(data[variable]))
                        variable_levels = [str(level) for level in variable_levels]
                        interaction_terms.append(variable)
                        interaction_terms_levels.append(variable_levels)

                    else:
                        interaction_terms.append(intfact)
                        interaction_terms_levels.append([intfact])

                interaction_combos = list(itertools.product(*interaction_terms_levels))
                interaction_combos = [":".join(level) for level in interaction_combos]

                factor_info[':'.join(interaction_terms)] = interaction_combos
                high_level_term_info[factor] = ':'.join(interaction_terms)

    # Build column name mapping
    mapping: Dict[str, str] = {}
    for column_name in column_names:
        mapping[column_name] = Term.clean_column_name(column_name)

    return high_level_term_info, mapping, factor_info


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
    current_terms = (pd.DataFrame.from_dict(mapping_info, orient="index")).reset_index()
    current_terms.columns = [dv, "term_level_cleaned"]
    current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in mapping_info.keys()]

    # Joining the tables together #
    table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                     how="left",
                     on="term_cleaned")

    table = pd.merge(table, current_terms,
                     how="left",
                     on=["term_cleaned", "term_level_cleaned"])

    #table = pd.merge(table, pd.DataFrame.from_dict(reg_table), how="left", on=dv)
    table = pd.merge(table, pd.DataFrame.from_dict(reg_table).astype(object), how="left", on=dv)  # From dev3.7.1

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
