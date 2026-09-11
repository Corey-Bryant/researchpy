from researchpy.utility import rounder, return_numeric, patsy_column_cleaner

def test_rounder():
    # Test rounding functionality
    lst = [1.12345, 2.67891, 3.98765]
    rounder(lst, decimals=3)
    assert lst == [1.123, 2.679, 3.988]

def test_return_numeric():
    # Test with numeric types
    assert return_numeric(5.5) == 5.5
    assert return_numeric(10) == 10

    # Test with non-numeric type
    assert return_numeric("string") == "string"

def test_patsy_column_cleaner():
    # Test with categorical variable
    result = patsy_column_cleaner("C(drug, Treatment(2))[T.1]")
    assert result == "1"

    # Test with non-categorical variable
    result = patsy_column_cleaner("disease")
    assert result == "disease"

    # Test with interaction term
    result = patsy_column_cleaner("C(drug, Treatment(2))[T.1]:disease")
    assert result == "1:disease"
