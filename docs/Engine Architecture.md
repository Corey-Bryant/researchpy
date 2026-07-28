Scope: We are starting fresh with our syntax engine implementation - we are to ignore anything previously done and come
at this fresh (completely ignore #file:syntax_engine.py and #file:matrix_engine.py ). This new syntax engine
will live within `engine/syntax.py` and the main class is called `FormulaSyntax`.


Objective: To build a completely new implementation of our engine system (formula, matrix, and table). Some new methods,
classes, and functions will be created, some existing ones will be used, and some will be modified. The goal is to
create a new engine system that is more efficient, more flexible, and more powerful than the previous implementation. 
The new engine system will be used for all of the regression models, descriptive statistics (numerical and categorical), 
and everything in between. The new engine system will be built on top of the `formulaic` package, which is a powerful 
and flexible formula parsing library that is designed to work with pandas DataFrames. The new engine system will also be 
built on top of the existing classes and methods in the `researchpy` package, in particular, the `DesignMatrix` class 
from #file:matrix_design.py , the `ModelDesignSpec` and `ModelTerm` classes from #file:multivariable.py , and 
the `Family` class (and subclasses) from #file:families.py .

**Syntax Engine (`engine/syntax.py`)**: Primary focus. This will be the universal input parsing layer that will be used 
for the regression models, descriptive statistics (numeric and categorical), and elsewhere. The new formula parsing 
engine that heavily relies on `formulaic`, and utilizes existing classes/methods, in particular, the `DesignMatrix` class from
#file:matrix_design.py , the `ModelDesignSpec` and `ModelTerm` classes from #file:multivariable.py , and
the `Family` class (and subclasses) from #file:families.py - that is not to say others are to be used, there are
known, for example #file:table.py can contain classes and functions for fitting the final table.


The syntax engine will be responsible for taking the input from the user (formula string, from_args, data, 
and other parameters) and parsing it to be able to be used by the matrix engine, and the `statistics/` submodule.


```markdown
FormulaSyntax — the universal input parsing layer.

Every researchpy function (descriptive, inferential, modeling) calls
``FormulaSyntax.from_args()`` (or its subclass override) first to normalize
any of the supported calling conventions into a single ``FormulaSyntax``
dataclass. The computation engine then operates exclusively on the spec.

Subclasses (e.g., AnovaSpec, TTestSpec) override ``from_args()`` to add
domain-specific validation while inheriting the full parsing logic.

Supported calling conventions:
    1. FormulaSyntax(df[['y']])                          → DataFrame/array, no groups
       FormulaSyntax(df[['y', 'z']])                     → DataFrame/array, no groups (if array has c>1 then compute for each c)
    2. FormulaSyntax(["y", "k", "c"], df)                → column list + DataFrame
    3. FormulaSyntax(df['y'])                            → Series/array, no groups
       FormulaSyntax('y', df)                            → Series/array, no groups
    4. FormulaSyntax("y ~ C(x)", df)                     → formula string + DataFrame
       FormulaSyntax("y ~ x", df)                        → formula string + DataFrame
    5. FormulaSyntax(dv="y", by="x", data=df)            → explicit keywords (cell grouping)
       FormulaSyntax(dv="y", iv=["x","k"], data=df)      → explicit keywords (marginal) (results stacked)
       FormulaSyntax(dv="y", by="x", over="k", data=df)  → pivot layout
    6. FormulaSyntax("y ~ C(x):C(k)", df)                → cell means (MultiIndex rows)
       FormulaSyntax("y ~ C(x)*C(k)", df)                → pivot layout

Formula operator semantics for descriptive stats:
    + : marginal (compute separately for each factor, stack results)
    : : cell (compute for each unique combination, MultiIndex rows)
    * : pivot (first factor → rows, second → columns)

Reverse mapping (to_formula):
    dv="y", iv=["x","k"]         → "dv ~ C(x) + C(k)"
    dv="y", by=["x"]             → "dv ~ C(x)"
    dv="y", by=["x","k"]         → "dv ~ C(x):C(k)"
    dv="y", by=["x"], over=["k"] → "dv ~ C(x)*C(k)"

    iv=["x","k"]         → "C(x) + C(k)"
    by=["x"]             → "C(x)"
    by=["x","k"]         → "C(x):C(k)"
    by=["x"], over=["k"] → "C(x)*C(k)"
```


**Matrix Engine (`engine/matrix.py`)**: The matrix engine will be responsible for taking
the output of the syntax engine and building the design matrix (and other matrices) that will be used in the
regression models, descriptive statistics, and elsewhere. This should heavily use `formulaic.Formula.get_model_matrix`, 
the `DesignMatrix` class from #file:matrix_design.py , and the `ModelTerm` class from #file:multivariable.py.

All calculations will use the `DesignMatrix` class, which is a wrapper around the `formulaic` package that provides a 
consistent interface for building design matrices from formula strings and pandas DataFrames. 




**Table Engine (`engine/table.py`)**: The table engine will be the final step after the matrix engine is complete. It 
    will be responsible for taking the output of the matrix engine and building the tables that will be used in the 
    regression models, descriptive statistics, and elsewhere.  

This should heavily use the `Table` class 
    from #file:table.py , and the `ModelTerm` class from #file:multivariable.py.

Requirements: Calling conventions must support formula string

Design: The syntax engine must support the use of 
