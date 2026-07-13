# Codebase Architecture Review

## Current State
| Layer | File | Role | Status |
|-------|------|------|--------|
| Input parsing | core/syntax_engine.py (SyntaxSpec) | Intended replacement for spec.py | Written but not wired in — nothing imports it yet |
| Matrix (descriptive) | core/matrix_engine.py | Indicator matrix + grouped stat computations | Active — used by statistics/_compute.py |
| Matrix (model) | core/matrix_design.py (DMatrix) | Formulaic-backed design matrices for regression | Active — used by core/model.py (BaseModel) |
| Base model | core/model.py (BaseModel) | Inherits DMatrix; shared regression infrastructure | Active — parent of GeneralModel, LinearModel |
| Models | models/base.py (CoreModel) | Standalone regression base (doesn't inherit DMatrix) | Active — exported as rp.CoreModel |
| Syntax utilities | core/utility.py | variable_information, patsy_term_cleaner, patsy_column_cleaner, base_table | Active — used by matrix_design.py and core/model.py |

```
   syntax_engine.py  (no researchpy deps)
        ↑
   matrix_design.py  (imports syntax_engine for variable_information/term parsing)
   matrix_engine.py  (imports syntax_engine for SyntaxSpec resolution)
        ↑
   BaseModel (core/model.py)  (inherits DMatrix from matrix_design)
        ↑
   GeneralModel, LinearModel, Regress, Anova, Logistic
```