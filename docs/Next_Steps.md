# Plan (Next Steps)
1. **core/matrix_engine.py expansion** — Add build_indicator_matrix() support for weighted observations and sparse matrices (when needed for high-cardinality groups). 
2. **Wire remaining descriptive functions** — Upgrade skewness(), kurtosis(), confidence_interval(), iqr() to support all 5 conventions via _route_computation() with the iteration fallback.
3. **Refactor _dispatcher.py (summarize)** — Replace the current manual type-checking with resolve() so summarize() uses the same unified input gate.
4. **CoreModel gradual migration** — Eventually have CoreModel.__init__ delegate to resolve() + matrix_engine.DesignMatrices instead of calling Patsy directly. Remove legacy variable_information() calls once ModelTerms fully replaces them.
5. **Tests** — Write formal pytest tests for resolve(), build_indicator_matrix(), grouped_statistic(), and the 5-convention API for each updated function.
6. **tabulate() refinement** — Wire into resolve() for formula-based grouping if needed (e.g., tabulate("gender ~ C(treatment)", df) to get frequency tables per treatment group).
7. **Deprecation path** — Add deprecation warnings to summary.py's summarize(), summary_cont(), summary_cat(), codebook() pointing users to the new descriptive submodule equivalents.