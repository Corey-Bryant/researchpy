# -*- coding: utf-8 -*-
"""
Tests for Phase 1: TermSpec, ComputeSpec.sub_specs, mixed-formula parsing,
and malformed formula validation in core/spec.py.
"""

import pytest
import numpy as np
import pandas as pd

from researchpy.core.spec import (
    ComputeSpec,
    TermSpec,
    resolve,
    _parse_formula,
    _is_star_expansion,
    _build_sub_specs,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grouped_df():
    """DataFrame with two grouping variables and a numeric DV."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "k": ["lo", "hi", "lo", "hi", "lo", "hi", "lo", "hi"],
        "z": ["m", "m", "f", "f", "f", "f", "m", "m"],
    })


@pytest.fixture
def simple_df():
    """Simple DataFrame with one grouping variable."""
    return pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "b", "b", "c", "c"],
    })


# ============================================================================
# TermSpec dataclass
# ============================================================================

class TestTermSpec:
    """Tests for TermSpec creation and attributes."""

    def test_basic_creation(self):
        ts = TermSpec(term_name="x", term_raw="C(x)", layout="iv", variables=["x"])
        assert ts.term_name == "x"
        assert ts.term_raw == "C(x)"
        assert ts.layout == "iv"
        assert ts.variables == ["x"]

    def test_interaction_term(self):
        ts = TermSpec(
            term_name="k:z",
            term_raw="C(k):C(z)",
            layout="by",
            variables=["k", "z"],
        )
        assert ts.term_name == "k:z"
        assert ts.term_raw == "C(k):C(z)"
        assert ts.layout == "by"
        assert ts.variables == ["k", "z"]


# ============================================================================
# ComputeSpec.sub_specs field
# ============================================================================

class TestComputeSpecSubSpecs:
    """Tests for the sub_specs field on ComputeSpec."""

    def test_default_is_none(self):
        spec = ComputeSpec(dv=["y"])
        assert spec.sub_specs is None

    def test_can_assign_sub_specs(self):
        ts = TermSpec(term_name="x", term_raw="C(x)", layout="iv", variables=["x"])
        spec = ComputeSpec(dv=["y"], sub_specs=[ts])
        assert spec.sub_specs is not None
        assert len(spec.sub_specs) == 1
        assert spec.sub_specs[0].term_name == "x"


# ============================================================================
# Malformed formula validation
# ============================================================================

class TestMalformedFormula:
    """Tests that RHS terms not wrapped in C() raise informative errors."""

    def test_bare_variable_raises(self, simple_df):
        with pytest.raises(ValueError, match="not wrapped in C\\(\\)"):
            resolve("y ~ g", simple_df)

    def test_bare_variable_suggests_fix(self, simple_df):
        with pytest.raises(ValueError, match="Use: 'y ~ C\\(g\\)'"):
            resolve("y ~ g", simple_df)

    def test_bare_variable_in_interaction_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not wrapped in C\\(\\)"):
            resolve("y ~ C(x):k", grouped_df)

    def test_wrapped_variable_passes(self, simple_df):
        # Should NOT raise
        spec = resolve("y ~ C(g)", simple_df)
        assert spec.by == ["g"]

    def test_wrapped_interaction_passes(self, grouped_df):
        # Should NOT raise
        spec = resolve("y ~ C(x):C(k)", grouped_df)
        assert spec.by == ["x", "k"]


# ============================================================================
# Existing formula patterns still work
# ============================================================================

class TestExistingPatternsPreserved:
    """Ensure existing formula patterns produce the same results as before."""

    def test_single_factor(self, simple_df):
        spec = resolve("y ~ C(g)", simple_df)
        assert spec.dv == ["y"]
        assert spec.by == ["g"]
        assert spec.iv is None
        assert spec.over is None
        assert spec.sub_specs is None

    def test_multiple_main_effects(self, grouped_df):
        spec = resolve("y ~ C(x) + C(k)", grouped_df)
        assert spec.dv == ["y"]
        assert spec.iv == ["x", "k"]
        assert spec.by is None
        assert spec.over is None
        assert spec.sub_specs is None

    def test_interaction(self, grouped_df):
        spec = resolve("y ~ C(x):C(k)", grouped_df)
        assert spec.dv == ["y"]
        assert spec.by == ["x", "k"]
        assert spec.iv is None
        assert spec.over is None
        assert spec.sub_specs is None

    def test_star_expansion(self, grouped_df):
        spec = resolve("y ~ C(x)*C(k)", grouped_df)
        assert spec.dv == ["y"]
        assert spec.by == ["x"]
        assert spec.over == ["k"]
        assert spec.iv is None
        assert spec.sub_specs is None

    def test_series_convention(self):
        s = pd.Series([1, 2, 3], name="val")
        spec = resolve(s)
        assert spec.dv == ["val"]
        assert spec.by is None
        assert spec.sub_specs is None

    def test_column_list_convention(self, grouped_df):
        spec = resolve(["y"], grouped_df)
        assert spec.dv == ["y"]
        assert spec.by is None
        assert spec.sub_specs is None

    def test_keyword_convention_by(self, simple_df):
        spec = resolve(dv="y", by="g", data=simple_df)
        assert spec.dv == ["y"]
        assert spec.by == ["g"]
        assert spec.sub_specs is None

    def test_keyword_convention_iv(self, grouped_df):
        spec = resolve(dv="y", iv=["x", "k"], data=grouped_df)
        assert spec.dv == ["y"]
        assert spec.iv == ["x", "k"]
        assert spec.sub_specs is None

    def test_keyword_convention_pivot(self, grouped_df):
        spec = resolve(dv="y", by="x", over="k", data=grouped_df)
        assert spec.dv == ["y"]
        assert spec.by == ["x"]
        assert spec.over == ["k"]
        assert spec.sub_specs is None


# ============================================================================
# Mixed formula → sub_specs
# ============================================================================

class TestMixedFormula:
    """Tests for mixed formula parsing into sub_specs."""

    def test_main_plus_interaction(self, grouped_df):
        """y ~ C(x) + C(k):C(z) is NOT a star expansion → sub_specs."""
        spec = resolve("y ~ C(x) + C(k):C(z)", grouped_df)
        assert spec.sub_specs is not None
        assert len(spec.sub_specs) == 2
        assert spec.iv is None
        assert spec.by is None
        assert spec.over is None

    def test_sub_spec_main_effect(self, grouped_df):
        spec = resolve("y ~ C(x) + C(k):C(z)", grouped_df)
        main_spec = spec.sub_specs[0]
        assert main_spec.term_name == "x"
        assert main_spec.term_raw == "C(x)"
        assert main_spec.layout == "iv"
        assert main_spec.variables == ["x"]

    def test_sub_spec_interaction(self, grouped_df):
        spec = resolve("y ~ C(x) + C(k):C(z)", grouped_df)
        interaction_spec = spec.sub_specs[1]
        assert interaction_spec.term_name == "k:z"
        assert interaction_spec.term_raw == "C(k):C(z)"
        assert interaction_spec.layout == "by"
        assert interaction_spec.variables == ["k", "z"]

    def test_multiple_main_plus_interaction(self, grouped_df):
        """y ~ C(x) + C(z) + C(x):C(k) → mixed (x and z are main, but
        interaction has x:k, and z is NOT in the interaction)."""
        spec = resolve("y ~ C(x) + C(z) + C(x):C(k)", grouped_df)
        assert spec.sub_specs is not None
        assert len(spec.sub_specs) == 3

        # First two are main effects (iv layout)
        assert spec.sub_specs[0].layout == "iv"
        assert spec.sub_specs[0].term_name == "x"
        assert spec.sub_specs[1].layout == "iv"
        assert spec.sub_specs[1].term_name == "z"

        # Third is interaction (by layout)
        assert spec.sub_specs[2].layout == "by"
        assert spec.sub_specs[2].term_name == "x:k"
        assert spec.sub_specs[2].variables == ["x", "k"]

    def test_star_is_not_mixed(self, grouped_df):
        """y ~ C(x)*C(k) should NOT produce sub_specs (it's a star expansion)."""
        spec = resolve("y ~ C(x)*C(k)", grouped_df)
        assert spec.sub_specs is None
        assert spec.by == ["x"]
        assert spec.over == ["k"]


# ============================================================================
# _is_star_expansion helper
# ============================================================================

class TestIsStarExpansion:
    """Tests for the _is_star_expansion detection logic."""

    def test_true_for_matching_components(self):
        """main=[x, k], interaction=[x:k] → True"""
        from researchpy.containers.multivariable import Term

        main = [Term(term="C(x)"), Term(term="C(k)")]
        interaction = [Term(term="C(x):C(k)")]
        assert _is_star_expansion(main, interaction) is True

    def test_false_when_main_not_in_interaction(self):
        """main=[x], interaction=[k:z] → False (x not in k:z)"""
        from researchpy.containers.multivariable import Term

        main = [Term(term="C(x)")]
        interaction = [Term(term="C(k):C(z)")]
        assert _is_star_expansion(main, interaction) is False

    def test_false_when_extra_main_effect(self):
        """main=[x, z], interaction=[x:k] → False (z not in any interaction)"""
        from researchpy.containers.multivariable import Term

        main = [Term(term="C(x)"), Term(term="C(z)")]
        interaction = [Term(term="C(x):C(k)")]
        assert _is_star_expansion(main, interaction) is False


# ============================================================================
# GroupBy convention still works
# ============================================================================

class TestGroupByPreserved:
    """Ensure GroupBy resolution still works."""

    def test_series_groupby(self, simple_df):
        grouped = simple_df.groupby("g")["y"]
        spec = resolve(grouped)
        assert spec.dv == ["y"]
        assert spec.by == ["g"]
        assert spec.sub_specs is None

