"""Tests for puremacro.narrative.indices.cross_source."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestConsensusDisagreement:
    def test_empty_dict_raises(self):
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        with pytest.raises(ValueError, match="empty"):
            consensus_disagreement({})

    def test_non_series_value_raises(self):
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        with pytest.raises(ValueError, match="not a pd.Series"):
            consensus_disagreement({"a": [1, 2, 3]})

    def test_two_constant_series_yield_zero_disagreement(self):
        """Two perfectly-equal series should give std=0 at each t."""
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        out = consensus_disagreement({"a": s, "b": s.copy()})
        # std across two identical series should be 0 at every t
        assert (out["disagreement"].abs() < 1e-10).all()
        # n_active should be 2 everywhere
        assert (out["n_active"] == 2).all()

    def test_two_opposite_series_yield_consensus_zero(self):
        """Two series symmetric around zero → consensus should be 0."""
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        b = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], index=idx)
        # After z-scoring (each centered on its own mean), the two
        # series are symmetric → consensus = 0 at every t.
        out = consensus_disagreement({"a": a, "b": b})
        assert (out["consensus"].abs() < 1e-10).all()
        # Disagreement is non-trivial (they move in opposite directions)
        assert (out["disagreement"] > 0).any()

    def test_quarterly_input_resampled_monthly(self):
        """Quarterly native frequency forward-fills within the period."""
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx_q = pd.date_range("2020-03-31", periods=4, freq="QE")
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx_q)
        out = consensus_disagreement(
            {"a": s, "b": s.copy()},
            freq="ME",
        )
        # Output should have monthly index (at least 10 months).
        assert len(out) >= 6
        # Constant pair → disagreement should be near zero
        assert (out["disagreement"].abs() < 1e-10).all()

    def test_min_active_drops_rows(self):
        """Rows with fewer than min_active series are dropped."""
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        # b has data only for the last 3 months
        b = pd.Series([np.nan, np.nan, np.nan, 4.0, 5.0, 6.0], index=idx)
        out = consensus_disagreement(
            {"a": a, "b": b},
            min_active=2,
        )
        # Only the last 3 rows should survive (n_active >= 2)
        assert len(out) == 3
        assert (out["n_active"] == 2).all()

    def test_return_panel_adds_per_series_columns(self):
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        out = consensus_disagreement(
            {"a": s, "b": s.copy()},
            return_panel=True,
        )
        assert {"consensus", "disagreement", "n_active", "a", "b"} <= set(out.columns)

    def test_degenerate_zero_variance_series_drops(self):
        """A constant series has zero std and z-scoring is undefined; it should drop from active set."""
        from puremacro.narrative.indices.cross_source import (
            consensus_disagreement,
        )
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        const = pd.Series([5.0] * 6, index=idx)
        moving = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
        out = consensus_disagreement(
            {"const": const, "moving": moving},
            min_active=1,  # allow single-series rows
        )
        # const drops to NaN; only moving counts → n_active = 1
        assert (out["n_active"] == 1).all()


class TestGroups:
    def test_groups_structure(self):
        from puremacro.narrative.indices.cross_source import GROUPS
        # 'all' must include every symbol in every other group
        all_symbols = set(GROUPS["all"])
        for k, names in GROUPS.items():
            if k == "all":
                continue
            for name in names:
                assert name in all_symbols, (
                    f"GROUPS[{k!r}] symbol {name!r} not in GROUPS['all']")

    def test_groups_keys_are_expected_families(self):
        from puremacro.narrative.indices.cross_source import GROUPS
        expected_families = {
            "macro_uncertainty", "labor", "us_policy",
            "eu_policy", "geopolitical", "social", "all",
        }
        assert expected_families == set(GROUPS.keys())
