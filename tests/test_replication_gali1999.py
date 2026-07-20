"""Replication tests for ``puremacro.examples.gali_1999_hours``.

Offline: reads the frozen FRED snapshot
``puremacro/replication/data/gali1999.csv``. Assertions are qualitative
(signs / orderings) so they survive a snapshot regeneration with a newer
FRED vintage; the exact points are documented in the example docstring.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gali_result():
    from puremacro.examples.gali_1999_hours import run_gali_replication
    # Small bootstrap: only the point IRFs are asserted tightly.
    return run_gali_replication(n_boot=30, seed=0)


def test_data_loads_offline():
    from puremacro.examples.gali_1999_hours import load_gali_data

    d = load_gali_data()
    assert {"dprod", "h", "dh"} <= set(d.columns)
    assert len(d) > 200  # quarterly, 1948 onwards
    assert np.isfinite(d.to_numpy()).all()


def test_hours_fall_on_impact_in_difference_spec(gali_result):
    """Gali's headline: hours per capita fall on impact after a positive
    technology shock when hours enter in first differences."""
    impact = gali_result["impact_diff"]
    assert impact < -0.05, (
        f"expected clearly negative hours impact in the difference spec "
        f"(Gali 1999), got {impact:+.3f}"
    )


def test_levels_spec_flips_or_attenuates(gali_result):
    """CEV critique: with hours in levels the impact response is well above
    the difference-spec response (flips positive on the 1948-2007 sample)."""
    d, l = gali_result["impact_diff"], gali_result["impact_levels"]
    assert l > d + 0.1, (
        f"expected levels-spec impact to flip/attenuate vs difference spec, "
        f"got levels {l:+.3f} vs diff {d:+.3f}"
    )


def test_technology_shock_raises_productivity_long_run(gali_result):
    """The identifying restriction's sign normalisation: the tech shock has
    a positive long-run effect on labor productivity."""
    assert gali_result["lr_prod"] > 0.0


def test_bands_bracket_point_diff_spec(gali_result):
    res = gali_result["diff"]
    assert res.irf_point.shape == res.irf_lower.shape == res.irf_upper.shape
    assert np.all(res.irf_lower <= res.irf_upper + 1e-12)


def test_impact_reads_match_cumulated_convention(gali_result):
    """At h=0 the cumulated and raw IRFs coincide; the dict fields must
    agree with the underlying result objects."""
    res_d = gali_result["diff"]
    assert gali_result["impact_diff"] == pytest.approx(
        float(res_d.irf_point[0, 1, 0])
    )
    raw0 = float(gali_result["levels"].irf_point[0, 1, 0])
    assert gali_result["impact_levels"] == pytest.approx(raw0)
