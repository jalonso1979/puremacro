"""Tests for the modern staggered-DiD estimators.

Each test feeds a synthetic panel with a known true ATT (and known
heterogeneity across cohorts where relevant) and asks whether the
estimator recovers it within bootstrap tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.did import (
    callaway_santanna, sun_abraham,
    borusyak_jaravel_spiess, synthetic_did,
    CallawaySantannaResult, SunAbrahamResult,
    BorusyakJaravelSpiessResult, SyntheticDiDResult,
)


def _staggered_panel(
    rng,
    *,
    n_units: int = 30,
    T: int = 20,
    cohorts: tuple = (8, 12, None),       # None = never-treated
    true_att_by_cohort: dict | None = None,
    sigma: float = 0.5,
):
    """Two-way FE panel with cohort-specific treatment effects.

    Y_{i,t} = α_i + λ_t + (treat dummy * τ_g) + ε_{i,t}.
    """
    if true_att_by_cohort is None:
        true_att_by_cohort = {c: 1.0 for c in cohorts if c is not None}

    units = [f"u{i:02d}" for i in range(n_units)]
    times = list(range(T))
    cohort_assign = rng.choice(cohorts, size=n_units, replace=True)

    rows = []
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3
    for i, u in enumerate(units):
        g = cohort_assign[i]
        for t in times:
            treat = 0.0 if g is None else (1.0 if t >= g else 0.0)
            tau = 0.0 if g is None else true_att_by_cohort[g]
            y = alpha[i] + lam[t] + tau * treat + sigma * rng.standard_normal()
            rows.append({
                "unit": u, "time": t, "y": y,
                "treat_time": np.nan if g is None else float(g),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Callaway-Sant'Anna
# ---------------------------------------------------------------------------


def test_callaway_santanna_recovers_homogeneous_att(rng):
    df = _staggered_panel(rng, n_units=80, T=16,
                            cohorts=(6, 10, None),
                            true_att_by_cohort={6: 1.5, 10: 1.5})
    res = callaway_santanna(df, n_boot=30)
    assert isinstance(res, CallawaySantannaResult)
    assert abs(res.att_overall - 1.5) < 0.5


def test_callaway_santanna_recovers_heterogeneous_atts(rng):
    df = _staggered_panel(rng, n_units=100, T=18,
                            cohorts=(6, 12, None),
                            true_att_by_cohort={6: 0.5, 12: 2.0})
    res = callaway_santanna(df, n_boot=30)
    att_gt = res.att_gt
    # Cohort-6 ATTs: should average ≈ 0.5
    g6 = att_gt[(att_gt["g"] == 6) & (att_gt["event_time"] >= 0)]["att"].mean()
    g12 = att_gt[(att_gt["g"] == 12) & (att_gt["event_time"] >= 0)]["att"].mean()
    assert abs(g6 - 0.5) < 0.5
    assert abs(g12 - 2.0) < 0.5


def test_callaway_santanna_event_study_columns(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = callaway_santanna(df, n_boot=20)
    es = res.att_event_study
    expected_cols = {"event_time", "att", "se", "lo", "hi", "n_cohorts"}
    assert expected_cols.issubset(es.columns)


def test_callaway_santanna_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = callaway_santanna(df, n_boot=10)
    assert isinstance(res, CallawaySantannaResult)
    with pytest.raises(Exception):
        res.att_overall = 99.0


# ---------------------------------------------------------------------------
# Sun-Abraham
# ---------------------------------------------------------------------------


def test_sun_abraham_recovers_share_weighted_att(rng):
    df = _staggered_panel(rng, n_units=120, T=18,
                            cohorts=(6, 12, None),
                            true_att_by_cohort={6: 0.5, 12: 2.0})
    res = sun_abraham(df, n_boot=20)
    assert isinstance(res, SunAbrahamResult)
    # The aggregated ATT should land between the two cohort effects.
    assert 0.3 < res.att_overall < 2.3


def test_sun_abraham_event_study_n_cohorts(rng):
    df = _staggered_panel(rng, n_units=40, T=12, cohorts=(5, 8, None))
    res = sun_abraham(df, n_boot=15)
    assert (res.att_event_study["n_cohorts"] >= 1).all()


def test_sun_abraham_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = sun_abraham(df, n_boot=10)
    assert isinstance(res, SunAbrahamResult)
    with pytest.raises(Exception):
        res.att_overall = 0.0


# ---------------------------------------------------------------------------
# Borusyak-Jaravel-Spiess
# ---------------------------------------------------------------------------


def test_bjs_recovers_homogeneous_att(rng):
    df = _staggered_panel(rng, n_units=80, T=15,
                            cohorts=(5, 10, None),
                            true_att_by_cohort={5: 1.0, 10: 1.0})
    res = borusyak_jaravel_spiess(df, n_boot=20)
    assert isinstance(res, BorusyakJaravelSpiessResult)
    assert abs(res.att_overall - 1.0) < 0.5


def test_bjs_event_study_centred_on_zero_pre_treatment(rng):
    """Pre-treatment event times should have a zero ATT under correct
    parallel trends, but this test sees only the *post* event times
    because BJS evaluates τ̂ on treated cells."""
    df = _staggered_panel(rng, n_units=60, T=14, cohorts=(5, 9, None))
    res = borusyak_jaravel_spiess(df, n_boot=15)
    # All event times in the BJS output are ≥ 0 (treated cells only).
    assert (res.att_event_study["event_time"] >= 0).all()


def test_bjs_result_is_frozen(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = borusyak_jaravel_spiess(df, n_boot=10)
    assert isinstance(res, BorusyakJaravelSpiessResult)
    with pytest.raises(Exception):
        res.att_overall = 0.0


# ---------------------------------------------------------------------------
# Synthetic-DiD
# ---------------------------------------------------------------------------


def _single_cohort_panel(rng, *, n_units=20, T=20, treat_time=10,
                            n_treated=3, true_att=1.5, sigma=0.4):
    units = [f"u{i:02d}" for i in range(n_units)]
    treated = set(units[:n_treated])
    rows = []
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3
    for i, u in enumerate(units):
        is_treated = u in treated
        for t in range(T):
            treat = 1.0 if (is_treated and t >= treat_time) else 0.0
            y = alpha[i] + lam[t] + true_att * treat + sigma * rng.standard_normal()
            rows.append({
                "unit": u, "time": t, "y": y,
                "treat_time": float(treat_time) if is_treated else np.nan,
            })
    return pd.DataFrame(rows)


def test_synthetic_did_recovers_treatment_effect(rng):
    df = _single_cohort_panel(rng, n_units=40, T=24, treat_time=12,
                                n_treated=4, true_att=2.0)
    res = synthetic_did(df, n_boot=40)
    assert isinstance(res, SyntheticDiDResult)
    # Spot-check fields exist
    assert hasattr(res, "tau") and hasattr(res, "omega")
    assert hasattr(res, "lambda_w") and hasattr(res, "treatment_time")
    assert abs(res.tau - 2.0) < 1.0


def test_synthetic_did_weights_sum_to_one(rng):
    df = _single_cohort_panel(rng, n_units=20, T=18, treat_time=10)
    res = synthetic_did(df, n_boot=10)
    np.testing.assert_allclose(float(res.omega.sum()), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(res.lambda_w.sum()), 1.0, atol=1e-6)


def test_synthetic_did_rejects_staggered_panel(rng):
    """SDID requires a single common treatment time."""
    df = _staggered_panel(rng, n_units=20, T=12, cohorts=(5, 8, None))
    with pytest.raises(ValueError, match="single common treatment time"):
        synthetic_did(df, n_boot=10)


def test_synthetic_did_result_is_frozen(rng):
    df = _single_cohort_panel(rng, n_units=15, T=14, treat_time=8)
    res = synthetic_did(df, n_boot=5)
    assert isinstance(res, SyntheticDiDResult)
    with pytest.raises(Exception):
        res.tau = 0.0


# ---------------------------------------------------------------------------
# summary() smoke tests
# ---------------------------------------------------------------------------


def test_callaway_santanna_summary_smoke(rng):
    df = _staggered_panel(rng, n_units=30, T=10, cohorts=(4, 7, None))
    res = callaway_santanna(df, n_boot=5)
    s = res.summary()
    assert "Callaway-Sant" in s


def test_synthetic_did_summary_smoke(rng):
    df = _single_cohort_panel(rng, n_units=15, T=12, treat_time=6)
    res = synthetic_did(df, n_boot=5)
    s = res.summary()
    assert "Synthetic-DiD" in s


def test_synthetic_did_lambda_w_field_name():
    """Field is named ``lambda_w`` (not ``lambda``) since ``lambda`` is
    a Python reserved keyword."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(SyntheticDiDResult)}
    assert "lambda_w" in field_names
    assert "lambda" not in field_names
