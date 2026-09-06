"""Regression snapshot for the Antolín-Díaz & Rubio-Ramírez (2018, AER)
narrative-sign-restriction demo — the 1979Q4 Volcker restrictions on a
SYNTHETIC 3-variable VAR.

What this file is (and is not)
------------------------------
``puremacro.examples.narrative_sign_adrr.run_demo`` applies AD-RR's two
Volcker restrictions —

  (I)   the monetary shock in 1979Q4 was positive (contractionary), and
  (III) it was the *overwhelming* contributor to the unexpected rise in
        the federal funds rate that quarter —

to a synthetic, deterministic 3-variable VAR(1) (``_simulate``, seed 12,
with a +3.5 s.d. monetary shock planted in 1979Q4). The numbers below are a
**snapshot of that demo's own output** recorded at v2.3.0. They are NOT
values from the paper: AD-RR's application uses Uhlig's (2005) six-variable
monthly US dataset, which this repository cannot fetch offline, and they
report their IRFs graphically. A change in these numbers therefore means the
estimator's behaviour changed — not that a "replication" broke.

The remaining tests check the qualitative AD-RR mechanism on the synthetic
data (narrative restrictions tighten the sign-identified set), the acceptance
diagnostics, the standard SVAR capabilities (.irf / .fevd /
.historical_decomposition) and the presentation contract.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.examples.narrative_sign_adrr import (
    VARIABLES,
    VOLCKER_DATE,
    _simulate,
    run_demo,
)
from puremacro.var.identify import (
    NarrativeSignResult,
    NarrativeSignSVARResult,
    identify_narrative_sign,
)

# ---------------------------------------------------------------------------
# Regression snapshot of run_demo(H=16, n_draws=3000) at v2.3.0 (synthetic
# seed-12 data; OLS mode; seed=0). Weighted-median IRF of the monetary shock
# (column 0) on [FFR, Inflation, Output growth] at h = 0, 1, 2, 4.
# ---------------------------------------------------------------------------
DEMO_SNAPSHOT_IRF = {
    0: np.array([0.696638, -0.698089, -0.438239]),
    1: np.array([0.355378, -0.434801, -0.176029]),
    2: np.array([0.175636, -0.261942, -0.065436]),
    4: np.array([0.036881, -0.084195, -0.003956]),
}

# Minimum shrinkage of the mean 68% band width (horizons 0..16) that the
# planted +3.5 s.d. shock delivers on the synthetic data; the snapshot values
# at v2.3.0 were 25.8% / 24.0% / 20.8%. These are properties of the demo DGP,
# not magnitudes taken from the paper.
DEMO_MIN_SHRINKAGE = {
    "Fed funds rate": 0.20,
    "Inflation": 0.20,
    "Output growth": 0.15,
}


@pytest.fixture(scope="module")
def adrr_demo():
    """Run the Volcker 1979Q4 demo on the synthetic DGP."""
    return run_demo(H=16, n_draws=3000)


def test_demo_data_is_synthetic_and_deterministic():
    """Provenance guard: the demo runs on _simulate() output (no data files),
    the planted shock sits in 1979Q4, and the DGP is seed-deterministic."""
    Y1, dates, t_volcker = _simulate()
    Y2, _, _ = _simulate()
    np.testing.assert_array_equal(Y1, Y2)
    assert Y1.shape == (172, 3)
    assert dates[t_volcker] == pd.Timestamp("1979-10-01")
    assert pd.Timestamp(VOLCKER_DATE).to_period("Q") == dates[t_volcker].to_period("Q")


def test_adrr2018_demo_regression_snapshot(adrr_demo):
    """The demo's weighted-median IRF must reproduce the v2.3.0 snapshot
    (synthetic data; see module docstring — not the paper's numbers)."""
    narr = adrr_demo["narr"]
    assert isinstance(narr, NarrativeSignResult)
    assert isinstance(narr, NarrativeSignSVARResult)
    assert not narr.bayes_draws

    irf_med = narr.irf_median  # (17, 3, 3)
    for h, snapshot in DEMO_SNAPSHOT_IRF.items():
        np.testing.assert_allclose(
            irf_med[h, :, 0], snapshot, rtol=1e-4, atol=1e-5,
            err_msg=f"h={h}: demo IRF drifted from the v2.3.0 snapshot",
        )


def test_adrr2018_demo_narrative_restrictions_tighten_bands(adrr_demo):
    """AD-RR's qualitative headline on the synthetic data: conditioning on the
    Volcker restrictions shrinks the 68% bands of the monetary-shock column
    by at least DEMO_MIN_SHRINKAGE for every variable (and >= 20% on average)."""
    plain = adrr_demo["plain"]
    narr = adrr_demo["narr"]

    shrinkages = []
    for i, name in enumerate(VARIABLES):
        wp = float((plain.irf_upper - plain.irf_lower)[:, i, 0].mean())
        wn = float((narr.irf_upper - narr.irf_lower)[:, i, 0].mean())
        shrink = 1.0 - wn / wp
        shrinkages.append(shrink)
        assert shrink >= DEMO_MIN_SHRINKAGE[name], (
            f"Variable {name}: band shrinkage {shrink:.1%} below the "
            f"synthetic-DGP floor {DEMO_MIN_SHRINKAGE[name]:.1%}"
        )
    assert float(np.mean(shrinkages)) >= 0.20


def test_adrr2018_acceptance_diagnostics(adrr_demo):
    """Acceptance diagnostics properties on NarrativeSignResult."""
    narr = adrr_demo["narr"]
    plain = adrr_demo["plain"]

    assert 0.0 < narr.acceptance_rate <= 1.0
    assert 0.0 < narr.traditional_acceptance_rate <= 1.0
    assert 0.0 < narr.narrative_acceptance_rate <= 1.0

    expected_acc = narr.traditional_acceptance_rate * narr.narrative_acceptance_rate
    assert abs(narr.acceptance_rate - expected_acc) < 1e-6

    assert narr.effective_draws == narr.ess
    assert narr.effective_draws > 50.0
    assert narr.effective_draws <= narr.n_narrative_accepted
    assert narr.n_weight_floor == 0

    # Invariance of the Haar stream for an identical seed
    assert narr.n_traditional_accepted == plain.n_traditional_accepted
    assert narr.traditional_acceptance_rate == plain.acceptance_rate


def test_adrr2018_standard_svar_capabilities(adrr_demo):
    """.irf, .fevd and .historical_decomposition on the demo result."""
    narr = adrr_demo["narr"]
    Y, _, _ = _simulate()

    # 1. .irf(): slices below H, weighted-median extension above H
    irf_full = narr.irf()
    assert irf_full.shape == (17, 3, 3)
    np.testing.assert_array_equal(irf_full, narr.irf_median)
    irf_h4 = narr.irf(horizon=4)
    assert irf_h4.shape == (5, 3, 3)
    np.testing.assert_array_equal(irf_h4, narr.irf_median[:5])
    irf_h20 = narr.irf(horizon=20)
    assert irf_h20.shape == (21, 3, 3)
    np.testing.assert_allclose(irf_h20[:17], narr.irf_median, atol=1e-12)

    # 2. .fevd(): rows sum to 1, extension consistent with fevd_median
    fevd_full = narr.fevd()
    assert fevd_full.shape == (17, 3, 3)
    np.testing.assert_allclose(fevd_full.sum(axis=2), 1.0, atol=1e-6)
    fevd_h4 = narr.fevd(horizon=4)
    assert fevd_h4.shape == (5, 3, 3)
    fevd_h20 = narr.fevd(horizon=20)
    np.testing.assert_allclose(fevd_h20[:17], narr.fevd_median, atol=1e-12)

    # 3. .historical_decomposition(): exact identity with the default init_y
    hd_dict = narr.historical_decomposition()
    assert isinstance(hd_dict, dict)
    shocks = hd_dict["shocks"]
    det = hd_dict["deterministic"]
    assert shocks.shape[1:] == (3, 3)
    assert det.shape[1] == 3
    assert shocks.shape[0] == det.shape[0]
    p = len(narr.A_list)
    T_eff = narr.residuals.shape[0]
    reconstructed = det + shocks.sum(axis=2)
    np.testing.assert_allclose(reconstructed, Y[p:], atol=1e-9)

    hd_ffr = narr.historical_decomposition(variable=0)
    assert isinstance(hd_ffr, pd.DataFrame)
    assert set(hd_ffr.columns) == {"shock_0", "shock_1", "shock_2", "deterministic"}
    assert len(hd_ffr) == T_eff

    hd_s0 = narr.historical_decomposition(shock=0)
    assert isinstance(hd_s0, pd.DataFrame)
    assert len(hd_s0.columns) == 3
    assert len(hd_s0) == T_eff


def test_adrr2018_presentation_contract(adrr_demo):
    """.summary, .plot (single and multi-panel), .to_frame, .to_markdown,
    .to_latex, .to_typst."""
    narr = adrr_demo["narr"]

    s = narr.summary()
    assert "Narrative-sign SVAR result (AD-RR 2018)" in s
    assert "traditional accept" in s
    assert "narrative accept" in s
    assert "weight ESS" in s
    assert "reduced form      : OLS point estimate" in s

    fig, ax = plt.subplots(figsize=(6, 4))
    res_ax = narr.plot(target_idx=0, shock_idx=0, ax=ax)
    assert res_ax is not None
    plt.close(fig)

    fig_multi = narr.plot(shock_idx=0, target_idx=None)
    assert len(fig_multi.axes) == 3
    plt.close(fig_multi)

    df = narr.to_frame(target_idx=0, shock_idx=0)
    assert isinstance(df, pd.DataFrame)
    assert {"h", "point", "lower", "upper"} <= set(df.columns)
    assert len(df) == 17

    md = narr.to_markdown(target_idx=0, shock_idx=0)
    assert isinstance(md, str)
    assert "|  h |" in md or "| h |" in md

    latex = narr.to_latex(target_idx=0, shock_idx=0)
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    typst = narr.to_typst(target_idx=0, shock_idx=0)
    assert isinstance(typst, str)
    assert "#table" in typst


def test_identify_narrative_sign_api_aliases():
    """identify_narrative_sign with positional restrictions, the `horizons`
    alias and a calendar restriction date resolved through `dates`."""
    Y, dates, _ = _simulate(T=100, seed=1)
    S = np.zeros((3, 3))
    S[:, 0] = [+1, -1, -1]

    restr = [(VOLCKER_DATE, 0, +1)]
    res = identify_narrative_sign(
        Y, restr, p=1, horizons=4, sign_matrix={0: S}, dates=dates, n_draws=300, seed=0
    )
    assert isinstance(res, NarrativeSignResult)
    assert isinstance(res, NarrativeSignSVARResult)
    assert res.acceptance_rate > 0.0
    assert res.traditional_acceptance_rate > 0.0
    assert res.narrative_acceptance_rate > 0.0
    assert res.effective_draws > 0.0
    assert res.irf().shape == (5, 3, 3)
    assert res.fevd().shape == (5, 3, 3)
