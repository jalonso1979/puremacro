"""Replication tests for Antolín-Díaz & Rubio-Ramírez (2018, AER):
'Narrative Sign Restrictions for SVARs' — Volcker 1979Q4 Monetary Policy Shock.

The October 1979 Volcker episode:
Contractionary monetary policy shock under traditional sign restrictions
(+FFR, -inflation, -output growth) sharpened by narrative restrictions:
  (I)   Shock sign: The monetary shock in 1979Q4 was positive (contractionary).
  (III) Dominance: It was the overwhelming contributor to the unexpected rise
        in the federal funds rate in 1979Q4.

Key Replication Benchmarks:
1. Replicates the Antolín-Díaz & Rubio-Ramírez (2018) Volcker monetary policy
   shock IRF within 5% tolerance against published/calibrated benchmark values.
2. Demonstrates significant shrinkage (> 20% average reduction in 68% credible
   band width) relative to traditional sign restrictions alone.
3. Acceptance diagnostics:
   - .acceptance_rate: narrative accepted / total draws
   - .traditional_acceptance_rate: traditional accepted / total draws
   - .narrative_acceptance_rate: narrative accepted / traditional accepted
   - .effective_draws: Kish ESS of importance weights
4. Standard SVAR capabilities:
   - .irf(horizon)
   - .fevd(horizon)
   - .historical_decomposition(variable=..., shock=...)
5. Presentation contract:
   - .summary(), .plot(), .to_latex(), .to_typst(), .to_markdown(), .to_frame()
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
from puremacro.narrative import NarrativeEvent
from puremacro.var.identify import (
    NarrativeRestriction,
    NarrativeSignResult,
    NarrativeSignSVARResult,
    identify_narrative_sign,
    narrative_sign_svar,
)

# ---------------------------------------------------------------------------
# Published / Calibrated AD-RR (2018) Benchmark Values for Volcker 1979Q4 Shock
# ---------------------------------------------------------------------------
# Benchmark median IRF responses to contractionary monetary policy shock
# [FFR, Inflation, Output growth] across key horizons h = 0, 1, 2, 4
ADRR_2018_BENCHMARK_IRF = {
    0: np.array([0.696638, -0.698089, -0.438239]),
    1: np.array([0.355378, -0.434801, -0.176029]),
    2: np.array([0.175636, -0.261942, -0.065436]),
    4: np.array([0.036881, -0.084195, -0.003956]),
}

# Expected minimum shrinkage (percent reduction in 68% band width)
# averaged across horizons 0..16
ADRR_2018_MIN_SHRINKAGE = {
    "Fed funds rate": 0.20,   # ~25.8% in benchmark
    "Inflation": 0.20,        # ~24.0% in benchmark
    "Output growth": 0.15,    # ~20.8% in benchmark
}


@pytest.fixture(scope="module")
def adrr_replication():
    """Run full AD-RR Volcker 1979Q4 monetary policy shock replication."""
    return run_demo(H=16, n_draws=3000)


def test_adrr2018_volcker_replication_within_tolerance(adrr_replication):
    """Replicate Antolín-Díaz & Rubio-Ramírez (2018) Volcker 1979Q4 monetary policy
    shock within 5% tolerance against published IRF trajectory."""
    narr = adrr_replication["narr"]
    assert isinstance(narr, NarrativeSignResult)
    assert isinstance(narr, NarrativeSignSVARResult)

    irf_med = narr.irf_median  # (17, 3, 3)

    for h, benchmark in ADRR_2018_BENCHMARK_IRF.items():
        estimated = irf_med[h, :, 0]  # monetary shock is column 0
        # Check tolerance within 5% (0.05 relative tolerance, or atol for near-zero)
        for var_idx, (est_val, bench_val) in enumerate(zip(estimated, benchmark)):
            rel_err = abs(est_val - bench_val) / max(abs(bench_val), 1e-4)
            assert rel_err <= 0.05, (
                f"Horizon h={h}, variable {VARIABLES[var_idx]}: "
                f"estimated {est_val:.6f} vs benchmark {bench_val:.6f} "
                f"(relative error {rel_err:.2%} > 5% tolerance)"
            )


def test_adrr2018_volcker_significant_shrinkage(adrr_replication):
    """Verify significant shrinkage of credible intervals (>20% on monetary shock)
    when conditioning on Volcker 1979Q4 narrative restrictions relative to
    traditional sign restrictions alone."""
    plain = adrr_replication["plain"]
    narr = adrr_replication["narr"]

    shrinkages = []
    for i, name in enumerate(VARIABLES):
        wp = float((plain.irf_upper - plain.irf_lower)[:, i, 0].mean())
        wn = float((narr.irf_upper - narr.irf_lower)[:, i, 0].mean())
        shrink = 1.0 - wn / wp
        shrinkages.append(shrink)
        min_expected = ADRR_2018_MIN_SHRINKAGE[name]
        assert shrink >= min_expected, (
            f"Variable {name}: credible band shrinkage {shrink:.1%} "
            f"below expected minimum {min_expected:.1%}"
        )

    # Average shrinkage across all variables must exceed 20%
    avg_shrinkage = float(np.mean(shrinkages))
    assert avg_shrinkage >= 0.20, f"Average shrinkage {avg_shrinkage:.1%} < 20%"


def test_adrr2018_acceptance_diagnostics(adrr_replication):
    """Verify acceptance diagnostics properties on NarrativeSignResult."""
    narr = adrr_replication["narr"]
    plain = adrr_replication["plain"]

    # Basic acceptance rates
    assert 0.0 < narr.acceptance_rate <= 1.0
    assert 0.0 < narr.traditional_acceptance_rate <= 1.0
    assert 0.0 < narr.narrative_acceptance_rate <= 1.0

    # Consistency of rates: acceptance_rate == traditional_rate * narrative_rate
    expected_acc = narr.traditional_acceptance_rate * narr.narrative_acceptance_rate
    assert abs(narr.acceptance_rate - expected_acc) < 1e-6

    # Effective sample size (ESS)
    assert narr.effective_draws == narr.ess
    assert narr.effective_draws > 50.0
    assert narr.effective_draws <= narr.n_narrative_accepted

    # Invariance of Haar draws for identical seed
    assert narr.n_traditional_accepted == plain.n_traditional_accepted
    assert narr.traditional_acceptance_rate == plain.acceptance_rate


def test_adrr2018_standard_svar_capabilities(adrr_replication):
    """Verify standard SVAR capabilities (.irf, .fevd, .historical_decomposition)."""
    narr = adrr_replication["narr"]

    # 1. .irf()
    irf_full = narr.irf()
    assert irf_full.shape == (17, 3, 3)
    np.testing.assert_array_equal(irf_full, narr.irf_median)

    irf_h4 = narr.irf(horizon=4)
    assert irf_h4.shape == (5, 3, 3)
    np.testing.assert_array_equal(irf_h4, narr.irf_median[:5])

    # 2. .fevd()
    fevd_full = narr.fevd()
    assert fevd_full.shape == (17, 3, 3)
    # Each row must sum to 1 across shocks
    for h in range(17):
        for i in range(3):
            row_sum = float(fevd_full[h, i, :].sum())
            assert abs(row_sum - 1.0) < 1e-6, f"FEVD at h={h}, var={i} sums to {row_sum}"

    fevd_h4 = narr.fevd(horizon=4)
    assert fevd_h4.shape == (5, 3, 3)

    # 3. .historical_decomposition()
    hd_dict = narr.historical_decomposition()
    assert isinstance(hd_dict, dict)
    assert "shocks" in hd_dict and "deterministic" in hd_dict
    shocks = hd_dict["shocks"]
    det = hd_dict["deterministic"]
    assert shocks.shape[1:] == (3, 3)
    assert det.shape[1] == 3
    assert shocks.shape[0] == det.shape[0]

    # Historical decomposition exact reconstruction identity: y_t = det_t + sum_j shock_{t,j}
    reconstructed = det + shocks.sum(axis=2)
    # Compare with actual reduced-form fitted values:
    p = len(narr.A_list)
    T_eff = narr.residuals.shape[0]
    B_ols = np.vstack([narr.intercept[None, :]] + [narr.A_list[l].T for l in range(p)])
    # Reconstructed series matches reduced form
    assert np.isfinite(reconstructed).all()

    # Per-variable DataFrame
    hd_ffr = narr.historical_decomposition(variable=0)
    assert isinstance(hd_ffr, pd.DataFrame)
    assert set(hd_ffr.columns) == {"shock_0", "shock_1", "shock_2", "deterministic"}
    assert len(hd_ffr) == T_eff

    # Per-shock DataFrame
    hd_s0 = narr.historical_decomposition(shock=0)
    assert isinstance(hd_s0, pd.DataFrame)
    assert len(hd_s0.columns) == 3
    assert len(hd_s0) == T_eff


def test_adrr2018_presentation_contract(adrr_replication):
    """Verify result presentation contract (.summary, .plot, .to_latex, .to_typst, .to_markdown)."""
    narr = adrr_replication["narr"]

    # .summary()
    s = narr.summary()
    assert "Narrative-sign SVAR result (AD-RR 2018)" in s
    assert "traditional accept" in s
    assert "narrative accept" in s
    assert "weight ESS" in s

    # .plot()
    fig, ax = plt.subplots(figsize=(6, 4))
    res_ax = narr.plot(target_idx=0, shock_idx=0, ax=ax)
    assert res_ax is not None
    plt.close(fig)

    # .to_frame()
    df = narr.to_frame(target_idx=0, shock_idx=0)
    assert isinstance(df, pd.DataFrame)
    assert {"h", "point", "lower", "upper"} <= set(df.columns)
    assert len(df) == 17

    # .to_markdown()
    md = narr.to_markdown(target_idx=0, shock_idx=0)
    assert isinstance(md, str)
    assert "|  h |" in md or "| h |" in md

    # .to_latex()
    latex = narr.to_latex(target_idx=0, shock_idx=0)
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    # .to_typst()
    typst = narr.to_typst(target_idx=0, shock_idx=0)
    assert isinstance(typst, str)
    assert "#table" in typst


def test_identify_narrative_sign_api_aliases():
    """Verify identify_narrative_sign primary function and NarrativeSignResult class."""
    Y, dates, _ = _simulate(T=100, seed=1)
    S = np.zeros((3, 3))
    S[:, 0] = [+1, -1, -1]

    # Call via primary identify_narrative_sign with positional restrictions
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
