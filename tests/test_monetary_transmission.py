"""Comprehensive unit and integration tests for MonetaryTransmissionSimulator.

Verifies:
1. Side-by-side HANK vs RANK transmission under conventional interest rate shocks.
2. Kaplan-Moll-Violante (2018) direct vs indirect transmission decomposition:
   dC_total = dC_direct + dC_indirect to machine precision (< 10^-14).
3. In RANK, indirect channel is identically 0.0%; in HANK, indirect channel is positive.
4. Heterogeneous MPC ladder across 10 wealth deciles in HANK vs flat MPC in RANK.
5. Balance-sheet / targeted transfer interventions showing HANK amplification over RANK.
6. Flexible shock sequence support via simulate_transmission.
7. Presentation export methods (.summary, .to_frame, .to_markdown, .to_latex, .to_typst, .plot).
"""
from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from puremacro.models.monetary_transmission import (
    MonetaryTransmissionResult,
    MonetaryTransmissionSimulator,
)


@pytest.fixture(scope="module")
def simulator() -> MonetaryTransmissionSimulator:
    """Instantiate a cached MonetaryTransmissionSimulator."""
    return MonetaryTransmissionSimulator(
        beta=0.985,
        gamma=1.0,
        r_ss=0.01,
        phi_pi=1.5,
        kappa=0.1,
        n_a=50,
        a_max=30.0,
    )


def test_rate_shock_contraction(simulator: MonetaryTransmissionSimulator) -> None:
    """Monetary policy rate hike causes output and inflation contraction in both HANK and RANK."""
    res = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

    assert isinstance(res, MonetaryTransmissionResult)
    assert res.horizon == 40
    assert res.shock_type == "rate"

    # Both models show contraction on impact
    assert res.irf_output_hank[0] < 0.0
    assert res.irf_output_rank[0] < 0.0
    assert res.irf_inflation_hank[0] < 0.0
    assert res.irf_inflation_rank[0] < 0.0

    # Real interest rate increases on impact
    assert res.irf_rate_hank[0] > 0.0
    assert res.irf_rate_rank[0] > 0.0


def test_kmv_decomposition_exact_identity(simulator: MonetaryTransmissionSimulator) -> None:
    """KMV decomposition dC = dC_direct + dC_indirect holds to machine precision in both models."""
    res = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

    # HANK identity
    sum_hank = res.direct_channel_hank + res.indirect_channel_hank
    diff_hank = np.abs(res.irf_consumption_hank - sum_hank)
    np.testing.assert_allclose(diff_hank, 0.0, atol=1e-14)

    # In HANK, indirect channel is positive (meaning it contributes to the drop)
    assert res.indirect_channel_hank[0] < 0.0
    assert res.indirect_share_hank > 5.0  # Significant indirect amplification

    # RANK identity
    sum_rank = res.direct_channel_rank + res.indirect_channel_rank
    diff_rank = np.abs(res.irf_consumption_rank - sum_rank)
    np.testing.assert_allclose(diff_rank, 0.0, atol=1e-14)

    # In RANK, indirect channel is identically zero
    np.testing.assert_allclose(res.indirect_channel_rank, 0.0, atol=1e-14)
    assert res.indirect_share_rank == 0.0


def test_mpc_decile_distribution(simulator: MonetaryTransmissionSimulator) -> None:
    """MPC ladder exhibits steep wealth gradient in HANK and flat line in RANK."""
    res = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

    mpc_h = res.mpc_deciles_hank.values
    mpc_r = res.mpc_deciles_rank.values

    # HANK has 10 deciles
    assert len(mpc_h) == 10
    assert len(mpc_r) == 10

    # Decile 1 (poorest / constrained) has highest MPC in HANK
    assert mpc_h[0] > 0.40
    # Decile 10 (richest) has lowest MPC in HANK
    assert mpc_h[-1] < 0.06

    # Strictly monotonically non-increasing across deciles
    for d in range(9):
        assert mpc_h[d] >= mpc_h[d + 1]

    # In RANK, MPC is uniform across all deciles and equals 1 - beta
    expected_rank_mpc = 1.0 - simulator.beta
    np.testing.assert_allclose(mpc_r, expected_rank_mpc, atol=1e-10)

    # Aggregate MPC in HANK exceeds RANK
    assert res.aggregate_mpc_hank > res.aggregate_mpc_rank


def test_balance_sheet_intervention(simulator: MonetaryTransmissionSimulator) -> None:
    """Targeted transfer to borrowers causes massive consumption response in HANK relative to RANK."""
    res = simulator.simulate_balance_sheet_intervention(amount=1.0, target="borrowers", T=40)

    assert res.shock_type == "balance_sheet"

    # Initial consumption response in HANK dwarfs RANK (> 10x)
    assert res.irf_consumption_hank[0] > res.irf_consumption_rank[0] * 10.0
    assert res.irf_output_hank[0] > 0.20  # Over 20% impact multiplier


def test_simulate_transmission_custom_path(simulator: MonetaryTransmissionSimulator) -> None:
    """simulate_transmission accepts custom exogenous shock sequences."""
    shock_path = [0.005, 0.003, 0.001, 0.0]
    res = simulator.simulate_transmission(shock_type="rate", shock_path=shock_path)

    assert res.horizon == 4
    assert len(res.irf_output_hank) == 4
    assert len(res.irf_output_rank) == 4


def test_presentation_export_methods(simulator: MonetaryTransmissionSimulator) -> None:
    """Presentation methods (.summary, .to_frame, .to_markdown, .to_latex, .to_typst, .plot) work."""
    res = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

    # Summary
    s = res.summary()
    assert isinstance(s, str)
    assert "HANK vs RANK" in s
    assert "Kaplan-Moll-Violante" in s

    # to_frame
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40
    assert "output_hank" in df.columns
    assert "output_rank" in df.columns

    # Markdown
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| quarter |" in md

    # LaTeX
    tex = res.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{table}" in tex or "\\begin{tabular}" in tex

    # Typst
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ

    # Plot
    fig_all = res.plot(kind="all")
    assert isinstance(fig_all, matplotlib.figure.Figure)

    fig_irf = res.plot(kind="irf")
    assert isinstance(fig_irf, matplotlib.figure.Figure)

    fig_mpc = res.plot(kind="mpc")
    assert isinstance(fig_mpc, matplotlib.figure.Figure)

    fig_kmv = res.plot(kind="kmv")
    assert isinstance(fig_kmv, matplotlib.figure.Figure)
