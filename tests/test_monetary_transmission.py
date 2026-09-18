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

    with pytest.raises(ValueError, match="Unknown plot kind"):
        res.plot(kind="invalid_kind")


def test_expansionary_rate_cut(simulator: MonetaryTransmissionSimulator) -> None:
    """Expansionary monetary policy (-25 bps) stimulates output and inflation."""
    res = simulator.simulate_rate_shock(magnitude=-0.0025, rho=0.7, T=40)

    assert res.irf_output_hank[0] > 0.0
    assert res.irf_output_rank[0] > 0.0
    assert res.irf_inflation_hank[0] > 0.0
    assert res.irf_rate_hank[0] < 0.0

    # KMV identity holds
    diff = np.abs(res.irf_consumption_hank - (res.direct_channel_hank + res.indirect_channel_hank))
    np.testing.assert_allclose(diff, 0.0, atol=1e-14)
    assert res.indirect_share_hank > 0.0


def test_zero_rate_shock(simulator: MonetaryTransmissionSimulator) -> None:
    """Zero shock produces zero response with machine-precision identity."""
    res = simulator.simulate_rate_shock(magnitude=0.0, rho=0.7, T=40)

    np.testing.assert_allclose(res.irf_output_hank, 0.0, atol=1e-14)
    np.testing.assert_allclose(res.irf_output_rank, 0.0, atol=1e-14)
    assert res.indirect_share_hank == 0.0


def test_persistence_extremes(simulator: MonetaryTransmissionSimulator) -> None:
    """Test transitory shock (rho=0) and persistent shock (rho=0.9)."""
    res_trans = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.0, T=40)
    assert res_trans.irf_output_hank[0] < 0.0
    # Shock dies down immediately after impact
    assert abs(res_trans.irf_output_hank[-1]) < 1e-4

    res_pers = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.9, T=40)
    assert res_pers.irf_output_hank[0] < 0.0
    assert abs(res_pers.irf_output_hank[10]) > abs(res_trans.irf_output_hank[10])


def test_caching_and_convenience_properties(simulator: MonetaryTransmissionSimulator) -> None:
    """Simulator caches model and exposes steady-state MPC and peak responses."""
    # steady_state_mpc property
    mpc_ss = simulator.steady_state_mpc
    assert 0.01 < mpc_ss < 0.50

    # Running with smaller T uses cached model without re-solving
    res = simulator.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=20)
    assert res.horizon == 20
    assert simulator._cached_hank is not None

    # peak_responses method
    peaks = res.peak_responses()
    assert isinstance(peaks, dict)
    assert "output_hank" in peaks
    assert "inflation_hank" in peaks
    assert "consumption_hank" in peaks
    assert peaks["output_hank"] < 0.0

    # repr
    rep = repr(res)
    assert "MonetaryTransmissionResult" in rep
    assert "rate" in rep
    assert "indirect_share_hank" in rep


def test_balance_sheet_different_targets(simulator: MonetaryTransmissionSimulator) -> None:
    """Borrowers target has much higher initial consumption than unconstrained target."""
    res_borrowers = simulator.simulate_balance_sheet_intervention(amount=1.0, target="borrowers", T=30)
    res_unconstr = simulator.simulate_balance_sheet_intervention(amount=1.0, target="unconstrained", T=30)
    res_all = simulator.simulate_balance_sheet_intervention(amount=1.0, target="all", T=30)

    assert res_borrowers.irf_consumption_hank[0] > res_unconstr.irf_consumption_hank[0]
    assert res_borrowers.irf_consumption_hank[0] > res_all.irf_consumption_hank[0]


def test_monetary_validation_errors(simulator: MonetaryTransmissionSimulator) -> None:
    """Validation properly raises errors on illegal inputs."""
    # Rate shock validation
    with pytest.raises(ValueError, match="must be at least 2 quarters"):
        simulator.simulate_rate_shock(T=1)

    with pytest.raises(ValueError, match="magnitude must be a finite float"):
        simulator.simulate_rate_shock(magnitude=float("inf"))

    with pytest.raises(ValueError, match="0.0 <= rho < 1.0"):
        simulator.simulate_rate_shock(rho=1.0)

    with pytest.raises(ValueError, match="0.0 <= rho < 1.0"):
        simulator.simulate_rate_shock(rho=-0.5)

    # Balance sheet validation
    with pytest.raises(ValueError, match="must be at least 2 quarters"):
        simulator.simulate_balance_sheet_intervention(T=1)

    with pytest.raises(ValueError, match="positive finite float"):
        simulator.simulate_balance_sheet_intervention(amount=-1.0)

    with pytest.raises(ValueError, match="positive finite float"):
        simulator.simulate_balance_sheet_intervention(amount=0.0)

    with pytest.raises(ValueError, match="Unknown target group"):
        simulator.simulate_balance_sheet_intervention(target="nonexistent_group")

    # Custom shock path validation
    with pytest.raises(ValueError, match="must have length >= 2"):
        simulator.simulate_transmission(shock_type="rate", shock_path=[0.01])

    with pytest.raises(ValueError, match="contains NaN or infinite"):
        simulator.simulate_transmission(shock_type="rate", shock_path=[0.01, float("nan")])

    with pytest.raises(ValueError, match="Unknown shock_type"):
        simulator.simulate_transmission(shock_type="foreign_exchange")
