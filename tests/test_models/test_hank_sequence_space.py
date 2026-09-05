"""Tests for Sequence-Space HANK solver (Auclert et al. 2021)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.hank_sequence_space import (
    SequenceSpaceHANKResult,
    FakeNewsResult,
    FiscalTransferResult,
    solve_hank_sequence_space,
    fake_news_algorithm,
    simulate_targeted_transfer,
)


def test_solve_hank_sequence_space_basic():
    res = solve_hank_sequence_space(T=30, n_a=30)
    
    assert isinstance(res, SequenceSpaceHANKResult)
    assert len(res.irf_output) == 30
    assert len(res.irf_consumption) == 30
    assert len(res.irf_inflation) == 30
    assert len(res.irf_rate) == 30
    assert res.jacobian_c_r.shape == (30, 30)
    assert res.jacobian_c_y.shape == (30, 30)
    
    # Check that MPC is positive and reasonable (e.g. 0.05 to 0.6)
    assert 0.01 <= res.steady_state_mpc <= 0.8
    assert len(res.mpc_distribution) == 10
    
    # Check monetary contraction leads to negative output response initially
    assert res.irf_output[0] < 0.0
    assert res.irf_consumption[0] < 0.0
    
    summary_text = res.summary()
    assert "Sequence-Space HANK" in summary_text
    assert "MPC by Wealth Decile" in summary_text


def test_fake_news_algorithm():
    res = solve_hank_sequence_space(T=25, n_a=25)
    fn = res.fake_news()
    
    assert isinstance(fn, FakeNewsResult)
    assert fn.horizon == 25
    assert fn.jacobian.shape == (25, 25)
    assert fn.fake_news.shape == (25, 25)
    assert fn.expectation_vectors.shape[0] == 25
    
    # Auclert et al. (2021) Proposition 1 fundamental identity:
    # J_{t, s} = J_{t-1, s-1} + F_{t, s}
    for t in range(1, 25):
        for s in range(1, 25):
            expected = fn.jacobian[t - 1, s - 1] + fn.fake_news[t, s]
            assert np.isclose(fn.jacobian[t, s], expected, atol=1e-12)
            
    # Cumulation along diagonals:
    # J_{t, s} = sum_{k=0}^{min(t, s)} F_{t-k, s-k}
    for t in range(25):
        for s in range(25):
            k_max = min(t, s)
            sum_f = sum(fn.fake_news[t - k, s - k] for k in range(k_max + 1))
            assert np.isclose(fn.jacobian[t, s], sum_f, atol=1e-12)
            
    # Reporting & Export formats
    summary = fn.summary()
    assert "Fake News Algorithm Decomposition" in summary
    assert "Jacobian Frobenius Norm" in summary
    
    df_jac = fn.to_frame("jacobian")
    assert df_jac.shape == (25, 25)
    df_fn = fn.to_frame("fake_news")
    assert df_fn.shape == (25, 25)
    
    md = fn.to_markdown()
    assert "|" in md
    tex = fn.to_latex()
    assert "\\begin{tabular}" in tex or "\\toprule" in tex
    typ = fn.to_typst()
    assert "#table" in typ
    
    # Plotting
    fig = fn.plot()
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_simulate_targeted_transfer():
    res = solve_hank_sequence_space(T=30, n_a=30)
    
    # 1. Borrowers / Hand-to-Mouth
    trans_borrowers = res.simulate_transfer(target="borrowers", amount=1.0)
    assert isinstance(trans_borrowers, FiscalTransferResult)
    assert trans_borrowers.target_group == "borrowers"
    assert len(trans_borrowers.irf_consumption) == 30
    assert trans_borrowers.impact_mpc > 0.0
    assert trans_borrowers.cumulative_multiplier > 0.0
    assert len(trans_borrowers.decile_incidence) == 10
    
    # 2. Unconstrained Savers
    trans_unconstrained = res.simulate_transfer(target="unconstrained", amount=1.0)
    assert isinstance(trans_unconstrained, FiscalTransferResult)
    assert trans_unconstrained.target_group == "unconstrained"
    
    # Macroeconomic core property:
    # Transfers targeted to constrained borrowers generate significantly higher
    # immediate MPC and cumulative fiscal multiplier than transfers to unconstrained
    assert trans_borrowers.impact_mpc > 1.5 * trans_unconstrained.impact_mpc
    assert trans_borrowers.cumulative_multiplier > trans_unconstrained.cumulative_multiplier
    
    # 3. Universal lump-sum
    trans_all = res.simulate_transfer(target="all", amount=1.0)
    assert isinstance(trans_all, FiscalTransferResult)
    assert trans_unconstrained.impact_mpc <= trans_all.impact_mpc <= trans_borrowers.impact_mpc
    
    # Export formats
    summary = trans_borrowers.summary()
    assert "Targeted Fiscal Transfer Simulation" in summary
    assert "Cumulative Fiscal Multiplier" in summary
    
    df = trans_borrowers.to_frame()
    assert df.shape == (10, 3)  # Transfer, Consumption, Decile_MPC
    
    md = trans_borrowers.to_markdown()
    assert "|" in md
    tex = trans_borrowers.to_latex()
    assert "\\begin{tabular}" in tex or "\\toprule" in tex
    typ = trans_borrowers.to_typst()
    assert "#table" in typ
    
    # Plotting
    fig = trans_borrowers.plot()
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_simulate_targeted_transfer_invalid_target():
    res = solve_hank_sequence_space(T=10, n_a=15)
    with pytest.raises(ValueError, match="Unknown target group"):
        res.simulate_transfer(target="invalid_target_category")
