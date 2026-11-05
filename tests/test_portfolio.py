"""Comprehensive test suite for ``puremacro.portfolio`` (Track 6 / 4.0)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.portfolio import (
    QUADRANTS,
    AllocationResult,
    FactorAttributionResult,
    MacroRegimePortfolioResult,
    PortfolioBacktestResult,
    WealthSimulationResult,
    all_weather_portfolio,
    backtest_portfolio,
    black_litterman,
    classify_macro_quadrants,
    factor_attribution,
    hierarchical_risk_parity,
    markowitz_frontier,
    optimize_portfolio,
    shrink_covariance,
    simulate_wealth_glidepath,
)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Generate realistic multi-asset monthly returns DataFrame (T=120, N=4)."""
    rng = np.random.default_rng(2026)
    T = 120
    mu_monthly = np.array([0.008, 0.004, 0.006, 0.003])
    cov_monthly = np.array(
        [
            [0.0020, -0.0003, 0.0008, 0.0002],
            [-0.0003, 0.0005, -0.0001, 0.0001],
            [0.0008, -0.0001, 0.0016, 0.0003],
            [0.0002, 0.0001, 0.0003, 0.0009],
        ]
    )
    L = np.linalg.cholesky(cov_monthly)
    draws = rng.standard_normal((T, 4)) @ L.T + mu_monthly
    dates = pd.date_range("2015-01-31", periods=T, freq="ME")
    return pd.DataFrame(draws, index=dates, columns=["Equities", "Bonds", "Commodities", "Gold"])


# ---------------------------------------------------------------------------
# 1. Covariance Shrinkage Tests
# ---------------------------------------------------------------------------


def test_shrink_covariance_high_dim_psd_and_condition_number() -> None:
    """Ledoit-Wolf and RMT must produce strictly PD matrices when N > T."""
    rng = np.random.default_rng(42)
    # T=18 observations, N=30 assets -> sample covariance is rank-deficient
    X = rng.standard_normal((18, 30)) * 0.03
    X_c = X - X.mean(axis=0, keepdims=True)
    raw_sample = (X_c.T @ X_c) / 17.0
    raw_eigs = np.linalg.eigvalsh(raw_sample)
    assert np.min(raw_eigs) < 1e-10  # Singular when N > T

    cov_lw = shrink_covariance(X, method="ledoit_wolf")
    eigs_lw = np.linalg.eigvalsh(cov_lw)
    assert cov_lw.shape == (30, 30)
    assert np.allclose(cov_lw, cov_lw.T)
    assert np.min(eigs_lw) > 1e-6
    cond_lw = float(np.max(eigs_lw) / np.min(eigs_lw))
    assert cond_lw < 500.0

    cov_rmt = shrink_covariance(X, method="rmt")
    eigs_rmt = np.linalg.eigvalsh(cov_rmt)
    assert cov_rmt.shape == (30, 30)
    assert np.allclose(cov_rmt, cov_rmt.T)
    assert np.min(eigs_rmt) > 1e-6
    # Trace and diagonal variances are preserved under RMT
    assert np.isclose(np.trace(cov_rmt), np.trace(raw_sample), rtol=1e-5)
    assert np.allclose(np.diag(cov_rmt), np.diag(raw_sample), rtol=1e-5)


def test_shrink_covariance_sample_and_validation(synthetic_returns: pd.DataFrame) -> None:
    """Sample covariance matches pandas cov and invalid inputs raise ValueError."""
    cov_s = shrink_covariance(synthetic_returns, method="sample")
    expected = synthetic_returns.cov().to_numpy()
    assert np.allclose(cov_s, expected, atol=1e-12)

    with pytest.raises(ValueError, match="Unknown covariance shrinkage"):
        shrink_covariance(synthetic_returns, method="invalid_method")

    with pytest.raises(ValueError, match="2D array"):
        shrink_covariance(np.ones(10))


# ---------------------------------------------------------------------------
# 2. Portfolio Optimization Tests (All 8 Methods)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "max_sharpe",
        "min_variance",
        "target_return",
        "risk_parity",
        "hrp",
        "min_cvar",
        "kelly",
        "equal_weight",
    ],
)
def test_optimize_portfolio_all_methods_invariants(
    synthetic_returns: pd.DataFrame, method: str
) -> None:
    """Every optimization method returns valid weights, risk contributions, and metrics."""
    res = optimize_portfolio(synthetic_returns, method=method, rf=0.02)
    assert isinstance(res, AllocationResult)
    assert res.method == method
    assert res.asset_names == ("Equities", "Bonds", "Commodities", "Gold")
    assert res.weights.shape == (4,)
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-8)
    assert np.all(res.weights >= -1e-9)
    assert np.all(res.weights <= 1.0 + 1e-9)
    assert np.isclose(np.sum(res.risk_contributions), 1.0, atol=1e-6)
    assert res.volatility > 0.0
    assert res.diversification_ratio >= 1.0 - 1e-6
    assert np.isclose(res.effective_n_assets, 1.0 / np.sum(res.weights ** 2), atol=1e-8)


def test_optimize_portfolio_risk_parity_exact_erc(synthetic_returns: pd.DataFrame) -> None:
    """Equal Risk Contribution must equalize risk_contributions to 1/N within < 1e-6."""
    res = optimize_portfolio(synthetic_returns, method="risk_parity")
    n_assets = len(res.asset_names)
    target_rc = np.full(n_assets, 1.0 / n_assets)
    max_err = float(np.max(np.abs(res.risk_contributions - target_rc)))
    assert max_err < 1e-6


def test_optimize_portfolio_dominance_and_target_return(
    synthetic_returns: pd.DataFrame,
) -> None:
    """min_variance minimizes volatility; max_sharpe maximizes Sharpe; target_return hits target."""
    res_mv = optimize_portfolio(synthetic_returns, method="min_variance", rf=0.01)
    res_ew = optimize_portfolio(synthetic_returns, method="equal_weight", rf=0.01)
    res_ms = optimize_portfolio(synthetic_returns, method="max_sharpe", rf=0.01)

    assert res_mv.volatility <= res_ew.volatility + 1e-9
    assert res_mv.volatility <= res_ms.volatility + 1e-9
    assert res_ms.sharpe_ratio >= res_ew.sharpe_ratio - 1e-7
    assert res_ms.sharpe_ratio >= res_mv.sharpe_ratio - 1e-7

    target = 0.5 * (res_mv.expected_return + res_ms.expected_return)
    res_tr = optimize_portfolio(
        synthetic_returns, method="target_return", target_return=target
    )
    assert np.isclose(res_tr.expected_return, target, atol=1e-5)


def test_optimize_portfolio_bounds_and_short_selling(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Box constraints (min_weight, max_weight) and long_only=False work properly."""
    res_bounded = optimize_portfolio(
        synthetic_returns,
        method="max_sharpe",
        min_weight=0.10,
        max_weight=0.40,
    )
    assert np.all(res_bounded.weights >= 0.10 - 1e-6)
    assert np.all(res_bounded.weights <= 0.40 + 1e-6)
    assert np.isclose(np.sum(res_bounded.weights), 1.0, atol=1e-8)

    res_short = optimize_portfolio(
        synthetic_returns,
        method="min_variance",
        long_only=False,
        max_weight=1.5,
    )
    assert np.isclose(np.sum(res_short.weights), 1.0, atol=1e-8)

    with pytest.raises(ValueError, match="Infeasible weight bounds"):
        optimize_portfolio(synthetic_returns, max_weight=0.15)


def test_optimize_portfolio_1d_mu_and_cov_inputs() -> None:
    """1D expected returns vector with explicit covariance matrix works across methods."""
    mu = np.array([0.10, 0.04, 0.07])
    cov = np.diag([0.04, 0.005, 0.02])
    for m in ("max_sharpe", "min_variance", "risk_parity", "min_cvar", "hrp", "kelly"):
        res = optimize_portfolio(mu, cov, method=m, asset_names=["A", "B", "C"])
        assert np.isclose(np.sum(res.weights), 1.0, atol=1e-8)
        assert res.asset_names == ("A", "B", "C")


# ---------------------------------------------------------------------------
# 3. Hierarchical Risk Parity & Markowitz Frontier Tests
# ---------------------------------------------------------------------------


def test_hierarchical_risk_parity_diagonal_and_clustered() -> None:
    """HRP reproduces inverse-variance weights on diagonal cov and splits correlated clusters."""
    # Diagonal covariance with variances [0.04, 0.01] -> IVP weights [0.20, 0.80]
    cov_diag = np.diag([0.04, 0.01])
    res_diag = hierarchical_risk_parity(cov_diag, is_cov=True, asset_names=["HighVol", "LowVol"])
    assert np.allclose(res_diag.weights, [0.20, 0.80], atol=1e-6)

    # 3 assets: two nearly identical assets (0, 1) and one uncorrelated asset (2) of equal variance
    cov_block = np.array(
        [
            [0.04, 0.039, 0.0],
            [0.039, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ]
    )
    res_block = hierarchical_risk_parity(
        cov_block, is_cov=True, linkage_method="single", asset_names=["A1", "A2", "B"]
    )
    assert np.isclose(np.sum(res_block.weights), 1.0, atol=1e-8)
    # Uncorrelated asset B receives ~50% while redundant cluster (A1, A2) splits the other ~50%
    assert res_block.weights[2] > res_block.weights[0]
    assert np.isclose(res_block.weights[0], res_block.weights[1], atol=1e-6)


def test_markowitz_frontier_monotonicity(synthetic_returns: pd.DataFrame) -> None:
    """Markowitz frontier has monotonically non-decreasing returns and volatilities."""
    res = markowitz_frontier(synthetic_returns, n_points=20, rf=0.01)
    assert res.frontier_returns is not None
    assert res.frontier_vols is not None
    assert res.frontier_weights is not None
    assert res.frontier_returns.shape == (20,)
    assert res.frontier_vols.shape == (20,)
    assert res.frontier_weights.shape == (20, 4)

    # Returns and volatilities increase monotonically along the upper efficient frontier
    assert np.all(np.diff(res.frontier_returns) >= -1e-9)
    assert np.all(np.diff(res.frontier_vols) >= -1e-6)
    assert np.allclose(np.sum(res.frontier_weights, axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Black-Litterman Tests
# ---------------------------------------------------------------------------


def test_black_litterman_recovers_market_weights_and_tilts_monotonically(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Black-Litterman recovers market weights with zero/no views and tilts monotonically with confidence."""
    cov = shrink_covariance(synthetic_returns, method="ledoit_wolf") * 12.0
    w_mkt = np.array([0.45, 0.30, 0.15, 0.10])

    # 1. Empty views recover market weights
    res_empty = black_litterman(
        cov,
        w_mkt,
        P=np.empty((0, 4)),
        Q=np.array([]),
        asset_names=synthetic_returns.columns,
    )
    assert np.allclose(res_empty.weights, w_mkt, atol=1e-6)

    # 2. Zero confidence views recover market weights
    P_view = np.array([[1.0, 0.0, 0.0, 0.0]])  # Bullish view on Equities (asset 0)
    Q_view = np.array([0.045])
    res_c0 = black_litterman(
        cov,
        w_mkt,
        P=P_view,
        Q=Q_view,
        confidences=[0.0],
        asset_names=synthetic_returns.columns,
    )
    assert np.allclose(res_c0.weights, w_mkt, atol=1e-6)

    # 3. Monotonic tilt toward Equities as confidence increases
    res_c25 = black_litterman(cov, w_mkt, P=P_view, Q=Q_view, confidences=[0.25])
    res_c60 = black_litterman(cov, w_mkt, P=P_view, Q=Q_view, confidences=[0.60])
    res_c95 = black_litterman(cov, w_mkt, P=P_view, Q=Q_view, confidences=[0.95])

    assert res_c0.weights[0] < res_c25.weights[0] < res_c60.weights[0] < res_c95.weights[0]

    # 4. Default He-Litterman (confidences=None) also tilts toward bullish view
    res_hl = black_litterman(cov, w_mkt, P=P_view, Q=Q_view)
    assert res_hl.weights[0] > w_mkt[0]


# ---------------------------------------------------------------------------
# 5. Macro Regimes, All-Weather, and Factor Attribution Tests
# ---------------------------------------------------------------------------


def test_classify_macro_quadrants_exact() -> None:
    """Verify exact classification of all four growth-inflation quadrants."""
    # Using 'demean' with zero-mean symmetric levels:
    # t=0: g=+1, pi=-1 -> goldilocks
    # t=1: g=+1, pi=+1 -> reflation
    # t=2: g=-1, pi=+1 -> stagflation
    # t=3: g=-1, pi=-1 -> deflationary_bust
    growth = pd.Series([1.0, 1.0, -1.0, -1.0])
    inflation = pd.Series([-1.0, 1.0, 1.0, -1.0])

    labels_demean = classify_macro_quadrants(growth, inflation, method="demean")
    assert tuple(labels_demean) == QUADRANTS

    labels_zscore = classify_macro_quadrants(growth, inflation, method="zscore")
    assert tuple(labels_zscore) == QUADRANTS

    labels_hp = classify_macro_quadrants(growth, inflation, method="hp")
    assert len(labels_hp) == 4
    assert set(labels_hp).issubset(set(QUADRANTS))


def test_all_weather_portfolio_quadrants(synthetic_returns: pd.DataFrame) -> None:
    """All-Weather portfolio constructs regime-conditional allocations across all 4 quadrants."""
    rng = np.random.default_rng(99)
    T = len(synthetic_returns)
    growth = pd.Series(np.cumsum(rng.normal(0.0, 0.5, size=T)), index=synthetic_returns.index)
    inflation = pd.Series(np.cumsum(rng.normal(0.0, 0.3, size=T)), index=synthetic_returns.index)

    res = all_weather_portfolio(synthetic_returns, growth, inflation, classify_method="diff")
    assert isinstance(res, MacroRegimePortfolioResult)
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-8)
    assert np.all(res.weights > 0.0)
    assert set(res.quadrant_weights.keys()) == set(QUADRANTS)
    for q in QUADRANTS:
        assert np.isclose(np.sum(res.quadrant_weights[q]), 1.0, atol=1e-8)
    assert np.isclose(sum(res.regime_shares.values()), 1.0, atol=1e-8)
    assert list(res.quadrant_returns.index) == list(QUADRANTS)
    assert list(res.quadrant_vols.index) == list(QUADRANTS)

    # Custom quadrant_probs putting 100% on goldilocks recovers goldilocks sub-portfolio
    res_gold = all_weather_portfolio(
        synthetic_returns,
        growth,
        inflation,
        quadrant_probs={
            "goldilocks": 1.0,
            "reflation": 0.0,
            "stagflation": 0.0,
            "deflationary_bust": 0.0,
        },
    )
    assert np.allclose(res_gold.weights, res_gold.quadrant_weights["goldilocks"], atol=1e-8)


def test_factor_attribution_exact_recovery() -> None:
    """Factor attribution recovers true annualized alpha, factor betas, and attribution shares."""
    rng = np.random.default_rng(1234)
    T = 180
    periods_per_year = 12
    true_alpha_ann = 0.036  # 3.6% annualized alpha (0.3% per month)
    true_betas = np.array([1.15, -0.45, 0.30])
    factors = pd.DataFrame(
        rng.normal(0.004, 0.03, size=(T, 3)),
        columns=["MKT", "SMB", "HML"],
    )
    noise = rng.normal(0.0, 1e-5, size=T)
    port_ret = (true_alpha_ann / periods_per_year) + factors.to_numpy() @ true_betas + noise

    res = factor_attribution(
        port_ret,
        factors,
        rf=0.0,
        hac_lags=4,
        rolling_window=36,
        periods_per_year=periods_per_year,
    )
    assert isinstance(res, FactorAttributionResult)
    assert np.isclose(res.alpha, true_alpha_ann, atol=1e-3)
    assert np.allclose(res.betas, true_betas, atol=1e-3)
    assert res.r_squared > 0.999
    assert res.information_ratio > 5.0
    assert res.rolling_betas is not None
    assert res.rolling_betas.shape == (T - 36 + 1, 4)

    # Sum of annualized return contributions equals annualized mean portfolio return
    total_attrib = sum(res.return_contributions.values())
    ann_mean_ret = float(np.mean(port_ret)) * periods_per_year
    assert np.isclose(total_attrib, ann_mean_ret, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Backtesting & Wealth Glidepath Simulation Tests
# ---------------------------------------------------------------------------


def test_backtest_portfolio_drift_and_transaction_costs(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Verify exact weight drift in buy-and-hold and transaction cost drag under rebalancing."""
    w_target = {"Equities": 0.50, "Bonds": 0.30, "Commodities": 0.10, "Gold": 0.10}

    # Buy-and-hold has zero turnover and zero transaction costs
    bt_bh = backtest_portfolio(
        synthetic_returns, w_target, rebalance="buy_and_hold", tc_bps=25.0
    )
    assert bt_bh.total_tc_paid == 0.0
    assert np.all(bt_bh.turnover_series == 0.0)
    # Weights drift away from initial target
    assert not np.allclose(bt_bh.weight_history[-1], [0.50, 0.30, 0.10, 0.10])

    # Monthly rebalancing resets weights to target every month and pays transaction costs
    bt_m0 = backtest_portfolio(
        synthetic_returns, w_target, rebalance="monthly", tc_bps=0.0
    )
    bt_m50 = backtest_portfolio(
        synthetic_returns,
        w_target,
        rebalance="monthly",
        tc_bps=50.0,
        benchmark=synthetic_returns["Equities"],
    )
    assert np.allclose(bt_m0.weight_history, [0.50, 0.30, 0.10, 0.10], atol=1e-10)
    assert bt_m50.total_tc_paid > 0.0
    assert float(bt_m50.equity_curve.iloc[-1]) < float(bt_m0.equity_curve.iloc[-1])
    assert bt_m50.benchmark_equity is not None
    assert bt_m50.var_95 <= bt_m50.cvar_95 + 1e-12
    assert 0.0 <= bt_m50.max_drawdown <= 1.0

    # Drift-band rebalancing only rebalances when drift exceeds threshold
    bt_drift = backtest_portfolio(
        synthetic_returns, w_target, rebalance="drift_band", drift_threshold=0.03, tc_bps=10.0
    )
    assert np.all(np.max(np.abs(bt_drift.weight_history - [0.5, 0.3, 0.1, 0.1]), axis=1) <= 0.03 + 1e-9)


def test_simulate_wealth_glidepath_ruin_monotonicity_and_swr(
    synthetic_returns: pd.DataFrame,
) -> None:
    """prob_ruin increases monotonically with withdrawal; 95% SWR keeps ruin <= 5%."""
    alloc = optimize_portfolio(synthetic_returns, method="risk_parity")

    sim_low = simulate_wealth_glidepath(
        alloc, initial_wealth=100_000.0, horizon_years=30, annual_withdrawal=2_000.0, n_sims=600, seed=7
    )
    sim_mid = simulate_wealth_glidepath(
        alloc, initial_wealth=100_000.0, horizon_years=30, annual_withdrawal=6_000.0, n_sims=600, seed=7
    )
    sim_high = simulate_wealth_glidepath(
        alloc, initial_wealth=100_000.0, horizon_years=30, annual_withdrawal=14_000.0, n_sims=600, seed=7
    )

    assert sim_low.prob_ruin <= sim_mid.prob_ruin <= sim_high.prob_ruin
    assert sim_low.median_terminal_wealth > sim_high.median_terminal_wealth
    assert 0.005 < sim_low.safe_withdrawal_rate_95 < 0.20

    # Verify that withdrawing exactly safe_withdrawal_rate_95 * initial_wealth gives <= 5% ruin
    swr_withdrawal = sim_low.safe_withdrawal_rate_95 * 100_000.0
    sim_swr = simulate_wealth_glidepath(
        alloc,
        initial_wealth=100_000.0,
        horizon_years=30,
        annual_withdrawal=swr_withdrawal * 0.9999,
        n_sims=600,
        seed=7,
    )
    assert sim_swr.prob_ruin <= 0.05 + 1e-6


# ---------------------------------------------------------------------------
# 7. Result-Object Presentation Standard 2.0+ Compliance (All 5 Classes)
# ---------------------------------------------------------------------------


def _assert_presentation_standard(result_obj: object) -> None:
    """Verify all 7 Presentation Standard 2.0+ methods and frozen dataclass immutability."""
    df_sum = result_obj.summary()  # type: ignore[attr-defined]
    df_full = result_obj.to_dataframe()  # type: ignore[attr-defined]
    df_frame = result_obj.to_frame()  # type: ignore[attr-defined]

    assert isinstance(df_sum, pd.DataFrame) and not df_sum.empty
    assert isinstance(df_full, pd.DataFrame) and not df_full.empty
    assert isinstance(df_frame, pd.DataFrame) and not df_frame.empty

    md_str = result_obj.to_markdown()  # type: ignore[attr-defined]
    tex_str = result_obj.to_latex()  # type: ignore[attr-defined]
    typ_str = result_obj.to_typst()  # type: ignore[attr-defined]

    assert isinstance(md_str, str) and "|" in md_str
    assert isinstance(tex_str, str) and "\\begin{tabular}" in tex_str
    assert isinstance(typ_str, str) and "#table" in typ_str

    fig1 = result_obj.plot(show=False)  # type: ignore[attr-defined]
    assert isinstance(fig1, matplotlib.figure.Figure)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    out_fig = result_obj.plot(ax=ax2, show=False)  # type: ignore[attr-defined]
    assert isinstance(out_fig, matplotlib.figure.Figure)
    plt.close(fig2)

    with pytest.raises(FrozenInstanceError):
        result_obj.metadata = {}  # type: ignore[attr-defined]


def test_all_five_result_classes_presentation_standard(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Every result dataclass in puremacro.portfolio satisfies Presentation Standard 2.0+."""
    res_alloc = markowitz_frontier(synthetic_returns, n_points=10)
    _assert_presentation_standard(res_alloc)

    growth = synthetic_returns["Equities"]
    inflation = synthetic_returns["Commodities"]
    res_macro = all_weather_portfolio(synthetic_returns, growth, inflation)
    _assert_presentation_standard(res_macro)

    res_factor = factor_attribution(
        synthetic_returns["Equities"],
        synthetic_returns[["Bonds", "Commodities", "Gold"]],
    )
    _assert_presentation_standard(res_factor)

    res_bt = backtest_portfolio(
        synthetic_returns, [0.25, 0.25, 0.25, 0.25], rebalance="quarterly"
    )
    _assert_presentation_standard(res_bt)

    res_wealth = simulate_wealth_glidepath(
        res_alloc, initial_wealth=100_000.0, horizon_years=20, n_sims=200
    )
    _assert_presentation_standard(res_wealth)


# ---------------------------------------------------------------------------
# 8. Extended Edge Cases, Parametrized Policies & Validation Tests
# ---------------------------------------------------------------------------


def test_shrink_covariance_identical_columns() -> None:
    """Ledoit-Wolf handles collinear/identical asset columns cleanly."""
    x = np.linspace(-0.05, 0.05, 30)
    X = np.column_stack([x, x, x])
    cov = shrink_covariance(X, method="ledoit_wolf")
    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_optimize_portfolio_cvar_and_kelly_sensitivity(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Verify cvar_alpha, kelly_fraction, and shrink options in optimize_portfolio."""
    res_cvar_95 = optimize_portfolio(
        synthetic_returns, method="min_cvar", cvar_alpha=0.05, shrink="ledoit_wolf"
    )
    res_cvar_99 = optimize_portfolio(
        synthetic_returns, method="min_cvar", cvar_alpha=0.01, shrink="rmt"
    )
    assert res_cvar_99.cvar_95 >= res_cvar_95.cvar_95 - 1e-6

    res_half_kelly = optimize_portfolio(
        synthetic_returns, method="kelly", kelly_fraction=0.5
    )
    res_full_kelly = optimize_portfolio(
        synthetic_returns, method="kelly", kelly_fraction=1.0
    )
    assert res_full_kelly.expected_return >= res_half_kelly.expected_return - 1e-6

    with pytest.raises(ValueError, match="Unknown optimization method"):
        optimize_portfolio(synthetic_returns, method="nonexistent")


@pytest.mark.parametrize("linkage_method", ["single", "ward", "complete", "average"])
def test_hierarchical_risk_parity_all_linkages(
    synthetic_returns: pd.DataFrame, linkage_method: str
) -> None:
    """HRP produces valid allocations across all hierarchical clustering linkage methods."""
    res = hierarchical_risk_parity(synthetic_returns, linkage_method=linkage_method)
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-8)
    assert np.all(res.weights > 0.0)
    assert res.metadata["linkage_method"] == linkage_method


def test_hierarchical_risk_parity_single_asset() -> None:
    """Single-asset HRP returns weight [1.0]."""
    res = hierarchical_risk_parity(np.array([[0.04]]), is_cov=True, asset_names=["Solo"])
    assert np.allclose(res.weights, [1.0])
    assert res.asset_names == ("Solo",)


def test_markowitz_frontier_validation(synthetic_returns: pd.DataFrame) -> None:
    """markowitz_frontier raises ValueError when n_points < 2."""
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        markowitz_frontier(synthetic_returns, n_points=1)


def test_black_litterman_custom_omega_and_validation(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Black-Litterman accepts custom 1D/2D omega and validates shapes."""
    cov = shrink_covariance(synthetic_returns, method="ledoit_wolf") * 12.0
    w_mkt = [0.25, 0.25, 0.25, 0.25]
    P = np.array([1.0, -1.0, 0.0, 0.0])  # 1D relative view: Equities outperform Bonds
    Q = [0.05]

    res_om1d = black_litterman(cov, w_mkt, P=P, Q=Q, omega=np.array([0.001]))
    res_om2d = black_litterman(cov, w_mkt, P=P, Q=Q, omega=np.array([[0.001]]))
    assert np.allclose(res_om1d.weights, res_om2d.weights, atol=1e-10)
    assert res_om1d.weights[0] > res_om1d.weights[1]

    with pytest.raises(ValueError, match="market_weights length"):
        black_litterman(cov, [0.5, 0.5], P=P, Q=Q)

    with pytest.raises(ValueError, match="P must have shape"):
        black_litterman(cov, w_mkt, P=np.ones((2, 4)), Q=[0.05])


def test_classify_macro_quadrants_validation() -> None:
    """classify_macro_quadrants validates mismatched lengths and unknown methods."""
    with pytest.raises(ValueError, match="must match"):
        classify_macro_quadrants([1.0, 2.0], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Unknown classify_macro_quadrants method"):
        classify_macro_quadrants([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], method="bad")


def test_factor_attribution_1d_and_validation(synthetic_returns: pd.DataFrame) -> None:
    """factor_attribution supports 1D factor array and validates inputs."""
    y = synthetic_returns["Equities"].to_numpy()
    f1d = synthetic_returns["Bonds"].to_numpy()
    res = factor_attribution(y, f1d, factor_names=["Bonds"])
    assert res.factor_names == ("Bonds",)
    assert len(res.betas) == 1

    with pytest.raises(ValueError, match="must match"):
        factor_attribution(y, f1d[:-2])

    with pytest.raises(ValueError, match="rolling_window"):
        factor_attribution(y, f1d, rolling_window=2)


@pytest.mark.parametrize(
    "rebalance_policy",
    ["monthly", "quarterly", "annual", "buy_and_hold", "drift_band"],
)
def test_backtest_portfolio_all_rebalance_policies(
    synthetic_returns: pd.DataFrame, rebalance_policy: str
) -> None:
    """All rebalance policies run cleanly on both DataFrame and ndarray inputs."""
    bt = backtest_portfolio(
        synthetic_returns.to_numpy(),
        [0.4, 0.3, 0.2, 0.1],
        rebalance=rebalance_policy,
        asset_names=["EQ", "FI", "CM", "AU"],
    )
    assert isinstance(bt.equity_curve, np.ndarray)
    assert bt.weight_history.shape == (len(synthetic_returns), 4)
    assert np.allclose(np.sum(bt.weight_history, axis=1), 1.0, atol=1e-8)


def test_simulate_wealth_glidepath_accumulation_and_ndarray_inputs(
    synthetic_returns: pd.DataFrame,
) -> None:
    """Wealth glidepath works with 2D/1D arrays and positive annual contributions."""
    sim_2d = simulate_wealth_glidepath(
        synthetic_returns.to_numpy(),
        weights=[0.5, 0.3, 0.1, 0.1],
        initial_wealth=50_000.0,
        horizon_years=15,
        annual_contribution=5_000.0,
        annual_withdrawal=0.0,
        n_sims=200,
    )
    assert sim_2d.prob_ruin == 0.0
    assert sim_2d.median_terminal_wealth > 50_000.0 + 15 * 5_000.0

    sim_1d = simulate_wealth_glidepath(
        synthetic_returns["Equities"].to_numpy(),
        initial_wealth=50_000.0,
        horizon_years=10,
        n_sims=100,
    )
    assert sim_1d.wealth_paths.shape == (100, 11)
    assert set(sim_1d.quantiles.keys()) == {5, 10, 25, 50, 75, 90, 95}

    with pytest.raises(ValueError, match="initial_wealth must be > 0"):
        simulate_wealth_glidepath(synthetic_returns, initial_wealth=0.0)

