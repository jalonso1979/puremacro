import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.did import (
    callaway_santanna,
    honest_did,
    honest_did_sensitivity,
    HonestDiDResult,
)


def test_honest_did_basic_relative_magnitude():
    # Synthetic event-study series
    # pre-periods: -3, -2, -1 (base)
    # post-periods: 0, 1, 2
    event_time = [-3, -2, -1, 0, 1, 2]
    # pre-trends: max abs deviation = 0.2
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]

    res = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=0,
        method="relative_magnitude",
        m_grid=[0.0, 0.5, 1.0, 2.0],
        ci=0.95,
    )

    assert isinstance(res, HonestDiDResult)
    assert res.pre_trend_max == pytest.approx(0.20)
    assert res.method == "relative_magnitude"
    assert res.ci == 0.95

    tbl = res.to_frame()
    assert len(tbl) == 4
    assert list(tbl["M"]) == [0.0, 0.5, 1.0, 2.0]

    # At M=0, identified set is point [1.50, 1.50]
    m0_row = tbl[tbl["M"] == 0.0].iloc[0]
    assert m0_row["id_lo"] == pytest.approx(1.50)
    assert m0_row["id_hi"] == pytest.approx(1.50)
    # 95% CI is 1.50 +/- 1.95996 * 0.15
    assert m0_row["ci_lo"] == pytest.approx(1.50 - 1.95996 * 0.15, rel=1e-3)
    assert m0_row["ci_hi"] == pytest.approx(1.50 + 1.95996 * 0.15, rel=1e-3)
    assert bool(m0_row["significant"]) is True

    # At M=1.0, identified set is [1.50 - 0.20, 1.50 + 0.20] = [1.30, 1.70]
    m1_row = tbl[tbl["M"] == 1.0].iloc[0]
    assert m1_row["id_lo"] == pytest.approx(1.30)
    assert m1_row["id_hi"] == pytest.approx(1.70)
    assert m1_row["ci_lo"] < 1.30
    assert m1_row["ci_hi"] > 1.70

    # Breakdown value M*
    # 1.50 - M* * 0.20 - c(M*) * 0.15 = 0
    # For large M*, c(M*) ~ 1.645
    # 1.50 - M* * 0.20 - 1.645 * 0.15 = 0 => M* ~ (1.50 - 0.24675) / 0.20 = 6.266
    assert isinstance(res.breakdown_value, float)
    assert 6.0 < res.breakdown_value < 6.5


def test_honest_did_with_callaway_santanna_result():
    # Build simple panel
    rng = np.random.default_rng(42)
    units = []
    for u in range(16):
        treat_yr = 2012 if u < 6 else (2014 if u < 12 else np.nan)
        for yr in range(2009, 2016):
            d = 1.0 if not np.isnan(treat_yr) and yr >= treat_yr else 0.0
            y = 3.0 * d + 0.2 * (yr - 2009) + rng.standard_normal() * 0.5
            units.append(
                {"unit": f"U{u}", "year": yr, "treat_time": treat_yr, "outcome": y}
            )

    df_panel = pd.DataFrame(units)
    cs_res = callaway_santanna(
        df_panel, unit="unit", time="year", outcome="outcome", treat_time="treat_time"
    )

    sens = honest_did_sensitivity(cs_res, target_horizon=0, ci=0.90)
    assert isinstance(sens, HonestDiDResult)
    assert sens.target_horizons == [0]
    assert len(sens.table) > 0

    # Formatting checks
    summ = sens.summary()
    assert "Honest DiD Sensitivity Analysis" in summ
    assert "Breakdown Value" in summ

    md = sens.to_markdown()
    assert "| M |" in md or "| M" in md or "M" in md

    ltx = sens.to_latex()
    assert "\\begin{tabular}" in ltx

    typ = sens.to_typst()
    assert "#table" in typ

    ascii_art = sens.plot_ascii()
    assert "Honest DiD Confidence Intervals vs M" in ascii_art
    assert "|" in ascii_art


def test_honest_did_smoothness_method():
    event_time = [-3, -2, -1, 0, 1]
    beta = [0.1, -0.05, 0.0, 1.2, 1.5]
    se = [0.1, 0.1, 0.0, 0.2, 0.2]

    sens = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=[0, 1],
        method="smoothness",
        m_grid=[0.0, 0.1, 0.2],
    )
    assert sens.method == "smoothness"
    assert sens.pre_trend_slope is not None
    assert isinstance(sens.breakdown_value, dict)
    assert 0 in sens.breakdown_value
    assert 1 in sens.breakdown_value


def test_honest_did_already_insignificant():
    # If baseline estimate is not significant, breakdown value M* = 0.0
    event_time = [-2, -1, 0]
    beta = [0.1, 0.0, 0.05]
    se = [0.1, 0.0, 0.5]  # large standard error

    sens = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=0,
    )
    assert sens.breakdown_value == 0.0
    assert "already not statistically distinguishable" in sens.summary()


def test_honest_did_validation_errors():
    with pytest.raises(ValueError, match="method must be"):
        honest_did_sensitivity(
            event_time=[-1, 0], beta=[0, 1], se=[0, 0.1], method="unknown"
        )

    with pytest.raises(ValueError, match="must provide either 'result'"):
        honest_did_sensitivity()

    with pytest.raises(ValueError, match="at least one pre-treatment period"):
        honest_did_sensitivity(
            event_time=[-1, 0, 1],
            beta=[0.0, 1.0, 1.2],
            se=[0.0, 0.1, 0.1],
            base_period=-1,
        )

    with pytest.raises(ValueError, match="equal lengths"):
        honest_did_sensitivity(event_time=[-2, -1, 0], beta=[0.1, 0.0], se=[0.1, 0.1])


# =============================================================================
# Benzarti & Carloni (2019) Canonical Benchmark Data from Rambachan & Roth (2023)
# =============================================================================
BC_BETAHAT = np.array(
    [
        0.0066963518,
        0.0293450337,
        -0.0064729722,
        0.0730149895,
        0.1959611177,
        0.3120639026,
        0.2395415455,
        0.1260425001,
    ]
)
BC_YEARS = [2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012]
BC_REF_YEAR = 2008
BC_SIGMA = np.array(
    [
        [
            0.0008428358,
            0.0004768687,
            0.0002618051,
            0.0002354220,
            0.0001676371,
            0.0001128708,
            0.0000199282,
            -0.0001368265,
        ],
        [
            0.0004768687,
            0.0006425420,
            0.0003987425,
            0.0002435515,
            0.0002201960,
            0.0001804591,
            0.0000384377,
            -0.0000296042,
        ],
        [
            0.0002618051,
            0.0003987425,
            0.0005229950,
            0.0002117686,
            0.0001840722,
            0.0001458528,
            0.0000700520,
            0.0000595299,
        ],
        [
            0.0002354220,
            0.0002435515,
            0.0002117686,
            0.0003089595,
            0.0001197866,
            0.0001334081,
            0.0001016335,
            0.0001079052,
        ],
        [
            0.0001676371,
            0.0002201960,
            0.0001840722,
            0.0001197866,
            0.0003599704,
            0.0002478819,
            0.0001749579,
            0.0001654257,
        ],
        [
            0.0001128708,
            0.0001804591,
            0.0001458528,
            0.0001334081,
            0.0002478819,
            0.0004263950,
            0.0002171438,
            0.0002892748,
        ],
        [
            0.0000199282,
            0.0000384377,
            0.0000700520,
            0.0001016335,
            0.0001749579,
            0.0002171438,
            0.0004886698,
            0.0003805322,
        ],
        [
            -0.0001368265,
            -0.0000296042,
            0.0000595299,
            0.0001079052,
            0.0001654257,
            0.0002892748,
            0.0003805322,
            0.0007617394,
        ],
    ]
)


def test_honest_did_primary_api():
    """Verify honest_did primary entry point with both smoothness and relative magnitude."""
    b_hat = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    et = [-3, -2, -1, 0, 1, 2]

    # Smoothness method
    res_sd = honest_did(
        b_hat=b_hat,
        se=se,
        event_time=et,
        method="smoothness",
        m_vec=[0.0, 0.1, 0.2],
        base_period=-1,
        alpha=0.05,
    )
    assert isinstance(res_sd, HonestDiDResult)
    assert res_sd.method == "smoothness"
    assert res_sd.ci == 0.95
    assert len(res_sd.table) == 3
    assert np.isfinite(res_sd.breakdown_value)
    assert res_sd.breakdown_value > 0.0

    # Relative magnitude method
    res_rm = honest_did(
        b_hat=b_hat,
        se=se,
        event_time=et,
        method="relative_magnitude",
        m_vec=[0.0, 0.5, 1.0, 2.0],
        base_period=-1,
        alpha=0.05,
    )
    assert isinstance(res_rm, HonestDiDResult)
    assert res_rm.method == "relative_magnitude"
    assert 6.0 < res_rm.breakdown_value < 6.5

    # Check presentation interface
    summ = res_sd.summary()
    assert "Honest DiD Sensitivity Analysis" in summ
    assert "Smoothness" in res_sd.summary() or "smoothness" in res_sd.method


def test_honest_did_covariance_matrix_sigma():
    """Verify honest_did with full covariance matrix Sigma and custom linear combination l_vec."""
    # 2 pre-periods (-2, -1), 2 post-periods (0, 1)
    b_hat = np.array([0.05, 0.0, 1.2, 1.6])
    event_time = [-2, -1, 0, 1]

    # Covariance with positive correlation between post-treatment horizons
    cov = np.array(
        [
            [0.04, 0.01, 0.00, 0.00],
            [0.01, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.04, 0.02],
            [0.00, 0.00, 0.02, 0.09],
        ]
    )

    # 1. Horizon 0 (basis vector e_0)
    res_h0 = honest_did(
        b_hat=b_hat,
        sigma=cov,
        event_time=event_time,
        method="smoothness",
        target_horizon=0,
        m_vec=[0.0, 0.1],
    )
    assert res_h0.table["orig_se"].iloc[0] == pytest.approx(np.sqrt(0.04), rel=1e-5)

    # 2. Average effect over post-periods: l_vec = [0.5, 0.5]
    l_avg = np.array([0.5, 0.5])
    # Expected variance: 0.25*0.04 + 0.25*0.09 + 2*0.25*0.02 = 0.01 + 0.0225 + 0.01 = 0.0425
    expected_se = np.sqrt(0.0425)
    res_avg = honest_did(
        b_hat=b_hat,
        sigma=cov,
        event_time=event_time,
        method="smoothness",
        l_vec=l_avg,
        m_vec=[0.0, 0.1],
    )
    assert res_avg.table["orig_estimate"].iloc[0] == pytest.approx(
        0.5 * 1.2 + 0.5 * 1.6
    )
    assert res_avg.table["orig_se"].iloc[0] == pytest.approx(expected_se, rel=1e-4)


def test_honest_did_plot_method():
    """Verify HonestDiDResult.plot graphical method and elements."""
    event_time = [-3, -2, -1, 0, 1, 2]
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]

    res = honest_did(
        b_hat=beta,
        se=se,
        event_time=event_time,
        method="relative_magnitude",
        m_vec=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        target_horizon=0,
    )

    # 1. Default call returns Axes
    ax = res.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    plt.close("all")

    # 2. return_fig=True returns (Figure, Axes)
    fig, ax2 = res.plot(return_fig=True, title="Custom Test Title")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax2, plt.Axes)
    assert ax2.get_title() == "Custom Test Title"

    # Verify plot elements: collections for shaded bands, lines for bounds
    assert len(ax2.collections) >= 2  # identified set band + robust CI band
    # Verify zero line exists
    zero_lines = [line for line in ax2.lines if line.get_linestyle() == ":"]
    assert len(zero_lines) >= 1
    plt.close("all")

    # 3. Supply existing axis
    fig_custom, ax_custom = plt.subplots(figsize=(10, 6))
    ax_out = res.plot(ax=ax_custom)
    assert ax_out is ax_custom
    plt.close("all")


def test_honest_did_replication_benzarti_carloni_2019():
    """Replicate Rambachan & Roth (2023) Section 6.1 on Benzarti & Carloni (2019) VAT study.

    Analyzes the incidence of the French sit-down restaurant VAT cut (July 2009) on firm profits.
    Reference period: 2008 (event time -1).
    Pre-periods: 2004..2007.
    Post-periods: 2009..2012.
    """
    se_vec = np.sqrt(np.diag(BC_SIGMA))

    # 1. Baseline regression estimate for 2009 (first post-treatment year)
    b_2009 = BC_BETAHAT[4]
    se_2009 = se_vec[4]
    assert b_2009 == pytest.approx(0.1960, abs=1e-3)
    assert se_2009 == pytest.approx(0.0190, abs=1e-3)
    t_stat = b_2009 / se_2009
    assert t_stat > 10.0  # Highly statistically significant at baseline

    # Pre-treatment max abs deviation: max(|0.0067|, |0.0293|, |-0.0065|, |0.0730|) = 0.073015
    expected_pre_max = np.max(np.abs(BC_BETAHAT[:4]))
    assert expected_pre_max == pytest.approx(0.073015, abs=1e-4)

    # 2. Relative Magnitude Sensitivity: Delta^RM(M_bar)
    res_rm = honest_did(
        b_hat=BC_BETAHAT,
        sigma=BC_SIGMA,
        event_time=BC_YEARS,
        base_period=BC_REF_YEAR,
        method="relative_magnitude",
        m_vec=[0.0, 0.5, 1.0, 1.5, 2.0],
        target_horizon=2009,
        alpha=0.05,
    )
    assert res_rm.method == "relative_magnitude"
    assert res_rm.pre_trend_max == pytest.approx(expected_pre_max, abs=1e-4)

    tbl_rm = res_rm.to_frame()
    # At M=0: point identified [0.1960, 0.1960]
    row_m0 = tbl_rm[tbl_rm["M"] == 0.0].iloc[0]
    assert row_m0["id_lo"] == pytest.approx(b_2009, abs=1e-3)
    assert row_m0["id_hi"] == pytest.approx(b_2009, abs=1e-3)
    assert row_m0["ci_lo"] == pytest.approx(b_2009 - 1.95996 * se_2009, abs=2e-3)
    assert row_m0["ci_hi"] == pytest.approx(b_2009 + 1.95996 * se_2009, abs=2e-3)
    assert bool(row_m0["significant"]) is True

    # At M=1.0: identified set [0.1960 - 0.0730, 0.1960 + 0.0730] = [0.1230, 0.2690]
    row_m1 = tbl_rm[tbl_rm["M"] == 1.0].iloc[0]
    assert row_m1["id_lo"] == pytest.approx(b_2009 - expected_pre_max, abs=1e-3)
    assert row_m1["id_hi"] == pytest.approx(b_2009 + expected_pre_max, abs=1e-3)
    assert row_m1["ci_lo"] > 0.08
    assert bool(row_m1["significant"]) is True

    # At M=2.0: effect remains strictly positive and significant
    row_m2 = tbl_rm[tbl_rm["M"] == 2.0].iloc[0]
    assert row_m2["ci_lo"] > 0.0
    assert bool(row_m2["significant"]) is True

    # Breakdown value M* for relative magnitudes is > 2.0
    assert isinstance(res_rm.breakdown_value, float)
    assert 2.0 < res_rm.breakdown_value < 2.5

    # 3. Smoothness Sensitivity: Delta^SD(M)
    res_sd = honest_did(
        b_hat=BC_BETAHAT,
        sigma=BC_SIGMA,
        event_time=BC_YEARS,
        base_period=BC_REF_YEAR,
        method="smoothness",
        m_vec=[0.0, 0.05, 0.10, 0.15],
        target_horizon=2009,
        alpha=0.05,
    )
    assert res_sd.method == "smoothness"
    tbl_sd = res_sd.to_frame()

    # At M=0: linear extrapolation of pre-trend
    row_sd0 = tbl_sd[tbl_sd["M"] == 0.0].iloc[0]
    assert row_sd0["ci_lo"] > 0.10
    assert bool(row_sd0["significant"]) is True

    # Effect remains significant at M=0.10
    row_sd10 = tbl_sd[tbl_sd["M"] == 0.10].iloc[0]
    assert row_sd10["ci_lo"] > 0.0
    assert bool(row_sd10["significant"]) is True

    # Breakdown value M* is ~0.14 - 0.16
    assert 0.12 < res_sd.breakdown_value < 0.18


def test_honest_did_breakdown_value_brentq():
    """Verify brentq root finder precision and edge cases."""
    event_time = [-3, -2, -1, 0]
    beta = [0.10, -0.10, 0.0, 1.0]
    se = [0.05, 0.05, 0.0, 0.10]

    res = honest_did(
        b_hat=beta,
        se=se,
        event_time=event_time,
        method="relative_magnitude",
        target_horizon=0,
    )
    m_star = res.breakdown_value
    assert isinstance(m_star, float)
    assert 8.0 < m_star < 10.0

    # Check that evaluating exactly at m_star gives ci_lo ~ 0.0
    res_at_mstar = honest_did(
        b_hat=beta,
        se=se,
        event_time=event_time,
        method="relative_magnitude",
        m_vec=[m_star],
        target_horizon=0,
    )
    row = res_at_mstar.table.iloc[0]
    assert abs(row["ci_lo"]) < 1e-4


def test_honest_did_multi_horizon_evaluation():
    """Verify multi-horizon sensitivity analysis returning dictionary of breakdown values."""
    event_time = [-2, -1, 0, 1, 2]
    beta = [0.1, 0.0, 1.2, 1.4, 1.6]
    se = [0.1, 0.0, 0.2, 0.25, 0.3]

    res = honest_did(
        b_hat=beta,
        se=se,
        event_time=event_time,
        method="smoothness",
        target_horizon=[0, 1, 2],
        m_vec=[0.0, 0.1],
    )
    assert isinstance(res.breakdown_value, dict)
    assert set(res.breakdown_value.keys()) == {0, 1, 2}
    for h in [0, 1, 2]:
        assert res.breakdown_value[h] > 0.0

    tbl = res.to_frame()
    assert len(tbl) == 6  # 3 horizons * 2 M values
    assert set(tbl["horizon"].unique()) == {0, 1, 2}


def test_honest_did_relative_magnitude_first_diff():
    """Verify relative magnitude with first-difference slope restriction."""
    event_time = [-3, -2, -1, 0, 1]
    beta = [0.10, -0.10, 0.0, 1.0, 1.2]
    se = [0.05, 0.05, 0.0, 0.10, 0.15]

    res = honest_did(
        b_hat=beta,
        se=se,
        event_time=event_time,
        method="relative_magnitude",
        bound="deviation from pre-trend slope",
        m_vec=[0.0, 0.5, 1.0],
        target_horizon=0,
    )
    assert res.method == "relative_magnitude"
    assert len(res.table) == 3
    assert res.breakdown_value > 0.0


def test_honest_did_pre_post_periods_syntax():
    """Verify honest_did invocation with pre_periods and post_periods integer counts."""
    # 3 pre-periods, 2 post-periods
    b_hat = [0.05, -0.02, 0.04, 1.2, 1.5]
    se = [0.05, 0.05, 0.05, 0.10, 0.12]

    res = honest_did(
        b_hat=b_hat,
        se=se,
        pre_periods=3,
        post_periods=2,
        method="smoothness",
        m_vec=[0.0, 0.1],
    )
    assert isinstance(res, HonestDiDResult)
    assert len(res.table) == 2
