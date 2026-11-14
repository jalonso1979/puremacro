"""Tests for Pesaran (2006) Common Correlated Effects Mean Group (CCE-MG) panel LP."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.cce import cce_mg_panel_lp, cce_panel_lp


def _simulate_heterogeneous_factor_panel(
    *,
    n_entities: int = 24,
    n_periods: int = 80,
    seed: int = 202606,
    integrated_y: bool = False,
    heterogeneous_slopes: bool = True,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Simulate panel with correlated heterogeneous factor loadings lambda_i * f_t."""
    rng = np.random.default_rng(seed)
    lambda_x = np.linspace(0.4, 2.6, n_entities)
    # Correlate y-loadings nonlinearly/monotonically with x-loadings so pooled FE suffers omitted-factor bias
    lambda_y = 2.2 * lambda_x + rng.normal(0.0, 0.15, size=n_entities)

    if heterogeneous_slopes:
        beta0_i = 0.50 + rng.uniform(-0.25, 0.25, size=n_entities)
        beta1_i = 0.30 + rng.uniform(-0.15, 0.15, size=n_entities)
        beta2_i = 0.15 + rng.uniform(-0.10, 0.10, size=n_entities)
    else:
        beta0_i = np.full(n_entities, 0.50)
        beta1_i = np.full(n_entities, 0.30)
        beta2_i = np.full(n_entities, 0.15)

    # Persistent AR(1) unobserved common factor
    f = np.zeros(n_periods)
    for t in range(1, n_periods):
        f[t] = 0.8 * f[t - 1] + rng.normal(0.0, 1.0)

    frames: list[pd.DataFrame] = []
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="QS")
    for i in range(n_entities):
        alpha_i = 0.4 * (i - n_entities / 2.0)
        eps_x = rng.normal(0.0, 0.55, size=n_periods)
        eps_y = rng.normal(0.0, 0.25, size=n_periods)
        z_i = 0.6 * lambda_x[i] * f + rng.normal(0.0, 0.5, size=n_periods)
        x_i = lambda_x[i] * f + eps_x
        dy_struct = np.zeros(n_periods)
        for t in range(n_periods):
            val = beta0_i[i] * eps_x[t] + 0.10 * z_i[t] + lambda_y[i] * f[t] + eps_y[t]
            if not integrated_y:
                if t >= 1:
                    val += beta1_i[i] * eps_x[t - 1]
                if t >= 2:
                    val += beta2_i[i] * eps_x[t - 2]
            dy_struct[t] = val
        y_i = alpha_i + (np.cumsum(dy_struct) if integrated_y else dy_struct)
        frames.append(
            pd.DataFrame(
                {"code": f"E{i:02d}", "date": dates, "y": y_i, "x": x_i, "z": z_i}
            )
        )

    df = pd.concat(frames, ignore_index=True).set_index(["code", "date"]).sort_index()
    if integrated_y:
        true_mg = {0: float(np.mean(beta0_i)), 1: float(np.mean(beta0_i)), 2: float(np.mean(beta0_i))}
    else:
        true_mg = {
            0: float(np.mean(beta0_i)),
            1: float(np.mean(beta1_i)),
            2: float(np.mean(beta2_i)),
        }
    return df, true_mg


def _simulate_stationary_unit_loading_dgp(
    *, n_entities: int = 32, n_periods: int = 140, seed: int = 31415
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Simulate correlated heterogeneous factor loadings directly on horizon differences."""
    rng = np.random.default_rng(seed)
    lambda_x = np.linspace(0.3, 2.8, n_entities)
    # Zero cross-sectional average lambda_y doesn't remove pooled bias because lambda_x * lambda_y covariance != 0
    lambda_y = 2.0 * (lambda_x - np.mean(lambda_x))
    beta0_i = 0.60 + rng.uniform(-0.20, 0.20, size=n_entities)
    beta1_i = 0.35 + rng.uniform(-0.12, 0.12, size=n_entities)
    beta2_i = 0.15 + rng.uniform(-0.08, 0.08, size=n_entities)

    f = np.zeros(n_periods)
    for t in range(1, n_periods):
        f[t] = 0.8 * f[t - 1] + rng.normal(0.0, 1.0)

    frames = []
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="QS")
    for i in range(n_entities):
        eps_x = rng.normal(0.0, 0.55, size=n_periods)
        eps_y = rng.normal(0.0, 0.25, size=n_periods)
        x_i = lambda_x[i] * f + eps_x
        y_i = np.zeros(n_periods)
        for t in range(n_periods):
            val = beta0_i[i] * eps_x[t] + lambda_y[i] * f[t] + eps_y[t]
            if t >= 1:
                val += beta1_i[i] * eps_x[t - 1]
            if t >= 2:
                val += beta2_i[i] * eps_x[t - 2]
            y_i[t] = val
        frames.append(pd.DataFrame({"code": f"U{i:02d}", "date": dates, "y": y_i, "x": x_i}))
    df = pd.concat(frames, ignore_index=True).set_index(["code", "date"]).sort_index()
    true_mg = {0: float(np.mean(beta0_i)), 1: float(np.mean(beta1_i)), 2: float(np.mean(beta2_i))}
    return df, true_mg


def test_cce_mg_recovers_true_irf_while_pooled_suffers_heterogeneous_loading_bias():
    df, true_mg = _simulate_stationary_unit_loading_dgp()

    res_pooled = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, estimator="pooled")
    res_mg = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, estimator="mean_group")
    res_helper = cce_mg_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1)

    # Helper and estimator="mean_group" produce identical estimates and provenance
    pd.testing.assert_frame_equal(res_mg, res_helper)
    assert res_mg.attrs["specification"] == res_helper.attrs["specification"]

    for h in (0, 1, 2):
        b_mg = float(res_mg.loc[res_mg["h"] == h, "beta"].iloc[0])
        se_mg = float(res_mg.loc[res_mg["h"] == h, "se"].iloc[0])
        b_pool = float(res_pooled.loc[res_pooled["h"] == h, "beta"].iloc[0])
        err_mg = abs(b_mg - true_mg[h])
        err_pool = abs(b_pool - true_mg[h])

        assert np.isfinite(se_mg) and se_mg > 0.0
        assert err_mg < 0.08, f"CCE-MG error at h={h}: est={b_mg:.3f} vs true={true_mg[h]:.3f}"
        # Correlated heterogeneous factor loadings create severe CCEP bias relative to CCE-MG
        assert err_pool > 0.20, f"Expected CCEP bias on correlated loadings at h={h}, got {err_pool:.3f}"
        assert err_mg < 0.35 * err_pool


def test_cce_mg_with_integrated_y_and_controls_reduces_ccep_bias():
    df, true_mg = _simulate_heterogeneous_factor_panel(
        n_entities=24, n_periods=100, seed=777, integrated_y=True, heterogeneous_slopes=True
    )

    res_pooled = cce_panel_lp(
        df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, controls=["z"], estimator="pooled"
    )
    res_mg = cce_panel_lp(
        df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, controls=["z"], estimator="mean_group"
    )

    for h in (0, 1, 2):
        b_mg = float(res_mg.loc[res_mg["h"] == h, "beta"].iloc[0])
        b_pool = float(res_pooled.loc[res_pooled["h"] == h, "beta"].iloc[0])
        err_mg = abs(b_mg - true_mg[h])
        err_pool = abs(b_pool - true_mg[h])
        assert err_mg < 0.12, f"h={h}: b_mg={b_mg:.3f}, true={true_mg[h]:.3f}"
        assert err_pool > 0.50, f"h={h}: expected large CCEP bias under I(1) het loadings, got {b_pool:.3f}"
        assert err_mg < 0.25 * err_pool


def test_cce_mg_zero_lags_and_nonparametric_se_exact_formula():
    df, true_mg = _simulate_stationary_unit_loading_dgp(n_entities=12, n_periods=60, seed=42)
    res_mg = cce_panel_lp(df, y="y", x="x", horizons=[0, 1], n_lags=0, estimator="mean_group", alpha=0.05)

    # Manually replicate h=0 unit OLS to verify Pesaran (2006) eq. (56) cross-unit mean & SE
    work = df.sort_index().copy()
    work["dy0"] = work.groupby(level="code")["y"].shift(0) - work.groupby(level="code")["y"].shift(1)
    work["cs_dy0"] = work.groupby(level="date")["dy0"].transform("mean")
    work["cs_x"] = work.groupby(level="date")["x"].transform("mean")
    sub = work[["dy0", "x", "cs_dy0", "cs_x"]].dropna()

    unit_betas = []
    for _, grp in sub.groupby(level="code"):
        Z = np.column_stack([np.ones(len(grp)), grp[["x", "cs_dy0", "cs_x"]].to_numpy(dtype=float)])
        coef, *_ = np.linalg.lstsq(Z, grp["dy0"].to_numpy(dtype=float), rcond=None)
        unit_betas.append(float(coef[1]))

    expected_beta = float(np.mean(unit_betas))
    expected_se = float(np.sqrt(np.var(unit_betas, ddof=1) / len(unit_betas)))

    row0 = res_mg.loc[res_mg["h"] == 0].iloc[0]
    assert np.isclose(row0["beta"], expected_beta, atol=1e-11)
    assert np.isclose(row0["se"], expected_se, atol=1e-11)
    assert int(row0["n_entities_used"]) == 12
    assert abs(expected_beta - true_mg[0]) < 0.10


def test_cce_baseline_pooled_preserved_and_specification_metadata():
    df, _ = _simulate_stationary_unit_loading_dgp(n_entities=8, n_periods=45, seed=99)

    res_default = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, controls=())
    res_pooled = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, controls=(), estimator="pooled")
    pd.testing.assert_frame_equal(res_default, res_pooled)

    assert list(res_pooled.columns) == ["h", "beta", "se", "lo", "hi"]
    assert res_pooled.method == "cce_panel_lp"
    spec_p = res_pooled.attrs["specification"]
    meta_p = res_pooled.metadata
    assert spec_p["estimator"] == "cce_panel_lp"
    assert spec_p["cce_estimator"] == "cce_pooled"
    assert spec_p["factor_loadings"] == "homogeneous"
    assert spec_p["cov_type"] == "cluster"
    assert meta_p["cce_estimator"] == "cce_pooled"
    assert meta_p["factor_loadings"] == "homogeneous"

    res_mg = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, estimator="mean_group")
    assert list(res_mg.columns) == ["h", "beta", "se", "lo", "hi", "n_entities_used"]
    assert res_mg.method == "cce_panel_lp"
    spec_mg = res_mg.attrs["specification"]
    meta_mg = res_mg.metadata
    assert spec_mg["estimator"] == "cce_mean_group"
    assert spec_mg["cce_estimator"] == "cce_mean_group"
    assert spec_mg["factor_loadings"] == "heterogeneous"
    assert spec_mg["cov_type"] == "pesaran_mg"
    assert spec_mg["n_entities_by_horizon"] == {0: 8, 1: 8, 2: 8}
    assert spec_mg["excluded_entities_by_horizon"] == {}
    assert meta_mg["estimator"] == "cce_mean_group"
    assert meta_mg["cce_estimator"] == "cce_mean_group"
    assert meta_mg["factor_loadings"] == "heterogeneous"


def test_cce_mg_long_form_and_row_order_invariance():
    df, _ = _simulate_stationary_unit_loading_dgp(n_entities=6, n_periods=40, seed=55)
    res_wide = cce_panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1, estimator="mean_group")

    df_long = df.reset_index().rename(columns={"code": "iso", "date": "quarter"})
    res_long = cce_panel_lp(
        df_long, y="y", x="x", horizons=[0, 1, 2], n_lags=1,
        estimator="mean_group", unit_col="iso", time_col="quarter"
    )
    pd.testing.assert_frame_equal(res_wide, res_long)

    # Row-order permutation invariance
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(df))
    df_shuffled = df.iloc[perm]
    res_shuffled = cce_panel_lp(df_shuffled, y="y", x="x", horizons=[0, 1, 2], n_lags=1, estimator="mean_group")
    pd.testing.assert_frame_equal(res_wide, res_shuffled)


def test_cce_insufficient_obs_and_rank_deficient_units_explicitly_warned_and_recorded():
    df, _ = _simulate_stationary_unit_loading_dgp(n_entities=6, n_periods=45, seed=808)
    work = df.copy()
    dates = work.index.get_level_values("date").unique().sort_values()

    # Make U00 short (only 8 non-NaN periods -> insufficient obs)
    work.loc[("U00", dates[8:]), "y"] = np.nan
    # Make U01 constant x (rank-deficient / collinear with intercept)
    work.loc["U01", "x"] = 1.0

    with pytest.warns(UserWarning) as caught:
        res_mg = cce_panel_lp(work, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="mean_group")

    messages = [str(w.message) for w in caught]
    assert any("U00" in m and "insufficient observations" in m for m in messages)
    assert any("U01" in m and ("collinear" in m or "rank-deficient" in m) for m in messages)

    assert list(res_mg["n_entities_used"]) == [4, 4]
    excl = res_mg.attrs["specification"]["excluded_entities_by_horizon"]
    assert "U00" in excl[0] and "U01" in excl[0]
    assert "U00" in excl[1] and "U01" in excl[1]

    # If remaining usable entities fall below 3, fail closed with ValueError
    work_too_few = work.drop(index=["U02", "U03"], level="code")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="insufficient valid entities"):
            cce_panel_lp(work_too_few, y="y", x="x", horizons=[0], n_lags=1, estimator="mean_group")


def test_cce_mg_rejects_fewer_than_three_entities_and_bad_min_obs():
    df, _ = _simulate_stationary_unit_loading_dgp(n_entities=4, n_periods=40, seed=17)
    df_two = df.loc[["U00", "U01"]]

    with pytest.raises(ValueError, match="at least 3 entities"):
        cce_panel_lp(df_two, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="mean_group")

    with pytest.raises(ValueError, match="min_obs"):
        cce_panel_lp(df, y="y", x="x", horizons=[0], n_lags=1, estimator="mean_group", min_obs=0)

    with pytest.raises(ValueError, match="min_obs"):
        cce_panel_lp(df, y="y", x="x", horizons=[0], n_lags=1, estimator="mean_group", min_obs=True)  # type: ignore[arg-type]

    # When min_obs forces all units out, fail closed with ValueError
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="insufficient valid entities"):
            cce_panel_lp(df, y="y", x="x", horizons=[0], n_lags=1, estimator="mean_group", min_obs=500)


def test_cce_input_and_chronology_error_contracts():
    df, _ = _simulate_stationary_unit_loading_dgp(n_entities=4, n_periods=35, seed=31)

    with pytest.raises(ValueError, match="estimator"):
        cce_panel_lp(df, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="horizons"):
        cce_panel_lp(df, y="y", x="x", horizons=[], n_lags=1)

    with pytest.raises(ValueError, match="horizons"):
        cce_panel_lp(df, y="y", x="x", horizons=[-1, 0], n_lags=1)

    with pytest.raises(ValueError, match="duplicate"):
        cce_panel_lp(df, y="y", x="x", horizons=[0, 0], n_lags=1)

    with pytest.raises(ValueError, match="distinct"):
        cce_panel_lp(df, y="y", x="y", horizons=[0, 1], n_lags=1)

    with pytest.raises(ValueError, match="conflicting"):
        cce_panel_lp(df, y="y", x="x", horizons=[0, 2], horizon=1, n_lags=1)

    with pytest.raises(ValueError, match="conflicting"):
        cce_panel_lp(df, y="y", x="x", horizons=[0, 1], n_lags=1, lags=2)

    with pytest.raises(ValueError, match="conflicting"):
        cce_panel_lp(df, y="y", x="x", horizons=[0, 1], n_lags=1, alpha=0.10, ci=0.95)

    with pytest.raises(KeyError, match="missing_col"):
        cce_panel_lp(df, y="y", x="missing_col", horizons=[0, 1], n_lags=1)

    # Duplicate (entity, time) index rows rejected
    df_dup = pd.concat([df, df.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        cce_panel_lp(df_dup, y="y", x="x", horizons=[0, 1], n_lags=1)

    # Infinite values rejected
    df_inf = df.copy()
    df_inf.iloc[0, df_inf.columns.get_loc("x")] = np.inf
    with pytest.raises(ValueError, match="inf|finite"):
        cce_panel_lp(df_inf, y="y", x="x", horizons=[0, 1], n_lags=1)

    # Dated index gap rejected by default, allowed with allow_index_gaps=True
    dates = df.index.get_level_values("date").unique().sort_values()
    df_gap = df.drop(index=("U00", dates[10]))
    with pytest.raises(ValueError, match="irregular gap|allow_index_gaps"):
        cce_panel_lp(df_gap, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="mean_group")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res_gapped = cce_panel_lp(
            df_gap, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="mean_group", allow_index_gaps=True
        )
    assert len(res_gapped) == 2

    # Missing time label (NaT) rejected
    df_nat = df.copy()
    bad_dates = df_nat.index.get_level_values("date").to_numpy().copy()
    bad_dates[0] = np.datetime64("NaT")
    df_nat.index = pd.MultiIndex.from_arrays(
        [df_nat.index.get_level_values("code"), bad_dates], names=["code", "date"]
    )
    with pytest.raises(ValueError, match="missing entity or time"):
        cce_panel_lp(df_nat, y="y", x="x", horizons=[0, 1], n_lags=1, estimator="mean_group")
