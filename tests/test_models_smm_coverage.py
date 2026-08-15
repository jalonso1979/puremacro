"""Coverage tests for puremacro.models.smm.

Tests cover:
- MomentVector dataclass (defaults, custom horizons)
- load_empirical_moments: missing CSV, partial data, se=0, wrong state filter,
  partial horizons, vacancy block, xs Q4-Q1 math, xs se=0, xs missing quartile
- load_empirical_moments_monthly: missing CSV, values/weights, custom horizons,
  vacancy-block stays NaN, missing se columns, xs block shared with quarterly loader
- theoretical_moments: output shapes, finite values, uniform weights, block
  composition against dmp_irf, AIOE xs-sign properties
- smm_objective: zero objective at truth, all-NaN ignores, infeasible parameters
  returns 1e10, weighted sum formula, combined ts+xs computation, weight scaling
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.models.dmp_regime_dependent import DMPParameters, dmp_irf
from puremacro.models.smm import (
    MomentVector,
    load_empirical_moments,
    load_empirical_moments_monthly,
    smm_objective,
    theoretical_moments,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_params() -> DMPParameters:
    """Canonical calibration matching the DMP tests."""
    return DMPParameters(
        beta_bar=0.996, alpha=0.5, eta_b=0.5, s=0.034, z=0.51,
        y_bar=1.0, c=0.5, mu=0.5, eta=0.067, kappa=0.040,
        rho_sigma=0.7, psi=0.5,
    )


@pytest.fixture
def empty_root() -> Path:
    """Temporary directory with no CSVs (simulates pre-LP run)."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def root_with_logurate(tmp_path: Path) -> Path:
    """Root with logurate_main_lp.csv populated with rate_state rows."""
    tables = tmp_path / "notebooks" / "output_tables"
    tables.mkdir(parents=True)
    h_ts = list(range(9))
    rows = {
        "state": ["rate_state"] * 9,
        "h": h_ts,
        "beta_H": [0.1 * i for i in h_ts],
        "beta_L": [-0.05 * i for i in h_ts],
        "se_H": [0.01 + 0.002 * i for i in h_ts],
        "se_L": [0.02 + 0.001 * i for i in h_ts],
    }
    pd.DataFrame(rows).to_csv(tables / "logurate_main_lp.csv", index=False)
    return tmp_path


@pytest.fixture
def root_full(tmp_path: Path) -> Path:
    """Root with all three CSVs: logurate, vacancies, and XS quartile."""
    tables = tmp_path / "notebooks" / "output_tables"
    tables.mkdir(parents=True)

    # logurate
    h9 = list(range(9))
    pd.DataFrame({
        "state": ["rate_state"] * 9, "h": h9,
        "beta_H": [0.1 * i for i in h9], "beta_L": [-0.05 * i for i in h9],
        "se_H": [0.01] * 9, "se_L": [0.02] * 9,
    }).to_csv(tables / "logurate_main_lp.csv", index=False)

    # vacancies
    pd.DataFrame({
        "h": h9,
        "beta_H": [0.3 * i for i in h9], "beta_L": [-0.1 * i for i in h9],
        "se_H": [0.01] * 9, "se_L": [0.02] * 9,
    }).to_csv(tables / "36_lp_state_dep_log_vacancies.csv", index=False)

    # cross-section quartile
    rows = []
    for regime in ["tight", "loose"]:
        for h in [6, 12, 18]:
            for q in [1, 2, 3, 4]:
                rows.append({"h_months": h, "regime": regime,
                              "aioe_quartile": q, "beta": 0.01 * q, "se": 0.005})
    pd.DataFrame(rows).to_csv(
        tables / "41_state_bartik_urate_quartile.csv", index=False
    )
    return tmp_path


@pytest.fixture
def root_with_monthly(tmp_path: Path) -> Path:
    """Root with paper_monthly_lp.csv at quarterly-spaced horizons 0..24."""
    tables = tmp_path / "notebooks" / "output_tables"
    tables.mkdir(parents=True)
    h_months = list(range(0, 25, 3))
    pd.DataFrame({
        "h": h_months,
        "beta_H": [0.1 * i for i in range(len(h_months))],
        "beta_L": [-0.05 * i for i in range(len(h_months))],
        "se_H": [0.01] * len(h_months),
        "se_L": [0.02] * len(h_months),
    }).to_csv(tables / "paper_monthly_lp.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# MomentVector — dataclass contract
# ---------------------------------------------------------------------------

class TestMomentVector:
    def test_default_horizons(self):
        """Default horizons_ts=0..8, horizons_xs=(6,12,18)."""
        mv = MomentVector(
            time_series=np.zeros(36),
            cross_section=np.zeros(6),
            weights_time_series=np.ones(36),
            weights_cross_section=np.ones(6),
        )
        assert mv.horizons_ts == tuple(range(9))
        assert mv.horizons_xs == (6, 12, 18)

    def test_custom_horizons(self):
        """Custom horizon tuples are stored verbatim."""
        mv = MomentVector(
            time_series=np.zeros(12),
            cross_section=np.zeros(4),
            weights_time_series=np.ones(12),
            weights_cross_section=np.ones(4),
            horizons_ts=(0, 3, 6),
            horizons_xs=(3, 6),
        )
        assert mv.horizons_ts == (0, 3, 6)
        assert mv.horizons_xs == (3, 6)


# ---------------------------------------------------------------------------
# load_empirical_moments — missing CSVs
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsMissing:
    def test_all_nan_when_no_csvs(self, empty_root):
        """When no CSVs exist all moments are NaN and all weights are 0."""
        mv = load_empirical_moments(empty_root)
        assert mv.time_series.shape == (36,)
        assert mv.cross_section.shape == (6,)
        assert np.all(np.isnan(mv.time_series))
        assert np.all(np.isnan(mv.cross_section))
        assert np.all(mv.weights_time_series == 0)
        assert np.all(mv.weights_cross_section == 0)

    def test_default_horizons_preserved(self, empty_root):
        """MomentVector returned by loader carries the canonical horizon tuples."""
        mv = load_empirical_moments(empty_root)
        assert mv.horizons_ts == tuple(range(9))
        assert mv.horizons_xs == (6, 12, 18)


# ---------------------------------------------------------------------------
# load_empirical_moments — logurate block
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsLogurate:
    def test_beta_values_loaded(self, root_with_logurate):
        """beta_H and beta_L are read from the CSV and placed in the ts vector."""
        mv = load_empirical_moments(root_with_logurate)
        # u_H block: indices 0..8
        for i in range(9):
            assert abs(mv.time_series[i] - 0.1 * i) < 1e-10, f"h={i} beta_H mismatch"
        # u_L block: indices 9..17
        for i in range(9):
            assert abs(mv.time_series[9 + i] - (-0.05 * i)) < 1e-10, f"h={i} beta_L mismatch"

    def test_inverse_variance_weights(self, root_with_logurate):
        """Weights equal 1/se^2 for all non-zero standard errors."""
        mv = load_empirical_moments(root_with_logurate)
        se_H_0 = 0.01  # h=0 se_H
        expected_w0 = 1.0 / se_H_0 ** 2
        assert abs(mv.weights_time_series[0] - expected_w0) < 1e-6

    def test_vacancy_block_is_nan(self, root_with_logurate):
        """When vacancy CSV is absent, vacancy slots 18..35 remain NaN."""
        mv = load_empirical_moments(root_with_logurate)
        assert np.all(np.isnan(mv.time_series[18:]))
        assert np.all(mv.weights_time_series[18:] == 0)

    def test_other_state_filtered_out(self, tmp_path):
        """Rows where state != 'rate_state' are silently ignored."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        pd.DataFrame({
            "state": ["other_state"] * 9, "h": list(range(9)),
            "beta_H": [0.1] * 9, "beta_L": [-0.05] * 9,
            "se_H": [0.01] * 9, "se_L": [0.02] * 9,
        }).to_csv(tables / "logurate_main_lp.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        assert np.all(np.isnan(mv.time_series[:18]))
        assert np.all(mv.weights_time_series[:18] == 0)

    def test_zero_se_yields_zero_weight(self, tmp_path):
        """se_H = 0 yields weight = 0 (no division by zero)."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        pd.DataFrame({
            "state": ["rate_state"] * 9, "h": list(range(9)),
            "beta_H": [0.1] * 9, "beta_L": [-0.05] * 9,
            "se_H": [0.0] * 9,   # all zero
            "se_L": [0.01] * 9,
        }).to_csv(tables / "logurate_main_lp.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        assert np.all(mv.weights_time_series[:9] == 0), "zero se_H should give zero weight"
        assert np.all(mv.weights_time_series[9:18] > 0), "positive se_L should give nonzero weight"

    def test_partial_horizons_only_h0_and_h1(self, tmp_path):
        """When only h=0,1 are present, h=2..8 remain NaN with zero weight."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        pd.DataFrame({
            "state": ["rate_state", "rate_state"], "h": [0, 1],
            "beta_H": [0.1, 0.2], "beta_L": [-0.05, -0.10],
            "se_H": [0.01, 0.02], "se_L": [0.01, 0.02],
        }).to_csv(tables / "logurate_main_lp.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        assert abs(mv.time_series[0] - 0.1) < 1e-10
        assert abs(mv.time_series[1] - 0.2) < 1e-10
        assert np.all(np.isnan(mv.time_series[2:9]))
        assert mv.weights_time_series[0] > 0
        assert np.all(mv.weights_time_series[2:9] == 0)


# ---------------------------------------------------------------------------
# load_empirical_moments — vacancy block
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsVacancies:
    def test_vacancy_block_loaded(self, tmp_path):
        """When vacancy CSV exists, v_H and v_L are populated at indices 18..35."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        h9 = list(range(9))
        pd.DataFrame({
            "h": h9, "beta_H": [0.3 * i for i in h9],
            "beta_L": [-0.1 * i for i in h9],
            "se_H": [0.01] * 9, "se_L": [0.02] * 9,
        }).to_csv(tables / "36_lp_state_dep_log_vacancies.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        # v_H at indices 18..26
        for i in range(9):
            assert abs(mv.time_series[18 + i] - 0.3 * i) < 1e-10
        # v_L at indices 27..35
        for i in range(9):
            assert abs(mv.time_series[27 + i] - (-0.1 * i)) < 1e-10
        assert np.all(mv.weights_time_series[18:] > 0)


# ---------------------------------------------------------------------------
# load_empirical_moments — cross-section block
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsXS:
    def test_q4_minus_q1_computed(self, tmp_path):
        """xs vector = beta_Q4 - beta_Q1 for each (regime, horizon) pair."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = []
        for regime in ["tight", "loose"]:
            for h in [6, 12, 18]:
                for q in [1, 2, 3, 4]:
                    rows.append({"h_months": h, "regime": regime,
                                  "aioe_quartile": q, "beta": 0.01 * q, "se": 0.005})
        pd.DataFrame(rows).to_csv(tables / "41_state_bartik_urate_quartile.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        # Q4 - Q1 = 0.04 - 0.01 = 0.03 for all six entries
        assert np.allclose(mv.cross_section, 0.03)

    def test_xs_inverse_variance_weights(self, tmp_path):
        """xs weights = 1 / (se4^2 + se1^2)."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = []
        se = 0.005
        for regime in ["tight", "loose"]:
            for h in [6, 12, 18]:
                for q in [1, 4]:
                    rows.append({"h_months": h, "regime": regime,
                                  "aioe_quartile": q, "beta": 0.01 * q, "se": se})
        pd.DataFrame(rows).to_csv(tables / "41_state_bartik_urate_quartile.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        expected_w = 1.0 / (2 * se ** 2)
        assert np.allclose(mv.weights_cross_section, expected_w)

    def test_xs_missing_quartile_gives_nan(self, tmp_path):
        """When Q4 row is absent the xs entry stays NaN and weight stays 0."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = []
        for regime in ["tight", "loose"]:
            for h in [6, 12, 18]:
                for q in [1, 2, 3]:  # no Q4
                    rows.append({"h_months": h, "regime": regime,
                                  "aioe_quartile": q, "beta": 0.01 * q, "se": 0.005})
        pd.DataFrame(rows).to_csv(tables / "41_state_bartik_urate_quartile.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        assert np.all(np.isnan(mv.cross_section))
        assert np.all(mv.weights_cross_section == 0)

    def test_xs_zero_se_gives_zero_weight(self, tmp_path):
        """When se=0 for both quartiles, var_diff=0 so weight stays zero."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = []
        for regime in ["tight", "loose"]:
            for h in [6, 12, 18]:
                for q in [1, 4]:
                    rows.append({"h_months": h, "regime": regime,
                                  "aioe_quartile": q, "beta": 0.01 * q, "se": 0.0})
        pd.DataFrame(rows).to_csv(tables / "41_state_bartik_urate_quartile.csv", index=False)
        mv = load_empirical_moments(tmp_path)
        # Values are still computed, only weights are zero
        assert np.allclose(mv.cross_section, 0.03)
        assert np.all(mv.weights_cross_section == 0)


# ---------------------------------------------------------------------------
# load_empirical_moments — all blocks together
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsFull:
    def test_full_load_shapes_and_finiteness(self, root_full):
        """When all CSVs are present everything is finite and shapes are correct."""
        mv = load_empirical_moments(root_full)
        assert mv.time_series.shape == (36,)
        assert mv.cross_section.shape == (6,)
        assert np.all(np.isfinite(mv.time_series))
        assert np.all(np.isfinite(mv.cross_section))
        assert np.all(mv.weights_time_series > 0)
        assert np.all(mv.weights_cross_section > 0)


# ---------------------------------------------------------------------------
# load_empirical_moments_monthly
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMomentsMonthly:
    def test_all_nan_when_no_csv(self, empty_root):
        """Missing paper_monthly_lp.csv gives all-NaN ts and zero weights."""
        mv = load_empirical_moments_monthly(empty_root)
        assert mv.time_series.shape == (36,)
        assert np.all(np.isnan(mv.time_series))
        assert np.all(mv.weights_time_series == 0)

    def test_values_loaded(self, root_with_monthly):
        """u_H and u_L blocks are read at the default quarterly horizons."""
        mv = load_empirical_moments_monthly(root_with_monthly)
        # u_H block (OFF_uH = 0): beta_H at h=0m is 0.0, h=3m is 0.1, ...
        assert abs(mv.time_series[0] - 0.0) < 1e-10   # h=0m  qi=0
        assert abs(mv.time_series[1] - 0.1) < 1e-10   # h=3m  qi=1
        assert abs(mv.time_series[8] - 0.8) < 1e-10   # h=24m qi=8
        # u_L block (OFF_uL = 9)
        assert abs(mv.time_series[9] - 0.0) < 1e-10
        assert abs(mv.time_series[10] - (-0.05)) < 1e-10

    def test_vacancy_blocks_are_nan(self, root_with_monthly):
        """v_H and v_L blocks (indices 18..35) are always NaN with zero weight."""
        mv = load_empirical_moments_monthly(root_with_monthly)
        assert np.all(np.isnan(mv.time_series[18:]))
        assert np.all(mv.weights_time_series[18:] == 0)

    def test_inverse_variance_weights(self, root_with_monthly):
        """Weights are 1/se^2 for positive standard errors."""
        mv = load_empirical_moments_monthly(root_with_monthly)
        # se_H=0.01 -> weight = 1/0.01^2 = 10000
        assert abs(mv.weights_time_series[0] - 10000.0) < 1e-6

    def test_zero_se_gives_zero_weight(self, tmp_path):
        """se_H = se_L = 0 yields zero weights (no division by zero)."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        h_months = list(range(0, 25, 3))
        pd.DataFrame({
            "h": h_months,
            "beta_H": [0.1 * i for i in range(len(h_months))],
            "beta_L": [-0.05 * i for i in range(len(h_months))],
            "se_H": [0.0] * len(h_months),
            "se_L": [0.0] * len(h_months),
        }).to_csv(tables / "paper_monthly_lp.csv", index=False)
        mv = load_empirical_moments_monthly(tmp_path)
        assert np.all(mv.weights_time_series[:18] == 0)
        # Values are still set
        assert not np.all(np.isnan(mv.time_series[:18]))

    def test_missing_se_columns_gives_zero_weight(self, tmp_path):
        """When se_H/se_L columns are absent weights default to zero."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        h_months = list(range(0, 25, 3))
        pd.DataFrame({
            "h": h_months,
            "beta_H": [0.1 * i for i in range(len(h_months))],
            "beta_L": [-0.05 * i for i in range(len(h_months))],
        }).to_csv(tables / "paper_monthly_lp.csv", index=False)
        mv = load_empirical_moments_monthly(tmp_path)
        assert np.all(mv.weights_time_series[:18] == 0)
        assert not np.all(np.isnan(mv.time_series[:18]))

    def test_custom_horizons_shape(self, tmp_path):
        """Custom ts_horizons_months changes the output ts vector length."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        pd.DataFrame({
            "h": list(range(25)),
            "beta_H": [0.1 * i for i in range(25)],
            "beta_L": [-0.05 * i for i in range(25)],
            "se_H": [0.01] * 25, "se_L": [0.02] * 25,
        }).to_csv(tables / "paper_monthly_lp.csv", index=False)
        custom = (0, 6, 12, 18)   # 4 horizons -> 4*4 = 16 elements
        mv = load_empirical_moments_monthly(tmp_path, ts_horizons_months=custom)
        assert mv.time_series.shape == (16,)

    def test_xs_block_loaded_from_quarterly_file(self, tmp_path):
        """Monthly loader shares the XS CSV with the quarterly loader."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = []
        for regime in ["tight", "loose"]:
            for h in [6, 12, 18]:
                for q in [1, 2, 3, 4]:
                    rows.append({"h_months": h, "regime": regime,
                                  "aioe_quartile": q, "beta": 0.01 * q, "se": 0.005})
        pd.DataFrame(rows).to_csv(tables / "41_state_bartik_urate_quartile.csv", index=False)
        mv = load_empirical_moments_monthly(tmp_path)
        assert mv.cross_section.shape == (6,)
        assert np.allclose(mv.cross_section, 0.03)

    def test_sparse_monthly_horizons_gives_nan_for_missing(self, tmp_path):
        """When the CSV only has h=0 and h=6 the other quarterly slots stay NaN."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        pd.DataFrame({
            "h": [0, 6], "beta_H": [0.0, 0.6], "beta_L": [-0.0, -0.3],
            "se_H": [0.01, 0.01], "se_L": [0.02, 0.02],
        }).to_csv(tables / "paper_monthly_lp.csv", index=False)
        mv = load_empirical_moments_monthly(tmp_path)
        # Default horizons 0,3,6,9,12,15,18,21,24 (step 3) -> qi=0 (h=0), qi=2 (h=6)
        assert abs(mv.time_series[0] - 0.0) < 1e-10   # qi=0 present
        assert np.isnan(mv.time_series[1])              # qi=1 (h=3) absent
        assert abs(mv.time_series[2] - 0.6) < 1e-10   # qi=2 present
        assert np.isnan(mv.time_series[3])              # qi=3 (h=9) absent

    def test_monthly_xs_missing_quartile_gives_nan(self, tmp_path):
        """When Q4 is absent from the monthly xs CSV, xs entries remain NaN."""
        tables = tmp_path / "notebooks" / "output_tables"
        tables.mkdir(parents=True)
        rows = [{"h_months": h, "regime": r, "aioe_quartile": q,
                 "beta": 0.01 * q, "se": 0.005}
                for r in ["tight", "loose"]
                for h in [6, 12, 18]
                for q in [1, 2, 3]]  # Q4 missing
        pd.DataFrame(rows).to_csv(
            tables / "41_state_bartik_urate_quartile.csv", index=False
        )
        mv = load_empirical_moments_monthly(tmp_path)
        assert np.all(np.isnan(mv.cross_section))
        assert np.all(mv.weights_cross_section == 0)


# ---------------------------------------------------------------------------
# theoretical_moments
# ---------------------------------------------------------------------------

class TestTheoreticalMoments:
    def test_output_shapes(self, baseline_params):
        """ts=(36,), xs=(6,), w_ts=(36,), w_xs=(6,) always."""
        mv = theoretical_moments(baseline_params)
        assert mv.time_series.shape == (36,)
        assert mv.cross_section.shape == (6,)
        assert mv.weights_time_series.shape == (36,)
        assert mv.weights_cross_section.shape == (6,)

    def test_all_finite(self, baseline_params):
        """All values and weights are finite for a valid calibration."""
        mv = theoretical_moments(baseline_params)
        assert np.all(np.isfinite(mv.time_series))
        assert np.all(np.isfinite(mv.cross_section))

    def test_uniform_weights(self, baseline_params):
        """Theoretical moments use uniform (all-ones) weights."""
        mv = theoretical_moments(baseline_params)
        assert np.all(mv.weights_time_series == 1.0)
        assert np.all(mv.weights_cross_section == 1.0)

    def test_ts_block_matches_dmp_irf(self, baseline_params):
        """ts = [u_H, u_L, v_H, v_L] matches direct dmp_irf calls."""
        mv = theoretical_moments(baseline_params)
        irf_H = dmp_irf(baseline_params, sigma_shock=1.0, regime="H", horizon=8)
        irf_L = dmp_irf(baseline_params, sigma_shock=1.0, regime="L", horizon=8)
        expected = np.concatenate([irf_H.log_u, irf_L.log_u, irf_H.log_v, irf_L.log_v])
        np.testing.assert_allclose(mv.time_series, expected, atol=1e-12)

    def test_h_regime_raises_urate(self, baseline_params):
        """H-regime (tight) shock raises u: u_H block should be positive."""
        mv = theoretical_moments(baseline_params)
        assert np.all(mv.time_series[:9] >= 0), (
            f"u_H should be >= 0 for tight regime; got {mv.time_series[:9]}"
        )

    def test_l_regime_lowers_urate(self, baseline_params):
        """L-regime (loose) shock lowers u: u_L block should be <= 0."""
        mv = theoretical_moments(baseline_params)
        assert np.all(mv.time_series[9:18] <= 0), (
            f"u_L should be <= 0 for loose regime; got {mv.time_series[9:18]}"
        )

    def test_xs_h_regime_positive(self, baseline_params):
        """Q4-Q1 AIOE difference under H-regime (tight) should be positive."""
        mv = theoretical_moments(baseline_params)
        # xs[0:3] = H-regime at h=2q,4q,6q
        assert np.all(mv.cross_section[:3] > 0), (
            f"H-regime xs should be > 0; got {mv.cross_section[:3]}"
        )

    def test_xs_l_regime_negative(self, baseline_params):
        """Q4-Q1 AIOE difference under L-regime (loose) should be negative."""
        mv = theoretical_moments(baseline_params)
        # xs[3:6] = L-regime
        assert np.all(mv.cross_section[3:] < 0), (
            f"L-regime xs should be < 0; got {mv.cross_section[3:]}"
        )

    def test_aioe_scaling_default(self, baseline_params):
        """Default aioe_quartile_scaling=(-1.5, +1.5) is symmetric."""
        mv_default = theoretical_moments(baseline_params)
        mv_explicit = theoretical_moments(
            baseline_params, aioe_quartile_scaling=(-1.5, 1.5)
        )
        np.testing.assert_allclose(
            mv_default.cross_section, mv_explicit.cross_section, atol=1e-12
        )


# ---------------------------------------------------------------------------
# smm_objective
# ---------------------------------------------------------------------------

class TestSmmObjective:
    def test_zero_at_truth(self, baseline_params):
        """When empirical == theoretical the objective is exactly zero."""
        th = theoretical_moments(baseline_params)
        assert smm_objective(baseline_params, th) == 0.0

    def test_all_nan_empirical_gives_zero(self, baseline_params):
        """All-NaN empirical moments (with zero weights) give objective = 0."""
        emp = MomentVector(
            time_series=np.full(36, np.nan),
            cross_section=np.full(6, np.nan),
            weights_time_series=np.zeros(36),
            weights_cross_section=np.zeros(6),
        )
        assert smm_objective(baseline_params, emp) == 0.0

    def test_infeasible_params_returns_1e10(self):
        """beta_bar > 1 violates admissibility; smm_objective must return 1e10."""
        p_bad = DMPParameters(
            beta_bar=2.0, alpha=0.5, eta_b=0.5, s=0.034, z=0.51,
            y_bar=1.0, c=0.5, mu=0.5, eta=0.067, kappa=0.040,
        )
        emp = MomentVector(
            time_series=np.zeros(36), cross_section=np.zeros(6),
            weights_time_series=np.ones(36), weights_cross_section=np.ones(6),
        )
        assert smm_objective(p_bad, emp) == 1e10

    def test_positive_with_noise(self, baseline_params):
        """Adding noise to empirical moments yields a strictly positive objective."""
        th = theoretical_moments(baseline_params)
        emp = MomentVector(
            time_series=th.time_series + 0.1,
            cross_section=th.cross_section + 0.1,
            weights_time_series=np.ones(36),
            weights_cross_section=np.ones(6),
        )
        assert smm_objective(baseline_params, emp) > 0.0

    def test_weighted_squared_error_ts(self, baseline_params):
        """ts term = sum_i w_i * (th_i - emp_i)^2 matches manual computation."""
        th = theoretical_moments(baseline_params)
        delta = 0.2
        emp = MomentVector(
            time_series=th.time_series + delta,
            cross_section=th.cross_section,
            weights_time_series=np.ones(36),
            weights_cross_section=np.zeros(6),  # ignore xs term
        )
        expected = 36 * delta ** 2
        assert abs(smm_objective(baseline_params, emp) - expected) < 1e-10

    def test_weighted_squared_error_xs(self, baseline_params):
        """xs term = sum_i w_i * (th_i - emp_i)^2 matches manual computation."""
        th = theoretical_moments(baseline_params)
        delta = 0.3
        emp = MomentVector(
            time_series=th.time_series,
            cross_section=th.cross_section + delta,
            weights_time_series=np.zeros(36),  # ignore ts term
            weights_cross_section=np.ones(6),
        )
        expected = 6 * delta ** 2
        assert abs(smm_objective(baseline_params, emp) - expected) < 1e-10

    def test_combined_ts_and_xs_terms(self, baseline_params):
        """Objective = ts_term + xs_term; both blocks contribute."""
        th = theoretical_moments(baseline_params)
        delta_ts, delta_xs = 0.1, 0.1
        emp = MomentVector(
            time_series=th.time_series + delta_ts,
            cross_section=th.cross_section + delta_xs,
            weights_time_series=np.ones(36),
            weights_cross_section=np.ones(6),
        )
        expected = 36 * delta_ts ** 2 + 6 * delta_xs ** 2
        assert abs(smm_objective(baseline_params, emp) - expected) < 1e-10

    def test_weight_scaling(self, baseline_params):
        """Objective scales linearly with uniform weight multiplier."""
        th = theoretical_moments(baseline_params)
        emp = MomentVector(
            time_series=th.time_series + 0.1,
            cross_section=th.cross_section,
            weights_time_series=np.ones(36),
            weights_cross_section=np.zeros(6),
        )
        obj_1 = smm_objective(baseline_params, emp)
        emp_heavy = MomentVector(
            time_series=emp.time_series,
            cross_section=emp.cross_section,
            weights_time_series=np.ones(36) * 10,
            weights_cross_section=np.zeros(6),
        )
        obj_10 = smm_objective(baseline_params, emp_heavy)
        assert abs(obj_10 / obj_1 - 10.0) < 1e-6

    def test_partial_nan_ignores_nan_entries(self, baseline_params):
        """Empirical NaN with zero weight contributes exactly zero to the sum."""
        th = theoretical_moments(baseline_params)
        # NaN out the second half of the ts block but zero out their weights
        ts_emp = th.time_series.copy()
        ts_emp[18:] = np.nan
        w_ts = np.ones(36)
        w_ts[18:] = 0.0
        emp = MomentVector(
            time_series=ts_emp,
            cross_section=th.cross_section,
            weights_time_series=w_ts,
            weights_cross_section=np.ones(6),
        )
        # At truth the non-NaN entries also have zero error -> total = 0
        assert smm_objective(baseline_params, emp) == 0.0
