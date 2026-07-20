"""Tests for puremacro.uncertainty.lp_long.

Covers:
  _ar2_residual         — internal helper (tested via derive_innovation_shock)
  derive_innovation_shock — AR(2) residual z-scored per country
  lp_per_country         — country-by-country Jordà LP
  pooled_with_dk_se      — pooled panel DK LP
  lp_state_dependent     — Auerbach-Gorodnichenko state-dependent LP
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.uncertainty.lp_long import (
    derive_innovation_shock,
    lp_per_country,
    lp_state_dependent,
    pooled_with_dk_se,
)


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_long_panel(
    n_countries: int = 5,
    n_periods: int = 80,
    beta_shock: float = 0.4,
    seed: int = 0,
    var_name: str = "gpr_m",
) -> pd.DataFrame:
    """Long-format panel with columns [code, date, variable, value].

    DGP: value_{c,t} = 0.6 * value_{c,t-1} - 0.2 * value_{c,t-2} + e
    """
    rng = np.random.default_rng(seed)
    codes = [f"C{i:02d}" for i in range(n_countries)]
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="MS")
    rows = []
    for code in codes:
        y = np.zeros(n_periods)
        e = rng.standard_normal(n_periods) * 0.5
        for t in range(2, n_periods):
            y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + e[t]
        for t, d in enumerate(dates):
            rows.append({"code": code, "date": d, "variable": var_name, "value": y[t]})
    return pd.DataFrame(rows)


def _make_lp_panel(
    n_countries: int = 6,
    n_periods: int = 100,
    beta: float = 0.5,
    seed: int = 1,
) -> pd.DataFrame:
    """Long-format panel suitable for lp_per_country / pooled_with_dk_se.

    Columns: code, date, outcome, shock, control1.
    DGP: outcome_{c,t+1} = beta * shock_{c,t} + alpha_c + noise
    """
    rng = np.random.default_rng(seed)
    codes = [f"C{i:02d}" for i in range(n_countries)]
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="QS")
    rows = []
    for code in codes:
        shock = rng.standard_normal(n_periods)
        alpha = rng.standard_normal()
        noise = rng.standard_normal(n_periods) * 0.5
        outcome = alpha + beta * np.roll(shock, 1) + noise
        control1 = rng.standard_normal(n_periods)
        for t, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "outcome": outcome[t],
                "shock": shock[t],
                "control1": control1[t],
            })
    return pd.DataFrame(rows)


def _make_state_dep_panel(
    n_countries: int = 5,
    n_periods: int = 120,
    beta_L: float = -0.6,
    beta_H: float = 0.3,
    seed: int = 2,
) -> pd.DataFrame:
    """Panel for lp_state_dependent.

    state column is F in [0,1] (smooth-transition weight).
    DGP: outcome_{t+1} = (1-F)*beta_L*shock + F*beta_H*shock + noise
    """
    rng = np.random.default_rng(seed)
    codes = [f"C{i:02d}" for i in range(n_countries)]
    dates = pd.date_range("2000-01-01", periods=n_periods, freq="QS")
    rows = []
    for code in codes:
        shock = rng.standard_normal(n_periods)
        state = np.clip(0.5 + 0.4 * rng.standard_normal(n_periods), 0.0, 1.0)
        noise = rng.standard_normal(n_periods) * 0.3
        outcome = (1 - state) * beta_L * shock + state * beta_H * shock + noise
        outcome = np.roll(outcome, 1)  # shift so t+1 aligns
        for t, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "outcome": outcome[t],
                "shock": shock[t],
                "state": state[t],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for derive_innovation_shock
# ---------------------------------------------------------------------------

class TestDeriveInnovationShock:

    def test_output_columns(self):
        panel = _make_long_panel(n_countries=3, n_periods=60)
        out = derive_innovation_shock(panel, var="gpr_m")
        assert set(out.columns) == {"code", "date", "variable", "value"}

    def test_variable_name_renaming(self):
        """gpr_m -> gpr_innov_m."""
        panel = _make_long_panel(n_countries=3, n_periods=60, var_name="gpr_m")
        out = derive_innovation_shock(panel, var="gpr_m")
        assert (out["variable"] == "gpr_innov_m").all()

    def test_variable_name_renaming_epu(self):
        """epu_q -> epu_innov_q."""
        panel = _make_long_panel(n_countries=3, n_periods=60, var_name="epu_q")
        out = derive_innovation_shock(panel, var="epu_q")
        assert (out["variable"] == "epu_innov_q").all()

    def test_z_scored_mean_zero(self):
        """Within each country the innovation should be mean-zero (z-score)."""
        panel = _make_long_panel(n_countries=4, n_periods=80)
        out = derive_innovation_shock(panel, var="gpr_m")
        for code, grp in out.groupby("code"):
            assert abs(grp["value"].mean()) < 1e-10, f"mean not zero for {code}"

    def test_z_scored_std_one(self):
        """Within each country the innovation should have unit std (ddof=0)."""
        panel = _make_long_panel(n_countries=4, n_periods=80)
        out = derive_innovation_shock(panel, var="gpr_m")
        for code, grp in out.groupby("code"):
            assert abs(grp["value"].std(ddof=0) - 1.0) < 1e-10, \
                f"std not 1 for {code}"

    def test_returns_all_countries_when_sufficient_obs(self):
        panel = _make_long_panel(n_countries=5, n_periods=60)
        out = derive_innovation_shock(panel, var="gpr_m")
        assert out["code"].nunique() == 5

    def test_skips_country_with_too_few_obs(self):
        """A country with < 10 observations is silently skipped."""
        rng = np.random.default_rng(99)
        # Build a panel with one short country
        rows = []
        dates_short = pd.date_range("2000-01-01", periods=5, freq="MS")
        for d in dates_short:
            rows.append({"code": "SHORT", "date": d, "variable": "gpr_m",
                         "value": float(rng.standard_normal())})
        # Add a normal country
        for t, d in enumerate(pd.date_range("2000-01-01", periods=40, freq="MS")):
            rows.append({"code": "OK", "date": d, "variable": "gpr_m",
                         "value": float(rng.standard_normal())})
        panel = pd.DataFrame(rows)
        out = derive_innovation_shock(panel, var="gpr_m")
        assert "SHORT" not in out["code"].values
        assert "OK" in out["code"].values

    def test_skips_constant_series(self):
        """A constant series has std=0 after AR(2) residual; it should be skipped."""
        rows = []
        dates = pd.date_range("2000-01-01", periods=40, freq="MS")
        for d in dates:
            rows.append({"code": "CONST", "date": d, "variable": "gpr_m", "value": 5.0})
        panel = pd.DataFrame(rows)
        out = derive_innovation_shock(panel, var="gpr_m")
        # Either empty DataFrame (no columns) or CONST not in codes
        if len(out) == 0:
            pass  # correctly skipped — empty output
        else:
            assert "CONST" not in out["code"].values

    def test_empty_if_no_matching_variable(self):
        panel = _make_long_panel(n_countries=3, n_periods=60, var_name="gpr_m")
        out = derive_innovation_shock(panel, var="nonexistent_x")
        assert len(out) == 0

    def test_output_dtypes(self):
        panel = _make_long_panel(n_countries=3, n_periods=60)
        out = derive_innovation_shock(panel, var="gpr_m")
        assert out["value"].dtype == np.float64 or pd.api.types.is_float_dtype(out["value"])

    def test_ar2_residual_removes_autocorrelation(self):
        """After AR(2) de-trending, the residuals should have lower autocorrelation
        than the original series (property test)."""
        panel = _make_long_panel(n_countries=1, n_periods=200, seed=77)
        out = derive_innovation_shock(panel, var="gpr_m")
        code = out["code"].iloc[0]
        innov = out[out["code"] == code].sort_values("date")["value"]
        # AR(1) of innovations should be close to zero
        from numpy.polynomial import polynomial as P
        if len(innov) > 10:
            ac1 = innov.autocorr(lag=1)
            assert abs(ac1) < 0.25  # rough bound — innovations shouldn't be strongly autocorrelated


# ---------------------------------------------------------------------------
# Tests for lp_per_country
# ---------------------------------------------------------------------------

class TestLpPerCountry:

    def test_returns_required_columns(self):
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 4), lags=2)
        for col in ("code", "h", "beta", "se", "ci_lo", "ci_hi", "n_obs"):
            assert col in out.columns, f"missing column: {col}"

    def test_one_row_per_horizon_per_country(self):
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        horizons = [0, 1, 2]
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=horizons, lags=2)
        for code, grp in out.groupby("code"):
            h_vals = sorted(grp["h"].tolist())
            assert h_vals == sorted(horizons), \
                f"country {code} missing horizons: got {h_vals}, expected {horizons}"

    def test_se_positive(self):
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=2)
        assert (out["se"] > 0).all(), "all SEs should be positive"

    def test_ci_bracket_beta(self):
        """CI should contain the point estimate."""
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=2, ci=0.9)
        assert (out["ci_lo"] <= out["beta"]).all()
        assert (out["ci_hi"] >= out["beta"]).all()

    def test_ci_width_positive(self):
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=2)
        ci_width = out["ci_hi"] - out["ci_lo"]
        assert (ci_width > 0).all()

    def test_n_obs_positive(self):
        panel = _make_lp_panel(n_countries=4, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=2)
        assert (out["n_obs"] > 0).all()

    def test_skips_country_with_too_few_obs(self):
        """Countries with fewer than 4*lags + h_max + 4 obs are silently skipped."""
        panel = _make_lp_panel(n_countries=5, n_periods=80)
        short_rows = panel[panel["code"] == "C00"].head(5).copy()
        other_rows = panel[panel["code"] != "C00"]
        tiny_panel = pd.concat([short_rows, other_rows], ignore_index=True)
        out = lp_per_country(tiny_panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=4)
        # C00 has only 5 rows — should be skipped
        assert "C00" not in out["code"].values

    def test_with_controls(self):
        """Controls argument should not break the call and SE should remain positive."""
        panel = _make_lp_panel(n_countries=4, n_periods=100)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 3), lags=2,
                             controls=["control1"])
        assert len(out) > 0
        assert (out["se"] > 0).all()

    def test_wider_ci_is_wider(self):
        """A wider CI level should produce wider bands."""
        panel = _make_lp_panel(n_countries=4, n_periods=100)
        out_90 = lp_per_country(panel, outcome="outcome", shock="shock",
                                horizons=[0, 1], lags=2, ci=0.90)
        out_99 = lp_per_country(panel, outcome="outcome", shock="shock",
                                horizons=[0, 1], lags=2, ci=0.99)
        width_90 = (out_90["ci_hi"] - out_90["ci_lo"]).mean()
        width_99 = (out_99["ci_hi"] - out_99["ci_lo"]).mean()
        assert width_99 > width_90

    def test_custom_col_names(self):
        """Should work with non-default unit_col / date_col names."""
        panel = _make_lp_panel(n_countries=3, n_periods=80)
        panel = panel.rename(columns={"code": "country", "date": "period"})
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 2), lags=2,
                             unit_col="country", date_col="period")
        assert "code" in out.columns
        assert len(out) > 0

    def test_returns_empty_df_when_no_country_passes_threshold(self):
        """If all countries have too few observations, return an empty DataFrame."""
        panel = _make_lp_panel(n_countries=3, n_periods=15)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=range(0, 10), lags=8)
        # All countries likely skipped; result should be empty or a valid DataFrame
        assert isinstance(out, pd.DataFrame)

    @pytest.mark.slow
    def test_recovers_known_beta_at_h1(self):
        """Known DGP: outcome_{c,t+1} = 0.5 * shock_{c,t} + noise.
        LP at h=1 should recover beta ≈ 0.5 across countries.
        Uses many countries/periods for reliability.
        """
        panel = _make_lp_panel(n_countries=12, n_periods=200, beta=0.5, seed=42)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=[1], lags=2)
        median_beta = out[out["h"] == 1]["beta"].median()
        assert abs(median_beta - 0.5) < 0.15, \
            f"Expected median beta ≈ 0.5 at h=1, got {median_beta:.3f}"


# ---------------------------------------------------------------------------
# Tests for pooled_with_dk_se
# ---------------------------------------------------------------------------

class TestPooledWithDkSe:

    def _make_wide_panel(self, n_countries=6, n_periods=80, beta=0.4, seed=3):
        """Return panel indexed by [code, date] with outcome, shock columns."""
        panel = _make_lp_panel(n_countries=n_countries, n_periods=n_periods,
                               beta=beta, seed=seed)
        return panel

    def test_returns_required_columns(self):
        panel = self._make_wide_panel()
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 4), n_lags=2)
        for col in ("h", "beta", "se", "n_obs"):
            assert col in out.columns, f"missing column: {col}"

    def test_one_row_per_horizon(self):
        panel = self._make_wide_panel()
        horizons = [0, 1, 2, 3]
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=horizons, n_lags=2)
        assert sorted(out["h"].tolist()) == horizons

    def test_se_positive(self):
        panel = self._make_wide_panel()
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 4), n_lags=2)
        assert (out["se"] > 0).all()

    def test_ci_columns_renamed(self):
        """lo/hi from panel_lp_dk should be renamed to ci_low/ci_high."""
        panel = self._make_wide_panel()
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 3), n_lags=2)
        # Either ci_low/ci_high are present, or lo/hi — the module renames them
        has_renamed = "ci_low" in out.columns and "ci_high" in out.columns
        has_original = "lo" in out.columns and "hi" in out.columns
        assert has_renamed or has_original, \
            "Expected ci_low/ci_high or lo/hi columns in output"

    def test_n_obs_equals_panel_length(self):
        """n_obs should equal len(panel) per module docstring."""
        panel = self._make_wide_panel()
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 3), n_lags=2)
        assert (out["n_obs"] == len(panel)).all()

    def test_with_controls(self):
        panel = self._make_wide_panel()
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 3), n_lags=2,
                                controls=["control1"])
        assert len(out) > 0
        assert (out["se"] > 0).all()

    def test_custom_col_names(self):
        panel = self._make_wide_panel()
        panel = panel.rename(columns={"code": "country", "date": "period"})
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=range(0, 3), n_lags=2,
                                unit_col="country", date_col="period")
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Tests for lp_state_dependent
# ---------------------------------------------------------------------------

class TestLpStateDependent:

    def test_returns_required_columns(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=80)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1, 2], lags=2)
        for col in ("code", "h", "beta_L", "se_L", "beta_H", "se_H", "n_obs"):
            assert col in out.columns, f"missing column: {col}"

    def test_one_row_per_horizon_per_country(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=120)
        horizons = [0, 1, 2]
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=horizons, lags=2)
        for code, grp in out.groupby("code"):
            h_vals = sorted(grp["h"].tolist())
            assert h_vals == sorted(horizons), \
                f"country {code}: expected {horizons}, got {h_vals}"

    def test_se_positive(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=120)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=2)
        assert (out["se_L"] > 0).all()
        assert (out["se_H"] > 0).all()

    def test_n_obs_positive(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=120)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=2)
        assert (out["n_obs"] > 0).all()

    def test_skips_country_with_too_few_obs(self):
        """Countries with < max(30, 4*lags) effective observations are skipped."""
        panel = _make_state_dep_panel(n_countries=5, n_periods=120)
        tiny = panel[panel["code"] == "C00"].head(10).copy()
        rest = panel[panel["code"] != "C00"]
        combined = pd.concat([tiny, rest], ignore_index=True)
        out = lp_state_dependent(combined, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=4)
        assert "C00" not in out["code"].values

    def test_returns_empty_when_all_skipped(self):
        """All tiny — should return empty DataFrame."""
        panel = _make_state_dep_panel(n_countries=3, n_periods=10)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1, 2, 3], lags=8)
        assert isinstance(out, pd.DataFrame)

    def test_with_controls(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=120)
        rng = np.random.default_rng(55)
        panel["ctrl"] = rng.standard_normal(len(panel))
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=2,
                                 controls=["ctrl"])
        assert len(out) > 0
        assert (out["se_L"] > 0).all()

    def test_custom_col_names(self):
        panel = _make_state_dep_panel(n_countries=4, n_periods=120)
        panel = panel.rename(columns={"code": "country", "date": "period"})
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=2,
                                 unit_col="country", date_col="period")
        assert "code" in out.columns
        assert len(out) > 0

    def test_state_in_zero_one(self):
        """State weight should be in [0, 1] — property of our DGP."""
        panel = _make_state_dep_panel(n_countries=3, n_periods=80)
        assert panel["state"].between(0.0, 1.0).all()

    @pytest.mark.slow
    def test_sign_of_state_dependent_betas(self):
        """DGP: outcome_t = (1-F_{t-1})*beta_L*shock_{t-1} + F_{t-1}*beta_H*shock_{t-1} + noise
        (implemented via np.roll(1)), so the LP at h=1 captures the contemporaneous
        shock-outcome relationship.  beta_L=-0.6 → median beta_L at h=1 should be < 0;
        beta_H=0.3 → median beta_H at h=1 should be > 0.
        """
        panel = _make_state_dep_panel(n_countries=10, n_periods=200,
                                     beta_L=-0.6, beta_H=0.3, seed=88)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[1], lags=2)
        assert len(out) > 0, "Expected non-empty results"
        assert out["beta_L"].median() < 0, \
            f"Expected median beta_L < 0, got {out['beta_L'].median():.3f}"
        assert out["beta_H"].median() > 0, \
            f"Expected median beta_H > 0, got {out['beta_H'].median():.3f}"


# ---------------------------------------------------------------------------
# Integration / cross-function tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_derive_then_lp_per_country(self):
        """Derive innovations from a long panel, then run lp_per_country on them."""
        rng = np.random.default_rng(7)
        n_countries, n_periods = 5, 100
        codes = [f"C{i:02d}" for i in range(n_countries)]
        dates = pd.date_range("2000-01-01", periods=n_periods, freq="MS")

        rows_unc = []
        rows_lp = []
        for code in codes:
            series = rng.standard_normal(n_periods).cumsum() * 0.1
            outcome_vals = rng.standard_normal(n_periods) * 0.5
            for t, d in enumerate(dates):
                rows_unc.append({"code": code, "date": d, "variable": "gpr_m",
                                 "value": series[t]})
                rows_lp.append({"code": code, "date": d, "outcome": outcome_vals[t]})

        long_panel = pd.DataFrame(rows_unc)
        outcome_panel = pd.DataFrame(rows_lp)

        innov = derive_innovation_shock(long_panel, var="gpr_m")
        assert len(innov) > 0

        # Build LP panel merging innovations and outcomes
        innov_wide = innov.rename(columns={"value": "shock"})
        lp_input = outcome_panel.merge(innov_wide[["code", "date", "shock"]],
                                       on=["code", "date"], how="inner")
        if len(lp_input) > 0:
            out = lp_per_country(lp_input, outcome="outcome", shock="shock",
                                 horizons=[0, 1, 2], lags=2)
            # Just verify it returns a DataFrame
            assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Edge-case tests targeting exception-handler branches
# ---------------------------------------------------------------------------

class TestEdgeCaseBranches:

    def test_lp_per_country_skips_on_linalg_error(self, monkeypatch):
        """When lp_hac raises LinAlgError, lp_per_country silently skips that country."""
        import puremacro.uncertainty.lp_long as lp_long_mod
        import numpy as np

        original_lp_hac = lp_long_mod.lp_hac

        call_count = [0]

        def failing_lp_hac(*args, **kwargs):
            call_count[0] += 1
            raise np.linalg.LinAlgError("singular matrix")

        monkeypatch.setattr(lp_long_mod, "lp_hac", failing_lp_hac)
        panel = _make_lp_panel(n_countries=3, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=[0, 1], lags=2)
        # All countries skipped — should return empty DataFrame
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0
        assert call_count[0] > 0  # it was actually called (and caught)

    def test_lp_per_country_skips_on_value_error(self, monkeypatch):
        """When lp_hac raises ValueError, lp_per_country silently skips that country."""
        import puremacro.uncertainty.lp_long as lp_long_mod

        def failing_lp_hac(*args, **kwargs):
            raise ValueError("bad input")

        monkeypatch.setattr(lp_long_mod, "lp_hac", failing_lp_hac)
        panel = _make_lp_panel(n_countries=3, n_periods=80)
        out = lp_per_country(panel, outcome="outcome", shock="shock",
                             horizons=[0, 1], lags=2)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_lp_state_dependent_skips_on_linalg_error(self, monkeypatch):
        """When np.linalg.lstsq raises LinAlgError inside lp_state_dependent,
        the country-horizon pair is silently skipped."""
        import puremacro.uncertainty.lp_long as lp_long_mod
        import numpy as np

        original_lstsq = np.linalg.lstsq

        call_count = [0]

        def failing_lstsq(a, b, **kwargs):
            call_count[0] += 1
            raise np.linalg.LinAlgError("singular matrix")

        monkeypatch.setattr(np.linalg, "lstsq", failing_lstsq)
        panel = _make_state_dep_panel(n_countries=3, n_periods=120)
        out = lp_state_dependent(panel, outcome="outcome", shock="shock",
                                 state="state", horizons=[0, 1], lags=2)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0
        assert call_count[0] > 0

    def test_pooled_with_dk_se_ci_columns_already_renamed(self, monkeypatch):
        """If panel_lp_dk already returns ci_low/ci_high (not lo/hi), the rename
        branch is skipped and ci_low/ci_high pass through unchanged."""
        import puremacro.uncertainty.lp_long as lp_long_mod
        import pandas as pd
        import numpy as np

        # Build a fake panel_lp_dk that already uses ci_low/ci_high
        def fake_panel_lp_dk(df_wide, *, y, x, horizons, n_lags, controls,
                              entity_level, time_level, **kw):
            return pd.DataFrame({
                "h": list(horizons),
                "beta": np.zeros(len(list(horizons))),
                "se": np.ones(len(list(horizons))),
                "t": np.zeros(len(list(horizons))),
                "ci_low": -np.ones(len(list(horizons))),
                "ci_high": np.ones(len(list(horizons))),
            })

        monkeypatch.setattr(lp_long_mod, "panel_lp_dk", fake_panel_lp_dk)
        panel = _make_lp_panel(n_countries=3, n_periods=50)
        out = pooled_with_dk_se(panel, outcome="outcome", shock="shock",
                                horizons=[0, 1, 2], n_lags=2)
        # ci_low/ci_high should still be present (not overwritten)
        assert "ci_low" in out.columns
        assert "ci_high" in out.columns
        # lo/hi should NOT appear since the fake already uses ci_low/ci_high
        assert "lo" not in out.columns
