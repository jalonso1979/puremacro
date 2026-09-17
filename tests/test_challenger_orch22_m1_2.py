"""Adversarial Empirical Challenge Suite for Milestone 1: Evaluation & Realtime Orchestrator.

Authored by orch22_challenger_m1_2 (Empirical Challenger 2).
Empirically stress-tests:
1. pit_uniformity_test():
   - Nominal size calibration under true uniform PITs (empirical rejection rate near alpha = 0.05).
   - Statistical power under alternatives: biased forecasts, underdispersed intervals,
     overdispersed intervals, and AR(1) autocorrelated forecast errors reliably reject (p < 0.01).
   - Boundary and stress conditions: minimal T=4, T<4 ValueError, constant/boundary PIT clipping,
     ensemble shape validation.
2. realtime_nowcast() orchestrator:
   - Irregular vintage cuts (unsorted dates, intermediate as_of cutoffs, single vintage,
     out-of-range as_of raising ValueError).
   - Missing target indicators (explicit KeyError, automatic candidate fallback, ragged edge extrapolation).
   - Corrupted cache and cartridge records (non-existent, zero-byte, corrupt zip headers).
   - News decomposition mathematical invariant |delta y - sum impact| < 1e-10 across randomized panels.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.realtime import VintagePanel, pack_realtime_cartridge
from puremacro.nowcast.evaluation import (
    FanChartResult,
    PITUniformityResult,
    fan_chart,
    pit_uniformity_test,
)
from puremacro.nowcast.realtime_nowcast import (
    COUNTRY_SPECS,
    RealtimeNowcastResult,
    realtime_nowcast,
)
from puremacro.pocket import CartridgeError


# ============================================================================
# 1. PIT Uniformity & Berkowitz (2001) LR Test Empirical Challenges
# ============================================================================

class TestPITUniformityEmpirical:
    """Empirical challenges for pit_uniformity_test and Berkowitz (2001) LR test."""

    def test_pit_uniformity_nominal_size_monte_carlo(self):
        """Verify empirical rejection rate under H0 is near nominal alpha (alpha=0.05)."""
        rng = np.random.default_rng(2026)
        n_sim = 100
        T = 250
        lr_rejections = 0
        ks_rejections = 0

        for _ in range(n_sim):
            mu = rng.normal(scale=0.5, size=T)
            sigma = rng.uniform(0.8, 1.3, size=T)
            y = mu + sigma * rng.normal(size=T)
            res = pit_uniformity_test(y, mu=mu, sigma=sigma)
            if res.lr_pvalue <= 0.05:
                lr_rejections += 1
            if res.ks_pvalue <= 0.05:
                ks_rejections += 1

        rej_rate_lr = lr_rejections / n_sim
        rej_rate_ks = ks_rejections / n_sim

        # At n=100 and nominal alpha=0.05, 99% binomial acceptance region is approx [0.01, 0.12]
        assert 0.01 <= rej_rate_lr <= 0.12, f"Berkowitz LR rejection rate {rej_rate_lr} diverges from 0.05"
        assert 0.01 <= rej_rate_ks <= 0.12, f"KS rejection rate {rej_rate_ks} diverges from 0.05"

    def test_pit_uniformity_biased_rejection_power(self):
        """Verify Berkowitz test reliably rejects biased forecasts (p < 0.01) with 100% power."""
        rng = np.random.default_rng(2027)
        n_sim = 50
        T = 200

        for bias in (0.5, 1.0, 1.5):
            rejections = 0
            for _ in range(n_sim):
                y = rng.normal(size=T)
                res = pit_uniformity_test(y, mu=np.full(T, bias), sigma=np.ones(T))
                if res.lr_pvalue < 0.01:
                    rejections += 1
            power = rejections / n_sim
            assert power >= 0.98, f"Insufficient power {power} against bias={bias}"

    def test_pit_uniformity_underdispersed_rejection_power(self):
        """Verify Berkowitz test reliably rejects underdispersed forecasts (intervals too narrow)."""
        rng = np.random.default_rng(2028)
        n_sim = 50
        T = 200

        for scale_factor in (0.5, 0.7):
            rejections = 0
            for _ in range(n_sim):
                y = rng.normal(scale=1.0, size=T)
                res = pit_uniformity_test(y, mu=np.zeros(T), sigma=np.full(T, scale_factor))
                if res.lr_pvalue < 0.01:
                    rejections += 1
            power = rejections / n_sim
            assert power == 1.0, f"Power against underdispersion {scale_factor} must be 1.0, got {power}"

    def test_pit_uniformity_overdispersed_rejection_power(self):
        """Verify Berkowitz test reliably rejects overdispersed forecasts (intervals too wide)."""
        rng = np.random.default_rng(2029)
        n_sim = 50
        T = 200

        for scale_factor in (1.5, 2.0):
            rejections = 0
            for _ in range(n_sim):
                y = rng.normal(scale=1.0, size=T)
                res = pit_uniformity_test(y, mu=np.zeros(T), sigma=np.full(T, scale_factor))
                if res.lr_pvalue < 0.01:
                    rejections += 1
            power = rejections / n_sim
            assert power >= 0.98, f"Power against overdispersion {scale_factor} must be >= 0.98, got {power}"

    def test_pit_uniformity_autocorrelated_errors_rejection_power(self):
        """Verify Berkowitz test reliably rejects AR(1) autocorrelated errors (p < 0.01)."""
        rng = np.random.default_rng(2030)
        n_sim = 50
        T = 250

        for rho in (0.5, 0.7, 0.9):
            rejections = 0
            for _ in range(n_sim):
                u = np.zeros(T)
                for t in range(1, T):
                    u[t] = rho * u[t - 1] + rng.normal(scale=np.sqrt(1.0 - rho**2))
                res = pit_uniformity_test(u, mu=np.zeros(T), sigma=np.ones(T))
                if res.lr_pvalue < 0.01:
                    rejections += 1
            power = rejections / n_sim
            assert power == 1.0, f"Power against autocorrelation rho={rho} must be 1.0, got {power}"

    def test_pit_uniformity_edge_cases_and_boundaries(self):
        """Verify minimum sample size, boundary clipping, and input shape errors."""
        # T < 4 raises informative ValueError
        with pytest.raises(ValueError, match="Need at least 4 observations"):
            pit_uniformity_test([1.0, 2.0, 3.0], mu=[0, 0, 0], sigma=[1, 1, 1])

        # T = 4 minimal sample succeeds
        res_min = pit_uniformity_test([1.0, 2.0, 0.5, 1.2], mu=[1, 1, 1, 1], sigma=[1, 1, 1, 1])
        assert res_min.n_obs == 4
        assert res_min.df == 3
        assert not np.isnan(res_min.lr_pvalue)

        # Boundary PIT values: 0.0 and 1.0 are clipped to eps / 1-eps
        # On constant degenerate arrays (std=0), KS test rejects with p < 1e-50 and is_uniform is False
        res_zero = pit_uniformity_test([0.0] * 10, ensemble_or_pit=[0.0] * 10)
        assert res_zero.is_uniform is False
        assert res_zero.ks_pvalue < 1e-10

        # Non-positive sigma handled safely by clipping
        res_sig0 = pit_uniformity_test([1.0, 2.0, 3.0, 4.0], mu=[0, 0, 0, 0], sigma=[0.0, -0.5, 1e-15, 1.0])
        assert not np.isnan(res_sig0.lr_stat)

        # Invalid ensemble dimensions
        with pytest.raises(ValueError, match="Invalid ensemble dimension"):
            pit_uniformity_test([1.0, 2.0, 3.0, 4.0], ensemble_or_pit=np.zeros((4, 5, 2)))


# ============================================================================
# 2. Real-Time Nowcasting Orchestrator Stress Tests
# ============================================================================

def _create_ragged_panel(
    vintages: list[str | pd.Timestamp],
    country: str = "MEX",
    variables: tuple[str, ...] = ("gdp", "igae", "ind_prod"),
    n_periods: int = 24,
) -> VintagePanel:
    """Construct multi-vintage panel with ragged edges and historical revisions."""
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    f = np.cumsum(rng.normal(scale=0.25, size=n_periods))

    rows = []
    for v_idx, v in enumerate(vintages):
        v_stamp = pd.Timestamp(v)
        for var in variables:
            load = 1.0 if var == "gdp" else 0.8
            y = 2.5 + load * f + rng.normal(scale=0.15, size=n_periods)

            for t_idx, d in enumerate(dates):
                val = float(y[t_idx])
                # Ragged edge: target gdp lags by 2 months in vintage 0, 1 month in vintage 1
                if var == "gdp" and t_idx >= n_periods - (2 - min(v_idx, 1)):
                    val = np.nan
                # Secondary variable lags by 1 month in vintage 0
                if var == "ind_prod" and v_idx == 0 and t_idx >= n_periods - 1:
                    val = np.nan
                # Revision at t=8 in later vintages
                if v_idx > 0 and t_idx == 8 and var == "igae":
                    val += 0.4

                if not np.isnan(val):
                    rows.append({
                        "country": country.upper(),
                        "variable": var,
                        "date": d,
                        "vintage": v_stamp,
                        "value": val,
                        "provider": "mock",
                        "series_id": f"MOCK_{var}",
                        "units": "rate",
                    })

    return VintagePanel(pd.DataFrame(rows))


class TestRealtimeNowcastStress:
    """Stress tests for realtime_nowcast() orchestrator."""

    def test_irregular_vintage_cuts_and_as_of(self):
        """Verify handling of unsorted vintages, intermediate as_of, and single-vintage panels."""
        # Unsorted vintage dates
        v_dates = ["2023-08-15", "2023-01-10", "2023-04-20", "2023-02-05"]
        vp = _create_ragged_panel(v_dates, country="MEX")

        res_latest = realtime_nowcast("MEX", panel=vp)
        assert pd.Timestamp(res_latest.latest_vintage) == pd.Timestamp("2023-08-15")
        assert not np.isnan(res_latest.nowcast)

        # Intermediate as_of cutoff
        res_inter = realtime_nowcast("MEX", panel=vp, as_of="2023-03-01")
        assert pd.Timestamp(res_inter.latest_vintage) == pd.Timestamp("2023-02-05")

        # Out-of-bounds as_of before any vintage raises ValueError
        with pytest.raises(ValueError, match="No vintages found as of 2020-01-01"):
            realtime_nowcast("MEX", panel=vp, as_of="2020-01-01")

        # Single vintage panel: nowcast computed, news decomposition gracefully omitted
        vp_single = _create_ragged_panel(["2023-01-10"], country="MEX")
        res_single = realtime_nowcast("MEX", panel=vp_single)
        assert not np.isnan(res_single.nowcast)
        assert res_single.news_decomposition is None

    def test_missing_target_indicators(self):
        """Verify error handling on non-existent target and automatic fallback."""
        vp = _create_ragged_panel(["2023-01-01", "2023-02-01"], country="MEX")

        # Non-existent target variable explicitly requested raises KeyError
        with pytest.raises(KeyError, match="Target variable 'missing_var' not found"):
            realtime_nowcast("MEX", target_variable="missing_var", panel=vp)

        # Panel with non-standard variable names falls back to first column
        vp_alt = _create_ragged_panel(
            ["2023-01-01", "2023-02-01"],
            country="MEX",
            variables=("alpha_series", "beta_series"),
        )
        res_fallback = realtime_nowcast("MEX", target_variable=None, panel=vp_alt)
        assert res_fallback.target_variable == "alpha_series"
        assert not np.isnan(res_fallback.nowcast)

    def test_corrupted_cartridge_handling(self, tmp_path):
        """Verify robust failure semantics when cartridges are missing or corrupted."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            realtime_nowcast("MEX", cartridge_path=tmp_path / "non_existent.pmz")

        # 0-byte cartridge
        empty_cart = tmp_path / "empty.pmz"
        empty_cart.write_bytes(b"")
        with pytest.raises(CartridgeError, match="not a cartridge"):
            realtime_nowcast("MEX", cartridge_path=empty_cart)

        # Truncated zip file
        corrupt_cart = tmp_path / "corrupt.pmz"
        corrupt_cart.write_bytes(b"PK\x03\x04\x00\x00\x00\x00" + b"\x42" * 40)
        with pytest.raises(CartridgeError, match="not a cartridge"):
            realtime_nowcast("MEX", cartridge_path=corrupt_cart)

    def test_multi_country_panel_routing(self):
        """Verify panel with multiple countries correctly routes to requested country."""
        vp_mex = _create_ragged_panel(["2023-01-01", "2023-02-01"], country="MEX")
        vp_bra = _create_ragged_panel(["2023-01-01", "2023-02-01"], country="BRA")
        df_combined = pd.concat([vp_mex.df, vp_bra.df], ignore_index=True)
        vp_multi = VintagePanel(df_combined)

        res_mex = realtime_nowcast("MEX", panel=vp_multi)
        assert res_mex.country == "MEX"
        assert res_mex.palette == "banxico"

        res_bra = realtime_nowcast("BRA", panel=vp_multi)
        assert res_bra.country == "BRA"
        assert res_bra.palette == "bcb"

    def test_news_decomposition_invariant_monte_carlo(self):
        """Verify |delta y - sum impact| < 1e-10 across 20 randomized vintage panels."""
        rng = np.random.default_rng(777)
        dates = pd.date_range("2021-01-01", periods=20, freq="MS")
        v1 = pd.Timestamp("2022-10-01")
        v2 = pd.Timestamp("2022-11-01")

        for trial in range(20):
            rows = []
            f = np.cumsum(rng.normal(scale=0.3, size=20))
            for var in ("gdp", "ip", "sales"):
                load = rng.uniform(0.6, 1.4)
                y = 2.0 + load * f + rng.normal(scale=0.1, size=20)
                for t_idx, d in enumerate(dates):
                    # v1
                    val_v1 = float(y[t_idx])
                    if var == "gdp" and t_idx >= 18:
                        val_v1 = np.nan
                    if not np.isnan(val_v1):
                        rows.append({
                            "country": "MEX", "variable": var, "date": d,
                            "vintage": v1, "value": val_v1, "provider": "mock",
                            "series_id": f"MOCK_{var}", "units": "rate",
                        })
                    # v2: revision and release
                    val_v2 = float(y[t_idx])
                    if t_idx == 10:
                        val_v2 += rng.normal(scale=0.3)
                    if not np.isnan(val_v2):
                        rows.append({
                            "country": "MEX", "variable": var, "date": d,
                            "vintage": v2, "value": val_v2, "provider": "mock",
                            "series_id": f"MOCK_{var}", "units": "rate",
                        })

            vp = VintagePanel(pd.DataFrame(rows))
            res = realtime_nowcast("MEX", target_variable="gdp", panel=vp)
            assert res.news_decomposition is not None
            err = res.news_decomposition.decomposition_error
            assert err < 1e-10, f"Trial {trial} violated analytical identity: err={err:.3e}"
