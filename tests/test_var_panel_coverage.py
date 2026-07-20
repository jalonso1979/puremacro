"""Comprehensive coverage tests for puremacro.var.panel.

Covers:
- PanelSVARResult dataclass fields and types
- mean_group_svar: cholesky identification (with and without ordering kwarg)
- mean_group_svar: bq identification
- mean_group_svar: proxy identification
- mean_group_svar: rigobon identification (all regime_indicator trimming branches)
- mean_group_svar: mean-group aggregation is correct (known-value)
- mean_group_svar: bands are percentiles of country_irfs (known-value)
- mean_group_svar: lo <= mean <= hi everywhere (property)
- mean_group_svar: wider ci -> wider bands (monotonicity)
- mean_group_svar: failing country is substituted with zeros + warning emitted
- mean_group_svar: error on < 2 countries (ValueError)
- mean_group_svar: error on mismatched variable counts (ValueError)
- mean_group_svar: error on unknown identification (KeyError)
- mean_group_svar: stores p, horizon, identification, ci on result
- mean_group_svar: heterogeneous T across countries
- mean_group_svar: n=3 variable system
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.panel import PanelSVARResult, mean_group_svar


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_countries: int = 4,
    T: int = 100,
    n: int = 2,
    rho: float = 0.4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a homogeneous stationary VAR(1) panel."""
    rng = np.random.default_rng(seed)
    A = rho * np.eye(n)
    panel: dict[str, np.ndarray] = {}
    for i in range(n_countries):
        Y = np.zeros((T, n))
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + rng.standard_normal(n) * 0.5
        panel[f"C{i}"] = Y
    return panel


def _make_panel_rich(
    n_countries: int = 6,
    T: int = 120,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Panel from a well-specified 2-var VAR(1) DGP with cross-correlation."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    panel: dict[str, np.ndarray] = {}
    for i in range(n_countries):
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(2)
        panel[f"C{i}"] = Y
    return panel


# ---------------------------------------------------------------------------
# PanelSVARResult dataclass
# ---------------------------------------------------------------------------

class TestPanelSVARResult:
    """Contract tests for the PanelSVARResult dataclass."""

    def _make_result(self, H=4, n=2, N=3) -> PanelSVARResult:
        return PanelSVARResult(
            irf_mean=np.ones((H + 1, n, n)),
            irf_lo=np.zeros((H + 1, n, n)),
            irf_hi=np.ones((H + 1, n, n)) * 2,
            country_irfs=np.ones((N, H + 1, n, n)),
            country_ids=["A", "B", "C"],
            identification="cholesky",
            p=1,
            horizon=H,
            ci=0.9,
        )

    def test_fields_accessible(self):
        res = self._make_result()
        assert res.irf_mean.shape == (5, 2, 2)
        assert res.irf_lo.shape == (5, 2, 2)
        assert res.irf_hi.shape == (5, 2, 2)
        assert res.country_irfs.shape == (3, 5, 2, 2)
        assert res.country_ids == ["A", "B", "C"]
        assert res.identification == "cholesky"
        assert res.p == 1
        assert res.horizon == 4
        assert res.ci == 0.9

    def test_country_ids_list(self):
        """country_ids must be stored as a list."""
        res = self._make_result()
        assert isinstance(res.country_ids, list)

    def test_ci_in_unit_interval(self):
        """Constructed ci should be in (0, 1)."""
        res = self._make_result()
        assert 0.0 < res.ci < 1.0

    def test_horizon_and_shape_consistent(self):
        """irf_mean first dim should equal horizon + 1."""
        res = self._make_result(H=7)
        assert res.irf_mean.shape[0] == res.horizon + 1


# ---------------------------------------------------------------------------
# mean_group_svar: error handling
# ---------------------------------------------------------------------------

class TestMeanGroupSVARErrors:

    def test_raises_on_one_country(self):
        panel = _make_panel(n_countries=1)
        with pytest.raises(ValueError, match="at least 2 countries"):
            mean_group_svar(panel, p=1, horizon=4)

    def test_raises_on_variable_count_mismatch(self):
        panel = _make_panel(n_countries=3)
        panel["D"] = np.zeros((100, 3))  # n=3 vs n=2 for others
        with pytest.raises(ValueError, match="same number of variables"):
            mean_group_svar(panel, p=1, horizon=4)

    def test_raises_on_unknown_identification(self):
        panel = _make_panel()
        with pytest.raises(KeyError, match="Unknown identification"):
            mean_group_svar(panel, p=1, horizon=4, identification="maxshare")

    def test_raises_on_empty_string_identification(self):
        panel = _make_panel()
        with pytest.raises(KeyError):
            mean_group_svar(panel, p=1, horizon=4, identification="")


# ---------------------------------------------------------------------------
# mean_group_svar: cholesky identification
# ---------------------------------------------------------------------------

class TestCholesky:

    def test_output_shape(self):
        panel = _make_panel()
        res = mean_group_svar(panel, p=1, horizon=5, identification="cholesky")
        assert res.irf_mean.shape == (6, 2, 2)
        assert res.irf_lo.shape == (6, 2, 2)
        assert res.irf_hi.shape == (6, 2, 2)
        assert res.country_irfs.shape == (4, 6, 2, 2)

    def test_metadata_stored_correctly(self):
        panel = _make_panel()
        p, H = 2, 6
        res = mean_group_svar(panel, p=p, horizon=H, identification="cholesky", ci=0.8)
        assert res.p == p
        assert res.horizon == H
        assert res.identification == "cholesky"
        assert res.ci == 0.8

    def test_country_ids_preserved(self):
        panel = _make_panel(n_countries=5)
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert len(res.country_ids) == 5
        assert set(res.country_ids) == {f"C{i}" for i in range(5)}

    def test_irf_mean_equals_country_mean(self):
        """Known-value: irf_mean must equal country_irfs.mean(axis=0)."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        expected_mean = res.country_irfs.mean(axis=0)
        assert np.allclose(res.irf_mean, expected_mean)

    def test_bands_are_correct_percentiles(self):
        """Known-value: irf_lo and irf_hi are the correct empirical percentiles."""
        panel = _make_panel_rich()
        ci = 0.9
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky", ci=ci)
        lo_q = (1.0 - ci) / 2.0 * 100.0
        hi_q = (1.0 - (1.0 - ci) / 2.0) * 100.0
        expected_lo = np.percentile(res.country_irfs, lo_q, axis=0)
        expected_hi = np.percentile(res.country_irfs, hi_q, axis=0)
        assert np.allclose(res.irf_lo, expected_lo)
        assert np.allclose(res.irf_hi, expected_hi)

    def test_lo_leq_mean_leq_hi(self):
        """Property: lo <= mean <= hi pointwise."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert np.all(res.irf_lo <= res.irf_mean + 1e-12)
        assert np.all(res.irf_mean <= res.irf_hi + 1e-12)

    def test_wider_ci_gives_wider_bands(self):
        """Monotonicity: wider ci -> wider cross-country bands."""
        panel = _make_panel_rich(n_countries=10)
        res_90 = mean_group_svar(panel, p=1, horizon=4, identification="cholesky", ci=0.9)
        res_50 = mean_group_svar(panel, p=1, horizon=4, identification="cholesky", ci=0.5)
        width_90 = res_90.irf_hi - res_90.irf_lo
        width_50 = res_50.irf_hi - res_50.irf_lo
        assert np.all(width_90 >= width_50 - 1e-10)

    def test_with_ordering_kwarg(self):
        """_cholesky_irf ordering branch: reordering variables should work."""
        panel = _make_panel()
        res_plain = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        res_ordered = mean_group_svar(
            panel, p=1, horizon=4, identification="cholesky", ordering=[1, 0]
        )
        assert res_ordered.irf_mean.shape == (5, 2, 2)
        # Reversed ordering changes the impact matrix so IRFs differ
        assert not np.allclose(res_plain.irf_mean, res_ordered.irf_mean)

    def test_heterogeneous_T_per_country(self):
        """Countries may have different sample lengths."""
        rng = np.random.default_rng(77)
        panel = {}
        for i, T_i in enumerate([60, 80, 100, 120]):
            Y = np.zeros((T_i, 2))
            for t in range(1, T_i):
                Y[t] = 0.4 * Y[t - 1] + rng.standard_normal(2) * 0.5
            panel[f"C{i}"] = Y
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert res.irf_mean.shape == (5, 2, 2)

    def test_three_variable_system(self):
        panel = _make_panel(n=3, n_countries=3, T=100)
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert res.irf_mean.shape == (5, 3, 3)
        assert res.country_irfs.shape == (3, 5, 3, 3)

    def test_minimum_two_countries(self):
        """With exactly 2 countries the function runs successfully."""
        panel = _make_panel(n_countries=2)
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert res.country_irfs.shape[0] == 2

    def test_failing_country_substituted_with_zeros_and_warns(self):
        """When a country's identification fails, zeros are substituted and a
        UserWarning is emitted."""
        rng = np.random.default_rng(0)
        panel: dict[str, np.ndarray] = {}
        # C0: all zeros -> Sigma singular -> LinAlgError in safe_cholesky
        panel["C0"] = np.zeros((80, 2))
        # C1, C2: normal data
        for i in range(1, 3):
            Y = np.zeros((80, 2))
            for t in range(1, 80):
                Y[t] = 0.4 * Y[t - 1] + rng.standard_normal(2) * 0.5
            panel[f"C{i}"] = Y

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")

        # At least one warning about C0
        assert any("C0" in str(w.message) for w in caught), \
            "Expected a warning about C0 identification failure"
        # Country 0 IRF should be zeros
        assert np.all(res.country_irfs[res.country_ids.index("C0")] == 0.0)
        # Result should still have correct shape
        assert res.irf_mean.shape == (5, 2, 2)


# ---------------------------------------------------------------------------
# mean_group_svar: BQ identification
# ---------------------------------------------------------------------------

class TestBQ:

    def test_output_shape_and_metadata(self):
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=8, identification="bq",
                              permanent_var_idx=0)
        assert res.irf_mean.shape == (9, 2, 2)
        assert res.identification == "bq"
        assert res.horizon == 8

    def test_mean_equals_country_mean(self):
        """Known-value: BQ mean-group is the mean of country BQ IRFs."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="bq",
                              permanent_var_idx=0)
        assert np.allclose(res.irf_mean, res.country_irfs.mean(axis=0))

    def test_lo_leq_mean_leq_hi(self):
        """Property: bands bracket the mean."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="bq",
                              permanent_var_idx=0)
        assert np.all(res.irf_lo <= res.irf_mean + 1e-12)
        assert np.all(res.irf_mean <= res.irf_hi + 1e-12)

    def test_permanent_var_idx_kwarg_forwarded(self):
        """permanent_var_idx=1 should run without error."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="bq",
                              permanent_var_idx=1)
        assert res.irf_mean.shape == (5, 2, 2)


# ---------------------------------------------------------------------------
# mean_group_svar: proxy identification
# ---------------------------------------------------------------------------

class TestProxy:

    def _make_proxy_panel(self, n_countries=4, T=100, seed=7):
        rng = np.random.default_rng(seed)
        panel: dict[str, np.ndarray] = {}
        for i in range(n_countries):
            Y = np.zeros((T, 2))
            for t in range(1, T):
                Y[t] = 0.4 * Y[t - 1] + rng.standard_normal(2) * 0.5
            panel[f"C{i}"] = Y
        instrument = rng.standard_normal(T)
        return panel, instrument

    def test_output_shape_and_metadata(self):
        panel, instrument = self._make_proxy_panel()
        res = mean_group_svar(
            panel, p=1, horizon=4,
            identification="proxy",
            instrument_series=instrument,
        )
        assert res.irf_mean.shape == (5, 2, 2)
        assert res.identification == "proxy"

    def test_mean_equals_country_mean(self):
        """Known-value: proxy mean-group is the mean of country proxy IRFs."""
        panel, instrument = self._make_proxy_panel(n_countries=5)
        res = mean_group_svar(
            panel, p=1, horizon=4,
            identification="proxy",
            instrument_series=instrument,
        )
        assert np.allclose(res.irf_mean, res.country_irfs.mean(axis=0))

    def test_lo_leq_mean_leq_hi(self):
        """Property: bands bracket the mean."""
        panel, instrument = self._make_proxy_panel(n_countries=5)
        res = mean_group_svar(
            panel, p=1, horizon=4,
            identification="proxy",
            instrument_series=instrument,
        )
        assert np.all(res.irf_lo <= res.irf_mean + 1e-12)
        assert np.all(res.irf_mean <= res.irf_hi + 1e-12)


# ---------------------------------------------------------------------------
# mean_group_svar: Rigobon identification (regime_indicator trimming branches)
# ---------------------------------------------------------------------------

class TestRigobon:

    def _make_rigobon_panel(self, n_countries=4, T=80, seed=5):
        rng = np.random.default_rng(seed)
        panel: dict[str, np.ndarray] = {}
        for i in range(n_countries):
            Y = np.zeros((T, 2))
            for t in range(1, T):
                Y[t] = 0.4 * Y[t - 1] + rng.standard_normal(2) * 0.5
            panel[f"C{i}"] = Y
        return panel

    def _make_regime(self, length: int) -> np.ndarray:
        """Half-half regime indicator of given length."""
        ri = np.zeros(length, dtype=int)
        ri[length // 2 :] = 1
        return ri

    def test_regime_indicator_length_equals_T(self):
        """Branch: regime_indicator.shape[0] == T  =>  ri = ri[p:]."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T)
        ri = self._make_regime(T)  # length == T
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert res.irf_mean.shape == (5, 2, 2)
        assert res.identification == "rigobon"

    def test_regime_indicator_length_equals_T_eff(self):
        """Branch: regime_indicator.shape[0] == T - p  (exact T_eff)."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T)
        T_eff = T - p
        ri = self._make_regime(T_eff)  # exact T_eff
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert res.irf_mean.shape == (5, 2, 2)

    def test_regime_indicator_too_long_is_trimmed(self):
        """Branch: regime_indicator.shape[0] > T_eff (and != T)  =>  ri[:T_eff]."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T)
        ri = self._make_regime(T + 10)  # longer than both T and T_eff
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert res.irf_mean.shape == (5, 2, 2)

    def test_regime_indicator_too_short_is_resized(self):
        """Branch: regime_indicator.shape[0] < T_eff  =>  np.resize(ri, T_eff)."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T)
        T_eff = T - p
        ri = self._make_regime(T_eff - 10)  # shorter than T_eff
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert res.irf_mean.shape == (5, 2, 2)

    def test_mean_equals_country_mean(self):
        """Known-value: rigobon mean-group is the mean of country Rigobon IRFs."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T, n_countries=5)
        ri = self._make_regime(T)
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert np.allclose(res.irf_mean, res.country_irfs.mean(axis=0))

    def test_lo_leq_mean_leq_hi(self):
        """Property: rigobon bands bracket the mean."""
        T, p = 80, 1
        panel = self._make_rigobon_panel(T=T, n_countries=5)
        ri = self._make_regime(T)
        res = mean_group_svar(
            panel, p=p, horizon=4,
            identification="rigobon",
            regime_indicator=ri,
        )
        assert np.all(res.irf_lo <= res.irf_mean + 1e-12)
        assert np.all(res.irf_mean <= res.irf_hi + 1e-12)


# ---------------------------------------------------------------------------
# Integration: structural properties shared across all schemes
# ---------------------------------------------------------------------------

class TestAggregationProperties:

    def test_impact_irf_horizon0_is_finite(self):
        """For cholesky, the horizon-0 impact matrix should be finite."""
        panel = _make_panel_rich()
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky")
        assert np.all(np.isfinite(res.irf_mean[0]))

    def test_irf_decays_for_stationary_var(self):
        """For a strongly stationary DGP, mean IRF should decay toward 0."""
        rng = np.random.default_rng(13)
        A = 0.3 * np.eye(2)  # strong mean-reversion
        panel: dict[str, np.ndarray] = {}
        for i in range(6):
            Y = np.zeros((150, 2))
            for t in range(1, 150):
                Y[t] = A @ Y[t - 1] + rng.standard_normal(2) * 0.3
            panel[f"C{i}"] = Y
        res = mean_group_svar(panel, p=1, horizon=20, identification="cholesky")
        # Frobenius norm at h=20 should be smaller than at h=1
        norm_h0 = np.linalg.norm(res.irf_mean[0])
        norm_h20 = np.linalg.norm(res.irf_mean[20])
        assert norm_h20 < norm_h0

    def test_ci_zero_point_nine_gives_sensible_band_fraction(self):
        """With 10 countries and ci=0.9, about 90% of country IRFs should
        fall inside [irf_lo, irf_hi] at each (h, i, j) cell.
        We test this on the diagonal cells (own-shock IRFs) at h=1."""
        panel = _make_panel_rich(n_countries=10)
        res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky", ci=0.9)
        H_plus1, n, _ = res.irf_mean.shape
        N = res.country_irfs.shape[0]
        # Check diagonal impulse-response at h=1
        h = 1
        for j in range(n):
            vals = res.country_irfs[:, h, j, j]
            inside = np.sum((vals >= res.irf_lo[h, j, j]) &
                            (vals <= res.irf_hi[h, j, j]))
            # With N=10 and ci=0.9, expect ~9 out of 10 inside
            assert inside >= int(0.7 * N), \
                f"Only {inside}/{N} country IRFs inside 90% band at h={h}, ({j},{j})"

    def test_result_has_all_required_attributes(self):
        """PanelSVARResult must expose all documented attributes."""
        panel = _make_panel()
        res = mean_group_svar(panel, p=1, horizon=4)
        for attr in ("irf_mean", "irf_lo", "irf_hi", "country_irfs",
                     "country_ids", "identification", "p", "horizon", "ci"):
            assert hasattr(res, attr), f"Missing attribute: {attr}"

    def test_dtype_float(self):
        """All IRF arrays should have float dtype."""
        panel = _make_panel()
        res = mean_group_svar(panel, p=1, horizon=4)
        for arr in (res.irf_mean, res.irf_lo, res.irf_hi, res.country_irfs):
            assert np.issubdtype(arr.dtype, np.floating), \
                f"Expected float dtype, got {arr.dtype}"
