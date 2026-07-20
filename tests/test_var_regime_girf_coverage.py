"""Focused coverage tests for puremacro.var.regime.girf (KPP 1996 GIRF).

Test strategy
-------------
- linear-limit oracle (THE validation anchor): a TVAR whose two regimes
  share identical coefficients and covariances is a linear VAR, and the
  common-random-numbers paired construction makes the GIRF equal the
  closed-form linear IRF (puremacro.var.irf) to machine precision — no
  Monte Carlo tolerance needed. In the same limit the between-regime
  difference is exactly zero, GI(-d) = -GI(d) (sign symmetry) and
  GI(2d) = 2*GI(d) (Kilian-Vigfusson size proportionality).
- planted regime dependence: a TVAR with different coefficient matrices
  across regimes must produce a detectably nonzero between-regime
  difference with the planted sign, and a bootstrap band that excludes
  zero at short horizons.
- MS-VAR smoke on Hamilton-style simulated data: shapes, finiteness, and
  the shared-A identity — the regime-conditional impact response equals
  chol(Sigma_k)[:, shock] exactly.
- TVECM smoke: shapes and finiteness on cointegrated data.
- seed reproducibility, frozen-result contract, __all__ exports, and
  diagnostic validation errors naming ``girf``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf as var_irf
from puremacro.var.regime import ms_var_fit, tvar_fit, tvecm_fit
from puremacro.var.regime.girf import RegimeGIRFResult, girf
from puremacro.var.regime.threshold import TVARResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_var_data(T: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable bivariate VAR(1) data; returns (Y, A_true, Sigma_true)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(2)
    return Y, A, Sigma


def _linear_limit_fit(Y: np.ndarray) -> tuple[TVARResult, object]:
    """TVAR whose two regimes are IDENTICAL copies of the fitted linear VAR."""
    est = estimate_var(Y, 1)
    A_hat = np.hstack(est.A_list)
    fit = TVARResult(
        A_low=A_hat, intercept_low=est.c, Sigma_low=est.Sigma,
        A_high=A_hat.copy(), intercept_high=est.c.copy(),
        Sigma_high=est.Sigma.copy(),
        threshold=float(np.median(Y[:, 0])), delay=1, threshold_var_idx=0,
        n_low=0, n_high=0, rss_total=0.0, grid={},
    )
    return fit, est


def _planted_tvar() -> tuple[TVARResult, np.ndarray]:
    """Two-regime TVAR(1) with regime-dependent propagation, plus its data.

    High regime (z > 0): own-persistence 0.8 and cross-effect -0.5;
    low regime: 0.5 and -0.1. Same identity covariance in both regimes,
    so ALL regime dependence is in the propagation, not the impact.
    """
    A_lo = np.array([[0.5, 0.0], [-0.1, 0.4]])
    A_hi = np.array([[0.8, 0.0], [-0.5, 0.4]])
    rng = np.random.default_rng(42)
    Y = np.zeros((500, 2))
    for t in range(1, 500):
        Ar = A_lo if Y[t - 1, 0] <= 0.0 else A_hi
        Y[t] = Ar @ Y[t - 1] + rng.standard_normal(2)
    fit = TVARResult(
        A_low=A_lo, intercept_low=np.zeros(2), Sigma_low=np.eye(2),
        A_high=A_hi, intercept_high=np.zeros(2), Sigma_high=np.eye(2),
        threshold=0.0, delay=1, threshold_var_idx=0,
        n_low=0, n_high=0, rss_total=0.0, grid={},
    )
    return fit, Y


def _hamilton_style_data(T: int = 220, seed: int = 1) -> np.ndarray:
    """Univariate 2-regime MS data: calm N(0, 0.5^2) vs turbulent N(1, 2^2)."""
    rng = np.random.default_rng(seed)
    P_true = np.array([[0.95, 0.05], [0.10, 0.90]])
    s = 0
    Y = np.zeros((T, 1))
    for t in range(1, T):
        s = rng.choice(2, p=P_true[s])
        Y[t, 0] = (0.0 if s == 0 else 1.0) + (0.5 if s == 0 else 2.0) * rng.standard_normal()
    return Y


def _cointegrated_data(T: int = 400, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T) * 0.5)
    y = x + rng.standard_normal(T) * 0.8
    return np.column_stack([y, x])


# ---------------------------------------------------------------------------
# Linear-limit oracle — the validation anchor
# ---------------------------------------------------------------------------

class TestLinearLimitOracle:
    @pytest.fixture(scope="class")
    def linear_case(self):
        Y, _, _ = _linear_var_data(T=300, seed=0)
        fit, est = _linear_limit_fit(Y)
        res = girf(fit, Y, shock=0, horizon=10, n_hist=12, n_sim=15,
                   shock_size=[1.0, -1.0, 2.0], n_boot=50, rng=3)
        return Y, fit, est, res

    def test_girf_equals_linear_irf(self, linear_case):
        """Identical regimes => GIRF == closed-form Cholesky IRF, exactly.

        With common random numbers and the impulse ADDED to the identified
        shock, the paired difference in a linear model is deterministic:
        y_shocked - y_base = Phi_h chol(Sigma) e_j * delta for every draw.
        """
        _, _, est, res = linear_case
        expected = var_irf(est.A_list, np.linalg.cholesky(est.Sigma), 10)[:, :, 0]
        assert np.allclose(res.girf_pooled[0], expected, atol=1e-10)

    def test_regime_conditional_girfs_coincide(self, linear_case):
        _, _, _, res = linear_case
        assert np.allclose(res.girf_by_regime[0], res.girf_by_regime[1], atol=1e-12)

    def test_difference_is_zero(self, linear_case):
        _, _, _, res = linear_case
        assert np.abs(res.difference).max() < 1e-12

    def test_difference_band_collapses_on_zero(self, linear_case):
        """Every bootstrap draw of the difference is exactly 0 in the limit."""
        _, _, _, res = linear_case
        assert np.abs(res.difference_lo).max() < 1e-12
        assert np.abs(res.difference_hi).max() < 1e-12

    def test_sign_symmetry(self, linear_case):
        """GI(-d) == -GI(d) in the linear limit (shock_sizes[1] = -1)."""
        _, _, _, res = linear_case
        assert np.allclose(res.girf_pooled[1], -res.girf_pooled[0], atol=1e-12)

    def test_kilian_vigfusson_size_proportionality(self, linear_case):
        """GI(2d) == 2*GI(d) in the linear limit (shock_sizes[2] = 2)."""
        _, _, _, res = linear_case
        assert np.allclose(res.girf_pooled[2], 2.0 * res.girf_pooled[0], atol=1e-12)

    def test_scaled_rows_identical(self, linear_case):
        """scaled() divides by delta: all rows coincide iff linear + symmetric."""
        _, _, _, res = linear_case
        sc = res.scaled()
        assert sc.shape == (3, 11, 2)
        assert np.allclose(sc, sc[0][None], atol=1e-12)

    def test_second_shock_column(self, linear_case):
        """shock=1 matches column 1 of the linear IRF (Cholesky ordering)."""
        Y, fit, est, _ = linear_case
        res = girf(fit, Y, shock=1, horizon=6, n_hist=8, n_sim=10,
                   n_boot=20, rng=5)
        expected = var_irf(est.A_list, np.linalg.cholesky(est.Sigma), 6)[:, :, 1]
        assert np.allclose(res.girf_pooled[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Planted regime dependence
# ---------------------------------------------------------------------------

class TestPlantedRegimeDependence:
    @pytest.fixture(scope="class")
    def planted(self):
        fit, Y = _planted_tvar()
        res = girf(fit, Y, shock=0, horizon=12, n_hist=30, n_sim=60,
                   shock_size=1.0, n_boot=200, rng=7)
        return fit, Y, res

    def test_labels_and_weights(self, planted):
        _, _, res = planted
        assert res.regime_labels == ("low", "high")
        assert res.pooled_weights.shape == (2,)
        assert abs(res.pooled_weights.sum() - 1.0) < 1e-12

    def test_difference_nonzero_correct_sign_own_variable(self, planted):
        """Own persistence 0.8 vs 0.5 => high-regime response larger at h>=1."""
        _, _, res = planted
        assert res.difference[0, 1, 0] > 0.1

    def test_difference_nonzero_correct_sign_cross_variable(self, planted):
        """Cross-effect -0.5 vs -0.1 => variable 1 falls MORE in the high regime."""
        _, _, res = planted
        assert res.difference[0, 1, 1] < -0.1

    def test_band_excludes_zero_at_short_horizons(self, planted):
        """The regime-dependence test: the 90% band excludes 0 for h = 1, 2."""
        _, _, res = planted
        for h in (1, 2):
            assert res.difference_hi[0, h, 1] < 0.0    # cross response
            assert res.difference_lo[0, h, 0] > 0.0    # own response

    def test_impact_identical_across_regimes(self, planted):
        """Sigma is the same in both regimes, so h=0 responses coincide."""
        _, _, res = planted
        assert np.allclose(res.girf_by_regime[0, 0, 0], res.girf_by_regime[1, 0, 0],
                           atol=1e-10)

    def test_fitted_tvar_reproduces_planted_sign(self):
        """End-to-end: tvar_fit on the planted data, then girf — same signs."""
        _, Y = _planted_tvar()
        fit = tvar_fit(Y, threshold_var_idx=0, p=1, delay_grid=(1,),
                       n_threshold_grid=40)
        res = girf(fit, Y, shock=0, horizon=8, n_hist=25, n_sim=40,
                   n_boot=100, rng=11)
        assert res.difference[0, 1, 0] > 0.05
        assert res.difference[0, 1, 1] < -0.05


# ---------------------------------------------------------------------------
# MS-VAR smoke (Hamilton-style data)
# ---------------------------------------------------------------------------

class TestMSVarGirf:
    @pytest.fixture(scope="class")
    def ms_case(self):
        Y = _hamilton_style_data(T=220, seed=1)
        fit = ms_var_fit(Y, K=2, p=1, n_iter=60)
        res = girf(fit, Y, shock=0, horizon=8, n_hist=15, n_sim=25,
                   n_boot=50, rng=5)
        return Y, fit, res

    def test_shapes(self, ms_case):
        _, _, res = ms_case
        assert res.girf_by_regime.shape == (2, 1, 9, 1)
        assert res.girf_pooled.shape == (1, 9, 1)
        assert res.difference.shape == (1, 9, 1)
        assert res.model == "ms_var"

    def test_finite(self, ms_case):
        _, _, res = ms_case
        for arr in (res.girf_by_regime, res.girf_pooled,
                    res.difference, res.difference_lo, res.difference_hi):
            assert np.all(np.isfinite(arr))

    def test_impact_equals_regime_cholesky(self, ms_case):
        """Shared-A spec: the paired difference is deterministic and the
        regime-k impact response is exactly chol(Sigma_k)[:, shock]."""
        _, fit, res = ms_case
        for k in range(2):
            Lk = np.linalg.cholesky(fit.Sigma[k])
            assert np.allclose(res.girf_by_regime[k, 0, 0], Lk[:, 0], atol=1e-8)

    def test_regime_labels(self, ms_case):
        _, _, res = ms_case
        assert res.regime_labels == ("regime 0", "regime 1")

    def test_pooled_weights_are_smoothed_prob_means(self, ms_case):
        _, fit, res = ms_case
        expected = fit.smoothed_probs.mean(axis=0)
        assert np.allclose(res.pooled_weights, expected / expected.sum(), atol=1e-10)


# ---------------------------------------------------------------------------
# TVECM smoke (cointegrated data)
# ---------------------------------------------------------------------------

class TestTVECMGirf:
    def test_shapes_and_finiteness(self):
        Y = _cointegrated_data(T=400, seed=9)
        fit = tvecm_fit(Y, p=2)
        res = girf(fit, Y, shock=0, horizon=8, n_hist=12, n_sim=20,
                   n_boot=40, rng=11)
        assert res.girf_by_regime.shape == (2, 1, 9, 2)
        assert res.model == "tvecm"
        assert np.all(np.isfinite(res.girf_by_regime))
        assert np.all(np.isfinite(res.difference_lo))

    def test_p1_no_short_run_lags(self):
        """p=1 TVECM (k_lags=0) must simulate without lagged differences."""
        Y = _cointegrated_data(T=300, seed=10)
        fit = tvecm_fit(Y, p=1)
        res = girf(fit, Y, shock=1, horizon=5, n_hist=8, n_sim=10,
                   n_boot=20, rng=2)
        assert res.girf_pooled.shape == (1, 6, 2)
        assert np.all(np.isfinite(res.girf_pooled))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical(self):
        fit, Y = _planted_tvar()
        kw = dict(shock=0, horizon=6, n_hist=10, n_sim=12, n_boot=30)
        r1 = girf(fit, Y, rng=123, **kw)
        r2 = girf(fit, Y, rng=123, **kw)
        assert np.array_equal(r1.girf_by_regime, r2.girf_by_regime)
        assert np.array_equal(r1.difference_lo, r2.difference_lo)

    def test_generator_accepted(self):
        fit, Y = _planted_tvar()
        gen = np.random.default_rng(99)
        res = girf(fit, Y, horizon=4, n_hist=6, n_sim=8, n_boot=20, rng=gen)
        assert isinstance(res, RegimeGIRFResult)

    def test_different_seeds_still_valid(self):
        fit, Y = _planted_tvar()
        res = girf(fit, Y, horizon=4, n_hist=6, n_sim=8, n_boot=20, rng=555)
        assert np.all(np.isfinite(res.girf_pooled))


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

class TestResultContract:
    @pytest.fixture(scope="class")
    def res(self):
        fit, Y = _planted_tvar()
        return girf(fit, Y, horizon=4, n_hist=6, n_sim=8, n_boot=20, rng=0)

    def test_frozen(self, res):
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.shock = 1

    def test_fields_present(self):
        expected = {
            "girf_by_regime", "girf_pooled", "difference",
            "difference_lo", "difference_hi", "shock", "shock_sizes",
            "horizon", "n_hist", "n_sim", "n_boot", "band",
            "regime_labels", "pooled_weights", "model",
        }
        actual = {f.name for f in dataclasses.fields(RegimeGIRFResult)}
        assert expected <= actual

    def test_summary_mentions_design(self, res):
        s = res.summary()
        assert isinstance(s, str)
        assert "Regime GIRF" in s and "tvar" in s

    def test_module_all(self):
        import importlib
        # NB: attribute access on the package returns the girf FUNCTION
        # (the __init__ re-export shadows the submodule name), so resolve
        # the module through importlib.
        mod = importlib.import_module("puremacro.var.regime.girf")
        assert set(mod.__all__) == {"girf", "RegimeGIRFResult"}

    def test_package_reexport(self):
        import puremacro.var.regime as pkg
        assert "girf" in pkg.__all__ and "RegimeGIRFResult" in pkg.__all__
        assert pkg.girf is girf


# ---------------------------------------------------------------------------
# Validation errors — diagnostic messages naming girf
# ---------------------------------------------------------------------------

class TestValidationErrors:
    @pytest.fixture(scope="class")
    def fit_and_Y(self):
        return _planted_tvar()

    def test_unsupported_fit_type(self, fit_and_Y):
        _, Y = fit_and_Y
        with pytest.raises(TypeError, match="girf: unsupported fit type"):
            girf({"not": "a fit"}, Y)

    def test_y_must_be_2d(self, fit_and_Y):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: Y must be 2-D"):
            girf(fit, Y[:, 0])

    def test_column_mismatch(self, fit_and_Y):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: Y has 3 columns"):
            girf(fit, np.column_stack([Y, Y[:, 0]]))

    def test_shock_out_of_range(self, fit_and_Y):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: shock must be an int"):
            girf(fit, Y, shock=2)

    def test_horizon_positive(self, fit_and_Y):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: horizon must be an int >= 1"):
            girf(fit, Y, horizon=0)

    @pytest.mark.parametrize("bad_kw", [
        {"n_hist": 0}, {"n_sim": 0}, {"n_boot": 0},
    ])
    def test_design_sizes_positive(self, fit_and_Y, bad_kw):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: n_"):
            girf(fit, Y, **bad_kw)

    def test_band_in_unit_interval(self, fit_and_Y):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: band must be in"):
            girf(fit, Y, band=1.2)

    @pytest.mark.parametrize("bad_size", [0.0, [1.0, 0.0], np.inf])
    def test_shock_size_nonzero_finite(self, fit_and_Y, bad_size):
        fit, Y = fit_and_Y
        with pytest.raises(ValueError, match="girf: every shock_size"):
            girf(fit, Y, shock_size=bad_size)

    def test_sample_too_short(self, fit_and_Y):
        fit, _ = fit_and_Y
        with pytest.raises(ValueError, match="girf: Y has T=1 rows"):
            girf(fit, np.zeros((1, 2)))

    def test_tvar_delay_outside_lag_window(self):
        """A hand-built fit with d > p must fail with a diagnostic error."""
        fit, Y = _planted_tvar()
        bad = dataclasses.replace(fit, delay=3)   # p = 1 < d = 3
        with pytest.raises(ValueError, match="girf: TVAR delay"):
            girf(bad, Y)

    def test_regime_never_visited(self):
        """Threshold below the sample range => no low-regime histories."""
        fit, Y = _planted_tvar()
        bad = dataclasses.replace(fit, threshold=float(Y[:, 0].min() - 10.0))
        with pytest.raises(ValueError, match="girf: no usable histories"):
            girf(bad, Y, n_hist=5, n_sim=5, n_boot=10)
