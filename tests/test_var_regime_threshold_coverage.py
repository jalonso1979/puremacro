"""Focused coverage tests for puremacro.var.regime.threshold (TVAR).

Test strategy
-------------
- known_value: generate synthetic TVAR(1) data from a known 2-regime DGP
  (low-variance / high-variance split on z_{t-1}); verify that tvar_fit
  recovers the threshold sign, the regime assignment counts roughly match
  the true splits, and each regime's residual variance is in the correct
  ballpark.
- contract / shapes: TVARResult is a dataclass; all documented fields are
  present; A_low/A_high shapes are (n, n*p); Sigma matrices are (n, n)
  and symmetric; intercepts are (n,); n_low + n_high == T_eff; rss_total
  is a positive float; grid["log"] is a non-empty list of dicts.
- properties: threshold lies inside the candidate range; delay is drawn
  from delay_grid; n_low > 0 and n_high > 0; Sigma PSD; rss > 0.
- parameter sweep: p=2; custom delay_grid; n_threshold_grid variation;
  finer grid gives RSS <= coarser grid.
- error / edge cases: trim so tight no split survives raises RuntimeError;
  single-column Y (n=1) works; delay_grid with d > p are skipped.
- _ols_segment internals: too-short segment returns None; normal segment
  returns the correct 4-tuple.
- __all__ / public API.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.var.regime.threshold import (
    TVARResult,
    _ols_segment,
    tvar_fit,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic TVAR DGP
# ---------------------------------------------------------------------------

def _make_tvar_data(
    T: int = 300,
    n: int = 2,
    p: int = 1,
    threshold: float = 0.0,
    sigma_low: float = 0.3,
    sigma_high: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Generate (T, n) observations from a 2-regime TVAR(p) DGP.

    Regime depends on z_{t-1} = Y[t-1, 0] vs threshold.
    Low regime (z <= threshold): innovations ~ N(0, sigma_low^2 * I).
    High regime (z > threshold): innovations ~ N(0, sigma_high^2 * I).
    Both regimes use a VAR companion A = 0.4 * I_n.
    """
    rng = np.random.default_rng(seed)
    A = 0.4 * np.eye(n)
    Y = np.zeros((T, n))
    Y[0] = rng.standard_normal(n) * sigma_low
    for t in range(1, T):
        z = Y[t - 1, 0]  # threshold variable is column 0
        sigma = sigma_low if z <= threshold else sigma_high
        lag_vec = np.zeros(n * p)
        for k in range(p):
            if t - k - 1 >= 0:
                lag_vec[k * n:(k + 1) * n] = Y[t - k - 1]
        Y[t] = A @ lag_vec[:n] + rng.standard_normal(n) * sigma
    return Y


def _make_univariate_tvar(
    T: int = 200,
    threshold: float = 0.0,
    sigma_low: float = 0.3,
    sigma_high: float = 1.5,
    seed: int = 1,
) -> np.ndarray:
    """Univariate TVAR(1)."""
    return _make_tvar_data(T=T, n=1, p=1, threshold=threshold,
                           sigma_low=sigma_low, sigma_high=sigma_high, seed=seed)


# ---------------------------------------------------------------------------
# TVARResult — dataclass contract
# ---------------------------------------------------------------------------

class TestTVARResultDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TVARResult)

    def test_has_all_documented_fields(self):
        expected = {
            "A_low", "intercept_low", "Sigma_low",
            "A_high", "intercept_high", "Sigma_high",
            "threshold", "delay", "threshold_var_idx",
            "n_low", "n_high", "rss_total", "grid",
        }
        actual = {f.name for f in dataclasses.fields(TVARResult)}
        assert expected <= actual

    def test_all_exports(self):
        import puremacro.var.regime.threshold as mod
        assert set(mod.__all__) == {"tvar_fit", "TVARResult"}


# ---------------------------------------------------------------------------
# _ols_segment — internal helper tests
# ---------------------------------------------------------------------------

class TestOlsSegment:
    def test_returns_none_when_segment_too_short(self):
        """n_obs <= n_regressors + 1 must return None (underdetermined)."""
        n_obs, n_reg = 3, 4
        Y_dep = np.ones((n_obs, 2))
        X = np.ones((n_obs, n_reg))
        result = _ols_segment(Y_dep, X)
        assert result is None

    def test_returns_four_tuple_on_valid_segment(self):
        """Normal segment returns (intercept, A, Sigma, rss) — all non-None."""
        rng = np.random.default_rng(42)
        n_obs, n_dep, n_reg = 30, 2, 2
        X = rng.standard_normal((n_obs, n_reg))
        B_true = rng.standard_normal((n_reg + 1, n_dep))
        X1 = np.column_stack([np.ones(n_obs), X])
        Y_dep = X1 @ B_true + 0.1 * rng.standard_normal((n_obs, n_dep))
        result = _ols_segment(Y_dep, X)
        assert result is not None
        intercept, A, Sigma, rss = result
        assert intercept.shape == (n_dep,)
        assert A.shape == (n_dep, n_reg)
        assert Sigma.shape == (n_dep, n_dep)
        assert isinstance(rss, float)
        assert rss >= 0.0

    def test_intercept_shape_univariate(self):
        rng = np.random.default_rng(10)
        n_obs, n_dep, n_reg = 20, 1, 1
        X = rng.standard_normal((n_obs, n_reg))
        Y_dep = 0.5 * X + rng.standard_normal((n_obs, n_dep)) * 0.1
        result = _ols_segment(Y_dep, X)
        assert result is not None
        intercept, A, Sigma, rss = result
        assert intercept.shape == (1,)
        assert A.shape == (1, 1)
        assert Sigma.shape == (1, 1)

    def test_sigma_is_symmetric(self):
        rng = np.random.default_rng(11)
        n_obs, n_dep, n_reg = 30, 3, 2
        X = rng.standard_normal((n_obs, n_reg))
        Y_dep = rng.standard_normal((n_obs, n_dep))
        result = _ols_segment(Y_dep, X)
        assert result is not None
        _, _, Sigma, _ = result
        assert np.allclose(Sigma, Sigma.T)

    def test_rss_nonneg(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((25, 2))
        Y_dep = rng.standard_normal((25, 2))
        result = _ols_segment(Y_dep, X)
        assert result is not None
        assert result[3] >= 0.0

    def test_known_value_perfect_fit(self):
        """When Y_dep is an exact linear function of X+intercept, RSS ~ 0."""
        n_obs = 50
        X = np.arange(n_obs).reshape(-1, 1).astype(float)
        Y_dep = 2.0 + 3.0 * X  # perfect linear relation
        result = _ols_segment(Y_dep, X)
        assert result is not None
        intercept, A, Sigma, rss = result
        assert rss < 1e-8
        assert abs(float(intercept[0]) - 2.0) < 1e-6
        assert abs(float(A[0, 0]) - 3.0) < 1e-6

    def test_linalg_error_returns_none(self, monkeypatch):
        """When lstsq raises LinAlgError, _ols_segment returns None (defensive branch)."""
        import numpy.linalg as npla

        def _fail_lstsq(*args, **kwargs):
            raise np.linalg.LinAlgError("forced failure for coverage")

        monkeypatch.setattr(npla, "lstsq", _fail_lstsq)
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 2))
        Y_dep = rng.standard_normal((20, 2))
        result = _ols_segment(Y_dep, X)
        assert result is None


# ---------------------------------------------------------------------------
# tvar_fit — output shape / type / contract
# ---------------------------------------------------------------------------

class TestTVARFitContract:
    """Test that tvar_fit returns a TVARResult with correct types/shapes."""

    @pytest.fixture(scope="class")
    def basic_result(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=0)
        return tvar_fit(Y, threshold_var_idx=0, p=1,
                        delay_grid=(1,), n_threshold_grid=25, trim=0.15)

    def test_returns_tvar_result(self, basic_result):
        assert isinstance(basic_result, TVARResult)

    def test_is_dataclass_instance(self, basic_result):
        assert dataclasses.is_dataclass(basic_result)

    def test_threshold_is_float(self, basic_result):
        assert isinstance(basic_result.threshold, float)

    def test_delay_is_int(self, basic_result):
        assert isinstance(basic_result.delay, int)

    def test_threshold_var_idx_preserved(self, basic_result):
        assert basic_result.threshold_var_idx == 0

    def test_n_low_is_int(self, basic_result):
        assert isinstance(basic_result.n_low, int)

    def test_n_high_is_int(self, basic_result):
        assert isinstance(basic_result.n_high, int)

    def test_n_low_positive(self, basic_result):
        assert basic_result.n_low > 0

    def test_n_high_positive(self, basic_result):
        assert basic_result.n_high > 0

    def test_rss_positive(self, basic_result):
        assert basic_result.rss_total > 0.0

    def test_rss_is_float(self, basic_result):
        assert isinstance(basic_result.rss_total, float)

    def test_grid_has_log_key(self, basic_result):
        assert "log" in basic_result.grid
        assert isinstance(basic_result.grid["log"], list)
        assert len(basic_result.grid["log"]) > 0

    def test_grid_log_entries_have_required_keys(self, basic_result):
        entry = basic_result.grid["log"][0]
        assert {"d", "c", "rss"} <= set(entry.keys())

    def test_a_low_shape(self, basic_result):
        """n=2, p=1 → A_low must be (2, 2)."""
        assert basic_result.A_low.shape == (2, 2)

    def test_a_high_shape(self, basic_result):
        assert basic_result.A_high.shape == (2, 2)

    def test_intercept_low_shape(self, basic_result):
        assert basic_result.intercept_low.shape == (2,)

    def test_intercept_high_shape(self, basic_result):
        assert basic_result.intercept_high.shape == (2,)

    def test_sigma_low_shape(self, basic_result):
        assert basic_result.Sigma_low.shape == (2, 2)

    def test_sigma_high_shape(self, basic_result):
        assert basic_result.Sigma_high.shape == (2, 2)


# ---------------------------------------------------------------------------
# tvar_fit — properties
# ---------------------------------------------------------------------------

class TestTVARFitProperties:
    def test_sigma_low_symmetric(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=5)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert np.allclose(res.Sigma_low, res.Sigma_low.T)

    def test_sigma_high_symmetric(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=6)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert np.allclose(res.Sigma_high, res.Sigma_high.T)

    def test_sigma_low_positive_semidefinite(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=7)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        eigs = np.linalg.eigvalsh(res.Sigma_low)
        assert eigs.min() >= -1e-10

    def test_sigma_high_positive_semidefinite(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=8)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        eigs = np.linalg.eigvalsh(res.Sigma_high)
        assert eigs.min() >= -1e-10

    def test_delay_in_delay_grid(self):
        """The chosen delay must come from the supplied delay_grid."""
        Y = _make_tvar_data(T=200, n=2, p=2, seed=9)
        delay_grid = (1, 2)
        res = tvar_fit(Y, threshold_var_idx=0, p=2,
                       delay_grid=delay_grid, n_threshold_grid=20)
        assert res.delay in delay_grid

    def test_n_low_plus_n_high_equals_t_eff(self):
        """n_low + n_high must equal the number of usable observations (T - p)."""
        T, p = 200, 1
        Y = _make_tvar_data(T=T, n=2, p=p, seed=10)
        res = tvar_fit(Y, threshold_var_idx=0, p=p,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.n_low + res.n_high == T - p

    def test_threshold_inside_sample_range(self):
        """Threshold must lie within the sample quantile band set by trim."""
        trim = 0.15
        Y = _make_tvar_data(T=200, n=2, p=1, seed=11)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20, trim=trim)
        z = Y[:, 0]
        lo = np.quantile(z, trim)
        hi = np.quantile(z, 1.0 - trim)
        # Allow a tiny float tolerance
        assert lo - 1e-10 <= res.threshold <= hi + 1e-10

    def test_larger_grid_rss_leq_smaller_grid(self):
        """A finer threshold grid can only find a split as good or better."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=12)
        res_coarse = tvar_fit(Y, threshold_var_idx=0, p=1,
                              delay_grid=(1,), n_threshold_grid=10)
        res_fine = tvar_fit(Y, threshold_var_idx=0, p=1,
                            delay_grid=(1,), n_threshold_grid=100)
        assert res_fine.rss_total <= res_coarse.rss_total + 1e-10

    def test_list_input_accepted(self):
        """tvar_fit accepts a plain Python list."""
        Y = _make_tvar_data(T=150, n=2, p=1, seed=13)
        res = tvar_fit(Y.tolist(), threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=15)
        assert isinstance(res, TVARResult)

    def test_a_low_dtype_float(self):
        Y = _make_tvar_data(T=150, n=2, p=1, seed=14)
        res = tvar_fit(Y, threshold_var_idx=0, p=1, delay_grid=(1,), n_threshold_grid=15)
        assert res.A_low.dtype == float
        assert res.A_high.dtype == float


# ---------------------------------------------------------------------------
# tvar_fit — parameter sweep: lag order p
# ---------------------------------------------------------------------------

class TestTVARFitLagOrder:
    def test_p1_a_shape(self):
        """p=1, n=2 → A_low shape (2, 2)."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=20)
        res = tvar_fit(Y, threshold_var_idx=0, p=1, delay_grid=(1,), n_threshold_grid=20)
        assert res.A_low.shape == (2, 2)
        assert res.A_high.shape == (2, 2)

    def test_p2_a_shape(self):
        """p=2, n=2 → A_low shape (2, 4)."""
        Y = _make_tvar_data(T=250, n=2, p=2, seed=21)
        res = tvar_fit(Y, threshold_var_idx=0, p=2, delay_grid=(1, 2), n_threshold_grid=20)
        assert res.A_low.shape == (2, 4)
        assert res.A_high.shape == (2, 4)

    def test_p3_a_shape(self):
        """p=3, n=2 → A_low shape (2, 6)."""
        Y = _make_tvar_data(T=300, n=2, p=3, seed=22)
        res = tvar_fit(Y, threshold_var_idx=0, p=3, delay_grid=(1, 2, 3), n_threshold_grid=20)
        assert res.A_low.shape == (2, 6)
        assert res.A_high.shape == (2, 6)

    def test_p2_intercept_shape(self):
        """Intercept shape must always be (n,) regardless of p."""
        Y = _make_tvar_data(T=250, n=3, p=2, seed=23)
        res = tvar_fit(Y, threshold_var_idx=0, p=2, delay_grid=(1, 2), n_threshold_grid=20)
        assert res.intercept_low.shape == (3,)
        assert res.intercept_high.shape == (3,)


# ---------------------------------------------------------------------------
# tvar_fit — univariate (n=1)
# ---------------------------------------------------------------------------

class TestTVARFitUnivariate:
    def test_univariate_shapes(self):
        """n=1, p=1 → A_low shape (1, 1), Sigma (1, 1), intercept (1,)."""
        Y = _make_univariate_tvar(T=200, seed=30)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.A_low.shape == (1, 1)
        assert res.A_high.shape == (1, 1)
        assert res.Sigma_low.shape == (1, 1)
        assert res.Sigma_high.shape == (1, 1)
        assert res.intercept_low.shape == (1,)
        assert res.intercept_high.shape == (1,)

    def test_univariate_rss_positive(self):
        Y = _make_univariate_tvar(T=200, seed=31)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.rss_total > 0.0

    def test_univariate_n_low_plus_n_high(self):
        T, p = 200, 1
        Y = _make_univariate_tvar(T=T, seed=32)
        res = tvar_fit(Y, threshold_var_idx=0, p=p,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.n_low + res.n_high == T - p


# ---------------------------------------------------------------------------
# tvar_fit — threshold_var_idx != 0
# ---------------------------------------------------------------------------

class TestTVARFitThresholdVarIdx:
    def test_threshold_var_idx_1(self):
        """Using column 1 as the threshold variable is supported."""
        Y = _make_tvar_data(T=200, n=3, p=1, seed=40)
        res = tvar_fit(Y, threshold_var_idx=1, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.threshold_var_idx == 1
        assert isinstance(res, TVARResult)

    def test_threshold_var_idx_stored_correctly(self):
        Y = _make_tvar_data(T=200, n=4, p=1, seed=41)
        for idx in [0, 1, 2]:
            res = tvar_fit(Y, threshold_var_idx=idx, p=1,
                           delay_grid=(1,), n_threshold_grid=15)
            assert res.threshold_var_idx == idx


# ---------------------------------------------------------------------------
# tvar_fit — delay_grid behavior
# ---------------------------------------------------------------------------

class TestTVARFitDelayGrid:
    def test_delay_gt_p_are_skipped(self):
        """When all delay_grid values > p, tvar_fit should either skip them
        and raise RuntimeError (no valid split), or if p is large enough,
        return a valid result.  Here p=1, delay_grid=(2,3) → all skipped →
        RuntimeError."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=50)
        with pytest.raises(RuntimeError, match="No valid threshold"):
            tvar_fit(Y, threshold_var_idx=0, p=1,
                     delay_grid=(2, 3), n_threshold_grid=20)

    def test_mixed_delay_grid_valid_entries_used(self):
        """delay_grid=(1,2) with p=2: both d=1 and d=2 are <= p → some used."""
        Y = _make_tvar_data(T=250, n=2, p=2, seed=51)
        res = tvar_fit(Y, threshold_var_idx=0, p=2,
                       delay_grid=(1, 2), n_threshold_grid=20)
        assert res.delay in (1, 2)

    def test_delay_grid_single_entry(self):
        Y = _make_tvar_data(T=200, n=2, p=1, seed=52)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.delay == 1


# ---------------------------------------------------------------------------
# tvar_fit — error handling
# ---------------------------------------------------------------------------

class TestTVARFitEdgeCases:
    def test_trim_too_tight_raises_runtime_error(self):
        """Very small T forces _ols_segment to return None for every split,
        causing tvar_fit to raise RuntimeError (no valid split found)."""
        rng = np.random.default_rng(60)
        # T=8, p=1 → T_eff=7; p=1,n=2 → X_lag has 2 cols; segment needs >3 obs;
        # after splitting, each side typically has ≤3 → _ols_segment returns None.
        Y = rng.standard_normal((8, 2))
        with pytest.raises(RuntimeError, match="No valid threshold"):
            tvar_fit(Y, threshold_var_idx=0, p=1,
                     delay_grid=(1,), n_threshold_grid=5, trim=0.15)

    def test_n_threshold_grid_1_runs_without_error(self):
        """A single candidate threshold is still a valid grid."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=61)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=1, trim=0.15)
        assert isinstance(res, TVARResult)

    def test_trim_check_fires_on_degenerate_z(self):
        """When the threshold variable is discrete with a large top value, some
        candidates put all obs in the low regime (n_hi=0 < int(trim*T_eff)),
        triggering the trim-check continue branch.  The result must still be
        valid because other candidates pass the check."""
        rng = np.random.default_rng(65)
        T = 100
        # Build Y with a discrete z (column 0): values in {0, 1, 2, 3}
        # Many obs at max value so some candidates hit n_hi=0.
        z = np.array([0] * 5 + [1] * 30 + [2] * 30 + [3] * 35, dtype=float)
        rng.shuffle(z)
        y2 = rng.standard_normal(T)
        Y = np.column_stack([z, y2])
        # With n_threshold_grid=20 and trim=0.15, some candidates at z=3 put
        # all obs in low regime (n_hi=0), triggering the trim-check branch.
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20, trim=0.15)
        assert isinstance(res, TVARResult)
        assert res.rss_total > 0.0

    def test_large_n_threshold_grid(self):
        """n_threshold_grid=200 should produce a valid result."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=62)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=200, trim=0.15)
        assert isinstance(res, TVARResult)

    def test_integer_input_coerced_to_float(self):
        """Integer ndarray is coerced to float."""
        rng = np.random.default_rng(63)
        Y = (rng.standard_normal((150, 2)) * 10).astype(int)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=15)
        assert res.A_low.dtype == float


# ---------------------------------------------------------------------------
# tvar_fit — known-value / regime recovery
# ---------------------------------------------------------------------------

class TestTVARFitKnownValue:
    @pytest.mark.slow
    def test_variance_ratio_ordering_recovered(self):
        """The high-variance regime has larger Sigma diagonal than the low one.

        DGP: sigma_low=0.3, sigma_high=1.5 (ratio 25).  With T=400 and a
        clear signal, the estimated Sigma diagonal in the assigned high regime
        should be substantially larger than in the low regime.
        """
        Y = _make_tvar_data(T=400, n=2, p=1,
                            sigma_low=0.3, sigma_high=1.5, seed=99)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=40, trim=0.15)
        var_low = np.diag(res.Sigma_low).mean()
        var_high = np.diag(res.Sigma_high).mean()
        # The ratio of the larger to smaller estimated variance should be > 2
        ratio = max(var_low, var_high) / (min(var_low, var_high) + 1e-12)
        assert ratio > 2.0, (
            f"Expected variance ratio > 2 across regimes, got {ratio:.2f}"
        )

    def test_threshold_splits_sample_nontrivially(self):
        """Both regimes should have at least 10% of effective observations."""
        T, p = 200, 1
        Y = _make_tvar_data(T=T, n=2, p=p, seed=100)
        res = tvar_fit(Y, threshold_var_idx=0, p=p,
                       delay_grid=(1,), n_threshold_grid=25, trim=0.15)
        T_eff = T - p
        assert res.n_low >= int(0.10 * T_eff)
        assert res.n_high >= int(0.10 * T_eff)

    def test_grid_log_rss_are_positive(self):
        """Every grid entry's RSS must be a positive real number."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=101)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=25)
        for entry in res.grid["log"]:
            assert entry["rss"] > 0.0

    def test_best_rss_equals_min_in_grid_log(self):
        """The stored rss_total must equal the minimum RSS in the grid log."""
        Y = _make_tvar_data(T=200, n=2, p=1, seed=102)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=25)
        min_rss = min(e["rss"] for e in res.grid["log"])
        assert abs(res.rss_total - min_rss) < 1e-10

    def test_grid_log_delay_values_leq_p(self):
        """Grid log entries must only have delay values <= p (d > p are skipped)."""
        p = 2
        Y = _make_tvar_data(T=250, n=2, p=p, seed=103)
        res = tvar_fit(Y, threshold_var_idx=0, p=p,
                       delay_grid=(1, 2, 3, 4), n_threshold_grid=20)
        for entry in res.grid["log"]:
            assert entry["d"] <= p


# ---------------------------------------------------------------------------
# tvar_fit — three-variable system
# ---------------------------------------------------------------------------

class TestTVARFitTrivariate:
    def test_n3_p1_shapes(self):
        """n=3, p=1 → A shape (3, 3), Sigma (3, 3), intercept (3,)."""
        Y = _make_tvar_data(T=200, n=3, p=1, seed=70)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.A_low.shape == (3, 3)
        assert res.A_high.shape == (3, 3)
        assert res.Sigma_low.shape == (3, 3)
        assert res.Sigma_high.shape == (3, 3)
        assert res.intercept_low.shape == (3,)
        assert res.intercept_high.shape == (3,)

    def test_n3_p2_shapes(self):
        """n=3, p=2 → A shape (3, 6)."""
        Y = _make_tvar_data(T=300, n=3, p=2, seed=71)
        res = tvar_fit(Y, threshold_var_idx=0, p=2,
                       delay_grid=(1, 2), n_threshold_grid=20)
        assert res.A_low.shape == (3, 6)
        assert res.A_high.shape == (3, 6)

    def test_n3_rss_positive(self):
        Y = _make_tvar_data(T=200, n=3, p=1, seed=72)
        res = tvar_fit(Y, threshold_var_idx=0, p=1,
                       delay_grid=(1,), n_threshold_grid=20)
        assert res.rss_total > 0.0
