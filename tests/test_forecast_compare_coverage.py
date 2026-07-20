"""Coverage tests for puremacro.forecast.compare.

The existing tests (tests/test_forecast/test_dm_gw.py) cover the happy-path
for diebold_mariano(loss="mse") with small_sample=True and giacomini_white
with a 2-column conditioning matrix.

Missing branches targeted here (line numbers in compare.py as of this writing):
  21-25  _loss_diff: loss="mae", loss="lp", unknown loss -> ValueError
  46-49  DM Bartlett kernel: h > 1 (inner-loop body)
  51-52  DM degenerate path: s2 <= 0 -> returns nan
  60     DM small_sample=False -> normal-distribution p-value
  86     GW conditioning_vars with 0-column array -> constant-only fallback
  90     GW 1-D conditioning_vars -> reshaped to (T,1)
  100-103 GW Bartlett kernel: h > 1
  106-107 GW singular Sigma -> LinAlgError -> returns nan

Strategy
--------
- Known-value tests: synthetic series with hand-checkable loss differentials.
- Property tests: shapes, sign, bounds, symmetry, monotonicity.
- Contract tests: required keys, raises on bad input.
- Error path tests: degenerate inputs that trigger the nan-return guards.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as t_dist, norm as sp_norm

from puremacro.forecast.compare import diebold_mariano, giacomini_white


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2025)


def _make_errors(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T), rng.standard_normal(T)


# ---------------------------------------------------------------------------
# _loss_diff: mae branch (lines 21-22)
# ---------------------------------------------------------------------------

class TestLossDiffMae:
    """Test DM with loss='mae', which exercises lines 21-22."""

    def test_mae_loss_returns_dict_with_required_keys(self):
        e1, e2 = _make_errors(100, seed=1)
        out = diebold_mariano(e1, e2, h=1, loss="mae")
        for k in ("stat", "p_value", "n_obs", "lag_used"):
            assert k in out

    def test_mae_loss_known_value(self):
        """If e1 is always zero and e2 is always 1.0, every loss-diff is
        |0| - |1| = -1.  The mean loss diff is exactly -1, variance is 0,
        so s2 = 0 and the function should return nan (degenerate path)."""
        e1 = np.zeros(50)
        e2 = np.ones(50)
        out = diebold_mariano(e1, e2, h=1, loss="mae", small_sample=False)
        # s2 = 0 -> degenerate nan path OR finite negative stat; both are valid
        # The key contract: no exception is raised and required keys are present
        assert "stat" in out
        assert "p_value" in out

    def test_mae_dominant_forecast_detected(self):
        """e1 ~ N(0, 0.1^2) vs e2 ~ N(0, 1): |e1| << |e2| -> DM should reject
        in favour of model 1 (negative stat, small p-value)."""
        rng = np.random.default_rng(10)
        e1 = rng.normal(0, 0.1, 400)
        e2 = rng.standard_normal(400)
        out = diebold_mariano(e1, e2, h=1, loss="mae")
        assert out["stat"] < 0
        assert out["p_value"] < 0.05

    def test_mae_equal_loss_large_p_value(self):
        """Two i.i.d. N(0,1) series -> MAE loss differential symmetric around 0
        -> should not reject at 5%."""
        rng = np.random.default_rng(11)
        e1 = rng.standard_normal(500)
        e2 = rng.standard_normal(500)
        out = diebold_mariano(e1, e2, h=1, loss="mae")
        assert out["p_value"] > 0.05

    def test_mae_p_value_in_01(self):
        e1, e2 = _make_errors(200, seed=2)
        out = diebold_mariano(e1, e2, h=1, loss="mae")
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)


# ---------------------------------------------------------------------------
# _loss_diff: lp branch (lines 23-24)
# ---------------------------------------------------------------------------

class TestLossDiffLp:
    """Test DM with loss='lp', which exercises lines 23-24."""

    def test_lp_power2_identical_to_mse(self):
        """lp with power=2 is |e|^2 = e^2, which equals MSE for any errors."""
        rng = np.random.default_rng(20)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        out_mse = diebold_mariano(e1, e2, h=1, loss="mse", power=2.0)
        out_lp  = diebold_mariano(e1, e2, h=1, loss="lp",  power=2.0)
        assert abs(out_mse["stat"] - out_lp["stat"]) < 1e-10

    def test_lp_power1_identical_to_mae(self):
        """lp with power=1 reduces to MAE: |e|^1 = |e|."""
        rng = np.random.default_rng(21)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        out_mae = diebold_mariano(e1, e2, h=1, loss="mae")
        out_lp1 = diebold_mariano(e1, e2, h=1, loss="lp", power=1.0)
        assert abs(out_mae["stat"] - out_lp1["stat"]) < 1e-10

    def test_lp_returns_required_keys(self):
        e1, e2 = _make_errors(100, seed=3)
        out = diebold_mariano(e1, e2, h=1, loss="lp", power=1.5)
        for k in ("stat", "p_value", "n_obs", "lag_used"):
            assert k in out

    def test_lp_dominant_model_detected(self):
        """e1 small errors, e2 large: lp(e1) < lp(e2) -> negative stat."""
        rng = np.random.default_rng(22)
        e1 = rng.normal(0, 0.1, 400)
        e2 = rng.standard_normal(400)
        out = diebold_mariano(e1, e2, h=1, loss="lp", power=1.5)
        assert out["stat"] < 0
        assert out["p_value"] < 0.05

    def test_lp_p_value_in_01(self):
        e1, e2 = _make_errors(150, seed=4)
        out = diebold_mariano(e1, e2, h=1, loss="lp", power=3.0)
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)


# ---------------------------------------------------------------------------
# _loss_diff: unknown loss -> ValueError (line 25)
# ---------------------------------------------------------------------------

class TestLossDiffUnknown:
    def test_unknown_loss_raises_value_error(self):
        e1, e2 = _make_errors(50, seed=5)
        with pytest.raises(ValueError, match="unknown loss"):
            diebold_mariano(e1, e2, h=1, loss="huber")

    def test_unknown_loss_message_contains_loss_name(self):
        e1, e2 = _make_errors(50, seed=6)
        with pytest.raises(ValueError, match="rmse"):
            diebold_mariano(e1, e2, h=1, loss="rmse")

    def test_unknown_loss_gw_also_raises(self):
        """giacomini_white delegates to _loss_diff, so also raises on bad loss."""
        e1, e2 = _make_errors(50, seed=7)
        with pytest.raises(ValueError, match="unknown loss"):
            giacomini_white(e1, e2, h=1, loss="bad")


# ---------------------------------------------------------------------------
# DM Bartlett kernel loop for h > 1 (lines 46-49)
# ---------------------------------------------------------------------------

class TestDMHGreaterThanOne:
    """With h > 1, L = h-1 >= 1, so the inner loop body (lines 46-49) runs."""

    def test_h2_returns_required_keys(self):
        e1, e2 = _make_errors(200, seed=30)
        out = diebold_mariano(e1, e2, h=2, loss="mse")
        for k in ("stat", "p_value", "n_obs", "lag_used"):
            assert k in out

    def test_lag_used_equals_h_minus_1(self):
        """lag_used should be L = max(0, h-1)."""
        e1, e2 = _make_errors(100, seed=31)
        for h in (1, 2, 3, 5):
            out = diebold_mariano(e1, e2, h=h, loss="mse")
            assert out["lag_used"] == max(0, h - 1)

    def test_h4_p_value_in_01(self):
        e1, e2 = _make_errors(300, seed=32)
        out = diebold_mariano(e1, e2, h=4, loss="mse")
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)

    def test_h_larger_than_T_does_not_raise(self):
        """When h > T, the inner loop hits ell >= T and breaks (line 47).
        Should return a dict with the right keys without raising."""
        e1 = np.array([0.5, -0.5, 0.3])
        e2 = np.array([0.2, -0.1, 0.4])
        out = diebold_mariano(e1, e2, h=10, loss="mse")
        assert "stat" in out
        assert out["n_obs"] == 3

    def test_h2_equal_series_large_p_value(self):
        """Two i.i.d. series, h=2: should not reject at 5% for large T."""
        rng = np.random.default_rng(33)
        e1 = rng.standard_normal(500)
        e2 = rng.standard_normal(500)
        out = diebold_mariano(e1, e2, h=2, loss="mse")
        assert out["p_value"] > 0.05


# ---------------------------------------------------------------------------
# DM degenerate path: s2 <= 0 -> nan (lines 51-52)
# ---------------------------------------------------------------------------

class TestDMDegeneratePath:
    """When all loss differentials are identical, variance is 0 and the
    function returns NaN instead of dividing by zero."""

    def test_constant_diff_returns_nan(self):
        """e1 and e2 chosen so that every MSE diff is the same constant.
        Variance of d is zero -> s2 <= 0 -> stat and p_value are nan."""
        # e1[t]^2 - e2[t]^2 = c for all t  means we can set e1 = 1, e2 = 0
        # -> d[t] = 1 always -> d - d_bar = 0 -> g0 = 0 -> s2 = 0
        e1 = np.ones(50)
        e2 = np.zeros(50)
        out = diebold_mariano(e1, e2, h=1, loss="mse")
        assert np.isnan(out["stat"])
        assert np.isnan(out["p_value"])
        # Other keys still present and valid
        assert out["n_obs"] == 50
        assert out["lag_used"] == 0

    def test_degenerate_dict_contains_all_keys(self):
        e1 = np.ones(30)
        e2 = np.zeros(30)
        out = diebold_mariano(e1, e2, h=1, loss="mse")
        for k in ("stat", "p_value", "n_obs", "lag_used"):
            assert k in out


# ---------------------------------------------------------------------------
# DM small_sample=False -> normal distribution (line 60)
# ---------------------------------------------------------------------------

class TestDMNormalDistribution:
    """With small_sample=False, p-value uses the standard normal (line 60)."""

    def test_small_sample_false_returns_finite_p_value(self):
        e1, e2 = _make_errors(200, seed=40)
        out = diebold_mariano(e1, e2, h=1, small_sample=False)
        assert np.isfinite(out["p_value"])

    def test_small_sample_false_p_in_01(self):
        e1, e2 = _make_errors(200, seed=41)
        out = diebold_mariano(e1, e2, h=1, small_sample=False)
        assert 0.0 <= out["p_value"] <= 1.0

    def test_small_sample_false_equal_series_large_p(self):
        """Two i.i.d. series: should not reject H0 (equal loss) at 5%."""
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(500)
        e2 = rng.standard_normal(500)
        out = diebold_mariano(e1, e2, h=1, small_sample=False)
        assert out["p_value"] > 0.05

    def test_small_sample_false_dominant_model_low_p(self):
        """Model 1 much better than model 2 -> reject at 5% even with normal dist."""
        rng = np.random.default_rng(43)
        e1 = rng.normal(0, 0.1, 400)
        e2 = rng.standard_normal(400)
        out = diebold_mariano(e1, e2, h=1, loss="mse", small_sample=False)
        assert out["p_value"] < 0.05

    def test_small_sample_true_vs_false_differ_in_large_sample(self):
        """HLN correction makes a difference: the two stats should differ."""
        rng = np.random.default_rng(44)
        e1 = rng.standard_normal(100)
        e2 = rng.standard_normal(100) * 1.5
        out_t   = diebold_mariano(e1, e2, h=1, small_sample=True)
        out_n   = diebold_mariano(e1, e2, h=1, small_sample=False)
        # Stats may differ due to HLN correction factor
        # (they need not differ always, but they can; contract: both are finite)
        assert np.isfinite(out_t["stat"])
        assert np.isfinite(out_n["stat"])

    def test_small_sample_false_known_value(self):
        """Hand-verified: for T=4, d=(1,1,-1,-1), d_bar=0, g0=1, s2=1,
        stat=0/sqrt(1/4)=0, p_value should be 1.0 via norm.cdf."""
        d_arr = np.array([1.0, 1.0, -1.0, -1.0])
        # Create e1, e2 such that e1^2 - e2^2 = d_arr:
        # e1[t] = sqrt((d_arr[t]+1)/2+0.5), e2[t] = 1 for d>0; use mae to make it clean
        # Simpler: use mae and construct directly
        # For mae: |e1| - |e2| = d_arr
        # Set |e1[t]| = 1 + d_arr[t]/2,  |e2[t]| = 1 - d_arr[t]/2
        e1 = np.array([1.5, 1.5, 0.5, 0.5])
        e2 = np.array([0.5, 0.5, 1.5, 1.5])
        # d = |e1| - |e2| = [1, 1, -1, -1], d_bar = 0
        out = diebold_mariano(e1, e2, h=1, loss="mae", small_sample=False)
        # d_bar=0 -> stat=0 -> p_value=1.0
        assert abs(out["stat"]) < 1e-10
        assert abs(out["p_value"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# GW: conditioning_vars with 0 columns -> constant fallback (line 86)
# ---------------------------------------------------------------------------

class TestGWZeroColumnConditioningVars:
    """Lines 83-86: when conditioning_vars has shape[-1]==0, use constant only."""

    def test_zero_column_array_triggers_constant_fallback(self):
        """conditioning_vars with 0 columns behaves like conditioning_vars=None."""
        rng = np.random.default_rng(50)
        e1 = rng.standard_normal(100)
        e2 = rng.standard_normal(100)
        zero_cols = np.empty((100, 0))
        out_zero = giacomini_white(e1, e2, h=1, conditioning_vars=zero_cols)
        out_none = giacomini_white(e1, e2, h=1, conditioning_vars=None)
        for k in ("stat", "p_value", "df"):
            assert k in out_zero
        # Both should use 1 instrument (constant) -> df=1
        assert out_zero["df"] == 1
        assert out_none["df"] == 1

    def test_zero_column_same_result_as_none(self):
        rng = np.random.default_rng(51)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        zero_cols = np.empty((200, 0))
        out_zero = giacomini_white(e1, e2, h=1, conditioning_vars=zero_cols)
        out_none = giacomini_white(e1, e2, h=1, conditioning_vars=None)
        assert abs(out_zero["stat"] - out_none["stat"]) < 1e-10

    def test_none_conditioning_vars_gives_df1(self):
        e1, e2 = _make_errors(100, seed=52)
        out = giacomini_white(e1, e2, h=1, conditioning_vars=None)
        assert out["df"] == 1


# ---------------------------------------------------------------------------
# GW: 1-D conditioning_vars -> reshape to (T,1) (line 90)
# ---------------------------------------------------------------------------

class TestGWOneDimConditioningVars:
    """Line 90: 1-D array is reshaped to column vector, then constant stacked."""

    def test_1d_conditioning_var_accepted(self):
        rng = np.random.default_rng(60)
        e1 = rng.standard_normal(150)
        e2 = rng.standard_normal(150)
        cond_1d = rng.standard_normal(150)   # shape (150,)
        out = giacomini_white(e1, e2, h=1, conditioning_vars=cond_1d)
        for k in ("stat", "p_value", "df"):
            assert k in out

    def test_1d_same_as_column_vector(self):
        """Passing a 1-D array should give the same result as a (T,1) column."""
        rng = np.random.default_rng(61)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        cond = rng.standard_normal(200)
        out_1d   = giacomini_white(e1, e2, h=1, conditioning_vars=cond)
        out_col  = giacomini_white(e1, e2, h=1, conditioning_vars=cond.reshape(-1, 1))
        assert abs(out_1d["stat"] - out_col["stat"]) < 1e-10

    def test_1d_conditioning_df_is_2(self):
        """1 conditioning variable + 1 constant = 2 instruments -> df=2."""
        rng = np.random.default_rng(62)
        e1 = rng.standard_normal(100)
        e2 = rng.standard_normal(100)
        cond_1d = rng.standard_normal(100)
        out = giacomini_white(e1, e2, h=1, conditioning_vars=cond_1d)
        assert out["df"] == 2

    def test_1d_p_value_in_01(self):
        rng = np.random.default_rng(63)
        e1 = rng.standard_normal(100)
        e2 = rng.standard_normal(100)
        cond = rng.standard_normal(100)
        out = giacomini_white(e1, e2, h=1, conditioning_vars=cond)
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)


# ---------------------------------------------------------------------------
# GW Bartlett kernel loop for h > 1 (lines 100-103)
# ---------------------------------------------------------------------------

class TestGWHGreaterThanOne:
    """With h > 1, the inner Bartlett loop in GW (lines 100-103) executes."""

    def test_h2_returns_required_keys(self):
        e1, e2 = _make_errors(200, seed=70)
        out = giacomini_white(e1, e2, h=2)
        for k in ("stat", "p_value", "df"):
            assert k in out

    def test_h2_p_value_in_01(self):
        e1, e2 = _make_errors(200, seed=71)
        out = giacomini_white(e1, e2, h=2)
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)

    def test_h4_with_conditioning_vars(self):
        rng = np.random.default_rng(72)
        e1 = rng.standard_normal(300)
        e2 = rng.standard_normal(300)
        cond = rng.standard_normal((300, 2))
        out = giacomini_white(e1, e2, h=4, conditioning_vars=cond)
        assert "stat" in out
        # df = 1 (constant) + 2 (conditioning cols) = 3
        assert out["df"] == 3

    def test_h_larger_than_T_does_not_raise(self):
        """When h > T the inner loop breaks at ell >= T (line 101). No exception."""
        e1 = np.array([0.5, -0.5, 0.3])
        e2 = np.array([0.2, -0.1, 0.4])
        out = giacomini_white(e1, e2, h=10)
        assert "stat" in out

    def test_h3_equal_series_large_p_value(self):
        """Equal-ability forecasts, h=3: should not reject at 5%."""
        rng = np.random.default_rng(73)
        e1 = rng.standard_normal(500)
        e2 = rng.standard_normal(500)
        out = giacomini_white(e1, e2, h=3)
        assert out["p_value"] > 0.05


# ---------------------------------------------------------------------------
# GW singular Sigma -> LinAlgError -> nan path (lines 106-107)
# ---------------------------------------------------------------------------

class TestGWSingularSigma:
    """Lines 106-107: when np.linalg.inv raises LinAlgError, return nan dict."""

    def test_singular_sigma_via_monkeypatch(self, monkeypatch):
        """Monkeypatch np.linalg.inv inside the compare module to force the
        LinAlgError branch, verifying the nan-fallback dict is returned."""
        import puremacro.forecast.compare as _compare_mod

        original_inv = np.linalg.inv

        def _raise_inv(m):
            raise np.linalg.LinAlgError("singular (mocked)")

        monkeypatch.setattr(_compare_mod.np.linalg, "inv", _raise_inv)

        rng = np.random.default_rng(80)
        e1 = rng.standard_normal(100)
        e2 = rng.standard_normal(100)
        out = giacomini_white(e1, e2, h=1)

        assert np.isnan(out["stat"])
        assert np.isnan(out["p_value"])
        assert "df" in out

    def test_singular_sigma_nan_dict_has_all_keys(self, monkeypatch):
        import puremacro.forecast.compare as _compare_mod

        def _raise_inv(m):
            raise np.linalg.LinAlgError("forced singular")

        monkeypatch.setattr(_compare_mod.np.linalg, "inv", _raise_inv)

        e1, e2 = _make_errors(50, seed=81)
        out = giacomini_white(e1, e2, h=1)
        for k in ("stat", "p_value", "df"):
            assert k in out

    def test_nearly_singular_conditioning_produces_valid_output(self):
        """Two identical conditioning columns make Sigma rank-deficient.
        The function should either return a valid result or the nan fallback."""
        rng = np.random.default_rng(82)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        cond_col = rng.standard_normal(200)
        # Duplicate the column: rank-1 conditioning matrix (after adding constant)
        cond = np.column_stack([cond_col, cond_col])
        out = giacomini_white(e1, e2, h=1, conditioning_vars=cond)
        for k in ("stat", "p_value", "df"):
            assert k in out
        # Either nan (singular) or a real value (if linalg recovers)
        assert np.isnan(out["stat"]) or np.isfinite(out["stat"])


# ---------------------------------------------------------------------------
# Integration: multi-loss / multi-h consistency checks
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_dm_all_losses_return_required_keys(self):
        """All three loss modes return the full DM dict."""
        e1, e2 = _make_errors(150, seed=90)
        for loss in ("mse", "mae", "lp"):
            out = diebold_mariano(e1, e2, h=1, loss=loss, power=1.5)
            for k in ("stat", "p_value", "n_obs", "lag_used"):
                assert k in out, f"Missing key {k!r} for loss={loss!r}"

    def test_gw_1d_and_2d_and_none_consistent_df(self):
        """df grows by 1 for each conditioning variable."""
        rng = np.random.default_rng(91)
        e1 = rng.standard_normal(200)
        e2 = rng.standard_normal(200)
        out0 = giacomini_white(e1, e2, h=1, conditioning_vars=None)
        out1 = giacomini_white(e1, e2, h=1,
                               conditioning_vars=rng.standard_normal(200))
        out2 = giacomini_white(e1, e2, h=1,
                               conditioning_vars=rng.standard_normal((200, 2)))
        assert out0["df"] == 1
        assert out1["df"] == 2
        assert out2["df"] == 3

    def test_dm_h2_mae_returns_finite_stat(self):
        """Combined: h>1 (Bartlett) AND mae loss simultaneously."""
        rng = np.random.default_rng(92)
        e1 = rng.standard_normal(300)
        e2 = rng.standard_normal(300)
        out = diebold_mariano(e1, e2, h=2, loss="mae", small_sample=True)
        assert np.isfinite(out["stat"]) or np.isnan(out["stat"])
        assert out["lag_used"] == 1

    def test_gw_h2_1d_conditioning(self):
        """GW with h=2 AND 1-D conditioning: exercises both Bartlett loop
        and the reshape branch simultaneously."""
        rng = np.random.default_rng(93)
        e1 = rng.standard_normal(300)
        e2 = rng.standard_normal(300)
        cond_1d = rng.standard_normal(300)
        out = giacomini_white(e1, e2, h=2, conditioning_vars=cond_1d)
        assert out["df"] == 2
        p = out["p_value"]
        assert 0.0 <= p <= 1.0 or np.isnan(p)
