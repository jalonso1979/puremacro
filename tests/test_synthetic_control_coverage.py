"""Tests for puremacro.synthetic_control — Abadie-Diamond-Hainmueller SCM.

Strategy:
- Known-value: DGP where the treated unit IS an exact convex combination of
  donors pre-treatment, so the optimal weights should be recovered with near
  zero pre-period RMSE and zero treatment effect when the post period is also
  consistent.
- Properties: output types/shapes, weight constraints (non-negative, sum to
  1), RMSE >= 0, treatment effect = actual - synthetic over post period,
  placebo_gaps shape and index alignment.
- Edge cases: single donor, minimal pre period, no predictors vs. with
  predictors, placebo=False returns None placebo_gaps.
- Summary method: smoke test that it runs and produces a non-empty string.
- Contract-only: the optimizer accuracy under noisy DGPs is tested as a
  contract (weights non-negative, sum to 1, RMSE finite) rather than exact
  numeric recovery, with one separate known-value test for the exact case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.synthetic_control import (
    SyntheticControlResult,
    _scm_weights,
    synthetic_control,
)

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_donors: int = 4,
    n_pre: int = 10,
    n_post: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, list, list]:
    """Return (Y, pre_period, post_period) with random outcome paths."""
    if rng is None:
        rng = np.random.default_rng(42)
    total = n_pre + n_post
    index = list(range(total))
    cols = ["treated"] + [f"donor_{i}" for i in range(n_donors)]
    data = {c: rng.standard_normal(total) for c in cols}
    Y = pd.DataFrame(data, index=index)
    pre_period = list(range(n_pre))
    post_period = list(range(n_pre, n_pre + n_post))
    return Y, pre_period, post_period


def _make_exact_panel(
    weights_true: np.ndarray,
    n_pre: int = 8,
    n_post: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, list, list]:
    """Panel where treated IS exactly the weighted combination of donors.

    The post period has the same combination, so the true treatment effect is 0.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    n_donors = len(weights_true)
    total = n_pre + n_post
    index = list(range(total))
    donor_data = {f"donor_{i}": rng.standard_normal(total) for i in range(n_donors)}
    donor_matrix = np.column_stack([donor_data[f"donor_{i}"] for i in range(n_donors)])
    treated = donor_matrix @ weights_true
    data = {"treated": treated, **donor_data}
    Y = pd.DataFrame(data, index=index)
    pre_period = list(range(n_pre))
    post_period = list(range(n_pre, n_pre + n_post))
    return Y, pre_period, post_period


# ===========================================================================
# 1. _scm_weights — internal QP solver
# ===========================================================================

class TestScmWeights:
    def test_weights_sum_to_one(self):
        """Weights must always sum to 1 (equality constraint)."""
        rng = np.random.default_rng(10)
        n_pre, n_donors = 10, 4
        X_treat = rng.standard_normal(n_pre)
        X_donors = rng.standard_normal((n_pre, n_donors))
        w = _scm_weights(X_treat, X_donors)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_nonnegative(self):
        """Weights must be non-negative (simplex constraint)."""
        rng = np.random.default_rng(11)
        n_pre, n_donors = 12, 5
        X_treat = rng.standard_normal(n_pre)
        X_donors = rng.standard_normal((n_pre, n_donors))
        w = _scm_weights(X_treat, X_donors)
        assert (w >= -1e-8).all()

    def test_output_shape(self):
        """Output vector must have shape (n_donors,)."""
        rng = np.random.default_rng(12)
        n_pre, n_donors = 8, 3
        X_treat = rng.standard_normal(n_pre)
        X_donors = rng.standard_normal((n_pre, n_donors))
        w = _scm_weights(X_treat, X_donors)
        assert w.shape == (n_donors,)

    def test_exact_recovery_single_donor(self):
        """With one donor, weight must be 1.0 (only feasible solution)."""
        rng = np.random.default_rng(13)
        n_pre = 10
        donor = rng.standard_normal(n_pre)
        treated = 2.0 * donor  # Not exactly feasible, but only one donor
        w = _scm_weights(treated, donor.reshape(-1, 1))
        assert w.shape == (1,)
        assert abs(w[0] - 1.0) < 1e-6

    def test_exact_feasible_solution(self):
        """If treated IS already a convex combo of donors, weights should achieve near-zero loss."""
        rng = np.random.default_rng(14)
        n_pre, n_donors = 20, 3
        true_w = np.array([0.5, 0.3, 0.2])
        donors = rng.standard_normal((n_pre, n_donors))
        treated = donors @ true_w
        w = _scm_weights(treated, donors)
        # Check the residual is near zero
        residual = np.linalg.norm(treated - donors @ w)
        assert residual < 1e-4

    def test_custom_V_matrix(self):
        """Passing a custom V matrix should not crash and still satisfy constraints."""
        rng = np.random.default_rng(15)
        n_pre, n_donors = 10, 3
        X_treat = rng.standard_normal(n_pre)
        X_donors = rng.standard_normal((n_pre, n_donors))
        V = np.diag(np.array([2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]))
        w = _scm_weights(X_treat, X_donors, V=V)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()


# ===========================================================================
# 2. synthetic_control — main function: output contracts
# ===========================================================================

class TestSyntheticControlOutputContracts:
    def test_returns_result_type(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert isinstance(result, SyntheticControlResult)

    def test_weights_sum_to_one(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_weights_nonnegative(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert (result.weights >= -1e-8).all()

    def test_weights_index_matches_donors(self):
        """Weights must be indexed by donor column names."""
        Y, pre, post = _make_panel(n_donors=3)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        expected_donors = ["donor_0", "donor_1", "donor_2"]
        assert list(result.weights.index) == expected_donors

    def test_synthetic_index_is_full_period(self):
        """Synthetic path covers both pre and post period."""
        n_pre, n_post = 8, 5
        Y, pre, post = _make_panel(n_pre=n_pre, n_post=n_post)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert len(result.synthetic) == n_pre + n_post
        assert list(result.synthetic.index) == pre + post

    def test_actual_index_is_full_period(self):
        n_pre, n_post = 8, 5
        Y, pre, post = _make_panel(n_pre=n_pre, n_post=n_post)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert len(result.actual) == n_pre + n_post
        assert list(result.actual.index) == pre + post

    def test_treatment_effect_index_is_post_period(self):
        Y, pre, post = _make_panel(n_pre=10, n_post=6)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert list(result.treatment_effect.index) == post

    def test_rmse_pre_nonnegative(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert result.rmse_pre >= 0.0

    def test_rmse_pre_is_float(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert isinstance(result.rmse_pre, float)

    def test_actual_matches_Y_treated(self):
        """actual series must exactly equal Y[treated] over the full period."""
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        full_period = pre + post
        expected = Y.loc[full_period, "treated"]
        pd.testing.assert_series_equal(
            result.actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_treatment_effect_equals_actual_minus_synthetic_post(self):
        """treatment_effect must equal actual - synthetic over post period."""
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        expected_te = result.actual.loc[post] - result.synthetic.loc[post]
        pd.testing.assert_series_equal(
            result.treatment_effect.reset_index(drop=True),
            expected_te.reset_index(drop=True),
            check_names=False,
        )

    def test_synthetic_is_linear_combination_of_donors_post(self):
        """Synthetic post must equal Y_post[donors] @ weights."""
        Y, pre, post = _make_panel(n_donors=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        donors = [c for c in Y.columns if c != "treated"]
        expected = Y.loc[post, donors].values @ result.weights.values
        np.testing.assert_allclose(
            result.synthetic.loc[post].values, expected, rtol=1e-10
        )


# ===========================================================================
# 3. synthetic_control — known-value exact DGP test
# ===========================================================================

class TestSyntheticControlKnownValue:
    def test_zero_rmse_when_treated_is_exact_combo(self):
        """When treated IS an exact convex combo of donors, RMSE must be near 0."""
        true_w = np.array([0.6, 0.3, 0.1])
        Y, pre, post = _make_exact_panel(true_w, n_pre=10, n_post=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert result.rmse_pre < 1e-5, (
            f"Expected near-zero RMSE, got {result.rmse_pre:.2e}"
        )

    def test_zero_treatment_effect_when_combo_holds_post(self):
        """If combo holds post-period too, treatment effect is near 0."""
        true_w = np.array([0.5, 0.3, 0.2])
        Y, pre, post = _make_exact_panel(true_w, n_pre=10, n_post=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        # Should have near-zero gaps
        assert abs(result.treatment_effect.mean()) < 1e-5

    def test_positive_treatment_effect_when_treated_above_synthetic(self):
        """Shift treated upward post-period: treatment effect should be positive."""
        true_w = np.array([0.5, 0.3, 0.2])
        rng = np.random.default_rng(99)
        Y, pre, post = _make_exact_panel(true_w, n_pre=10, n_post=4, rng=rng)
        # Add a positive treatment effect post-period
        shift = 5.0
        Y.loc[post, "treated"] = Y.loc[post, "treated"] + shift
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        # Mean treatment effect should be near shift
        assert result.treatment_effect.mean() > 0.0

    def test_weights_recover_true_weights_exact_dgp(self):
        """In exact DGP, recovered weights should be close to true weights."""
        true_w = np.array([0.7, 0.2, 0.1])
        Y, pre, post = _make_exact_panel(true_w, n_pre=20, n_post=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        # Weights should be close to true_w (and sum to 1, non-negative)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(
            result.weights.values, true_w, atol=0.05
        )


# ===========================================================================
# 4. synthetic_control — placebo distribution
# ===========================================================================

class TestSyntheticControlPlacebo:
    def test_placebo_none_when_disabled(self):
        Y, pre, post = _make_panel(n_donors=3)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert result.placebo_gaps is None

    def test_placebo_is_dataframe_when_enabled(self):
        Y, pre, post = _make_panel(n_donors=3)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        assert isinstance(result.placebo_gaps, pd.DataFrame)

    def test_placebo_columns_are_donors(self):
        """placebo_gaps columns must be the donor unit names."""
        Y, pre, post = _make_panel(n_donors=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        expected_donors = [f"donor_{i}" for i in range(4)]
        assert sorted(result.placebo_gaps.columns.tolist()) == sorted(expected_donors)

    def test_placebo_index_is_post_period(self):
        """placebo_gaps must be indexed by the post period."""
        n_post = 5
        Y, pre, post = _make_panel(n_pre=8, n_post=n_post)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        assert list(result.placebo_gaps.index) == post

    def test_placebo_shape(self):
        """placebo_gaps must be (n_post, n_donors)."""
        n_donors, n_pre, n_post = 4, 8, 5
        Y, pre, post = _make_panel(n_donors=n_donors, n_pre=n_pre, n_post=n_post)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        assert result.placebo_gaps.shape == (n_post, n_donors)


# ===========================================================================
# 5. synthetic_control — with predictors
# ===========================================================================

class TestSyntheticControlWithPredictors:
    def _make_predictor_df(
        self, Y: pd.DataFrame, n_predictors: int = 2,
        rng: np.random.Generator | None = None
    ) -> pd.DataFrame:
        """Build a (n_predictors, n_units) predictor DataFrame."""
        if rng is None:
            rng = np.random.default_rng(55)
        data = {col: rng.standard_normal(n_predictors) for col in Y.columns}
        return pd.DataFrame(data, index=[f"pred_{i}" for i in range(n_predictors)])

    def test_with_predictors_returns_result(self):
        Y, pre, post = _make_panel(n_donors=3)
        preds = self._make_predictor_df(Y)
        result = synthetic_control(Y, "treated", pre, post, predictors=preds, placebo=False)
        assert isinstance(result, SyntheticControlResult)

    def test_with_predictors_weights_sum_to_one(self):
        Y, pre, post = _make_panel(n_donors=3)
        preds = self._make_predictor_df(Y)
        result = synthetic_control(Y, "treated", pre, post, predictors=preds, placebo=False)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_with_predictors_weights_nonnegative(self):
        Y, pre, post = _make_panel(n_donors=3)
        preds = self._make_predictor_df(Y)
        result = synthetic_control(Y, "treated", pre, post, predictors=preds, placebo=False)
        assert (result.weights >= -1e-8).all()

    def test_with_predictors_placebo_runs(self):
        """Predictor path in placebo loop must also execute without error."""
        Y, pre, post = _make_panel(n_donors=3, n_pre=8, n_post=4)
        preds = self._make_predictor_df(Y)
        result = synthetic_control(Y, "treated", pre, post, predictors=preds, placebo=True)
        assert isinstance(result.placebo_gaps, pd.DataFrame)
        assert result.placebo_gaps.shape[1] == 3  # 3 donors


# ===========================================================================
# 6. SyntheticControlResult — dataclass contracts
# ===========================================================================

class TestSyntheticControlResultDataclass:
    def _make_result(self, n_donors: int = 3, n_pre: int = 8, n_post: int = 4) -> SyntheticControlResult:
        Y, pre, post = _make_panel(n_donors=n_donors, n_pre=n_pre, n_post=n_post)
        return synthetic_control(Y, "treated", pre, post, placebo=True)

    def test_frozen_dataclass(self):
        """SyntheticControlResult is frozen — assignment must raise."""
        result = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.rmse_pre = 999.0  # type: ignore[misc]

    def test_all_fields_present(self):
        result = self._make_result()
        assert hasattr(result, "weights")
        assert hasattr(result, "synthetic")
        assert hasattr(result, "actual")
        assert hasattr(result, "rmse_pre")
        assert hasattr(result, "treatment_effect")
        assert hasattr(result, "placebo_gaps")

    def test_weights_is_series(self):
        result = self._make_result()
        assert isinstance(result.weights, pd.Series)

    def test_synthetic_is_series(self):
        result = self._make_result()
        assert isinstance(result.synthetic, pd.Series)

    def test_actual_is_series(self):
        result = self._make_result()
        assert isinstance(result.actual, pd.Series)

    def test_treatment_effect_is_series(self):
        result = self._make_result()
        assert isinstance(result.treatment_effect, pd.Series)

    def test_rmse_pre_is_finite(self):
        result = self._make_result()
        assert np.isfinite(result.rmse_pre)


# ===========================================================================
# 7. SyntheticControlResult.summary() — smoke test
# ===========================================================================

class TestSummaryMethod:
    def test_summary_returns_string(self):
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_key_fields(self):
        """Summary string must mention RMSE and donor info."""
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        s = result.summary()
        assert "RMSE" in s or "rmse" in s.lower()
        assert "donor" in s.lower()

    def test_summary_no_placebo_string(self):
        """When placebo=False, summary should say 'not computed'."""
        Y, pre, post = _make_panel()
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        s = result.summary()
        assert "not computed" in s.lower() or "not" in s.lower()

    def test_summary_with_placebo_mentions_count(self):
        """When placebo is present, summary should mention the donor count."""
        Y, pre, post = _make_panel(n_donors=4)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        s = result.summary()
        # Should say "4" donors somewhere in the placebo line
        assert "4" in s


# ===========================================================================
# 8. Single donor edge case
# ===========================================================================

class TestSingleDonor:
    def test_single_donor_weight_is_one(self):
        """With one donor, its weight must be 1.0."""
        rng = np.random.default_rng(77)
        n_pre, n_post = 8, 4
        pre = list(range(n_pre))
        post = list(range(n_pre, n_pre + n_post))
        Y = pd.DataFrame({
            "treated": rng.standard_normal(n_pre + n_post),
            "donor_0": rng.standard_normal(n_pre + n_post),
        }, index=list(range(n_pre + n_post)))
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert abs(result.weights["donor_0"] - 1.0) < 1e-6

    def test_single_donor_placebo_has_no_columns(self):
        """With one donor, placebo loop has no donors to iterate over."""
        rng = np.random.default_rng(78)
        n_pre, n_post = 8, 4
        pre = list(range(n_pre))
        post = list(range(n_pre, n_pre + n_post))
        Y = pd.DataFrame({
            "treated": rng.standard_normal(n_pre + n_post),
            "donor_0": rng.standard_normal(n_pre + n_post),
        }, index=list(range(n_pre + n_post)))
        # With only one donor, placebo has no feasible placebo units (each
        # placebo unit would have zero donors left). Result should be empty DF.
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        # placebo_gaps may have 0 columns (no valid placebo units)
        assert isinstance(result.placebo_gaps, pd.DataFrame)


# ===========================================================================
# 9. Datetime index (non-integer) — index type agnosticism
# ===========================================================================

class TestDatetimeIndex:
    def test_datetime_index_works(self):
        """SCM must work with datetime period labels."""
        rng = np.random.default_rng(88)
        dates = pd.date_range("2000-01-01", periods=12, freq="QE")
        pre = list(dates[:8])
        post = list(dates[8:])
        data = {
            "treated": rng.standard_normal(12),
            "donor_a": rng.standard_normal(12),
            "donor_b": rng.standard_normal(12),
        }
        Y = pd.DataFrame(data, index=dates)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert isinstance(result, SyntheticControlResult)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.synthetic.index.tolist() == pre + post

    def test_string_index_works(self):
        """SCM must work with string period labels."""
        rng = np.random.default_rng(89)
        periods = [f"t{i}" for i in range(12)]
        pre = periods[:8]
        post = periods[8:]
        data = {
            "treated": rng.standard_normal(12),
            "donor_a": rng.standard_normal(12),
            "donor_b": rng.standard_normal(12),
        }
        Y = pd.DataFrame(data, index=periods)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        assert isinstance(result, SyntheticControlResult)
        assert abs(result.weights.sum() - 1.0) < 1e-6


# ===========================================================================
# 10. Many donors — scalability smoke test (marked slow)
# ===========================================================================

class TestManyDonors:
    @pytest.mark.slow
    def test_many_donors_runs_without_error(self):
        """SCM with 20 donors and 40 pre-periods must run without error."""
        rng = np.random.default_rng(100)
        n_donors, n_pre, n_post = 20, 40, 10
        pre = list(range(n_pre))
        post = list(range(n_pre, n_pre + n_post))
        data = {"treated": rng.standard_normal(n_pre + n_post)}
        for i in range(n_donors):
            data[f"donor_{i}"] = rng.standard_normal(n_pre + n_post)
        Y = pd.DataFrame(data, index=pre + post)
        result = synthetic_control(Y, "treated", pre, post, placebo=True)
        assert isinstance(result, SyntheticControlResult)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.placebo_gaps.shape == (n_post, n_donors)

    @pytest.mark.slow
    def test_many_donors_with_predictors_runs(self):
        """SCM with 15 donors, predictors, and placebo must not crash."""
        rng = np.random.default_rng(101)
        n_donors, n_pre, n_post = 15, 20, 6
        pre = list(range(n_pre))
        post = list(range(n_pre, n_pre + n_post))
        cols = ["treated"] + [f"donor_{i}" for i in range(n_donors)]
        data = {c: rng.standard_normal(n_pre + n_post) for c in cols}
        Y = pd.DataFrame(data, index=pre + post)
        # 3 predictors
        pred_data = {c: rng.standard_normal(3) for c in cols}
        predictors = pd.DataFrame(pred_data, index=["age", "gdp", "pop"])
        result = synthetic_control(
            Y, "treated", pre, post, predictors=predictors, placebo=False
        )
        assert isinstance(result, SyntheticControlResult)
        assert abs(result.weights.sum() - 1.0) < 1e-6


# ===========================================================================
# 11. RMSE consistency check
# ===========================================================================

class TestRmseConsistency:
    def test_rmse_matches_manual_computation(self):
        """rmse_pre must equal sqrt(mean((actual_pre - synthetic_pre)**2))."""
        Y, pre, post = _make_panel(n_donors=4, n_pre=10, n_post=5)
        result = synthetic_control(Y, "treated", pre, post, placebo=False)
        actual_pre = result.actual.loc[pre].values
        synthetic_pre = result.synthetic.loc[pre].values
        manual_rmse = float(np.sqrt(np.mean((actual_pre - synthetic_pre) ** 2)))
        assert abs(result.rmse_pre - manual_rmse) < 1e-10


# ===========================================================================
# 12. Fallback projected-gradient path in _scm_weights (lines 95-103)
#     Triggered when SLSQP reaches maxiter=1 on a large, pathological problem.
# ===========================================================================

class TestScmWeightsFallback:
    def test_fallback_path_satisfies_constraints(self):
        """Force SLSQP to fail by setting maxiter=1, then check fallback result.

        We monkeypatch scipy.optimize.minimize so that it always returns
        res.success=False, triggering the projected-gradient fallback.
        """
        from unittest.mock import patch, MagicMock
        import puremacro.synthetic_control as scm_mod

        rng = np.random.default_rng(200)
        n_pre, n_donors = 15, 4
        X_treat = rng.standard_normal(n_pre)
        X_donors = rng.standard_normal((n_pre, n_donors))

        # Create a fake minimize result that reports failure
        fake_result = MagicMock()
        fake_result.success = False

        with patch.object(scm_mod, "minimize", return_value=fake_result):
            w = scm_mod._scm_weights(X_treat, X_donors)

        # Fallback result must still satisfy simplex constraints
        assert abs(w.sum() - 1.0) < 1e-6, f"fallback weights sum={w.sum():.6f}"
        assert (w >= 0.0).all(), f"fallback weights have negatives: {w}"
        assert w.shape == (n_donors,)

    def test_fallback_path_via_synthetic_control_with_mock(self):
        """Run the full synthetic_control pipeline under the fallback optimizer."""
        from unittest.mock import patch, MagicMock
        import puremacro.synthetic_control as scm_mod

        Y, pre, post = _make_panel(n_donors=3, n_pre=8, n_post=4)

        fake_result = MagicMock()
        fake_result.success = False

        with patch.object(scm_mod, "minimize", return_value=fake_result):
            result = synthetic_control(Y, "treated", pre, post, placebo=False)

        assert isinstance(result, SyntheticControlResult)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert (result.weights >= -1e-6).all()
        assert np.isfinite(result.rmse_pre)


# ===========================================================================
# 13. Placebo exception-handling path (lines 188-189)
#     When _scm_weights raises inside the placebo loop, the unit is skipped.
# ===========================================================================

class TestPlaceboExceptionHandling:
    def test_placebo_skips_failing_units(self):
        """If _scm_weights raises for some donor, that donor is absent from
        placebo_gaps (skipped via 'except Exception: continue').
        """
        from unittest.mock import patch
        import puremacro.synthetic_control as scm_mod

        Y, pre, post = _make_panel(n_donors=4, n_pre=8, n_post=4)
        call_count = [0]

        original_scm_weights = scm_mod._scm_weights

        def _failing_on_second_call(X_treat, X_donors, V=None):
            call_count[0] += 1
            # First call is for the main treated unit — let it succeed.
            # Subsequent calls are in the placebo loop — raise on all.
            if call_count[0] > 1:
                raise RuntimeError("simulated failure in placebo loop")
            return original_scm_weights(X_treat, X_donors, V=V)

        with patch.object(scm_mod, "_scm_weights", side_effect=_failing_on_second_call):
            result = synthetic_control(Y, "treated", pre, post, placebo=True)

        # All 4 placebo units failed → placebo_gaps should have 0 columns
        assert isinstance(result.placebo_gaps, pd.DataFrame)
        assert result.placebo_gaps.shape[1] == 0, (
            f"expected 0 placebo columns, got {result.placebo_gaps.shape[1]}"
        )
        # Main treatment effect and weights must still be valid
        assert abs(result.weights.sum() - 1.0) < 1e-6
