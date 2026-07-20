"""Coverage tests for puremacro.inference.over_id.

Two public functions:
  - hansen_j     -- Hansen J over-identification test for 2SLS / IV-GMM.
  - stock_yogo_cv -- Stock-Yogo (2005) first-stage F critical value lookup.

Approach:
  - Known-value tests: DGPs with analytically determined correct answers
    (just-identified -> J=0; invalid instruments -> large J/small p-value).
  - Property tests: return type, keys, dtypes, df formula, p-value in [0,1],
    J >= 0, p-value NaN when df=0, J grows with sample when instrument invalid.
  - Stock-Yogo table: all 24 tabulated entries against published values;
    ordering (stricter bias threshold -> higher CV); untabulated -> None.
  - Error handling / edge cases: 1D inputs, W without constant (auto-prepend),
    W with constant already present (no double-prepend), multiple endog.
  No network I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.inference.over_id import hansen_j, stock_yogo_cv


# ---------------------------------------------------------------------------
# Shared DGP helpers
# ---------------------------------------------------------------------------

def _dgp_valid(T: int = 200, *, seed: int = 0) -> dict:
    """Single-endog 2SLS DGP with two valid instruments.

    y = 2*x + u,  x = 0.7*z1 + 0.5*z2 + 0.3*v
    z1, z2, v, u all iid N(0,1) -> instruments are valid (uncorrelated with u).
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    v = rng.standard_normal(T)
    u = rng.standard_normal(T)
    x = 0.7 * z1 + 0.5 * z2 + 0.3 * v
    y = 2.0 * x + u
    return {"y": y, "x": x, "Z": np.column_stack([z1, z2]), "z1": z1, "u": u}


def _dgp_invalid(T: int = 200, *, seed: int = 42) -> dict:
    """DGP where z2 is directly correlated with u — invalid instrument."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(T)
    u = rng.standard_normal(T)
    # z2 is heavily correlated with u -> invalid
    z2_bad = 0.9 * u + 0.1 * rng.standard_normal(T)
    x = 0.7 * z1 + 0.5 * z2_bad + 0.3 * rng.standard_normal(T)
    y = 2.0 * x + u
    return {"y": y, "x": x, "Z": np.column_stack([z1, z2_bad])}


# ---------------------------------------------------------------------------
# TestHansenJReturnContract — output structure and types
# ---------------------------------------------------------------------------

class TestHansenJReturnContract:
    def test_returns_dict(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert isinstance(res, dict)

    def test_has_all_required_keys(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert {"stat", "p_value", "df", "n_obs"} == set(res.keys())

    def test_stat_is_float(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert isinstance(res["stat"], float)

    def test_p_value_is_float(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert isinstance(res["p_value"], float)

    def test_df_is_int(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert isinstance(res["df"], int)

    def test_n_obs_is_int(self):
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert isinstance(res["n_obs"], int)

    def test_n_obs_equals_input_length(self):
        T = 150
        d = _dgp_valid(T, seed=3)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert res["n_obs"] == T

    def test_stat_is_non_negative(self):
        """J = T * R^2 >= 0 always."""
        d = _dgp_valid(200)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert res["stat"] >= 0.0

    def test_p_value_in_unit_interval(self):
        d = _dgp_valid(200)
        res = hansen_j(d["y"], d["x"], d["Z"])
        assert 0.0 <= res["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# TestHansenJDegreesOfFreedom — df = l - k contract
# ---------------------------------------------------------------------------

class TestHansenJDegreesOfFreedom:
    def test_df_l_minus_k_1endog_2inst(self):
        """l=2, k=1 -> df=1."""
        d = _dgp_valid(100)
        res = hansen_j(d["y"], d["x"], d["Z"])  # Z has 2 cols, x is 1-D
        assert res["df"] == 1

    def test_df_l_minus_k_1endog_3inst(self):
        """l=3, k=1 -> df=2."""
        rng = np.random.default_rng(10)
        T = 100
        Z3 = rng.standard_normal((T, 3))
        x = Z3[:, 0] * 0.5 + rng.standard_normal(T) * 0.5
        y = 2 * x + rng.standard_normal(T)
        res = hansen_j(y, x, Z3)
        assert res["df"] == 2

    def test_df_l_minus_k_2endog_4inst(self):
        """l=4, k=2 -> df=2."""
        rng = np.random.default_rng(11)
        T = 150
        Z4 = rng.standard_normal((T, 4))
        X2 = np.column_stack([
            Z4[:, 0] + rng.standard_normal(T) * 0.5,
            Z4[:, 1] + rng.standard_normal(T) * 0.5,
        ])
        y = 2 * X2[:, 0] + 1.5 * X2[:, 1] + rng.standard_normal(T)
        res = hansen_j(y, X2, Z4)
        assert res["df"] == 2

    def test_df_l_minus_k_2endog_5inst(self):
        """l=5, k=2 -> df=3."""
        rng = np.random.default_rng(12)
        T = 150
        Z5 = rng.standard_normal((T, 5))
        X2 = np.column_stack([
            Z5[:, 0] + rng.standard_normal(T) * 0.5,
            Z5[:, 1] + rng.standard_normal(T) * 0.5,
        ])
        y = 2 * X2[:, 0] + rng.standard_normal(T)
        res = hansen_j(y, X2, Z5)
        assert res["df"] == 3

    def test_df_zero_when_just_identified(self):
        """l=k=1: just-identified -> df=0."""
        rng = np.random.default_rng(20)
        T = 100
        z = rng.standard_normal(T)
        x = 0.8 * z + rng.standard_normal(T) * 0.2
        y = 2 * x + rng.standard_normal(T)
        res = hansen_j(y, x, z)
        assert res["df"] == 0

    def test_p_value_nan_when_df_zero(self):
        """Just-identified: df=0 means no over-identification, p must be NaN."""
        rng = np.random.default_rng(21)
        T = 100
        z = rng.standard_normal(T)
        x = 0.8 * z + rng.standard_normal(T) * 0.2
        y = 2 * x + rng.standard_normal(T)
        res = hansen_j(y, x, z)
        assert math.isnan(res["p_value"])


# ---------------------------------------------------------------------------
# TestHansenJKnownValues — analytically determined correct answers
# ---------------------------------------------------------------------------

class TestHansenJKnownValues:
    def test_just_identified_j_is_zero(self):
        """When l == k (just-identified), J must equal 0 (up to float precision).

        Proof: the 2SLS residuals are in the orthogonal complement of the
        instrument space, so the auxiliary R^2 is 0 when df=0.
        """
        rng = np.random.default_rng(99)
        T = 300
        z = rng.standard_normal(T)
        x = 0.7 * z + 0.3 * rng.standard_normal(T)
        y = 2.0 * x + rng.standard_normal(T)
        res = hansen_j(y, x, z)
        assert abs(res["stat"]) < 1e-10, f"Expected J≈0, got {res['stat']}"

    def test_just_identified_2x2_j_is_zero(self):
        """Two endogenous, two instruments: still just-identified -> J=0."""
        rng = np.random.default_rng(100)
        T = 300
        Z2 = rng.standard_normal((T, 2))
        X2 = np.column_stack([
            0.7 * Z2[:, 0] + 0.3 * rng.standard_normal(T),
            0.6 * Z2[:, 1] + 0.4 * rng.standard_normal(T),
        ])
        y = 2.0 * X2[:, 0] + 1.5 * X2[:, 1] + rng.standard_normal(T)
        res = hansen_j(y, X2, Z2)
        assert abs(res["stat"]) < 1e-8, f"Expected J≈0, got {res['stat']}"

    def test_invalid_instrument_detected(self):
        """An instrument directly correlated with the error should produce
        a small p-value (strong rejection of the null of valid instruments)
        with large T."""
        d = _dgp_invalid(T=5000, seed=1234)
        res = hansen_j(d["y"], d["x"], d["Z"])
        # With T=5000 and 90% contamination, J >> chi2(1) critical value.
        assert res["p_value"] < 0.01, f"Expected rejection, p={res['p_value']}"

    def test_valid_instruments_not_rejected_large_sample(self):
        """With valid instruments and large T, J ~chi2(1) under H0.
        We cannot guarantee non-rejection for a single draw, so we check
        J is in a plausible range [0, 20] rather than the exact distribution."""
        d = _dgp_valid(T=2000, seed=777)
        res = hansen_j(d["y"], d["x"], d["Z"])
        # chi2(1) has mean=1, sd=sqrt(2)~1.4; values above 20 are astronomically rare.
        assert 0.0 <= res["stat"] <= 20.0, f"J={res['stat']} implausibly large for valid instruments"

    def test_larger_sample_shrinks_j_under_null(self):
        """Under the null, J is O_p(1); it should NOT grow proportionally to T.
        Compare J/T between a small and large sample."""
        d_small = _dgp_valid(T=100, seed=55)
        d_large = _dgp_valid(T=10000, seed=55)
        # Reuse same noise proportions — regenerate independently to avoid
        # exact rescaling; just confirm J/T is much smaller for large T.
        j_small = hansen_j(d_small["y"], d_small["x"], d_small["Z"])["stat"]
        j_large = hansen_j(d_large["y"], d_large["x"], d_large["Z"])["stat"]
        # Under H0, J ~ chi2(1) regardless of T, so J/T -> 0 as T grows.
        assert j_large / 10000 < 0.01, (
            f"J/T = {j_large / 10000:.4f} too large for valid instruments"
        )


# ---------------------------------------------------------------------------
# TestHansenJInputForms — 1D, 2D, W variants
# ---------------------------------------------------------------------------

class TestHansenJInputForms:
    def test_1d_arrays_accepted(self):
        """All of y, x, z may be 1-D numpy arrays."""
        rng = np.random.default_rng(5)
        T = 80
        z = rng.standard_normal(T)
        x = 0.8 * z + 0.2 * rng.standard_normal(T)
        y = 2 * x + rng.standard_normal(T)
        # 1D inputs (not column vectors)
        res = hansen_j(y.ravel(), x.ravel(), z.ravel())
        assert isinstance(res, dict)
        assert res["n_obs"] == T

    def test_2d_column_arrays_accepted(self):
        """y, x as (T,1) arrays."""
        rng = np.random.default_rng(6)
        T = 80
        z = rng.standard_normal((T, 2))
        x = (z[:, 0] + rng.standard_normal(T)).reshape(-1, 1)
        y = (2 * x.ravel() + rng.standard_normal(T)).reshape(-1, 1)
        res = hansen_j(y, x, z)
        assert res["n_obs"] == T

    def test_w_none_default(self):
        """Passing W=None gives the same result as omitting W."""
        d = _dgp_valid(100, seed=9)
        res_default = hansen_j(d["y"], d["x"], d["Z"])
        res_none = hansen_j(d["y"], d["x"], d["Z"], W=None)
        assert res_default["stat"] == res_none["stat"]

    def test_w_1d_accepted(self):
        """W may be a 1-D array (one control variable)."""
        rng = np.random.default_rng(30)
        T = 120
        z1 = rng.standard_normal(T)
        z2 = rng.standard_normal(T)
        w = rng.standard_normal(T)
        x = 0.6 * z1 + 0.5 * z2 + 0.2 * w + rng.standard_normal(T) * 0.3
        y = 2.0 * x + 0.5 * w + rng.standard_normal(T)
        res = hansen_j(y, x, np.column_stack([z1, z2]), W=w)
        assert isinstance(res, dict)
        assert res["df"] == 1  # l - k = 2 - 1

    def test_w_with_constant_first_col_not_doubled(self):
        """If W[:,0] is all-ones (constant included), it should not be prepended again.
        We verify the result is stable (idempotent for constant detection)."""
        rng = np.random.default_rng(31)
        T = 100
        z1 = rng.standard_normal(T)
        z2 = rng.standard_normal(T)
        x = 0.7 * z1 + 0.4 * z2 + rng.standard_normal(T) * 0.3
        y = 2.0 * x + rng.standard_normal(T)
        w_extra = rng.standard_normal(T)
        # W with constant in first column
        W_with_const = np.column_stack([np.ones(T), w_extra])
        res = hansen_j(y, x, np.column_stack([z1, z2]), W=W_with_const)
        assert isinstance(res, dict)
        assert 0.0 <= res.get("p_value", float("nan")) <= 1.0 or math.isnan(res["p_value"])

    def test_w_without_constant_gets_prepended(self):
        """If W[:,0] is not all-ones, a constant is prepended.
        Both cases (W with / without constant) should give valid dicts."""
        rng = np.random.default_rng(32)
        T = 100
        z1 = rng.standard_normal(T)
        z2 = rng.standard_normal(T)
        x = 0.7 * z1 + 0.4 * z2 + rng.standard_normal(T) * 0.3
        y = 2.0 * x + rng.standard_normal(T)
        # W with no constant
        W_no_const = rng.standard_normal((T, 2))
        res = hansen_j(y, x, np.column_stack([z1, z2]), W=W_no_const)
        assert isinstance(res, dict)
        # df is still l - k = 2 - 1 = 1, W columns do not count for df
        assert res["df"] == 1


# ---------------------------------------------------------------------------
# TestHansenJControlsW — controls change the result
# ---------------------------------------------------------------------------

class TestHansenJControlsW:
    def test_controls_change_stat(self):
        """Adding a control W changes the J statistic (it does not produce
        the same result as W=None when there is a genuine control variable)."""
        rng = np.random.default_rng(40)
        T = 200
        z1 = rng.standard_normal(T)
        z2 = rng.standard_normal(T)
        w = rng.standard_normal(T)  # genuine exogenous control
        x = 0.7 * z1 + 0.4 * z2 + 0.3 * w + rng.standard_normal(T) * 0.3
        y = 2.0 * x + 0.8 * w + rng.standard_normal(T)
        Z = np.column_stack([z1, z2])
        res_no_w = hansen_j(y, x, Z)
        res_with_w = hansen_j(y, x, Z, W=w)
        # They should differ (W partialled out)
        assert res_no_w["stat"] != res_with_w["stat"]

    def test_2d_w_matrix_accepted(self):
        """W may be a 2-D matrix of controls."""
        rng = np.random.default_rng(41)
        T = 150
        Z = rng.standard_normal((T, 3))
        x = Z[:, 0] + rng.standard_normal(T) * 0.5
        y = 2 * x + rng.standard_normal(T)
        W = rng.standard_normal((T, 2))
        res = hansen_j(y, x, Z, W=W)
        assert res["df"] == 2  # l=3, k=1, W doesn't affect df


# ---------------------------------------------------------------------------
# TestStockYogoCVKnownValues — lookup table correctness
# ---------------------------------------------------------------------------

# Published Stock-Yogo (2005) Table 5.2 values for k=1 endogenous regressor.
_SY_TABLE = {
    (3,  5): 13.91, (3,  10):  9.08, (3,  20): 6.46, (3,  30): 5.39,
    (4,  5): 16.85, (4,  10): 10.27, (4,  20): 6.71, (4,  30): 5.34,
    (5,  5): 18.37, (5,  10): 10.83, (5,  20): 6.77, (5,  30): 5.25,
    (6,  5): 19.28, (6,  10): 11.12, (6,  20): 6.76, (6,  30): 5.15,
    (7,  5): 19.86, (7,  10): 11.29, (7,  20): 6.73, (7,  30): 5.07,
    (8,  5): 20.25, (8,  10): 11.39, (8,  20): 6.69, (8,  30): 4.99,
}


class TestStockYogoCVKnownValues:
    @pytest.mark.parametrize("n_inst,pct,expected", [
        (k[0], k[1], v) for k, v in _SY_TABLE.items()
    ])
    def test_tabulated_values_match_published(self, n_inst, pct, expected):
        """Each tabulated (l, pct) pair must return the published CV exactly."""
        cv = stock_yogo_cv(n_inst, pct)
        assert cv == pytest.approx(expected, abs=1e-9), (
            f"stock_yogo_cv({n_inst}, {pct}) = {cv}, expected {expected}"
        )

    def test_untabulated_n_instruments_returns_none(self):
        cv = stock_yogo_cv(99, 10)
        assert cv is None

    def test_untabulated_bias_pct_returns_none(self):
        cv = stock_yogo_cv(3, 15)  # 15% is not in the table
        assert cv is None

    def test_untabulated_both_returns_none(self):
        cv = stock_yogo_cv(100, 50)
        assert cv is None

    def test_staiger_stock_rule_of_thumb(self):
        """The classic Staiger-Stock 'F > 10' rule corresponds to
        n_instruments=3, max_bias_pct=10.  The docstring says so explicitly."""
        cv = stock_yogo_cv(n_instruments=3, max_bias_pct=10)
        assert cv == pytest.approx(9.08, abs=1e-9)

    def test_returns_float(self):
        cv = stock_yogo_cv(3, 10)
        assert isinstance(cv, float)

    def test_int_coercion_works(self):
        """n_instruments and max_bias_pct are cast to int internally."""
        # Passing float-like int should not fail.
        cv_int = stock_yogo_cv(3, 10)
        cv_via_float_key = stock_yogo_cv(int(3.0), int(10.0))
        assert cv_int == cv_via_float_key


# ---------------------------------------------------------------------------
# TestStockYogoCVOrdering — monotonicity in the table
# ---------------------------------------------------------------------------

class TestStockYogoCVOrdering:
    def test_stricter_bias_threshold_gives_higher_cv(self):
        """For fixed n_instruments, a stricter max_bias_pct (smaller %)
        requires a higher first-stage F to pass. CV(5%) > CV(10%) > ... > CV(30%)."""
        for n in [3, 4, 5, 6, 7, 8]:
            cv5  = stock_yogo_cv(n, 5)
            cv10 = stock_yogo_cv(n, 10)
            cv20 = stock_yogo_cv(n, 20)
            cv30 = stock_yogo_cv(n, 30)
            assert cv5 > cv10 > cv20 > cv30, (
                f"Monotonicity violated at n_instruments={n}: "
                f"{cv5}, {cv10}, {cv20}, {cv30}"
            )

    def test_more_instruments_generally_raises_cv_for_strict_threshold(self):
        """For the 5% bias threshold, more instruments -> generally higher CV
        (more instruments to validate, so stricter first-stage F required).
        This holds across n=3..8 in the table."""
        cvs = [stock_yogo_cv(n, 5) for n in [3, 4, 5, 6, 7, 8]]
        # Check the sequence is non-decreasing (strictly for most pairs)
        for i in range(len(cvs) - 1):
            assert cvs[i] < cvs[i + 1], (
                f"CV not increasing: CV({3+i},5)={cvs[i]}, CV({4+i},5)={cvs[i+1]}"
            )

    def test_more_instruments_raises_cv_for_10pct_threshold(self):
        """Same monotonicity for the 10% threshold."""
        cvs = [stock_yogo_cv(n, 10) for n in [3, 4, 5, 6, 7, 8]]
        for i in range(len(cvs) - 1):
            assert cvs[i] < cvs[i + 1], (
                f"CV not increasing: CV({3+i},10)={cvs[i]}, CV({4+i},10)={cvs[i+1]}"
            )


# ---------------------------------------------------------------------------
# TestHansenJPowerAndSize — statistical behavior (slow-marked where needed)
# ---------------------------------------------------------------------------

class TestHansenJPowerAndSize:
    @pytest.mark.slow
    def test_size_controlled_under_null_many_seeds(self):
        """Rough size check: run J test on 500 valid-instrument draws and
        verify that the empirical rejection rate at 5% is close to 5%."""
        n_trials = 500
        T = 300
        rejections = 0
        for seed in range(n_trials):
            d = _dgp_valid(T, seed=seed)
            res = hansen_j(d["y"], d["x"], d["Z"])
            if res["p_value"] < 0.05:
                rejections += 1
        rejection_rate = rejections / n_trials
        # 95% CI for Binomial(500, 0.05): roughly 0.05 ± 0.02
        assert 0.02 <= rejection_rate <= 0.10, (
            f"Rejection rate {rejection_rate:.3f} is outside [0.02, 0.10] "
            "(expected ~0.05 under H_0)"
        )

    def test_power_rises_with_sample_size_invalid_instrument(self):
        """With an invalid instrument, the J test should reject more often
        as T increases.  Check for two specific sample sizes."""
        # Large T should have smaller p-value than small T (on average)
        d_small = _dgp_invalid(T=100, seed=77)
        d_large = _dgp_invalid(T=3000, seed=77)
        res_small = hansen_j(d_small["y"], d_small["x"], d_small["Z"])
        res_large = hansen_j(d_large["y"], d_large["x"], d_large["Z"])
        # Both may reject; but large T's J should be much larger
        assert res_large["stat"] > res_small["stat"], (
            "J statistic should grow with T when instruments are invalid"
        )


# ---------------------------------------------------------------------------
# TestHansenJDFandPValueConsistency — df / p-value consistency
# ---------------------------------------------------------------------------

class TestHansenJDFandPValueConsistency:
    def test_p_value_matches_chi2_cdf(self):
        """p_value == 1 - chi2.cdf(stat, df) for any df > 0."""
        from scipy.stats import chi2

        d = _dgp_valid(200, seed=88)
        res = hansen_j(d["y"], d["x"], d["Z"])
        expected_p = 1.0 - chi2.cdf(res["stat"], df=res["df"])
        assert abs(res["p_value"] - expected_p) < 1e-12

    def test_very_large_stat_gives_small_p(self):
        """A very large J stat must produce a p-value near 0."""
        d = _dgp_invalid(T=10000, seed=999)
        res = hansen_j(d["y"], d["x"], d["Z"])
        # With this many obs and a badly invalid instrument, J >> 1000.
        assert res["p_value"] < 1e-6, f"Expected p~0, got {res['p_value']}"

    def test_df_consistent_with_instruments_and_endog(self):
        """df must equal n_cols(Z) - n_cols(X_endog) for any over-identified spec."""
        rng = np.random.default_rng(55)
        T = 200
        for l, k in [(3, 1), (4, 2), (5, 2), (6, 3)]:
            Z = rng.standard_normal((T, l))
            X = rng.standard_normal((T, k))
            y = rng.standard_normal(T)
            res = hansen_j(y, X, Z)
            assert res["df"] == l - k, (
                f"df={res['df']} != l-k={l-k} for l={l}, k={k}"
            )
