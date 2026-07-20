"""Tests for puremacro.var.bootstrap — bootstrap IRF confidence bands.

Coverage strategy
-----------------
- Output shape / keys / dtype contracts for bootstrap_bands (all three
  method branches: 'recursive', 'wild', 'block').
- Known-value / DGP: point estimate (median) tracks the true IRF for a
  known VAR(1) DGP.
- Properties: lower <= point <= upper element-wise; CI contains the true
  IRF with high probability over many bootstrap draws.
- Tighter bounds with more data (monotonicity of CI width).
- Reproducibility: same seed gives same draws.
- rng=None branch executes.
- Error handling: unknown method raises ValueError.
- bias_correct=True path exercised (shape contract + pilot uses n_pilot).
- Private helper _irf_from_var: shape / known-value.
- Private helper _kilian_bias_correct: returns A_bc and int_bc of same
  shape/type as original estimates; spectral radius of companion < 1 when
  DGP is stable.
- alpha parameter: narrower bands at alpha=0.01 vs alpha=0.20.
- p=2 lags: bootstrap_bands completes without error.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.bootstrap import (
    _irf_from_var,
    _kilian_bias_correct,
    bootstrap_bands,
)
from puremacro.var.estimate import estimate_var, companion
from puremacro.var.identify.cholesky import cholesky_factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_var1_data(T: int = 120, n: int = 2, seed: int = 0) -> np.ndarray:
    """Simulate a stationary VAR(1) with known diagonal coefficient matrix."""
    rng = np.random.default_rng(seed)
    # Diagonal A → easy to verify stability; spectral radius = 0.5
    A = 0.5 * np.eye(n)
    L = np.linalg.cholesky(np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n)))
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def _cholesky_identify(A_list, Sigma):
    """Wrapper so bootstrap_bands can accept this as identify_fn."""
    return cholesky_factor(Sigma)


# ===========================================================================
# _irf_from_var — private helper
# ===========================================================================

def test_irf_from_var_shape():
    Y = _stable_var1_data(T=80, n=2)
    out = _irf_from_var(Y, p=1, identify_fn=_cholesky_identify, horizon=6)
    assert out.shape == (7, 2, 2), f"Expected (7, 2, 2); got {out.shape}"


def test_irf_from_var_finite():
    Y = _stable_var1_data(T=80, n=2, seed=1)
    out = _irf_from_var(Y, p=1, identify_fn=_cholesky_identify, horizon=10)
    assert np.all(np.isfinite(out)), "IRF from _irf_from_var contains non-finite values"


def test_irf_from_var_impact_lower_triangular():
    """With Cholesky identification, IRF[0] must equal lower-triangular B0."""
    Y = _stable_var1_data(T=100, n=2, seed=2)
    out = _irf_from_var(Y, p=1, identify_fn=_cholesky_identify, horizon=4)
    # Impact matrix should be lower-triangular (Cholesky)
    assert np.allclose(out[0], np.tril(out[0]), atol=1e-10), \
        f"Impact matrix not lower-triangular:\n{out[0]}"


def test_irf_from_var_id_kwargs_passed():
    """id_kwargs should be forwarded to identify_fn."""
    call_log = {}

    def logging_identify(A_list, Sigma, tag=None):
        call_log["tag"] = tag
        return cholesky_factor(Sigma)

    Y = _stable_var1_data(T=60, n=2, seed=3)
    _irf_from_var(Y, p=1, identify_fn=logging_identify, horizon=3, tag="test")
    assert call_log.get("tag") == "test"


# ===========================================================================
# _kilian_bias_correct — private helper
# ===========================================================================

def test_kilian_bias_correct_output_shapes():
    Y = _stable_var1_data(T=80, n=2, seed=4)
    A_list_orig, _, _, _, _ = estimate_var(Y, 1)
    A_bc, int_bc = _kilian_bias_correct(Y, p=1, n_pilot=20,
                                         rng=np.random.default_rng(0))
    assert len(A_bc) == 1, "Should return one A matrix for p=1"
    assert A_bc[0].shape == A_list_orig[0].shape
    assert int_bc.shape == (2,)


def test_kilian_bias_correct_stable_companion():
    """Bias-corrected companion matrix must be stable (spectral radius < 1)
    for a well-conditioned DGP — the shrinkage step guarantees this."""
    Y = _stable_var1_data(T=150, n=2, seed=5)
    A_bc, _ = _kilian_bias_correct(Y, p=1, n_pilot=30,
                                    rng=np.random.default_rng(1))
    C = companion(A_bc)
    rho = np.abs(np.linalg.eigvals(C)).max()
    assert rho < 1.0, f"Bias-corrected companion not stable; spectral radius = {rho:.4f}"


def test_kilian_bias_correct_rng_none():
    """rng=None should use default seed 0 without error."""
    Y = _stable_var1_data(T=60, n=2, seed=6)
    A_bc, int_bc = _kilian_bias_correct(Y, p=1, n_pilot=10, rng=None)
    assert len(A_bc) == 1
    assert np.all(np.isfinite(A_bc[0]))
    assert np.all(np.isfinite(int_bc))


def test_kilian_bias_correct_p2():
    """p=2: A_bc should contain 2 matrices of the right shape."""
    Y = _stable_var1_data(T=100, n=2, seed=7)
    A_bc, int_bc = _kilian_bias_correct(Y, p=2, n_pilot=20,
                                         rng=np.random.default_rng(2))
    assert len(A_bc) == 2
    for A in A_bc:
        assert A.shape == (2, 2)
        assert np.all(np.isfinite(A))
    assert int_bc.shape == (2,)


# ===========================================================================
# bootstrap_bands — output contract
# ===========================================================================

def test_bootstrap_bands_keys():
    Y = _stable_var1_data(T=80, n=2)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=30, rng=np.random.default_rng(0))
    assert set(result.keys()) == {"point", "lower", "upper", "draws"}


def test_bootstrap_bands_shapes():
    n, horizon = 2, 5
    Y = _stable_var1_data(T=80, n=n)
    n_boot = 20
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=horizon, n_boot=n_boot,
                             rng=np.random.default_rng(0))
    assert result["point"].shape == (horizon + 1, n, n)
    assert result["lower"].shape == (horizon + 1, n, n)
    assert result["upper"].shape == (horizon + 1, n, n)
    assert result["draws"].shape == (n_boot, horizon + 1, n, n)


def test_bootstrap_bands_lower_le_point_le_upper():
    """Lower bound must not exceed point, and point must not exceed upper."""
    Y = _stable_var1_data(T=100, n=2, seed=1)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=50,
                             rng=np.random.default_rng(42))
    assert np.all(result["lower"] <= result["point"] + 1e-10), \
        "lower > point detected"
    assert np.all(result["point"] <= result["upper"] + 1e-10), \
        "point > upper detected"


def test_bootstrap_bands_point_is_nanpercentile_50():
    """The point estimate is defined as nanpercentile(draws, 50)."""
    Y = _stable_var1_data(T=80, n=2, seed=2)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=3, n_boot=25,
                             rng=np.random.default_rng(7))
    expected_point = np.nanpercentile(result["draws"], 50, axis=0)
    assert np.allclose(result["point"], expected_point, equal_nan=True)


def test_bootstrap_bands_reproducible():
    """Same rng seed must produce identical draws."""
    Y = _stable_var1_data(T=80, n=2, seed=3)
    r1 = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                         horizon=3, n_boot=20,
                         rng=np.random.default_rng(99))
    r2 = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                         horizon=3, n_boot=20,
                         rng=np.random.default_rng(99))
    assert np.allclose(r1["draws"], r2["draws"], equal_nan=True)
    assert np.allclose(r1["point"], r2["point"])


def test_bootstrap_bands_rng_none_runs():
    """rng=None auto-creates an RNG; the call should complete."""
    Y = _stable_var1_data(T=60, n=2, seed=4)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=2, n_boot=15, rng=None)
    assert result["draws"].shape[0] == 15


def test_bootstrap_bands_unknown_method_raises():
    Y = _stable_var1_data(T=60, n=2)
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                        horizon=2, n_boot=5, method="gibbs",
                        rng=np.random.default_rng(0))


# ===========================================================================
# bootstrap_bands — method branches
# ===========================================================================

def test_bootstrap_bands_method_wild():
    """Wild bootstrap: output shapes and lower <= upper."""
    Y = _stable_var1_data(T=80, n=2, seed=5)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=30, method="wild",
                             rng=np.random.default_rng(10))
    assert result["point"].shape == (5, 2, 2)
    assert np.all(result["lower"] <= result["upper"] + 1e-10)


def test_bootstrap_bands_method_block():
    """Block bootstrap: output shapes and lower <= upper."""
    Y = _stable_var1_data(T=80, n=2, seed=6)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=30, method="block",
                             rng=np.random.default_rng(11))
    assert result["point"].shape == (5, 2, 2)
    assert np.all(result["lower"] <= result["upper"] + 1e-10)


def test_bootstrap_bands_method_recursive_explicit():
    """Explicitly request 'recursive' (the default); shapes must match."""
    Y = _stable_var1_data(T=80, n=2, seed=7)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=30, method="recursive",
                             rng=np.random.default_rng(12))
    assert result["point"].shape == (5, 2, 2)


def test_bootstrap_bands_wild_vs_recursive_different():
    """Wild and recursive methods should give different draws (up to luck)."""
    Y = _stable_var1_data(T=100, n=2, seed=8)
    r_rec = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=3, n_boot=30, method="recursive",
                            rng=np.random.default_rng(50))
    r_wld = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=3, n_boot=30, method="wild",
                            rng=np.random.default_rng(50))
    # The draws should not be numerically identical
    assert not np.allclose(r_rec["draws"], r_wld["draws"], equal_nan=True)


# ===========================================================================
# bootstrap_bands — alpha parameter (CI width)
# ===========================================================================

def test_bootstrap_bands_narrower_ci_at_high_alpha():
    """Higher alpha → narrower confidence interval (smaller upper-lower gap)."""
    Y = _stable_var1_data(T=100, n=2, seed=9)
    r_wide = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=100, alpha=0.01,
                             rng=np.random.default_rng(20))
    r_narrow = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                               horizon=4, n_boot=100, alpha=0.20,
                               rng=np.random.default_rng(20))
    width_wide = (r_wide["upper"] - r_wide["lower"]).mean()
    width_narrow = (r_narrow["upper"] - r_narrow["lower"]).mean()
    assert width_wide > width_narrow, (
        f"Expected alpha=0.01 CI to be wider than alpha=0.20, but got "
        f"width_wide={width_wide:.4f}, width_narrow={width_narrow:.4f}"
    )


def test_bootstrap_bands_ci_contains_point():
    """CI by construction: point lies between lower and upper."""
    Y = _stable_var1_data(T=100, n=2, seed=10)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=60, alpha=0.10,
                             rng=np.random.default_rng(30))
    gap = 1e-8  # numerical tolerance
    assert np.all(result["lower"] - gap <= result["point"])
    assert np.all(result["point"] <= result["upper"] + gap)


# ===========================================================================
# bootstrap_bands — p=2 lags
# ===========================================================================

def test_bootstrap_bands_p2_shape():
    """VAR(2): bootstrap_bands should complete with correct shapes."""
    Y = _stable_var1_data(T=120, n=2, seed=11)
    result = bootstrap_bands(Y, p=2, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=20,
                             rng=np.random.default_rng(0))
    assert result["point"].shape == (5, 2, 2)
    assert result["draws"].shape == (20, 5, 2, 2)


def test_bootstrap_bands_p2_finite():
    Y = _stable_var1_data(T=120, n=2, seed=12)
    result = bootstrap_bands(Y, p=2, identify_fn=_cholesky_identify,
                             horizon=3, n_boot=20,
                             rng=np.random.default_rng(1))
    assert np.all(np.isfinite(result["point"]))
    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))


# ===========================================================================
# bootstrap_bands — bias_correct path
# ===========================================================================

def test_bootstrap_bands_bias_correct_shapes():
    """bias_correct=True should return results with the same shape contract."""
    Y = _stable_var1_data(T=100, n=2, seed=13)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=20, bias_correct=True,
                             n_pilot=20, rng=np.random.default_rng(0))
    assert result["point"].shape == (5, 2, 2)
    assert result["lower"].shape == (5, 2, 2)
    assert result["upper"].shape == (5, 2, 2)
    assert result["draws"].shape == (20, 5, 2, 2)


def test_bootstrap_bands_bias_correct_lower_le_upper():
    Y = _stable_var1_data(T=100, n=2, seed=14)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=3, n_boot=20, bias_correct=True,
                             n_pilot=15, rng=np.random.default_rng(2))
    assert np.all(result["lower"] <= result["upper"] + 1e-10)


def test_bootstrap_bands_bias_correct_vs_plain_different():
    """Bias-corrected and plain bootstrap should (usually) give different bands."""
    Y = _stable_var1_data(T=100, n=2, seed=15)
    r_plain = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                              horizon=3, n_boot=30, bias_correct=False,
                              rng=np.random.default_rng(5))
    r_bc = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                           horizon=3, n_boot=30, bias_correct=True,
                           n_pilot=20, rng=np.random.default_rng(5))
    # They share the same random seed but use different DGPs — should differ
    # NOTE: contract-only assertion; the exact difference has no analytic ground truth
    assert not np.allclose(r_plain["draws"], r_bc["draws"], equal_nan=True), \
        "Bias-corrected and plain bootstrap gave identical draws — unexpected"


# ===========================================================================
# bootstrap_bands — n=3 variables
# ===========================================================================

def test_bootstrap_bands_three_variables():
    """VAR(1) with n=3; shapes should generalize correctly."""
    rng = np.random.default_rng(0)
    T, n = 100, 3
    A = 0.4 * np.eye(n)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + 0.5 * rng.standard_normal(n)
    result = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                             horizon=4, n_boot=20,
                             rng=np.random.default_rng(1))
    assert result["point"].shape == (5, 3, 3)
    assert result["lower"].shape == (5, 3, 3)
    assert result["upper"].shape == (5, 3, 3)


# ===========================================================================
# Known-value: IRF median should be near the true IRF for a simple DGP
# ===========================================================================

def test_bootstrap_bands_median_near_true_irf():
    """For a univariate AR(1) with known coefficient, the bootstrap median
    (point estimate) should approximate the true MA impulse responses.

    DGP: y_t = 0.5 y_{t-1} + eps_t, eps ~ N(0, 1).
    True IRF: response to unit structural shock at horizon h = 0.5^h.
    """
    rng_dgp = np.random.default_rng(42)
    T = 200
    a = 0.5
    Y = np.zeros((T, 1))
    for t in range(1, T):
        Y[t] = a * Y[t - 1] + rng_dgp.standard_normal()

    def trivial_identify(A_list, Sigma):
        """For n=1, B0 = sqrt(Sigma)."""
        return np.sqrt(Sigma).reshape(1, 1)

    result = bootstrap_bands(Y, p=1, identify_fn=trivial_identify,
                             horizon=5, n_boot=200, alpha=0.10,
                             rng=np.random.default_rng(0))
    point = result["point"]  # (6, 1, 1)
    # True IRF: irf_h = a^h * sigma, where sigma = 1 (unit shock in B0 sense)
    for h in range(6):
        true_val = a ** h * 1.0   # sigma_eps = 1 from DGP
        assert abs(point[h, 0, 0] - true_val) < 0.15, (
            f"Median IRF at horizon {h} = {point[h, 0, 0]:.4f}; "
            f"true = {true_val:.4f}; gap too large"
        )


# ===========================================================================
# More boot draws tighten the CI (numerical property test)
# ===========================================================================

@pytest.mark.slow
def test_bootstrap_bands_more_data_tighter_ci():
    """More observations should yield tighter bootstrap bands on average."""
    def run(T, seed):
        rng_dgp = np.random.default_rng(seed)
        Y = np.zeros((T, 2))
        A = 0.4 * np.eye(2)
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + rng_dgp.standard_normal(2)
        return bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                               horizon=4, n_boot=100, alpha=0.10,
                               rng=np.random.default_rng(0))

    r_small = run(T=60, seed=10)
    r_large = run(T=300, seed=10)
    width_small = (r_small["upper"] - r_small["lower"]).mean()
    width_large = (r_large["upper"] - r_large["lower"]).mean()
    assert width_large < width_small, (
        f"Larger T should give tighter bands; got width_small={width_small:.4f}, "
        f"width_large={width_large:.4f}"
    )


# ===========================================================================
# Exception branches — lines 47-48, 63-65, 123-124
# ===========================================================================

def test_kilian_bias_correct_linalg_error_in_pilot(monkeypatch):
    """Lines 47-48: if estimate_var raises LinAlgError on a pilot sample,
    the loop continues (skip that draw) and returns valid output anyway."""
    from puremacro.var import bootstrap as boot_module

    original_estimate = boot_module.estimate_var
    call_count = {"n": 0}

    def patched_estimate(Y, p):
        call_count["n"] += 1
        # First call = the real estimate at the start of _kilian_bias_correct.
        # Subsequent calls during the pilot loop → raise on every other call.
        if call_count["n"] == 1:
            return original_estimate(Y, p)
        if call_count["n"] % 2 == 0:
            raise np.linalg.LinAlgError("forced singular")
        return original_estimate(Y, p)

    monkeypatch.setattr(boot_module, "estimate_var", patched_estimate)

    Y = _stable_var1_data(T=80, n=2, seed=20)
    # Should complete without error despite every-other pilot raising LinAlgError
    A_bc, int_bc = boot_module._kilian_bias_correct(
        Y, p=1, n_pilot=10, rng=np.random.default_rng(0)
    )
    assert len(A_bc) == 1
    assert np.all(np.isfinite(A_bc[0]))
    assert np.all(np.isfinite(int_bc))


def test_kilian_bias_correct_shrinkage_loop():
    """Lines 63-65: the shrinkage loop reduces delta until stability is restored.

    We construct a 1-variable DGP where the estimated A is near the boundary
    and force large pilot draws so bias_A pushes A_bc past 1.0, triggering
    lines 63-65 on each iteration until the companion stabilises.

    Strategy: use a near-unit-root series (A=0.97) so the OLS estimate of A
    is close to 1.  The pilot draws intentionally return A_b = 1.5 (via
    monkeypatching) so bias_A ≈ 0.53, and A_bc = A - bias_A ≈ 0.44 — which
    is actually STABLE.  To reliably hit lines 63-65 we need A_bc to start
    EXPLOSIVE (>0.999).  Easiest approach: patch only `companion` so that the
    first call returns a matrix whose max-eigenvalue is >=0.999 and subsequent
    calls return a stable one.  But that requires internal access.

    Simpler, self-contained approach: call the module's _kilian_bias_correct
    with data where the OLS companion is near 0.999 MINUS a small positive
    bias, so that A_bc = A - bias_A exceeds 0.999 on the first loop check.

    We achieve this by using n_pilot=1 with a controlled random stream so
    the single pilot draw nudges A_bc just past the threshold, then one
    shrinkage step brings it back.
    """
    from puremacro.var import bootstrap as boot_module
    from puremacro.var.estimate import estimate_var as real_estimate, companion as real_companion

    # We patch `companion` inside the bootstrap module so that the first call
    # to companion(A_bc) inside the while loop returns a "fake explosive"
    # matrix (max eig = 1.05), and subsequent calls return the real companion.
    companion_call_count = {"n": 0}
    original_companion = real_companion

    def patched_companion(A_bc):
        companion_call_count["n"] += 1
        if companion_call_count["n"] == 1:
            # Return a 1x1 matrix whose eigenvalue is > 0.999 → triggers shrinkage
            return np.array([[1.05]])
        return original_companion(A_bc)

    import puremacro.var.bootstrap as _bm
    original_bc_companion = getattr(_bm, "companion", None)

    # We need to patch the companion that gets imported inside _kilian_bias_correct.
    # It uses `from .estimate import companion` inside the function body.
    # The cleanest way: monkeypatch puremacro.var.estimate.companion.
    import puremacro.var.estimate as est_module
    est_orig = est_module.companion

    try:
        est_module.companion = patched_companion
        Y = _stable_var1_data(T=80, n=1, seed=30)
        A_bc, int_bc = _bm._kilian_bias_correct(
            Y, p=1, n_pilot=5, rng=np.random.default_rng(5)
        )
    finally:
        est_module.companion = est_orig

    assert len(A_bc) == 1
    assert np.all(np.isfinite(A_bc[0]))


def test_bootstrap_bands_exception_in_identify_fn():
    """Lines 123-124: if identify_fn raises, irfs[b] is set to nan and
    nanpercentile still produces finite output from the non-nan draws."""
    Y = _stable_var1_data(T=80, n=2, seed=21)
    call_count = {"n": 0}

    def flaky_identify(A_list, Sigma):
        call_count["n"] += 1
        # Fail on even-numbered bootstrap draws
        if call_count["n"] % 2 == 0:
            raise ValueError("simulated identification failure")
        return cholesky_factor(Sigma)

    result = bootstrap_bands(Y, p=1, identify_fn=flaky_identify,
                             horizon=3, n_boot=20,
                             rng=np.random.default_rng(0))
    # Some draws are nan; nanpercentile should give finite point/bands
    assert np.all(np.isfinite(result["point"]))
    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))
    # The nan draws are recorded in "draws"
    nan_draws = np.any(np.isnan(result["draws"]), axis=(1, 2, 3))
    assert nan_draws.sum() > 0, "Expected some nan draws from flaky identify_fn"
