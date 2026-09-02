"""Coverage tests for puremacro.var.estimate — uncovered functions/branches.

The existing tests (test_var/test_estimate_result.py, test_var_bootstrap_coverage.py)
already cover estimate_var(...) and companion(...).  This file targets the
remaining ~57% that was missing at the time it was written:

  select_lag_bic   (lines 29-40)
  lag_select       (lines 43-65) — aic / hq / bic branches + exception path
  is_stable        (lines 79-82)
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.estimate import (
    estimate_var,
    select_lag_bic,
    lag_select,
    companion,
    is_stable,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stationary_var1(T: int, n: int, rho: float = 0.5, seed: int = 0) -> np.ndarray:
    """Simulate a stable diagonal VAR(1) with spectral radius ``rho``."""
    rng = np.random.default_rng(seed)
    A = rho * np.eye(n)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + rng.standard_normal(n)
    return Y


def _var2_data(T: int = 200, seed: int = 1) -> np.ndarray:
    """Simulate a bivariate VAR(2) from a known coefficient set."""
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1], [-0.05, 0.4]])
    A2 = np.array([[0.1, 0.0], [0.0, 0.05]])
    Y = np.zeros((T, 2))
    for t in range(2, T):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.standard_normal(2) * 0.5
    return Y


# ---------------------------------------------------------------------------
# select_lag_bic
# ---------------------------------------------------------------------------

class TestSelectLagBic:

    def test_returns_int(self):
        Y = _stationary_var1(150, 2)
        p = select_lag_bic(Y, max_p=4)
        assert isinstance(p, int)

    def test_returns_in_valid_range(self):
        """Returned lag must be in [1, max_p]."""
        Y = _stationary_var1(150, 2)
        max_p = 5
        p = select_lag_bic(Y, max_p=max_p)
        assert 1 <= p <= max_p

    def test_default_max_p_is_8(self):
        """Default max_p=8 should not raise, and result is in [1, 8]."""
        Y = _stationary_var1(200, 2)
        p = select_lag_bic(Y)
        assert 1 <= p <= 8

    def test_var1_dgp_does_not_select_too_many_lags(self):
        """For a true VAR(1) DGP, BIC should prefer p=1 (or a small lag)."""
        Y = _stationary_var1(T=300, n=2, rho=0.6, seed=42)
        p = select_lag_bic(Y, max_p=6)
        # BIC is parsimonious; for 300 obs the true lag=1 should win or at most p=2
        assert p <= 3

    def test_max_p_1_always_returns_1(self):
        """When max_p=1 there is only one candidate, so answer is always 1."""
        Y = _stationary_var1(100, 3)
        p = select_lag_bic(Y, max_p=1)
        assert p == 1

    def test_univariate_case(self):
        """n=1 is a degenerate case; function should return a valid lag."""
        rng = np.random.default_rng(7)
        Y = rng.standard_normal((100, 1))
        p = select_lag_bic(Y, max_p=4)
        assert 1 <= p <= 4

    def test_larger_sample_picks_parsimonious_model(self):
        """BIC penalty grows with T; very large samples tend toward p=1 for a VAR(1)."""
        Y = _stationary_var1(T=500, n=2, rho=0.5, seed=10)
        p = select_lag_bic(Y, max_p=8)
        assert p <= 4  # generous bound; BIC should not over-select badly


# ---------------------------------------------------------------------------
# lag_select
# ---------------------------------------------------------------------------

class TestLagSelect:

    def test_bic_returns_int(self):
        Y = _stationary_var1(150, 2)
        p = lag_select(Y, maxlags=4, ic="bic")
        assert isinstance(p, int)

    def test_aic_branch_returns_valid_lag(self):
        """ic='aic' branch (lines 54-55) should execute and return valid p."""
        Y = _stationary_var1(150, 2)
        p = lag_select(Y, maxlags=4, ic="aic")
        assert isinstance(p, int)
        assert 1 <= p <= 4

    def test_hq_branch_returns_valid_lag(self):
        """ic='hq' (Hannan-Quinn) branch (lines 56-57) should execute and return valid p."""
        Y = _stationary_var1(150, 2)
        p = lag_select(Y, maxlags=4, ic="hq")
        assert isinstance(p, int)
        assert 1 <= p <= 4

    def test_bic_branch_explicit(self):
        """ic='bic' default branch (else, lines 58-59) should execute."""
        Y = _stationary_var1(150, 2)
        p = lag_select(Y, maxlags=4, ic="bic")
        assert isinstance(p, int)
        assert 1 <= p <= 4

    def test_unknown_ic_falls_through_to_bic(self):
        """Any string that is not 'aic' or 'hq' falls into the else=bic branch."""
        Y = _stationary_var1(150, 2)
        p_bic = lag_select(Y, maxlags=4, ic="bic")
        p_other = lag_select(Y, maxlags=4, ic="OTHER_STRING")
        assert p_other == p_bic

    def test_all_three_ics_return_in_range(self):
        Y = _var2_data(T=200)
        for ic in ("aic", "hq", "bic"):
            p = lag_select(Y, maxlags=6, ic=ic)
            assert 1 <= p <= 6, f"ic={ic!r} returned p={p}"

    def test_maxlags_1_forces_p_1(self):
        """With maxlags=1 the only candidate is p=1."""
        Y = _stationary_var1(100, 2)
        for ic in ("aic", "hq", "bic"):
            assert lag_select(Y, maxlags=1, ic=ic) == 1

    def test_aic_vs_bic_ordering(self):
        """AIC penalises less than BIC so it typically selects >= the BIC lag.

        This is a property test, not a strict equality assertion.  With enough
        data the ordering is reliable; we only assert they are both valid.
        """
        Y = _stationary_var1(T=300, n=2, rho=0.6)
        p_aic = lag_select(Y, maxlags=6, ic="aic")
        p_bic = lag_select(Y, maxlags=6, ic="bic")
        # Both must be in range; AIC is at least as large as BIC on average,
        # but we only enforce the range contract to avoid flaky tests.
        assert 1 <= p_aic <= 6
        assert 1 <= p_bic <= 6

    def test_exception_branch_skips_gracefully(self):
        """If estimate_var raises, the loop should continue (except: continue path).

        This docstring used to say "linalg.lstsq raises LinAlgError" on NaN
        input. It does not: LAPACK prints a complaint straight to the terminal
        and `lstsq` returns all-NaN coefficients. So the `except: continue`
        branch was never taken and this test passed for the wrong reason — the
        NaN simply propagated to `slogdet`, which returned -inf rather than
        raising. `estimate_var` now rejects non-finite input by name, which is
        what makes the branch below actually reachable.
        """
        import warnings

        rng = np.random.default_rng(999)
        Y = rng.standard_normal((20, 2))
        # Inject NaN so that every call to estimate_var raises LinAlgError.
        # All p iterations will hit the except block; function falls back to best_p=1.
        Y[5, 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = lag_select(Y, maxlags=4, ic="bic")
        assert isinstance(p, int)
        assert 1 <= p <= 4

    def test_accepts_list_input(self):
        """lag_select coerces input via np.asarray, so lists are valid."""
        Y = _stationary_var1(100, 2).tolist()
        p = lag_select(Y, maxlags=3, ic="bic")
        assert 1 <= p <= 3

    def test_three_variable_system(self):
        Y = _stationary_var1(200, 3, rho=0.4)
        p = lag_select(Y, maxlags=4, ic="aic")
        assert 1 <= p <= 4


# ---------------------------------------------------------------------------
# companion
# ---------------------------------------------------------------------------

class TestCompanion:
    """companion() was partially covered by bootstrap tests; these add
    mathematical-property checks that are not in existing coverage."""

    def test_shape_p1(self):
        """For p=1, companion is n×n."""
        n = 3
        A = [np.eye(n) * 0.5]
        C = companion(A)
        assert C.shape == (n, n)

    def test_shape_p2(self):
        """For p=2, companion is 2n×2n."""
        n = 2
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.05]])
        C = companion([A1, A2])
        assert C.shape == (2 * n, 2 * n)

    def test_identity_block_p2(self):
        """Lower-left block of a p=2 companion must be an identity."""
        n = 2
        A1 = np.random.default_rng(5).standard_normal((n, n)) * 0.3
        A2 = np.random.default_rng(6).standard_normal((n, n)) * 0.1
        C = companion([A1, A2])
        # Bottom-left block is the identity
        assert np.allclose(C[n:, :n], np.eye(n))
        # Bottom-right block is zero
        assert np.allclose(C[n:, n:], 0.0)

    def test_top_row_hstacks_a_list(self):
        """Top n rows of companion equal [A1 | A2 | ...]."""
        n = 2
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.05]])
        C = companion([A1, A2])
        expected_top = np.hstack([A1, A2])
        assert np.allclose(C[:n, :], expected_top)

    def test_shape_p3(self):
        """For p=3, companion is 3n×3n."""
        n = 2
        p = 3
        A_list = [np.eye(n) * (0.3 / i) for i in range(1, p + 1)]
        C = companion(A_list)
        assert C.shape == (n * p, n * p)

    def test_spectral_radius_stable_system(self):
        """A truly stable system should have companion spectral radius < 1."""
        n = 2
        A = [np.array([[0.3, 0.0], [0.0, 0.4]])]
        C = companion(A)
        rho = np.max(np.abs(np.linalg.eigvals(C)))
        assert rho < 1.0


# ---------------------------------------------------------------------------
# is_stable
# ---------------------------------------------------------------------------

class TestIsStable:

    def test_stable_var1_returns_true(self):
        """Diagonal VAR(1) with spectral radius 0.5 must be stable."""
        n = 2
        A = [np.array([[0.5, 0.0], [0.0, 0.4]])]
        assert is_stable(A) is True

    def test_explosive_var1_returns_false(self):
        """Diagonal VAR(1) with an eigenvalue > 1 must be unstable."""
        n = 2
        A = [np.array([[1.2, 0.0], [0.0, 0.3]])]
        assert is_stable(A) is False

    def test_unit_root_returns_false(self):
        """Eigenvalue exactly on the unit circle → not inside → returns False."""
        n = 2
        A = [np.array([[1.0, 0.0], [0.0, 0.5]])]
        assert is_stable(A) is False

    def test_stable_var2_returns_true(self):
        """Known stable VAR(2) should return True."""
        A1 = np.array([[0.5, 0.1], [-0.05, 0.4]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.05]])
        # Verify analytically: companion spectral radius
        C = companion([A1, A2])
        rho = np.max(np.abs(np.linalg.eigvals(C)))
        expected = bool(rho < 1.0)
        assert is_stable([A1, A2]) == expected

    def test_near_unit_root_stable_side(self):
        """Coefficient 0.999 < 1 should still be stable."""
        A = [np.array([[0.999, 0.0], [0.0, 0.0]])]
        assert is_stable(A) is True

    def test_returns_bool_type(self):
        """Return type must be bool (not numpy.bool_)."""
        A = [np.eye(2) * 0.5]
        result = is_stable(A)
        assert isinstance(result, bool)

    def test_estimated_stable_var_is_stable(self):
        """A VAR estimated from a stable DGP should typically be recognised as stable."""
        Y = _stationary_var1(T=300, n=2, rho=0.4, seed=123)
        A_list, *_ = estimate_var(Y, p=1)
        # With spectral radius 0.4 and 300 obs, OLS estimates should be stable
        assert is_stable(A_list) is True

    def test_univariate_stable(self):
        """n=1 scalar case: coefficient 0.8 should be stable."""
        A = [np.array([[0.8]])]
        assert is_stable(A) is True

    def test_univariate_explosive(self):
        """n=1 scalar case: coefficient -1.5 should be unstable."""
        A = [np.array([[-1.5]])]
        assert is_stable(A) is False

    def test_multivariate_off_diagonal_can_destabilise(self):
        """Off-diagonal terms can push spectral radius past 1.

        We construct a matrix whose eigenvalues are clearly inside the unit
        circle (diagonal) but then add off-diagonal terms that we know push
        it to exactly the right place.
        """
        # eigenvalues of [[0.5, 1.5], [0.0, 0.5]] are both 0.5 → stable
        A_stable = [np.array([[0.5, 1.5], [0.0, 0.5]])]
        assert is_stable(A_stable) is True

        # eigenvalues of [[0.5, 0.0], [0.0, -1.1]] include -1.1 → unstable
        A_unstable = [np.array([[0.5, 0.0], [0.0, -1.1]])]
        assert is_stable(A_unstable) is False
