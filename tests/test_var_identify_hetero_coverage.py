"""Tests for puremacro.var.identify.hetero — Rigobon (2003) identification.

Strategy
--------
- Known-value / DGP:
    * The exact algebraic identities B @ B.T = Sigma_0 and
      B @ diag(lambda) @ B.T = Sigma_1 must hold to machine precision.
    * With large T and a clear DGP (two-regime, known B, known variance ratios)
      the estimated variance_ratios converge toward the true values.

- Properties: shapes, dtypes, FEVD sums-to-one per row, variance_ratios >= 0,
  Sigma matrices are symmetric PSD, IRF at h=0 equals B,
  point attribute is the same object as irfs.

- HeteroResult contract: frozen dataclass, all fields present, lower/upper
  may be None when bootstrap is skipped.

- Error handling: ValueError on too-few regime observations, ValueError on
  mismatched regime_indicator length.

- Configuration: regime_indicator of length T or T-p are both accepted;
  n_boot=0 skips bootstrap (lower=upper=None); n=2,3,4 all produce
  correctly shaped outputs; block_len parameter is accepted.

Note: _HAS_MBB is False in this installation (moving_block_bootstrap is not
shipped), so bootstrap-band tests are limited to contract checks (lower=None,
upper=None). Tests that would require bootstrap inference are omitted rather
than patched.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.var.identify.hetero import (
    HeteroResult,
    _regime_covariances,
    _rigobon_impact,
    rigobon_svar,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _stable_var1(T: int, n: int, *, seed: int = 0) -> np.ndarray:
    """Simulate a stable VAR(1): Y_t = 0.4 * I_n * Y_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    A = 0.4 * np.eye(n)
    eps = rng.standard_normal((T, n))
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + eps[t]
    return Y


def _split_regime(T: int, p: int = 0) -> np.ndarray:
    """Return a 0/1 regime indicator that splits T-p obs in half."""
    T_eff = T - p
    ri = np.zeros(T_eff, dtype=int)
    ri[T_eff // 2 :] = 1
    return ri


# ---------------------------------------------------------------------------
# HeteroResult dataclass contract
# ---------------------------------------------------------------------------

class TestHeteroResultDataclass:
    """HeteroResult is a frozen dataclass with the expected fields."""

    def _make_result(self, H: int = 4, n: int = 2) -> HeteroResult:
        return HeteroResult(
            B=np.eye(n),
            variance_ratios=np.array([1.0, 2.0][:n]),
            irfs=np.zeros((H + 1, n, n)),
            fevd=np.zeros((H + 1, n, n)),
            lower=None,
            upper=None,
            point=np.zeros((H + 1, n, n)),
        )

    def test_is_frozen_dataclass(self):
        res = self._make_result()
        assert dataclasses.is_dataclass(res)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.B = np.zeros((2, 2))

    def test_all_fields_present(self):
        res = self._make_result(H=6, n=3)
        assert res.B.shape == (3, 3)
        assert res.variance_ratios.shape == (2,)  # we passed 2 elements for n=3 test
        assert res.irfs.shape == (7, 3, 3)
        assert res.fevd.shape == (7, 3, 3)
        assert res.lower is None
        assert res.upper is None
        assert res.point.shape == (7, 3, 3)

    def test_with_bootstrap_bands(self):
        H, n = 4, 2
        lower = np.zeros((H + 1, n, n))
        upper = np.ones((H + 1, n, n))
        res = HeteroResult(
            B=np.eye(n),
            variance_ratios=np.ones(n),
            irfs=np.zeros((H + 1, n, n)),
            fevd=np.zeros((H + 1, n, n)),
            lower=lower,
            upper=upper,
            point=np.zeros((H + 1, n, n)),
        )
        assert res.lower is not None
        assert res.upper is not None
        # When bands are supplied, lower <= upper is a meaningful contract
        assert (res.lower <= res.upper).all()


# ---------------------------------------------------------------------------
# _regime_covariances: properties and error handling
# ---------------------------------------------------------------------------

class TestRegimeCovariances:

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        n = 3
        residuals = rng.standard_normal((100, n))
        regime = np.array([0] * 50 + [1] * 50)
        S0, S1 = _regime_covariances(residuals, regime)
        assert S0.shape == (n, n)
        assert S1.shape == (n, n)

    def test_output_is_symmetric(self):
        rng = np.random.default_rng(1)
        n = 2
        residuals = rng.standard_normal((80, n))
        regime = np.array([0] * 40 + [1] * 40)
        S0, S1 = _regime_covariances(residuals, regime)
        np.testing.assert_allclose(S0, S0.T, atol=1e-14)
        np.testing.assert_allclose(S1, S1.T, atol=1e-14)

    def test_output_is_positive_semidefinite(self):
        rng = np.random.default_rng(2)
        n = 3
        residuals = rng.standard_normal((120, n))
        regime = np.array([0] * 60 + [1] * 60)
        S0, S1 = _regime_covariances(residuals, regime)
        eigvals0 = np.linalg.eigvalsh(S0)
        eigvals1 = np.linalg.eigvalsh(S1)
        assert (eigvals0 >= -1e-10).all(), f"S0 not PSD: min eigenval={eigvals0.min()}"
        assert (eigvals1 >= -1e-10).all(), f"S1 not PSD: min eigenval={eigvals1.min()}"

    def test_raises_if_regime0_too_few_observations(self):
        """Regime 0 must have at least n+1 observations."""
        rng = np.random.default_rng(0)
        n = 3
        residuals = rng.standard_normal((10, n))
        # Only 2 obs in regime 0 — need n+1 = 4
        regime = np.array([0] * 2 + [1] * 8)
        with pytest.raises(ValueError, match="Regime 0"):
            _regime_covariances(residuals, regime)

    def test_raises_if_regime1_too_few_observations(self):
        """Regime 1 must have at least n+1 observations."""
        rng = np.random.default_rng(0)
        n = 3
        residuals = rng.standard_normal((10, n))
        # Only 2 obs in regime 1 — need n+1 = 4
        regime = np.array([0] * 8 + [1] * 2)
        with pytest.raises(ValueError, match="Regime 1"):
            _regime_covariances(residuals, regime)

    def test_known_value_diagonal_case(self):
        """When residuals are i.i.d. within each regime the sample covariances
        converge toward the true population ones."""
        rng = np.random.default_rng(42)
        T0, T1, n = 2000, 2000, 2
        # Regime 0: unit-variance noise
        e0 = rng.standard_normal((T0, n))
        # Regime 1: scaled noise (diagonal Sigma_1 = diag(4, 9))
        e1 = rng.standard_normal((T1, n)) * np.array([2.0, 3.0])
        residuals = np.vstack([e0, e1])
        regime = np.array([0] * T0 + [1] * T1)
        S0, S1 = _regime_covariances(residuals, regime)
        np.testing.assert_allclose(np.diag(S0), [1.0, 1.0], atol=0.1)
        np.testing.assert_allclose(np.diag(S1), [4.0, 9.0], atol=0.5)


# ---------------------------------------------------------------------------
# _rigobon_impact: exact algebraic identities
# ---------------------------------------------------------------------------

class TestRigobonImpact:

    def _make_pd_matrix(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        return A @ A.T + n * np.eye(n)  # strictly PD

    def test_output_shapes(self):
        n = 3
        S0 = self._make_pd_matrix(n, seed=0)
        S1 = self._make_pd_matrix(n, seed=1)
        B, lam = _rigobon_impact(S0, S1)
        assert B.shape == (n, n)
        assert lam.shape == (n,)

    def test_identity_B_Bt_equals_Sigma0(self):
        """Exact algebraic identity: B @ B.T must equal Sigma_0."""
        n = 3
        S0 = self._make_pd_matrix(n, seed=10)
        S1 = self._make_pd_matrix(n, seed=11)
        B, _ = _rigobon_impact(S0, S1)
        np.testing.assert_allclose(B @ B.T, S0, atol=1e-10)

    def test_identity_B_diag_lam_Bt_equals_Sigma1(self):
        """Exact algebraic identity: B @ diag(lam) @ B.T must equal Sigma_1."""
        n = 2
        S0 = self._make_pd_matrix(n, seed=20)
        S1 = self._make_pd_matrix(n, seed=21)
        B, lam = _rigobon_impact(S0, S1)
        np.testing.assert_allclose(B @ np.diag(lam) @ B.T, S1, atol=1e-10)

    def test_identity_holds_for_n4(self):
        """Both identities hold for n=4."""
        n = 4
        S0 = self._make_pd_matrix(n, seed=30)
        S1 = self._make_pd_matrix(n, seed=31)
        B, lam = _rigobon_impact(S0, S1)
        np.testing.assert_allclose(B @ B.T, S0, atol=1e-10)
        np.testing.assert_allclose(B @ np.diag(lam) @ B.T, S1, atol=1e-10)

    def test_variance_ratios_are_real(self):
        """Eigenvalues of the ratio matrix must be real (scipy.linalg.eigh
        guarantees this for symmetric inputs)."""
        n = 3
        S0 = self._make_pd_matrix(n, seed=5)
        S1 = self._make_pd_matrix(n, seed=6)
        _, lam = _rigobon_impact(S0, S1)
        assert np.isrealobj(lam), "variance_ratios must be real"


# ---------------------------------------------------------------------------
# rigobon_svar: output shapes, properties, error handling
# ---------------------------------------------------------------------------

class TestRigobonSvar:

    def test_output_is_hetero_result(self):
        Y = _stable_var1(T=200, n=2, seed=0)
        ri = _split_regime(200)
        res = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0)
        assert isinstance(res, HeteroResult)

    def test_shapes_n2_p1_h4(self):
        T, n, H = 200, 2, 4
        Y = _stable_var1(T=T, n=n, seed=1)
        ri = _split_regime(T)
        res = rigobon_svar(Y, p=1, horizon=H, regime_indicator=ri, n_boot=0)
        assert res.B.shape == (n, n)
        assert res.irfs.shape == (H + 1, n, n)
        assert res.fevd.shape == (H + 1, n, n)
        assert res.point.shape == (H + 1, n, n)
        assert res.variance_ratios.shape == (n,)

    def test_shapes_n3_p2_h8(self):
        T, n, H, p = 400, 3, 8, 2
        Y = _stable_var1(T=T, n=n, seed=2)
        ri = _split_regime(T, p=p)
        res = rigobon_svar(Y, p=p, horizon=H, regime_indicator=ri, n_boot=0)
        assert res.B.shape == (n, n)
        assert res.irfs.shape == (H + 1, n, n)
        assert res.fevd.shape == (H + 1, n, n)
        assert res.variance_ratios.shape == (n,)

    def test_shapes_n4_p1_h6(self):
        T, n, H = 400, 4, 6
        Y = _stable_var1(T=T, n=n, seed=3)
        ri = _split_regime(T)
        res = rigobon_svar(Y, p=1, horizon=H, regime_indicator=ri, n_boot=0)
        assert res.B.shape == (n, n)
        assert res.irfs.shape == (H + 1, n, n)

    def test_irf_at_h0_equals_B(self):
        """IRF at horizon 0 is the structural impact matrix B."""
        Y = _stable_var1(T=300, n=2, seed=4)
        ri = _split_regime(300)
        res = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0)
        np.testing.assert_allclose(res.irfs[0], res.B, atol=1e-10)

    def test_point_is_same_object_as_irfs(self):
        """res.point must be the identical array as res.irfs."""
        Y = _stable_var1(T=200, n=2, seed=5)
        ri = _split_regime(200)
        res = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0)
        assert res.point is res.irfs

    def test_fevd_rows_sum_to_one_at_all_horizons(self):
        """FEVD rows must sum to 1 at every horizon."""
        T, n, H = 300, 3, 8
        Y = _stable_var1(T=T, n=n, seed=6)
        ri = _split_regime(T)
        res = rigobon_svar(Y, p=1, horizon=H, regime_indicator=ri, n_boot=0)
        for h in range(H + 1):
            row_sums = res.fevd[h].sum(axis=1)
            np.testing.assert_allclose(
                row_sums, np.ones(n), atol=1e-10,
                err_msg=f"FEVD rows do not sum to 1 at h={h}",
            )

    def test_variance_ratios_are_non_negative(self):
        """Variance ratios (eigenvalues of M) must be non-negative."""
        Y = _stable_var1(T=300, n=3, seed=7)
        ri = _split_regime(300)
        res = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0)
        assert (res.variance_ratios >= 0).all(), (
            f"Negative variance_ratio(s): {res.variance_ratios}"
        )

    def test_structural_identities_hold_on_sample_covariances(self):
        """B @ B.T == Sigma_0 and B @ diag(lambda) @ B.T == Sigma_1 exactly.

        Use the same regime_indicator slice that rigobon_svar uses internally:
        pass length-T indicator so the function trims it to T-p, then
        reconstruct Sigma_0/Sigma_1 from the same trimmed indicator.
        """
        from puremacro.var.estimate import estimate_var

        T, n, p = 300, 2, 1
        Y = _stable_var1(T=T, n=n, seed=8)
        ri_T = _split_regime(T, p=0)   # length T — function trims to T-p
        res = rigobon_svar(Y, p=p, horizon=4, regime_indicator=ri_T, n_boot=0)

        _, _, _, residuals, _ = estimate_var(Y, p)
        ri_eff = ri_T[p:]              # same trim: T-p elements
        Sigma_0, Sigma_1 = _regime_covariances(residuals, ri_eff)

        np.testing.assert_allclose(res.B @ res.B.T, Sigma_0, atol=1e-10)
        np.testing.assert_allclose(
            res.B @ np.diag(res.variance_ratios) @ res.B.T,
            Sigma_1,
            atol=1e-10,
        )

    def test_no_bootstrap_when_n_boot_zero(self):
        """n_boot=0 must leave lower and upper as None."""
        Y = _stable_var1(T=200, n=2, seed=9)
        ri = _split_regime(200)
        res = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0)
        assert res.lower is None
        assert res.upper is None

    def test_seed_determinism_when_no_bootstrap(self):
        """Determinism: same inputs → same output regardless of seed when
        n_boot=0 (no randomness is used)."""
        Y = _stable_var1(T=200, n=2, seed=10)
        ri = _split_regime(200)
        res_a = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0, seed=0)
        res_b = rigobon_svar(Y, p=1, horizon=4, regime_indicator=ri, n_boot=0, seed=99)
        np.testing.assert_array_equal(res_a.B, res_b.B)
        np.testing.assert_array_equal(res_a.irfs, res_b.irfs)

    def test_regime_indicator_length_T_is_accepted(self):
        """regime_indicator of length T (not T-p) must be silently trimmed."""
        T, n, p = 200, 2, 1
        Y = _stable_var1(T=T, n=n, seed=11)
        ri_T = _split_regime(T, p=0)     # length T
        ri_Tp = _split_regime(T, p=p)    # length T-p
        assert ri_T.shape[0] == T
        assert ri_Tp.shape[0] == T - p
        res_T = rigobon_svar(Y, p=p, horizon=4, regime_indicator=ri_T, n_boot=0)
        res_Tp = rigobon_svar(Y, p=p, horizon=4, regime_indicator=ri_Tp, n_boot=0)
        # Both should produce identical results (trim logic is deterministic)
        np.testing.assert_array_equal(res_T.B, res_Tp.B)

    def test_block_len_param_is_accepted(self):
        """block_len parameter must be accepted without error (n_boot=0 so
        it has no effect on output, but the argument path must be exercised)."""
        Y = _stable_var1(T=200, n=2, seed=12)
        ri = _split_regime(200)
        res = rigobon_svar(
            Y, p=1, horizon=4, regime_indicator=ri, n_boot=0, block_len=5
        )
        assert isinstance(res, HeteroResult)

    def test_raises_on_mismatched_regime_indicator_length(self):
        """ValueError when regime_indicator has the wrong length."""
        T, n = 100, 2
        Y = _stable_var1(T=T, n=n, seed=13)
        bad_ri = np.zeros(50, dtype=int)  # neither T nor T-p
        with pytest.raises(ValueError, match="regime_indicator length"):
            rigobon_svar(Y, p=1, horizon=4, regime_indicator=bad_ri, n_boot=0)

    def test_raises_when_regime0_has_insufficient_obs(self):
        """ValueError when regime 0 lacks enough observations after VAR fit."""
        rng = np.random.default_rng(14)
        T, n, p = 80, 3, 1
        Y = rng.standard_normal((T, n))
        # Only 2 obs in regime 0 (T-p = 79 obs; need at least n+1 = 4 per regime)
        ri = np.ones(T, dtype=int)
        ri[:2] = 0  # just 2 obs in regime 0 (after trimming 1 for p=1 → still 2)
        with pytest.raises(ValueError, match="Regime 0"):
            rigobon_svar(Y, p=p, horizon=4, regime_indicator=ri, n_boot=0)

    def test_raises_when_regime1_has_insufficient_obs(self):
        """ValueError when regime 1 lacks enough observations after VAR fit."""
        rng = np.random.default_rng(15)
        T, n, p = 80, 3, 1
        Y = rng.standard_normal((T, n))
        ri = np.zeros(T, dtype=int)
        ri[-2:] = 1  # just 2 obs in regime 1
        with pytest.raises(ValueError, match="Regime 1"):
            rigobon_svar(Y, p=p, horizon=4, regime_indicator=ri, n_boot=0)


# ---------------------------------------------------------------------------
# Known-DGP test: variance ratios converge toward truth
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestKnownDGP:
    """With a large sample and a clear DGP the estimated variance ratios should
    be within 25% of the true values.

    DGP: pure noise residuals in each regime (no VAR dynamics to fit) with
    known impact matrix B_true and known variance scaling per regime.
    """

    def test_variance_ratios_converge_to_truth(self):
        """Rigobon should recover variance ratios [2, 4] from a DGP where
        regime-1 shocks are scaled by [sqrt(2), 2] relative to regime 0."""
        rng = np.random.default_rng(999)
        T0, T1, n = 1000, 1000, 2
        B_true = np.array([[1.0, 0.0], [0.5, 0.8]])

        e0 = rng.standard_normal((T0, n)) @ B_true.T
        scale1 = np.array([np.sqrt(2.0), 2.0])  # variance ratios: 2 and 4
        e1 = (rng.standard_normal((T1, n)) * scale1) @ B_true.T

        residuals = np.vstack([e0, e1])
        regime = np.array([0] * T0 + [1] * T1)

        Sigma_0, Sigma_1 = _regime_covariances(residuals, regime)
        _, lam_hat = _rigobon_impact(Sigma_0, Sigma_1)
        lam_sorted = np.sort(lam_hat)
        true_ratios = np.array([2.0, 4.0])
        np.testing.assert_allclose(
            lam_sorted, true_ratios, rtol=0.25,
            err_msg=f"variance_ratios {lam_sorted} too far from true {true_ratios}",
        )

    def test_structural_identities_are_exact_on_any_DGP(self):
        """The algebraic identities B@B.T = S0 and B@diag(lam)@B.T = S1 must
        hold to machine precision independent of DGP."""
        rng = np.random.default_rng(1234)
        T0, T1, n = 500, 500, 3
        e0 = rng.standard_normal((T0, n))
        e1 = rng.standard_normal((T1, n)) * rng.uniform(0.5, 3.0, n)
        residuals = np.vstack([e0, e1])
        regime = np.array([0] * T0 + [1] * T1)
        Sigma_0, Sigma_1 = _regime_covariances(residuals, regime)
        B, lam = _rigobon_impact(Sigma_0, Sigma_1)
        np.testing.assert_allclose(B @ B.T, Sigma_0, atol=1e-10)
        np.testing.assert_allclose(B @ np.diag(lam) @ B.T, Sigma_1, atol=1e-10)


def test_mbb_import_resolves():
    """Regression: the bootstrap-bands import must resolve. It was silently
    dead from a bad relative path (..inference -> puremacro.var.inference,
    which does not exist), so rigobon_svar never produced bands."""
    import importlib, sys
    importlib.import_module("puremacro.var.identify.hetero")
    hetero_mod = sys.modules["puremacro.var.identify.hetero"]
    assert hetero_mod._HAS_MBB is True
