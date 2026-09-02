"""Smoke + return-shape tests for proxy_svar with the 0.4.0 ProxySVARResult API."""
import numpy as np
import pytest

from puremacro.var.identify.proxy import proxy_svar
from puremacro.var.identify._results import ProxySVARResult


def _synthetic_var_with_proxy(T=300, n=3, p=2, seed=0):
    rng = np.random.default_rng(seed)
    # Generate VAR(2) with a known structural shock
    A1 = 0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    A2 = 0.1 * np.eye(n) + 0.02 * rng.standard_normal((n, n))
    eps = rng.standard_normal((T, n)) * 0.5
    Y = np.zeros((T, n))
    for t in range(p, T):
        Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + eps[t]
    # Proxy: noisy version of the first structural shock
    z = eps[:, 0] + 0.3 * rng.standard_normal(T)
    return Y, z


def test_proxy_svar_returns_result_dataclass():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert isinstance(res, ProxySVARResult)


@pytest.mark.pyodide_smoke
def test_proxy_svar_irf_shape():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=12, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    n = Y.shape[1]
    assert res.irf_point.shape == (13, n, n)
    assert res.irf_lower.shape == (13, n, n)
    assert res.irf_upper.shape == (13, n, n)
    assert res.B.shape == (n, n)
    # Bootstrap percentile bands: lower <= upper (point may sit slightly
    # outside the percentile band when n_boot is small).
    assert (res.irf_lower <= res.irf_upper).all()


def test_proxy_svar_first_stage_F_is_finite_positive():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert np.isfinite(res.first_stage_F)
    assert res.first_stage_F > 0


def test_proxy_svar_strong_instrument_clears_op_cutoff():
    """Synthetic high-correlation instrument should produce F > 23 (Olea-Pflueger
    'strong' threshold)."""
    rng = np.random.default_rng(42)
    T = 500
    Y, _ = _synthetic_var_with_proxy(T=T, seed=42)
    # Instrument: dominant linear function of Y residuals' first column
    # Use a tighter proxy by reducing noise relative to signal
    eps = np.diff(Y, axis=0, prepend=Y[:1])  # rough residuals
    z = eps[:, 0] + 0.1 * rng.standard_normal(T)
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert res.first_stage_F > 23.0, f"strong proxy should clear F=23, got {res.first_stage_F:.2f}"


# ---------------------------------------------------------------------------
# Identification recovery. `_synthetic_var_with_proxy` above draws i.i.d. eps,
# so its B is a multiple of the identity and Sigma is proportional to I — and
# `Sigma @ Pi` is then proportional to `Pi`. The impact vector could be (and
# was) computed in the wrong metric without any of the tests above noticing.
# ---------------------------------------------------------------------------

def _proxy_dgp(T=200_000, seed=0):
    """A DGP whose Sigma is emphatically not proportional to the identity."""
    rng = np.random.default_rng(seed)
    B_true = np.array([[1.0, 0.0, 0.0],
                       [0.8, 1.0, 0.0],
                       [-0.5, 0.3, 1.0]])
    eps = rng.standard_normal((T, 3))
    u = eps @ B_true.T
    z = eps[:, 0] + 0.5 * rng.standard_normal(T)
    return B_true, u, np.cov(u, rowvar=False), z


def test_proxy_impact_recovers_the_true_column_when_sigma_is_not_scalar():
    """b_1 is identified by `Pi / sqrt(Pi' Sigma^-1 Pi)`, not `Sigma Pi / ...`.

    Sigma = BB' gives B' Sigma^-1 B = I, hence b_1' Sigma^-1 b_1 = 1; and
    Cov(u, z) is proportional to b_1 under the proxy assumptions. Normalising
    in the Sigma metric instead returns a vector proportional to `Sigma b_1`,
    which is b_1 only when b_1 is an eigenvector of Sigma.
    """
    from puremacro.var.identify.proxy import _proxy_impact_factory

    B_true, u, Sigma, z = _proxy_dgp()
    B = _proxy_impact_factory(z)(None, Sigma, u)
    got = B[:, 0]
    got = got if got[0] > 0 else -got            # sign is not identified
    assert np.allclose(got, B_true[:, 0], atol=5e-3), got


def test_proxy_impact_satisfies_the_covariance_identity():
    """BB' = Sigma held for the WRONG vector too, so it is not the guard.

    Kept as a positive control: it must still hold, and it must not be the
    only thing asserted about B.
    """
    from puremacro.var.identify.proxy import _proxy_impact_factory

    _, u, Sigma, z = _proxy_dgp(T=20_000)
    B = _proxy_impact_factory(z)(None, Sigma, u)
    assert np.abs(B @ B.T - Sigma).max() < 1e-8
    Si = np.linalg.inv(Sigma)
    assert float(B[:, 0] @ Si @ B[:, 0]) == pytest.approx(1.0, abs=1e-10)
