"""Tests for puremacro.inference.lewbel_iv."""
from __future__ import annotations

import numpy as np
import pytest


def test_lewbel_iv_result_dataclass_is_frozen():
    import dataclasses
    from puremacro.inference._results import LewbelIVResult

    res = LewbelIVResult(
        beta=np.array([1.0, 2.0]),
        se=np.array([0.1, 0.2]),
        t=np.array([10.0, 10.0]),
        n_obs=500,
        n_iv_constructed=3,
        first_stage_F=42.0,
        lewbel_diagnostic={"stat": 80.0, "p_value": 1e-12},
    )
    assert res.n_obs == 500
    assert res.first_stage_F == 42.0
    assert dataclasses.is_dataclass(res)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_obs = 600


def _lewbel_dgp(T: int, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lewbel (2012) DGP: y = β·x + u, x = z·e1 + e2,
    var(e1) = exp(0.5·z), so z drives heteroskedasticity in x."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(T)
    e1 = rng.standard_normal(T) * np.sqrt(np.exp(0.5 * z))
    e2 = rng.standard_normal(T)
    x = z * e1 + e2  # heteroskedastic in z
    u = e2 + 0.3 * rng.standard_normal(T)  # correlated with e2 → x endogenous
    y = beta * x + u
    return y, x.reshape(-1, 1), np.ones((T, 1)), z.reshape(-1, 1)


def test_lewbel_iv_recovers_known_beta():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=5000, beta=1.5, seed=0)
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(y, X_endog, X_exog, Z)
    # Coefficient on the endogenous regressor comes first.
    assert abs(res.beta[0] - 1.5) < 0.20, f"beta={res.beta[0]:.3f} far from 1.5"


def test_lewbel_iv_returns_dataclass():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=2000, beta=1.0, seed=1)
    from puremacro.inference.lewbel_iv import lewbel_iv
    from puremacro.inference._results import LewbelIVResult
    res = lewbel_iv(y, X_endog, X_exog, Z)
    assert isinstance(res, LewbelIVResult)
    assert res.n_obs == 2000
    assert res.beta.shape == (2,)  # 1 endog + 1 exog
    assert res.se.shape == (2,)
    assert res.t.shape == (2,)


def test_lewbel_iv_handles_two_endogenous():
    rng = np.random.default_rng(2)
    T = 4000
    z = rng.standard_normal(T)
    e1 = rng.standard_normal(T) * np.sqrt(np.exp(0.4 * z))
    e2 = rng.standard_normal(T) * np.sqrt(np.exp(-0.3 * z))
    x1 = z * e1 + rng.standard_normal(T)
    x2 = z * e2 + rng.standard_normal(T)
    u = e1 + e2
    y = 1.0 * x1 + 0.5 * x2 + u
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(
        y, np.column_stack([x1, x2]), np.ones((T, 1)), z.reshape(-1, 1),
    )
    assert res.beta.shape == (3,)  # 2 endog + 1 exog
    assert res.n_iv_constructed == 2  # k_z × k_endog = 1 × 2


def test_lewbel_iv_warns_on_weak_diagnostic():
    rng = np.random.default_rng(3)
    T = 1000
    # Homoskedastic DGP — heterosk_source carries no info.
    z = rng.standard_normal(T)
    x = rng.standard_normal(T)
    u = rng.standard_normal(T) + 0.5 * x
    y = 1.0 * x + u
    from puremacro.inference.lewbel_iv import lewbel_iv
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = lewbel_iv(y, x.reshape(-1, 1), np.ones((T, 1)), z.reshape(-1, 1))
    assert any("weak" in str(wi.message).lower() for wi in w)


def test_lewbel_iv_exported_from_inference_package():
    from puremacro.inference import lewbel_iv as exported
    from puremacro.inference.lewbel_iv import lewbel_iv as direct
    assert exported is direct
    from puremacro.inference import LewbelIVResult  # noqa: F401


def test_lewbel_iv_first_stage_F_finite_on_strong_data():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=4000, beta=1.0, seed=4)
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(y, X_endog, X_exog, Z)
    assert np.isfinite(res.first_stage_F)
    assert res.first_stage_F > 0
