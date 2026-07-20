import numpy as np
import pandas as pd
import pytest

from puremacro.garch.fit import garch11_fit
from puremacro.garch._results import GARCH11Result


def _simulate_garch(rng, T, omega, alpha, beta):
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / max(1e-8, 1 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1] ** 2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return pd.Series(eps, index=pd.date_range("1990-01-01", periods=T, freq="MS"))


def test_garch11_recovers_known_params():
    rng = np.random.default_rng(23)
    omega, alpha, beta = 0.05, 0.10, 0.85
    s = _simulate_garch(rng, T=2000, omega=omega, alpha=alpha, beta=beta)
    out = garch11_fit(s)
    # Verify documented attributes
    assert isinstance(out, GARCH11Result)
    for attr in ("omega", "alpha", "beta", "sigma", "loglik", "converged", "persistence"):
        assert hasattr(out, attr)
    assert out.converged
    # Persistence should be close to alpha + beta
    assert abs(out.persistence - (alpha + beta)) < 0.05
    # Sigma is a Series of length T with positive values
    assert len(out.sigma) == len(s)
    assert (out.sigma > 0).all()


def test_garch11_parity_with_arch():
    """Within 5pp of arch.arch_model — both fit Gaussian MLE so they
    converge to the same parameters on long enough data."""
    rng = np.random.default_rng(31)
    omega, alpha, beta = 0.04, 0.08, 0.88
    s = _simulate_garch(rng, T=3000, omega=omega, alpha=alpha, beta=beta)
    out = garch11_fit(s)
    from arch import arch_model
    am = arch_model(s.values, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
    # Tolerances reflect numerical optimizer differences
    assert abs(out.omega - am.params["omega"]) < 0.05
    assert abs(out.alpha - am.params["alpha[1]"]) < 0.05
    assert abs(out.beta - am.params["beta[1]"]) < 0.05


def test_garch11_result_is_frozen():
    rng = np.random.default_rng(0)
    s = _simulate_garch(rng, T=500, omega=0.05, alpha=0.10, beta=0.85)
    out = garch11_fit(s)
    assert isinstance(out, GARCH11Result)
    with pytest.raises(Exception):
        out.alpha = 99.0


def test_garch11_summary_smoke():
    rng = np.random.default_rng(0)
    s = _simulate_garch(rng, T=400, omega=0.05, alpha=0.10, beta=0.85)
    out = garch11_fit(s)
    txt = out.summary()
    assert "GARCH" in txt


def test_dcc_fit_returns_result_dataclass():
    from puremacro.garch.dcc import dcc_fit
    from puremacro.garch._results import DCCResult

    rng = np.random.default_rng(0)
    T = 600
    n = 3
    R = np.array([[1.0, 0.3, 0.1],
                  [0.3, 1.0, 0.2],
                  [0.1, 0.2, 1.0]])
    L = np.linalg.cholesky(R)
    rets = pd.DataFrame(
        (L @ rng.standard_normal((n, T))).T,
        columns=["A", "B", "C"],
        index=pd.date_range("1990-01-01", periods=T, freq="D"),
    )
    out = dcc_fit(rets)
    assert isinstance(out, DCCResult)
    assert 0.0 <= out.a <= 1.0
    assert 0.0 <= out.b <= 1.0
    assert out.R.shape == (T, n, n)
    assert out.H.shape == (T, n, n)
    assert out.sigma.shape == (T, n)
    with pytest.raises(Exception):
        out.a = 0.0
