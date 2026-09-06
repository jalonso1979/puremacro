"""Regression tests for puremacro.garch.fit.garch11_fit (audit C28 / M63 / M64)
and the GARCH result presentation contract (audit M65).

Before the fix ``garch11_fit`` ran L-BFGS-B on the raw series with the
starting point ``[0.05 * var, 0.10, 0.85]`` and default tolerances. At unit
variance that reached the MLE, but on decimal daily returns (variance ~1e-4)
or on x100 data L-BFGS-B exited after 4 iterations with 'RELATIVE REDUCTION
OF F <= FACTR*EPSMCH' far from the maximum -- a Nelder-Mead polish from the
returned point raised the log-likelihood by up to +31 units, the (alpha, beta)
estimates changed with the units of the input, one seed in twenty returned
the untouched starting values, and every one of those fits said
``converged=True``. NaN input silently produced an all-NaN sigma path with
the starting values for (alpha, beta), and ``mean='bogus'`` was silently
treated as ``'zero'``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from puremacro.garch import DCCResult, GARCH11Result, dcc_fit, garch11_fit
from puremacro.garch.fit import _filter_sigma2, _neg_loglik


def _sim_garch(T, omega, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    eps = np.empty(T)
    s2 = np.empty(T)
    s2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(s2[0]) * rng.standard_normal()
    for t in range(1, T):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return eps


def _nm_polish_gain(y, res) -> float:
    """Log-likelihood a tight Nelder-Mead can still add from the fitted point."""
    var0 = float(np.var(y))
    ll_fit = -_neg_loglik([res.omega, res.alpha, res.beta], y, var0)
    nm = minimize(_neg_loglik, [res.omega, res.alpha, res.beta], args=(y, var0),
                  method="Nelder-Mead",
                  options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000})
    return float(-nm.fun - ll_fit)


# ---------------------------------------------------------------------------
# C28: scale equivariance and "fit is the maximum of its own likelihood"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true", [(0.05, 0.10, 0.85), (0.1, 0.30, 0.60), (0.2, 0.05, 0.90)])
def test_garch11_fit_is_scale_equivariant(true):
    """x100 and /100 must give the same alpha/beta, omega scaled by c^2 and
    the log-likelihood shifted by exactly -T log c.

    Old code: (0.121, 0.822) -> (0.117, 0.835) -> (0.119, 0.832) across the
    three scales on the same series, converged=True each time.
    """
    y = _sim_garch(2000, *true, seed=3)
    base = garch11_fit(y)
    T = len(y)
    for c in (100.0, 0.01):
        r = garch11_fit(y * c)
        assert r.alpha == pytest.approx(base.alpha, abs=1e-5)
        assert r.beta == pytest.approx(base.beta, abs=1e-5)
        assert r.omega == pytest.approx(base.omega * c ** 2, rel=1e-4)
        assert r.loglik == pytest.approx(base.loglik - T * np.log(c), abs=1e-4)
        np.testing.assert_allclose(r.sigma.values, base.sigma.values * c, rtol=1e-5)
        assert r.converged


@pytest.mark.parametrize("scale", [1.0, 100.0, 0.01])
@pytest.mark.parametrize("true", [(0.05, 0.10, 0.85), (0.1, 0.30, 0.60), (0.2, 0.05, 0.90)])
def test_nelder_mead_polish_cannot_improve_loglik(true, scale):
    """A Nelder-Mead polish from the returned point must gain < 0.01 log-lik.

    Old code gained +30.95 at scale 0.01 for (0.1, 0.3, 0.6) and +11.67 at
    scale 100, while reporting converged=True.
    """
    y = _sim_garch(2000, *true, seed=1) * scale
    r = garch11_fit(y)
    assert r.converged
    assert _nm_polish_gain(y, r) < 0.01


def test_decimal_daily_returns_do_not_return_the_starting_values():
    """On sd-1% decimal returns the old fit returned x0 = (0.10, 0.85)
    untouched for some seeds and |beta - true MLE| > 0.01 in 16-19/20 seeds."""
    al, be = 0.05, 0.90
    om = 1e-4 * (1 - al - be)
    n_stuck = 0
    n_far = 0
    for seed in range(8):
        y = _sim_garch(2000, om, al, be, seed)
        r = garch11_fit(y)
        n_stuck += (abs(r.alpha - 0.10) < 1e-12 and abs(r.beta - 0.85) < 1e-12)
        n_far += _nm_polish_gain(y, r) > 0.01
        assert r.converged
    assert n_stuck == 0
    assert n_far == 0


def test_lfilter_recursion_matches_python_loop():
    """The C-loop filter must reproduce the textbook recursion to rounding."""
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(500) * 0.01
    omega, alpha, beta, var0 = 2e-6, 0.08, 0.9, float(np.var(eps))
    loop = np.empty(len(eps))
    loop[0] = var0
    for t in range(1, len(eps)):
        loop[t] = omega + alpha * eps[t - 1] ** 2 + beta * loop[t - 1]
    np.testing.assert_allclose(_filter_sigma2(eps, omega, alpha, beta, var0), loop,
                               rtol=1e-12)


def test_unit_scale_fit_unchanged_within_optimiser_precision():
    """The unit-scale answer must stay where it was (it was already at the
    MLE): the golden-DGP parameters move by < 1e-5, the log-lik by < 1e-6."""
    from puremacro.validation.cases_garch import garch_demo_data

    y = garch_demo_data()["y"]
    r = garch11_fit(y)
    # Values of the 2.3.0 estimator on this series (L-BFGS-B, no polish).
    old = (0.05327226, 0.07779402, 0.90381724, -9429.866556)
    assert r.omega == pytest.approx(old[0], abs=1e-5)
    assert r.alpha == pytest.approx(old[1], abs=1e-5)
    assert r.beta == pytest.approx(old[2], abs=1e-5)
    assert r.loglik == pytest.approx(old[3], abs=1e-4)
    assert r.loglik >= old[3] - 1e-6


# ---------------------------------------------------------------------------
# M63 / M64: input validation
# ---------------------------------------------------------------------------

def test_nan_input_raises_value_error():
    """Old code: one NaN -> omega=nan, alpha=0.1, beta=0.85 (x0), all-NaN
    sigma, no exception."""
    y = _sim_garch(300, 0.05, 0.1, 0.85, seed=2)
    y[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        garch11_fit(y)
    y[100] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        garch11_fit(pd.Series(y))


@pytest.mark.parametrize("bad", ["bogus", "const", "Zero", ""])
def test_invalid_mean_raises_value_error(bad):
    """Old code silently fitted a zero-mean model for any unknown `mean`."""
    y = _sim_garch(300, 0.05, 0.1, 0.85, seed=2) + 0.5
    with pytest.raises(ValueError, match="mean must be one of"):
        garch11_fit(y, mean=bad)


def test_mean_constant_still_demeans():
    y = _sim_garch(1500, 0.05, 0.1, 0.85, seed=5) + 0.5
    rc = garch11_fit(y, mean="constant")
    rz = garch11_fit(y, mean="zero")
    assert rc.loglik > rz.loglik + 10


def test_too_short_and_multidimensional_inputs_raise():
    with pytest.raises(ValueError):
        garch11_fit(np.array([1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        garch11_fit(np.ones((10, 3)))
    # A (T, 1) column is accepted.
    y = _sim_garch(300, 0.05, 0.1, 0.85, seed=2)
    assert garch11_fit(y.reshape(-1, 1)).alpha == pytest.approx(garch11_fit(y).alpha)


# ---------------------------------------------------------------------------
# M65: presentation contract on GARCH11Result and DCCResult; dcc ndarray input
# ---------------------------------------------------------------------------

def _dcc_panel(T=400, seed=0):
    rng = np.random.default_rng(seed)
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    L = np.linalg.cholesky(R)
    arr = (L @ rng.standard_normal((2, T))).T
    return pd.DataFrame(arr, columns=["A", "B"],
                        index=pd.date_range("2000-01-01", periods=T, freq="B"))


def test_garch11_result_presentation_contract():
    """to_markdown/to_latex/to_typst/plot were missing (only summary())."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    y = _sim_garch(400, 0.05, 0.1, 0.85, seed=2)
    r = garch11_fit(pd.Series(y, index=pd.date_range("2000-01-01", periods=400, freq="B")))
    assert isinstance(r, GARCH11Result)
    frame = r.to_frame()
    assert list(frame.columns) == ["parameter", "value"]
    assert set(frame["parameter"]) >= {"omega", "alpha", "beta", "persistence", "loglik"}
    md = r.to_markdown()
    assert md.startswith("|") and "alpha" in md
    tex = r.to_latex()
    assert tex.startswith("\\begin{tabular}") and tex.endswith("\\end{tabular}")
    typ = r.to_typst()
    assert typ.startswith("#table(") and "alpha" in typ
    fig = r.plot(title="vol")
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_dcc_result_presentation_contract_and_ndarray_input():
    """DCCResult lacked the table/plot methods and dcc_fit crashed with
    AttributeError on an ndarray panel (``returns.index``)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    panel = _dcc_panel()
    r = dcc_fit(panel)
    assert isinstance(r, DCCResult)
    frame = r.to_frame()
    assert list(frame.columns) == ["series", "parameter", "value"]
    assert {"A", "B", "DCC"} <= set(frame["series"])
    assert "a+b" in set(frame["parameter"])
    assert r.to_markdown().startswith("|")
    assert r.to_latex().startswith("\\begin{tabular}")
    assert r.to_typst().startswith("#table(")
    corr = r.correlations()
    assert list(corr.columns) == ["A-B"]
    np.testing.assert_allclose(corr["A-B"].values, r.R[:, 0, 1])
    fig = r.plot()
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)

    rn = dcc_fit(panel.values)
    assert rn.a == pytest.approx(r.a) and rn.b == pytest.approx(r.b)
    assert list(rn.sigma.columns) == [0, 1]
    assert len(rn.sigma.index) == len(panel)


def test_dcc_fit_rejects_nan_and_wrong_shape():
    panel = _dcc_panel(T=100)
    bad = panel.copy()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        dcc_fit(bad)
    with pytest.raises(ValueError, match="panel"):
        dcc_fit(panel["A"].values)
