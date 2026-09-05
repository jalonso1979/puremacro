"""Unit tests for DSGE Dynare decision rules, theoretical moments, and analytical FEVD."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build,
    DynareDR,
    TheoreticalMomentsResult,
)


@pytest.fixture
def rbc_model():
    """Standard calibrated RBC model solved via puremacro.dsge.build."""
    def eqs(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.eps,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    return build(
        eqs,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )


def test_dynare_decision_rules(rbc_model):
    dr = rbc_model.decision_rules()
    assert isinstance(dr, DynareDR)
    assert rbc_model.dynare_dr is not None

    assert list(dr.variable_names) == ["c", "k", "z"]
    assert list(dr.state_variables) == ["k", "z"]
    assert list(dr.shock_names) == ["eps"]

    # Dimensions
    assert dr.ghx.shape == (3, 2)
    assert dr.ghu.shape == (3, 1)

    # State transition rows match Klein G and N
    np.testing.assert_allclose(dr.ghx.loc["k", "k"], rbc_model.solution.G[0, 0])
    np.testing.assert_allclose(dr.ghu.loc["k", "eps"], rbc_model.solution.N[0, 0])

    # Dynare-style combined table
    frame = dr.to_frame()
    assert list(frame.columns) == ["c", "k", "z"]
    assert list(frame.index) == ["Constant", "k(-1)", "z(-1)", "eps"]
    assert frame.loc["Constant", "z"] == pytest.approx(1.0)

    # Formatted exports
    summ = dr.summary()
    assert "POLICY AND TRANSITION FUNCTIONS (Dynare Format)" in summ
    assert "k(-1)" in summ
    assert "\\begin{tabular}" in dr.to_latex()
    assert "#table" in dr.to_typst()


def test_theoretical_moments_and_autocorrelations(rbc_model):
    res = rbc_model.theoretical_moments(lags=5)
    assert isinstance(res, TheoreticalMomentsResult)

    # Moments table
    mom = res.moments
    assert list(mom.columns) == ["Mean", "Std.Dev.", "Variance"]
    assert mom.loc["z", "Mean"] == pytest.approx(1.0)
    # Analytical variance of AR(1) z with rho=0.95, sigma=1.0: 1 / (1 - 0.95^2) = 10.2564
    assert mom.loc["z", "Variance"] == pytest.approx(1.0 / (1.0 - 0.95**2), rel=1e-3)
    assert mom.loc["z", "Std.Dev."] == pytest.approx(np.sqrt(1.0 / (1.0 - 0.95**2)), rel=1e-3)

    # Correlation matrix properties
    corr = res.correlation
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    assert np.all(corr.to_numpy() >= -1.0)
    assert np.all(corr.to_numpy() <= 1.0)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)

    # Autocorrelation of AR(1) technology: rho(z, k) = rho^k
    ac = res.autocorr
    for k in range(1, 6):
        expected_rho_k = 0.95**k
        assert ac.loc["z", f"Lag {k}"] == pytest.approx(expected_rho_k, rel=1e-3)

    # Summary and table formatting
    summ = res.summary()
    assert "THEORETICAL MOMENTS (Dynare stoch_simul)" in summ
    assert "MATRIX OF CORRELATIONS" in summ
    assert "COEFFICIENTS OF AUTOCORRELATION" in summ
    assert "VARIANCE DECOMPOSITION" in summ

    assert "\\begin{tabular}" in res.to_latex()
    assert "#table" in res.to_typst()


def test_multi_shock_fevd():
    # Model with two shocks: technology (e_z) and preference (e_b)
    def eqs_2shock(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta) - e.eb,
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.ez,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    m2 = build(
        eqs_2shock,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["ez", "eb"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )

    mom2 = m2.theoretical_moments(fevd_horizons=[1, 4, 8, None], sigma=dict(ez=0.01, eb=0.005))
    fevd = mom2.fevd

    assert list(fevd.columns) == ["ez", "eb"]
    # At all horizons and for all variables, shares sum to 100%
    row_sums = fevd.sum(axis=1)
    np.testing.assert_allclose(row_sums, 100.0, atol=1e-4)


def test_non_stationary_model_error():
    # Model with unstable unit root or explosive transition
    def bad_eqs(xp, x, e, p):
        return [
            xp.k - 1.2 * x.k - e.eps,
            xp.c - x.c,
        ]

    with pytest.raises(Exception):
        m_bad = build(
            bad_eqs,
            variables=["k", "c"],
            states=["k"],
            shocks=["eps"],
            params=dict(),
            steady_state=dict(k=0.0, c=0.0),
        )
        m_bad.theoretical_moments()
