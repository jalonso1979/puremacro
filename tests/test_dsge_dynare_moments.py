"""Unit tests for DSGE Dynare decision rules, theoretical moments, and analytical FEVD."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build,
    compute_fevd,
    DynareDR,
    load_mod,
    TheoreticalMomentsResult,
)
from puremacro.dsge._moments import ZeroVarianceWarning


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


# ===========================================================================
# conditional_fevd kernel: input validation, stationarity, per-row zero test
# ===========================================================================

@pytest.fixture
def two_state_model():
    """AR(1) states of very different scale feeding a huge and a tiny control."""
    return load_mod(
        """
        var z1 z2 big small; varexo e1 e2;
        model;
          z1 = 0.5*z1(-1) + e1;
          z2 = 0.5*z2(-1) + e2;
          big   = 1e8*z1 + 1e8*z2;
          small = 1e-1*z1 + 1e-1*z2;
        end;
        shocks; var e1; stderr 1.0; var e2; stderr 1.0; end;
        """
    )


def test_zero_variance_test_is_per_row_not_matrix_wide(two_state_model):
    """A well-scaled row is not erased because another variable is 1e9 larger.

    The zero test used to compare every row's total against the largest entry
    of the whole matrix, so ``small`` (loadings 1e-1) and both unit-variance
    states came back NaN with a 'variance is zero' warning purely because
    ``big`` (loadings 1e8) shares the table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZeroVarianceWarning)
        table = compute_fevd(two_state_model, horizons=[1, 4, None]).table

    assert not table.isna().to_numpy().any()
    np.testing.assert_allclose(table.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
    # 'big' and 'small' are the same linear combination up to a scale factor,
    # so their shares must be identical.
    np.testing.assert_allclose(
        table.xs("big", level="Variable").to_numpy(),
        table.xs("small", level="Variable").to_numpy(),
        atol=1e-12,
    )
    # z1 and z2 are each driven by exactly one shock.
    np.testing.assert_allclose(table.loc[("z1", 1)].to_numpy(), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(table.loc[("z2", 1)].to_numpy(), [0.0, 1.0], atol=1e-12)


def test_asymptotic_fevd_refuses_a_non_stationary_transition():
    """A unit root has an infinite unconditional variance: say so, do not invent shares.

    ``solve_discrete_lyapunov`` returns a large finite number for a singular
    pencil, which used to be reported as a perfectly ordinary 100% / 0% split.
    """
    m = load_mod(
        """
        var z b x; varexo e eb;
        model;
          z = 1.0*z(-1) + e;
          b = 0.5*b(-1) + eb;
          x = 0.5*z + 0.5*b;
        end;
        shocks; var e; stderr 0.01; var eb; stderr 0.02; end;
        """
    )
    with pytest.raises(ValueError, match="non-stationary eigenvalues"):
        compute_fevd(m, horizons=[1, 4, None])
    with pytest.raises(ValueError, match="non-stationary eigenvalues"):
        m.fevd(horizons=[None])

    # Finite horizons remain meaningful and are still computed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZeroVarianceWarning)
        table = compute_fevd(m, horizons=[1, 4]).table
    np.testing.assert_allclose(table.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(table.loc[("x", 1)].to_numpy(), [0.2, 0.8], atol=1e-12)


def test_correlated_shocks_are_refused_rather_than_silently_diagonalised():
    """The FEVD must not decompose a total that differs from the model's variance.

    ``theoretical_moments`` uses the full declared covariance; the FEVD used to
    take only ``sqrt(diag(...))``, so with ``corr e1, e2 = 0.8`` the shares
    described a different total than the Variance column beside them.
    """
    m = load_mod(
        """
        var z1 z2 x; varexo e1 e2;
        model;
          z1 = 0.5*z1(-1) + e1;
          z2 = 0.8*z2(-1) + e2;
          x = z1 + z2;
        end;
        shocks; var e1; stderr 0.01; var e2; stderr 0.005; corr e1, e2 = 0.8; end;
        """
    )
    np.testing.assert_allclose(m._shock_cov, [[1e-4, 4e-5], [4e-5, 2.5e-5]])
    with pytest.raises(ValueError, match=r"not \s*diagonal|not diagonal"):
        compute_fevd(m, horizons=[1, 4, None])

    # Deliberately decomposing the diagonal part stays available via sigma=.
    table = compute_fevd(m, horizons=[1], sigma={"e1": 0.01, "e2": 0.005}).table
    np.testing.assert_allclose(table.loc[("x", 1)].to_numpy(), [0.8, 0.2], atol=1e-12)


def test_fevd_horizons_are_validated(rbc_model):
    """h <= 0, fractional h and repeated infinities are user errors, not NaN rows."""
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        rbc_model.fevd(horizons=[0])
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        rbc_model.fevd(horizons=[-3])
    with pytest.raises(ValueError, match="not an integer number of periods"):
        rbc_model.fevd(horizons=[4.7])
    with pytest.raises(TypeError, match="horizon must be an int"):
        rbc_model.fevd(horizons=["four"])

    # An integral float is accepted and means exactly that horizon.
    np.testing.assert_allclose(
        rbc_model.fevd(horizons=[4.0]).to_numpy(),
        rbc_model.fevd(horizons=[4]).to_numpy(),
    )

    # Repeated asymptotic entries collapse to one row: the index stays unique.
    table = rbc_model.fevd(horizons=[None, np.inf, "Infinity", 4, 4])
    assert table.index.is_unique
    assert list(table.index.get_level_values("Horizon").unique()) == ["Infinity", 4]
