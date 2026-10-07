"""Conditional policy losses and the ``loss_criterion`` of ``stabilization_bias``.

``PolicyResult.conditional_loss`` is (1 - beta) E_0 sum_t beta^t loss_t with the
economy starting at the steady state (zero lagged variables and multipliers).
The oracles are the Clarida, Gali and Gertler (1999) closed forms for a
unit-variance cost-push shock u_t = rho u_{t-1} + e_t:

* discretion: pi_t = alpha q u_t and x_t = -lam q u_t, so the conditional loss is
  alpha q^2 (alpha + lam^2) / (1 - beta rho^2);
* timeless commitment: x_t = delta x_{t-1} + c_u u_t and
  pi_t = -(alpha / lam)(x_t - x_{t-1}). Both responses are a delta^j + b rho^j,
  so the discounted sums are geometric series.

From the steady state the commitment plan is the Ramsey optimum, so the
conditional stabilization bias is non-negative, while the unconditional one
turns negative at low beta (Sauer 2010, Prop. 2).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.policy import discretionary_policy, lq_commitment, optimal_policy


def _cgg_model(beta, lam, alpha, rho, phi=1.0, mu=0.8):
    values = {"beta": beta, "phi": phi, "lam": lam, "alpha": alpha, "rho": rho, "mu": mu,
              "phi_pi": 1.5, "phi_x": 0.5}
    text = ("var x pi i g u;\nvarexo eps_g eps_u;\n"
            f"parameters {' '.join(values)};\n"
            + "".join(f"{k} = {float(v)!r};\n" for k, v in values.items())
            + "model;\n"
            "x = x(+1) - phi*(i - pi(+1)) + g;\n"
            "pi = lam*x + beta*pi(+1) + u;\n"
            "i = phi_pi*pi + phi_x*x;\n"
            "g = mu*g(-1) + eps_g;\nu = rho*u(-1) + eps_u;\nend;\n"
            "shocks;\nvar eps_g; stderr 1;\nvar eps_u; stderr 1;\nend;\n")
    return build_dynare(text)


def _closed_form_conditional_losses(beta, lam, alpha, rho):
    """CGG (1999) conditional losses (1 - beta) E_0 sum beta^t (pi^2 + alpha x^2)."""
    q = 1.0 / (lam**2 + alpha * (1.0 - beta * rho))
    disc = alpha * q**2 * (alpha + lam**2) / (1.0 - beta * rho**2)

    a = alpha / (alpha * (1.0 + beta) + lam**2)
    delta = (1.0 - np.sqrt(1.0 - 4.0 * beta * a**2)) / (2.0 * a * beta)
    c_u = -lam * delta / (alpha * (1.0 - delta * beta * rho))

    def geometric(p, r):  # sum_j beta^j (p delta^j + r rho^j)^2
        return (p**2 / (1.0 - beta * delta**2) + 2.0 * p * r / (1.0 - beta * delta * rho)
                + r**2 / (1.0 - beta * rho**2))

    x_delta, x_rho = c_u * delta / (delta - rho), -c_u * rho / (delta - rho)
    pi_delta = -(alpha / lam) * c_u * (delta - 1.0) / (delta - rho)
    pi_rho = -(alpha / lam) * c_u * (1.0 - rho) / (delta - rho)
    comm = geometric(pi_delta, pi_rho) + alpha * geometric(x_delta, x_rho)
    return disc, comm


def _solve(beta, lam, alpha, rho, **kwargs):
    model = _cgg_model(beta, lam, alpha, rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model, discretionary_policy(
            model, target_vars=["pi", "x"], weights={"pi": 1.0, "x": alpha},
            instruments="i", beta=beta, **kwargs)


CALIBRATIONS = [
    (0.99, 0.1, 0.25, 0.5),   # notebook 66 baseline
    (0.5, 0.1, 0.25, 0.0),    # unconditional bias negative (Sauer 2010, Prop. 2)
    (0.8, 0.3, 0.5, 0.9),
]


@pytest.mark.parametrize("beta,lam,alpha,rho", CALIBRATIONS)
def test_conditional_losses_match_cgg_closed_forms(beta, lam, alpha, rho):
    _, disc = _solve(beta, lam, alpha, rho)
    disc_cf, comm_cf = _closed_form_conditional_losses(beta, lam, alpha, rho)
    # The Dennis iteration stops at ||F_{k+1} - F_k|| < 1e-9; commitment is a direct QZ solve.
    np.testing.assert_allclose(disc.conditional_loss, disc_cf, rtol=1e-7)
    np.testing.assert_allclose(disc.commitment_result.conditional_loss, comm_cf, rtol=1e-10)


def test_notebook_66_baseline_values():
    _, disc = _solve(0.99, 0.1, 0.25, 0.5)
    np.testing.assert_allclose(disc.conditional_loss, 4.653008334868956, rtol=1e-7)
    np.testing.assert_allclose(disc.commitment_result.conditional_loss, 3.111201541849508, rtol=1e-10)


def test_default_criterion_is_unconditional_and_unchanged():
    _, disc = _solve(0.99, 0.1, 0.25, 0.5)
    assert disc.loss_criterion == "unconditional"
    assert disc.stabilization_bias == pytest.approx(disc.loss - disc.commitment_result.loss, abs=1e-12)


def test_conditional_criterion_is_positive_where_unconditional_is_negative():
    _, unc = _solve(0.5, 0.1, 0.25, 0.0)
    _, con = _solve(0.5, 0.1, 0.25, 0.0, loss_criterion="conditional")
    assert unc.stabilization_bias < -0.19
    assert con.loss_criterion == "conditional"
    assert con.stabilization_bias > 0.03
    assert con.stabilization_bias == pytest.approx(
        con.conditional_loss - con.commitment_result.conditional_loss, abs=1e-12)
    # The criterion changes only the bias, never the solved policy or its losses.
    assert con.loss == pytest.approx(unc.loss, abs=1e-12)
    assert con.conditional_loss == pytest.approx(unc.conditional_loss, abs=1e-12)


def test_conditional_bias_nonnegative_at_random_calibrations():
    rng = np.random.default_rng(20260923)
    for _ in range(12):
        beta = rng.uniform(0.3, 0.99)
        lam = rng.uniform(0.05, 0.5)
        alpha = rng.uniform(0.1, 1.0)
        rho = rng.uniform(0.0, 0.9)
        _, disc = _solve(beta, lam, alpha, rho, loss_criterion="conditional")
        disc_cf, comm_cf = _closed_form_conditional_losses(beta, lam, alpha, rho)
        assert disc.converged
        np.testing.assert_allclose(disc.conditional_loss, disc_cf, rtol=1e-6)
        np.testing.assert_allclose(disc.commitment_result.conditional_loss, comm_cf, rtol=1e-9)
        assert disc.stabilization_bias >= -1e-9


def test_conditional_loss_equals_discounted_irf_sum():
    """Cross-check the Lyapunov formula against discounted squared impulse responses."""
    beta, alpha = 0.95, 0.4
    model, disc = _solve(beta, 0.2, alpha, 0.7)
    weights = {"pi": 1.0, "x": alpha}
    horizon = 1500
    discount = beta ** np.arange(horizon + 1)
    for result in (disc, disc.commitment_result):
        total = 0.0
        for shock in result.linear_model.shocks:
            irf = result.linear_model.irf(shock, horizon=horizon)
            total += sum(w * np.sum(discount * irf[v].to_numpy() ** 2) for v, w in weights.items())
        np.testing.assert_allclose(result.conditional_loss, total, rtol=1e-9)


def test_optimal_policy_forwards_loss_criterion():
    model = _cgg_model(0.5, 0.1, 0.25, 0.0)
    weights = {"pi": 1.0, "x": 0.25}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc = optimal_policy(model, weights, "discretion", instruments="i", beta=0.5,
                              loss_criterion="conditional")
        comm = optimal_policy(model, weights, "commitment", instruments="i", beta=0.5)
    assert disc.loss_criterion == "conditional"
    assert disc.stabilization_bias > 0.0
    assert np.isfinite(comm.conditional_loss)
    assert comm.conditional_loss == pytest.approx(disc.commitment_result.conditional_loss, abs=1e-12)


def test_lq_commitment_reports_both_losses():
    model = _cgg_model(0.99, 0.1, 0.25, 0.5)
    comm = lq_commitment(model, ["pi", "x"], {"pi": 1.0, "x": 0.25}, "i", beta=0.99)
    _, comm_cf = _closed_form_conditional_losses(0.99, 0.1, 0.25, 0.5)
    assert np.isfinite(comm.loss)
    np.testing.assert_allclose(comm.conditional_loss, comm_cf, rtol=1e-10)
    assert "Conditional loss" in comm.summary()


def test_summary_names_the_criterion():
    _, con = _solve(0.8, 0.3, 0.5, 0.9, loss_criterion="conditional")
    text = con.summary()
    assert "Conditional loss (from s.s.)" in text
    assert "Stabilization bias (Loss^disc - Loss^comm, conditional)" in text


def test_invalid_loss_criterion_raises():
    model = _cgg_model(0.99, 0.1, 0.25, 0.5)
    with pytest.raises(ValueError, match="loss_criterion"):
        discretionary_policy(model, ["pi", "x"], {"pi": 1.0, "x": 0.25}, "i",
                             loss_criterion="ergodic")
