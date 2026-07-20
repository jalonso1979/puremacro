"""Bring-your-own-likelihood: Student-t MLE with sandwich SE.

Demonstrates the ``puremacro.numerics`` workflow:
  1. Write a per-observation negative log-likelihood as a numpy function.
  2. Pass it to ``mle_fit`` to get a point estimate plus a numerical
     Hessian.
  3. Pass the same per-obs function to get heteroskedasticity-robust
     ("sandwich") standard errors.

We fit a Student-t with parameters (μ, log σ, log ν) on simulated
heavy-tailed returns, then compare classical (Hessian-only) and
sandwich SE. Under correct specification they should be similar; we
also fit a *misspecified* Gaussian on the same data and show how the
sandwich SE adjusts upward to reflect the true heavy tails.

Run:
    python -m puremacro.examples.mle_sandwich_demo
"""
from __future__ import annotations

import numpy as np

from ..numerics import mle_fit
from ..reports import coef_table


def _simulate_t(n: int = 1000, mu: float = 0.05, sigma: float = 1.5,
                nu: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Student-t with location-scale: y = mu + sigma * t(nu)
    return mu + sigma * rng.standard_t(df=nu, size=n)


def _t_neg_ll_per_obs(theta, y):
    mu, log_sigma, log_nu = theta
    sigma = np.exp(log_sigma)
    nu = np.exp(log_nu) + 2.0  # ensure ν > 2 so variance exists
    z = (y - mu) / sigma
    from math import lgamma, log, pi
    log_const = (lgamma((nu + 1) / 2) - lgamma(nu / 2)
                  - 0.5 * log(nu * pi) - log_sigma)
    return -log_const + 0.5 * (nu + 1) * np.log(1 + z ** 2 / nu)


def _gauss_neg_ll_per_obs(theta, y):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    return 0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * ((y - mu) / sigma) ** 2


def main() -> None:
    y = _simulate_t()
    print("Bring-your-own-likelihood demo on n=1000 Student-t draws "
          "(true μ=0.05, σ=1.5, ν=5.0)")
    print()

    # 1. Correctly-specified Student-t fit.
    res_t = mle_fit(
        lambda t: float(np.sum(_t_neg_ll_per_obs(t, y))),
        theta_init=np.array([0.0, 0.5, 1.0]),
        sandwich=True,
        neg_loglik_per_obs=lambda t: _t_neg_ll_per_obs(t, y),
    )
    mu_hat = res_t["theta"][0]
    sigma_hat = np.exp(res_t["theta"][1])
    nu_hat = np.exp(res_t["theta"][2]) + 2.0
    print(f"Student-t MLE: μ̂={mu_hat:+.3f}, σ̂={sigma_hat:.3f}, ν̂={nu_hat:.2f}")
    print(f"Classical SE:  {np.round(res_t['se_classical'], 4).tolist()}")
    print(f"Sandwich SE:   {np.round(res_t['se_sandwich'], 4).tolist()}")
    print()
    print(coef_table(
        res_t["theta"], res_t["se_sandwich"],
        names=["μ", "log σ", "log(ν-2)"],
        digits=4, include_p=True, include_ci=True,
    ))
    print()

    # 2. Misspecified Gaussian fit on the same heavy-tailed data.
    res_g = mle_fit(
        lambda t: float(np.sum(_gauss_neg_ll_per_obs(t, y))),
        theta_init=np.array([0.0, 0.5]),
        sandwich=True,
        neg_loglik_per_obs=lambda t: _gauss_neg_ll_per_obs(t, y),
    )
    mu_g = res_g["theta"][0]; sigma_g = np.exp(res_g["theta"][1])
    print(f"Misspecified Gaussian fit on the same data: "
          f"μ̂={mu_g:+.3f}, σ̂={sigma_g:.3f}")
    print(f"Classical SE: {np.round(res_g['se_classical'], 4).tolist()}")
    print(f"Sandwich SE:  {np.round(res_g['se_sandwich'], 4).tolist()}  "
          f"← should be inflated relative to classical because the "
          f"Gaussian is misspecified")


if __name__ == "__main__":
    main()
