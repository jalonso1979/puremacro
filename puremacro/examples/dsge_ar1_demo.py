"""Toy AR(1) state-space — demonstrates puremacro.dsge.estimate_dsge.

This is a 2-parameter model — the smallest non-trivial Bayesian DSGE
estimation. It proves the generic engine is genuinely model-agnostic
(it's not silently coupled to SW07-specific structure).
"""
import numpy as np
import pandas as pd

from puremacro.dsge import estimate_dsge
from puremacro.state_space import StateSpaceModel


PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
}


def make_state_space(params: dict) -> StateSpaceModel:
    """y_t = x_t,  x_t = rho * x_{t-1} + sigma * eps_t."""
    rho = params["rho"]
    sigma = params["sigma"]
    return StateSpaceModel(
        T=np.array([[rho]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma ** 2]]),
        H=np.array([[1e-8]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate(rho_true: float, sigma_true: float, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def main() -> None:
    rho_true, sigma_true = 0.7, 0.5
    data = _simulate(rho_true, sigma_true, T=500, seed=0)
    res = estimate_dsge(
        data,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=["y"],
        initial_params={"rho": 0.5, "sigma": 0.4},
        model_name="AR1_demo",
        n_chains=1, n_draws=2000, burn_in=500, seed=0,
    )
    summary = res.summary()
    print(summary)
    print(f"\ntrue rho={rho_true:.3f}, posterior mean={summary.loc['rho', 'mean']:.3f}")
    print(f"true sigma={sigma_true:.3f}, posterior mean={summary.loc['sigma', 'mean']:.3f}")


if __name__ == "__main__":
    main()
