"""Simulation uncertainty in a Gaussian mean fit, with AR(1)/Markov checks.

Run: python -m puremacro.examples.simulation_aware_smm --output /tmp/smm
This is a synthetic reference experiment, not an empirical calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.structural import (
    MomentReplication, MomentTargets, SimulationPlan, fit_simulated_moments, simulate_moments,
)


def gaussian_mean(theta, rng, plan):
    return MomentReplication([theta[0]+rng.normal(size=plan.periods).mean()], counts=[plan.periods])


def ar1_moments(theta, rng, plan):
    """Mean, raw second moment and lag product, each with exactly T pairs.

    X_0 has the stationary Gaussian distribution. Retain X_{b+1},...,X_{b+T}
    and lag each against its immediate predecessor. Initialization is exact
    here; a generic simulation may require a burn-in bias study.
    """
    mu, rho, sigma = theta
    if abs(rho) >= 1 or sigma <= 0:
        return MomentReplication([np.nan]*3, status="nonstationary_or_invalid_scale")
    innovations = rng.normal(size=plan.periods+plan.burn_in+1)
    path = np.empty(len(innovations))
    path[0] = mu+sigma/np.sqrt(1-rho*rho)*innovations[0]
    for i in range(1, len(path)):
        path[i] = mu+rho*(path[i-1]-mu)+sigma*innovations[i]
    current, previous = path[plan.burn_in+1:], path[plan.burn_in:-1]
    return MomentReplication([current.mean(), (current*current).mean(), (current*previous).mean()],
                              counts=[plan.periods]*3)


def markov_moments(theta, rng, plan):
    """Stationary two-state chain: mean, lag product and switching frequency."""
    p01, p10 = theta
    if not 0 < p01 < 1 or not 0 < p10 < 1:
        return MomentReplication([np.nan]*3, status="invalid_transition")
    uniforms = rng.random(plan.periods+plan.burn_in+1)
    path = np.empty(len(uniforms), dtype=int)
    path[0] = int(uniforms[0] < p01/(p01+p10))
    for i in range(1, len(path)):
        path[i] = int(uniforms[i] < (p01 if path[i-1] == 0 else 1-p10))
    current, previous = path[plan.burn_in+1:], path[plan.burn_in:-1]
    return MomentReplication([current.mean(), (current*previous).mean(), (current != previous).mean()],
                              counts=[plan.periods]*3)


def ar1_mean_variance(rho: float, sigma: float, periods: int) -> float:
    lag = np.arange(1, periods)
    return float(sigma*sigma/(1-rho*rho) *
                 (periods+2*np.sum((periods-lag)*rho**lag))/(periods*periods))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulation-aware-smm"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = SimulationPlan(seed=20261108, replications=32, periods=10)
    targets = MomentTargets(np.array([.4]), np.array([[1/1000]]), ("mean",), units="level",
                            metadata={"example": "synthetic observed mean with known sampling variance"})
    fit = fit_simulated_moments(
        gaussian_mean, targets, [0.], plan=plan, definition_id="iid-normal-mean-v1",
        parameter_names=("mu",), jac=lambda t: [[1.]], assume_independent=True, assume_smooth=True,
        validation_plan=replace(plan, role="validation", replications=128),
    )
    assert fit.success and fit.inference_valid, fit.inference_reasons
    assert fit.validation is not None
    variance = targets.covariance[0, 0]+fit.simulation.covariance[0, 0]
    np.testing.assert_allclose(fit.covariance[0, 0], variance, rtol=1e-12)
    fit.to_frame().to_csv(args.output/"parameters.csv", index=False)
    fit.moment_fit().to_csv(args.output/"moments.csv", index=False)
    fit.moment_fit(validation=True).to_csv(args.output/"validation-moments.csv", index=False)
    pd.DataFrame(fit.simulation.replicates, columns=targets.labels).to_csv(args.output/"fit-replicates.csv", index=False)
    pd.DataFrame(fit.validation.replicates, columns=targets.labels).to_csv(args.output/"validation-replicates.csv", index=False)
    (args.output/"parameters.md").write_text(fit.to_markdown()+"\n", encoding="utf-8")
    (args.output/"parameters.tex").write_text(fit.to_latex()+"\n", encoding="utf-8")
    (args.output/"parameters.typ").write_text(fit.to_typst()+"\n", encoding="utf-8")

    ar_plan = SimulationPlan(seed=8801, replications=1024, periods=256, burn_in=20, role="validation")
    ar_theta = np.array([.4, .6, .5])
    ar_batch = simulate_moments(ar1_moments, ar_theta, ar_plan,
                               labels=("mean", "raw_second", "lag_product"), definition_id="stationary-ar1-pairs-v1")
    v = ar_theta[2]**2/(1-ar_theta[1]**2)
    exact_ar = np.array([ar_theta[0], ar_theta[0]**2+v, ar_theta[0]**2+ar_theta[1]*v])
    standardized_error = (ar_batch.mean-exact_ar)/np.sqrt(np.diag(ar_batch.covariance))
    exact_mean_variance = ar1_mean_variance(ar_theta[1], ar_theta[2], ar_plan.periods)/ar_plan.replications
    assert np.max(np.abs(standardized_error)) < 4
    assert abs(ar_batch.covariance[0, 0]/exact_mean_variance-1) < .2
    markov_plan = SimulationPlan(seed=8802, replications=256, periods=2048, burn_in=20, role="validation")
    transition = [.15, .2]
    markov_batch = simulate_moments(markov_moments, transition, markov_plan,
                                   labels=("mean", "lag_product", "switching"), definition_id="two-state-pairs-v1")
    pi = transition[0]/sum(transition)
    exact_markov = np.array([pi, pi*(1-transition[1]), (1-pi)*transition[0]+pi*transition[1]])
    markov_error = (markov_batch.mean-exact_markov)/np.sqrt(np.diag(markov_batch.covariance))
    assert np.max(np.abs(markov_error)) < 4
    for name, batch, exact in (("ar1", ar_batch, exact_ar), ("markov", markov_batch, exact_markov)):
        frame = batch.to_frame()
        frame["analytic"] = exact
        frame.to_csv(args.output/f"{name}-moments.csv", index=False)
        pd.DataFrame(batch.replicates, columns=batch.labels).to_csv(args.output/f"{name}-replicates.csv", index=False)
    report = {
        "scope": "synthetic Gaussian fit and stationary AR1/Markov moment checks; no general coverage or sovereign validation claim",
        "fit_plan": plan.to_dict(), "fit_plan_id": plan.plan_id, "definition_id": fit.simulation.definition_id,
        "validation_plan": fit.validation.plan.to_dict(), "validation_plan_id": fit.validation.plan.plan_id,
        "theta": fit.theta.tolist(), "standard_errors": fit.standard_errors.tolist(),
        "data_only_se": float(np.sqrt(targets.covariance[0, 0])),
        "sampling_variance": fit.sampling_covariance.tolist(), "simulation_variance": fit.simulation_covariance.tolist(),
        "total_variance": fit.covariance.tolist(), "objective": fit.objective,
        "success": fit.success, "inference_valid": fit.inference_valid,
        "inference_assumptions": fit.diagnostics["inference_assumptions"],
        "n_batches": fit.n_evals, "n_replications": fit.n_replications_evaluated,
        "ar1": {"plan": ar_plan.to_dict(), "theta": ar_theta.tolist(), "analytic": exact_ar.tolist(),
                "estimated": ar_batch.mean.tolist(), "error_in_mc_se": standardized_error.tolist(),
                "analytic_mean_variance": exact_mean_variance, "estimated_mean_variance": float(ar_batch.covariance[0, 0])},
        "markov": {"plan": markov_plan.to_dict(), "transition": transition, "analytic": exact_markov.tolist(),
                   "estimated": markov_batch.mean.tolist(), "error_in_mc_se": markov_error.tolist()},
    }
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(fit.summary().to_string(index=False))
    print(f"Data-only SE: {report['data_only_se']:.6f}; simulation-aware SE: {fit.standard_errors[0]:.6f}")
    print(f"AR1 standardized moment errors: {standardized_error}")
    print(f"Markov standardized moment errors: {markov_error}")
    print(f"Records: {args.output}")


if __name__ == "__main__":
    main()
