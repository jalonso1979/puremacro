"""Empirical LPs -> joint sampling covariance -> a three-equation NK model.

Run ``python -m puremacro.examples.structural_irf_matching --output DIR``.
Data are simulated from an independently derived closed-form equilibrium.
This is a parameter-recovery application, not an empirical NK replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro.dsge import load_mod
from puremacro.structural import fit_structural
from puremacro.structural.lp import lp_moment_targets

NK_MODEL = """
var y pi i v;
varexo eps;
parameters sigma kappa rho beta phi;
sigma=1.5; kappa=0.15; rho=0.6; beta=0.99; phi=1.5;
model;
 y = y(+1) - (i-pi(+1))/sigma;
 pi = beta*pi(+1) + kappa*y;
 i = phi*pi + v;
 v = rho*v(-1) + eps;
end;
"""
TRUTH = np.array([1.5, .15, .6])


def simulate_data(n: int = 4000, seed: int = 20261001) -> pd.DataFrame:
    """Independent closed-form solution with independent measurement errors."""
    if not isinstance(n, int) or n < 100:
        raise ValueError("n must be an integer of at least 100 observations")
    rng = np.random.default_rng(seed)
    sigma, kappa, rho = TRUTH
    a = -1 / (sigma * (1-rho) + (1.5-rho) * kappa / (1-.99*rho))
    b = kappa * a / (1-.99*rho)
    innovations = rng.normal(scale=.01, size=n+200)
    state = np.zeros(n+200)
    for t in range(1, len(state)):
        state[t] = rho*state[t-1] + innovations[t]
    return pd.DataFrame({
        "y": a*state[200:] + rng.normal(scale=.002, size=n),
        "pi": b*state[200:] + rng.normal(scale=.001, size=n),
        "eps": innovations[200:],
    }, index=pd.period_range("1000Q1", periods=n, freq="Q"))


def run_application(output: Path, *, n: int = 4000, seed: int = 20261001):
    """Fit sigma/kappa/rho and write a replayable parameter-recovery dossier."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = simulate_data(n=n, seed=seed)
    data_path = output / "simulated_data.csv"
    data.to_csv(data_path)
    targets = lp_moment_targets(
        data, responses=["y", "pi"], shock="eps", horizons=range(9), lags=2,
        response_units={"y": "log output deviation", "pi": "quarterly inflation fraction"},
        shock_unit="quarterly nominal-rate fraction", shock_size=.01, frequency="Q",
        bandwidth=9, metadata={"source": "independent closed-form NK simulation",
                               "is_synthetic": True, "seed": seed},
    )
    fitted_labels = [f"{r}:eps:h={h}" for r in ("y", "pi") for h in range(5)]
    held_labels = [f"{r}:eps:h={h}" for r in ("y", "pi") for h in range(5, 9)]
    fitted, held = targets.select(fitted_labels), targets.select(held_labels)

    def moments(theta, horizons):
        model = load_mod(NK_MODEL, params=dict(zip(("sigma", "kappa", "rho"), theta)))
        paths = model.irf("eps", horizon=max(horizons), size=.01)
        return np.array([paths.loc[h, response] for response in ("y", "pi") for h in horizons])

    result = fit_structural(
        lambda theta: moments(theta, range(5)), fitted, [1.2, .1, .45],
        parameter_names=("sigma", "kappa", "rho"), moment_labels=fitted.labels,
        bounds=[(.5, 3.), (.03, .4), (.05, .9)], held_out=held,
        held_out_moments_at=lambda theta: moments(theta, range(5, 9)),
        assume_correct_specification=True,
    )
    if not result.success or not result.inference_valid:
        raise RuntimeError(f"NK fit did not meet numerical inference conditions: {result.message}")
    parameters = result.summary()
    parameters["simulation_truth"] = TRUTH
    parameters.to_csv(output / "parameters.csv", index=False)
    targets.to_frame().to_csv(output / "lp_targets.csv", index=False)
    pd.DataFrame(targets.covariance, index=targets.labels, columns=targets.labels).to_csv(output / "joint_covariance.csv")
    result.moment_fit().to_csv(output / "fitted_moments.csv", index=False)
    result.moment_fit(held_out=True).to_csv(output / "held_out_moments.csv", index=False)
    manifest = {
        "package_version": __version__, "application": "NK IRF parameter recovery",
        "evidence_kind": "synthetic_parameter_recovery", "is_empirical_replication": False,
        "seed": seed, "observations": n, "true_parameters": dict(zip(result.parameter_names, TRUTH.tolist())),
        "estimated_parameters": dict(zip(result.parameter_names, result.theta.tolist())),
        "model": NK_MODEL, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "covariance": "joint aligned-score Bartlett HAC of LP coefficient estimators; bandwidth 9; no diagonal approximation",
        "shock_normalization": "0.01 quarterly nominal-rate innovation (one percentage point)",
        "training_horizons": list(range(5)), "held_out_horizons": list(range(5, 9)),
        "objective": result.objective, "identification_rank": result.identification_rank,
        "jacobian_condition": result.condition_number, "inference_valid": result.inference_valid,
        "j_pvalue": result.j_pvalue,
        "limitations": ["synthetic exogenous shocks, not estimated historical monetary-policy shocks",
                        "local asymptotic uncertainty; no finite-sample coverage claim",
                        "held-out horizons use the same data and are descriptive, not an independent validation sample"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from puremacro.reports import df_to_markdown
    (output / "report.md").write_text(
        "# Empirical-to-structural parameter recovery\n\n"
        "Simulated data generated from a closed-form New Keynesian equilibrium; "
        "LP moments are fitted by solving the model's equations with puremacro. "
        "This demonstrates the estimation workflow, not a historical empirical replication.\n\n"
        + df_to_markdown(parameters, index=False)
        + f"\n\nJoint moment covariance retained; local Jacobian rank {result.identification_rank}/3. "
        "Horizons 0–4 fit the model; horizons 5–8 are descriptive checks. "
        "The manifest records the data hash, model, seed, shock normalization, and limitations.\n",
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    prediction = moments(result.theta, range(9)).reshape(2, 9)
    for j, (response, ax) in enumerate(zip(("Output", "Inflation"), axes)):
        selected = slice(j*9, (j+1)*9)
        ax.errorbar(range(9), 100*targets.values[selected],
                    yerr=1.96*100*np.sqrt(np.diag(targets.covariance)[selected]),
                    fmt="o", color="#256D85", label="LP, pointwise 95% intervals")
        ax.plot(range(9), 100*prediction[j], color="#AC4C31", label="Fitted NK model")
        ax.axvline(4.5, linestyle=":", color="gray")
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set(xlabel="Quarters after a 1 pp policy innovation",
               ylabel="Percent deviation" if j == 0 else "Percentage points per quarter",
               title=response)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "irf_fit.png", dpi=170)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research_output/structural_bridge"))
    parser.add_argument("--observations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    result = run_application(args.output, n=args.observations, seed=args.seed)
    print(result.to_markdown())
    print(f"Saved study to {args.output}")


if __name__ == "__main__":
    main()
