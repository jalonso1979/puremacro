"""Disconnected structural confidence sets and a grid that misses them.

Run: python -m puremacro.examples.identification_robust_sets --output /tmp/weak-id
Known-covariance Gaussian moment illustration; no empirical model is estimated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from puremacro.structural import (
    MomentTargets, QuadraticMomentCurve, SimulationPlan, simulate_moments,
    moment_restriction_test, simulated_moment_test, invert_quadratic_moments,
    evaluate_moment_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("identification-robust-sets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = MomentTargets(np.array([1.]), np.array([[.01]]), ("second",))
    curve = QuadraticMomentCurve(np.array([0.]), np.array([1.]), targets.labels,
                                 "gaussian-theta-squared-v1", linear=0., quadratic=1.)
    confidence = invert_quadratic_moments(curve, targets, calibration="gaussian_known")
    assert confidence.resolved and len(confidence.intervals) == 2
    low, high = np.sqrt([1-norm.ppf(.975)*.1, 1+norm.ppf(.975)*.1])
    np.testing.assert_allclose(confidence.intervals, [[-high, -low], [low, high]], atol=1e-12)
    point = lambda theta: moment_restriction_test(targets, curve(theta), calibration="gaussian_known")
    coarse = evaluate_moment_grid(point, [-1.5, 0., 1.5])
    assert not coarse.accepted.any() and not coarse.continuous_complete
    clipped = invert_quadratic_moments(curve, targets, domain=(0., 1.), calibration="gaussian_known")
    assert clipped.truncated and clipped.boundary_contact == (False, True)
    constant = QuadraticMomentCurve(np.zeros(1), np.zeros(1), ("mean",), "constant-moment-v1")
    absent = invert_quadratic_moments(constant, MomentTargets(np.zeros(1), np.eye(1), constant.labels), domain=(-3., 3.),
                                     calibration="gaussian_known")
    assert absent.global_geometry == "all_real" and absent.truncated
    confidence.to_frame().to_csv(args.output/"components.csv", index=False)
    confidence.summary().to_csv(args.output/"set-status.csv", index=False)
    coarse.to_frame().to_csv(args.output/"coarse-grid.csv", index=False)
    coarse.unresolved_cells().to_csv(args.output/"unresolved-cells.csv", index=False)
    clipped.to_frame().to_csv(args.output/"clipped-components.csv", index=False)
    absent.to_frame().to_csv(args.output/"unidentified-clipped-components.csv", index=False)
    for suffix, content in (("md", confidence.to_markdown()), ("tex", confidence.to_latex()), ("typ", confidence.to_typst())):
        (args.output/f"components.{suffix}").write_text(content+"\n", encoding="utf-8")

    plan = SimulationPlan(seed=618, replications=64, periods=20)
    empirical = MomentTargets(np.array([.3]), np.array([[.02]]), ("mean",))
    def simulated(theta, rng, budget):
        return [theta[0]+(1+theta[0]**2)*rng.normal(size=budget.periods).mean()]
    def test_at(theta):
        batch = simulate_moments(simulated, [theta], plan, labels=empirical.labels, definition_id="heteroskedastic-normal-mean-v1")
        return simulated_moment_test(empirical, batch, assume_independent=True)
    simulation_grid = evaluate_moment_grid(test_at, np.linspace(-.5, 1., 61))
    assert simulation_grid.valid.all() and not simulation_grid.continuous_complete
    simulation_grid.to_frame().to_csv(args.output/"simulation-grid.csv", index=False)
    simulation_grid.accepted_runs().to_csv(args.output/"simulation-accepted-nodes.csv", index=False)
    report = {"scope": "declared Gaussian quadratic moment curve; general callback grid remains incomplete",
              "calibration": confidence.calibration, "alpha": confidence.alpha, "df": confidence.df,
              "definition_id": curve.definition_id, "components": confidence.intervals.tolist(),
              "max_endpoint_relative_residual": confidence.diagnostics["max_endpoint_relative_residual"],
              "coarse_grid_nodes": coarse.grid.tolist(), "coarse_grid_accepted": coarse.accepted.tolist(),
              "coarse_grid_continuous_complete": coarse.continuous_complete,
              "clipped": {"domain": clipped.domain, "components": clipped.intervals.tolist(),
                          "truncated": clipped.truncated, "global_geometry": clipped.global_geometry},
              "unidentified": {"global_components": [["-inf", "inf"]], "domain": absent.domain,
                               "displayed_components": absent.intervals.tolist(), "truncated": absent.truncated},
              "simulation": {"plan": plan.to_dict(), "plan_id": plan.plan_id,
                             "calibration": simulation_grid.tests[0].calibration,
                             "n_valid_nodes": int(simulation_grid.valid.sum()),
                             "n_accepted_nodes": int(simulation_grid.accepted.sum()),
                             "continuous_complete": simulation_grid.continuous_complete}}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(-1.5, 1.5, 601)
    statistics = (1-grid**2)**2/.01
    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    ax.plot(grid, statistics, color="#285e77", label="Point-null S statistic")
    ax.axhline(confidence.critical_value, color="#b16033", linestyle="--", label="5% rejection cutoff")
    for i, (left, right) in enumerate(confidence.intervals):
        ax.axvspan(left, right, color="#78aa80", alpha=.45, label="Accepted continuous components" if i == 0 else None)
    ax.scatter(coarse.grid, [r.statistic for r in coarse.tests], marker="x", color="black", s=50,
               zorder=4, label="Three sampled nodes: all rejected")
    ax.set(xlabel="Parameter, θ", ylabel="S statistic", ylim=(-2, 165),
           title="Two confidence-set components missed by a coarse grid")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="upper center")
    fig.savefig(args.output/"confidence-set.png", dpi=180)
    fig.savefig(args.output/"confidence-set.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(confidence.summary().to_string(index=False))
    print(confidence.to_frame().to_string(index=False))
    print("Coarse grid accepts no nodes; continuous set has two components.")
    print(f"Simulation grid: {int(simulation_grid.accepted.sum())} accepted nodes, continuous interiors unresolved.")
    print(f"Records: {args.output}")


if __name__ == "__main__":
    main()
