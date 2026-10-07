"""Finite-sample SW07 covariance-matching diagnostics and reviewable evidence.

Run ``python -m puremacro.examples.sw07_finite_sample --output DIR``.
Simulation uncertainty is reported separately from parameter uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro import __version__
from puremacro.structural.sw07_finite_sample import run_sw07_finite_sample


SCENARIOS = ("baseline_fixed", "published_fixed", "baseline_fitted", "published_fitted")
TABLES = ("summary", "draws", "moment_audit", "covariance_audit", "observed_fits")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _cell(value: Any) -> str:
    value = _json_value(value)
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(_cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def _plot_criterion(study, output: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = study.summary["scenario"].tolist()
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 3*len(scenarios)),
                             constrained_layout=True, squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        selected = study.draws.loc[study.draws["scenario"] == scenario]
        criterion = selected.loc[~selected["unresolved"], "objective"].to_numpy(dtype=float)
        finite = np.sort(criterion[np.isfinite(criterion)])
        if len(finite):
            ax.step(finite, np.arange(1, len(finite)+1)/len(finite), where="post",
                    color="#2864a0", label=f"Usable simulated criteria (n={len(finite)})")
        else:
            ax.text(.5, .5, "No finite simulated criteria", ha="center", transform=ax.transAxes)
        observed = study.summary.loc[study.summary["scenario"] == scenario].iloc[0]
        if np.isfinite(float(observed["observed_objective"])):
            ax.axvline(float(observed["observed_objective"]), color="#b0472f", linewidth=1.5,
                       label="Observed-data criterion")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set(title=str(scenario), xlabel="Minimized criterion (symmetric log scale)",
               ylabel="Empirical cumulative fraction", ylim=(0, 1.03))
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(output / "criterion_distribution.png", dpi=160)
    plt.close(fig)


def run_application(output: Path, *, replications: int = 399, seed: int = 20261001,
                    bandwidth: int = 8, scenarios=None, progress=None, resume: bool = True):
    """Run the diagnostic and export every draw, covariance, and comparison."""
    study = run_sw07_finite_sample(replications=replications, seed=seed, bandwidth=bandwidth,
                                  scenarios=scenarios, progress=progress,
                                  checkpoint_dir=Path(output)/"_checkpoints", resume=resume)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(study, name).to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "draw_evidence.npz", moments=study.draw_moments,
                        covariances=study.draw_covariances)
    manifest = {**study.metadata,
        "package_version": __version__,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__},
        "array_shapes": {"moments": list(study.draw_moments.shape),
                         "covariances": list(study.draw_covariances.shape)},
        "summary": study.summary.to_dict(orient="records"),
        "observed_fits": study.observed_fits.to_dict(orient="records"),
        "json_missing_values": "Unavailable or nonfinite numerical diagnostics are null; retained draw failures are not successful results.",
        "artifacts": {f"{name}.csv": hashlib.sha256((output / f"{name}.csv").read_bytes()).hexdigest()
                      for name in TABLES}}
    manifest["artifacts"]["draw_evidence.npz"] = hashlib.sha256((output / "draw_evidence.npz").read_bytes()).hexdigest()
    _plot_criterion(study, output)
    manifest["artifacts"]["criterion_distribution.png"] = hashlib.sha256((output / "criterion_distribution.png").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(_json_value(manifest), indent=2,
                                                     allow_nan=False)+"\n", encoding="utf-8")
    computational_failures = int(study.summary["unresolved_draws"].sum())
    status = "computational outcomes unresolved" if computational_failures else "completed"
    outcome_columns = ["scenario", "replications", "usable_fits", "unresolved_draws",
                       "boundary_fits", "numerically_regular_fits", "observed_objective",
                       "criterion_median", "criterion_q95"]
    tail_columns = ["scenario", "tail_exceedances", "tail_fraction_lower", "tail_fraction_upper",
                    "tail_mc_95_lower", "tail_mc_95_upper"]
    chi2_columns = ["scenario", "naive_chi2_rate_lower", "naive_chi2_rate_upper",
                    "numerically_regular_fits", "regular_j_rate_conditional"]
    parameter_columns = ["scenario", "true_crr", "bias_crr_usable", "rmse_crr_usable",
                         "true_em", "bias_em_usable", "rmse_em_usable"]
    observed_columns = ["calibration", "crr", "em", "objective", "optimizer_success",
                        "numerically_regular", "boundary", "j_pvalue", "unresolved"]
    small_run_note = ("**Small simulation run:** fewer than 199 replications per scenario; "
                      "use this dossier for workflow checks and coarse diagnostics, not precise tail conclusions.\n\n"
                      if study.metadata.get("small_replication_warning", False) else "")
    report = (
        "# SW07 finite-sample covariance-matching diagnostic\n\n"
        f"Computation status: **{status}**. Unresolved draws: **{computational_failures}**.\n\n"
        f"Requested simulations per scenario: **{replications}**. Seed: **{seed}**. "
        f"Bartlett HAC bandwidth used for fitting: **{bandwidth}**.\n\n"
        + small_run_note + "Each simulation repeats the 156-quarter sample, full-sample demeaning, common "
        "lag window, covariance estimation, parameter bounds, and four optimizer starts. "
        "Nine moments estimate monetary-policy smoothing and innovation volatility; six "
        "unused same-sample moments remain descriptive checks. Simulation starts from the "
        "stationary model distribution and excludes economic measurement error.\n\n"
        "## Scenario results\n\n" + _markdown_table(study.summary[outcome_columns]) + "\n\n"
        "## Tail fractions and Monte Carlo uncertainty\n\n"
        + _markdown_table(study.summary[tail_columns]) + "\n\n"
        "Fixed-calibration scenarios examine a specified Gaussian model and its fixed "
        "parameters. Fitted scenarios use values estimated from the observed data; their "
        "tail fractions are descriptive plug-in diagnostics, not validated bootstrap "
        "p-values or composite-null tests. Published calibration values use overlapping "
        "historical data and do not provide independent validation.\n\n"
        "Binomial Monte Carlo intervals describe uncertainty from a finite number of "
        "simulation draws. They are not parameter confidence intervals and do not include "
        "calibration, model, historical-regime, or data-vintage uncertainty. A small "
        "simulation count supports only a coarse tail estimate; zero exceedances do not "
        "establish a zero population tail probability.\n\n"
        "The lower and upper tail fractions count unresolved draws as no exceedances "
        "and all exceedances, respectively. The 95% Clopper–Pearson limits include "
        "both possibilities.\n\n"
        "## Asymptotic-rule diagnostics\n\n"
        + _markdown_table(study.summary[chi2_columns]) + "\n\n"
        "The naive chi-square(7) rule deliberately includes usable boundary fits and "
        "must not be interpreted as valid boundary inference. The regular J rejection "
        "rate uses only the reported numerically regular subset; selection changes "
        "its denominator. Neither rate proves general test size or coverage.\n\n"
        "## Parameter recovery among usable fits\n\n"
        + _markdown_table(study.summary[parameter_columns]) + "\n\n"
        "Bias and RMSE are conditional on usable optimization and are not parameter "
        "confidence intervals.\n\n"
        "## Observed-data fits\n\n" + _markdown_table(study.observed_fits[observed_columns]) + "\n\n"
        "The criterion distribution plot shows the usable simulated criteria and the "
        "observed-data criterion on a common scale within each scenario. Read it together "
        "with the summary failure counts: unavailable draws are retained in draws.csv "
        "and must not be treated as successful non-exceedances.\n\n"
        "## Moment and covariance evidence\n\n"
        "moment_audit.csv compares observed moments, stationary population targets, exact "
        "finite-sample expectations, and simulated empirical moments. covariance_audit.csv compares "
        "exact Gaussian finite-sample covariance, simulation covariance of empirical "
        "moment estimators, and mean Bartlett HAC estimates. Bandwidths 4, 8, and 12 "
        "are audited without reestimating parameters for each audit bandwidth. "
        "These diagnostics expose finite-sample demeaning and weighting "
        "effects; they do not alter the original targets or declare a rejected model valid.\n\n"
        "The declared Pfeifer initialization and the published SW07 mode represent "
        "different fixed calibrations. The independent data/mapping audit found no "
        "quarterly-percent unit mismatch. It identified the declared initialization's "
        "large risk-premium innovation variance as a major source of its poor empirical "
        "fit. Persistence and finite-sample demeaning remain separate diagnostic questions.\n\n"
        "The source snapshot is revised FRED data for 1966Q1–2004Q4, not the original "
        "SW07 vintage. This exercise neither repeats the paper's Bayesian estimation "
        "nor externally identifies a monetary shock.\n\n"
        "Files: summary.csv, draws.csv, observed_fits.csv, moment_audit.csv, "
        "covariance_audit.csv, draw_evidence.npz, criterion_distribution.png, and "
        "manifest.json. The manifest contains seeds, scenario definitions, source "
        "provenance, parameter choices, diagnostics, array shapes, and artifact hashes. "
        "The array_row column in draws.csv maps each replication to its NPZ row. "
        "Moment-audit draw counts can exceed usable fit counts when moments are "
        "available but optimization remains unresolved.\n\n"
        "Sources: [SW07 model and data appendix](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf), "
        "[Pfeifer model source](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod).\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return study


def _progress(scenario: str, completed: int, total: int) -> None:
    if completed in (1, total) or completed % max(1, total//20) == 0:
        print(f"{scenario}: {completed}/{total}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/sw07_finite_sample"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append", dest="scenarios",
                        help="repeat to select scenarios; by default run all four")
    parser.add_argument("--replications", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--bandwidth", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true", help="start fresh, replacing checkpoints for the selected scenarios")
    args = parser.parse_args(argv)
    study = run_application(args.output, replications=args.replications, seed=args.seed,
                            bandwidth=args.bandwidth, scenarios=args.scenarios, progress=_progress, resume=not args.no_resume)
    print(study.summary.to_string(index=False))
    print(f"Evidence: {args.output}")
    return int(study.summary["unresolved_draws"].gt(0).any())


if __name__ == "__main__":
    raise SystemExit(main())
