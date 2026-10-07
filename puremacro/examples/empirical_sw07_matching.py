"""Observed revised-FRED data -> conditional SW07 minimum-distance study.

Run ``python -m puremacro.examples.empirical_sw07_matching --output DIR``.
This is an empirical study, not a replication of published SW07 estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro.structural.empirical_sw07 import (
    fit_empirical_sw07, load_empirical_sw07_data,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_application(output: Path, *, bandwidth: int = 8, profile_points: int = 11,
                    allow_conditional_inference: bool = False):
    """Write observed data, covariance, full diagnostics and an honest report."""
    study = fit_empirical_sw07(bandwidth=bandwidth, profile_points=profile_points,
                              allow_conditional_inference=allow_conditional_inference)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, _ = load_empirical_sw07_data()
    data_path = output / "observed_data.csv"
    data.to_csv(data_path)
    result = study.fit
    result.summary().to_csv(output / "parameters.csv", index=False)
    study.targets.to_frame().to_csv(output / "empirical_moments.csv", index=False)
    pd.DataFrame(study.targets.covariance, index=study.targets.labels,
                 columns=study.targets.labels).to_csv(output / "joint_covariance.csv")
    for reserved, filename in ((False, "fitted_moments.csv"), (True, "held_out_moments.csv")):
        comparison = result.moment_fit(held_out=reserved)
        comparison["residual_over_target_se"] = comparison["residual"] / comparison["target_se"]
        comparison.to_csv(output / filename, index=False)
    study.multistart.to_csv(output / "multistart.csv", index=False)
    study.profiles.to_csv(output / "profiles.csv", index=False)
    study.bandwidth_sensitivity.to_csv(output / "bandwidth_sensitivity.csv", index=False)
    study.sample_split.to_csv(output / "sample_split_moments.csv", index=False)
    study.calibration_sensitivity.to_csv(output / "calibration_sensitivity.csv", index=False)
    study.calibration_sensitivity_moments.to_csv(output / "calibration_sensitivity_moments.csv", index=False)
    manifest = {**study.metadata,
                "package_version": __version__, "evidence_kind": "observed_data_conditional_structural_estimation",
                "observed_data_artifact_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "target_metadata": dict(study.targets.metadata),
                "estimates": dict(zip(result.parameter_names, result.theta)),
                "objective": result.objective, "optimizer_success": result.success,
                "reported_inference_available": result.inference_valid,
                "conditional_null_j_statistic": result.j_statistic,
                "conditional_null_j_df": result.j_df,
                "conditional_null_j_pvalue": result.j_pvalue,
                "diagnostics": dict(result.diagnostics),
                "fitted_moment_labels": list(result.targets.labels),
                "held_out_moment_labels": list(result.held_out_targets.labels)}
    (output / "manifest.json").write_text(json.dumps(_json_value(manifest), indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    p_text = f"{result.j_pvalue:.6g}" if np.isfinite(result.j_pvalue) else "unavailable (nonregular fit)"
    report = (
        "# Observed-data conditional SW07 covariance matching\n\n"
        f"Study status: **{study.metadata['status']}**. Objective: {result.objective:.6g}. "
        f"Conditional correct-specification null J p-value: {p_text}.\n\n"
        "This study uses 156 observed US quarters, 1966Q1–2004Q4, from the revised FRED "
        "snapshot built 2026-09-30. It is not a reproduction of the original SW07 Bayesian "
        "estimates or data vintage. GDP growth, inflation and the funds rate use quarterly "
        "percent units. Nine covariance moments fit two parameters; six unused same-sample "
        "moments provide descriptive checks. The full 15×15 empirical-estimator covariance "
        "retains cross-moment dependence.\n\n"
        + result.to_markdown() + "\n\n"
        "Any displayed SE is a conditional correct-specification local asymptotic SE. "
        "Ordinary confidence intervals are withheld by default, and also for boundary, "
        "rank, optimization, flat-profile or rejected-specification cases. A missing J "
        "p-value is not evidence of fit; the usual reference law is unavailable. Passing "
        "J does not validate the calibration or establish correct specification.\n\n"
        "All other structural parameters and independent latent-shock variances are held "
        "at the explicit Pfeifer model calibration in manifest.json. No externally identified "
        "monetary surprise enters this exercise. Population moments use an exact stationary "
        "Lyapunov solution and exclude the Kalman numerical measurement-error ridge.\n\n"
        "profiles.csv contains nuisance-reoptimized objective slices, not confidence sets. "
        "bandwidth_sensitivity.csv uses Bartlett bandwidths 4, 8 and 12. "
        "sample_split_moments.csv compares the Great Inflation and Great Moderation periods "
        "descriptively, without a formal break test. Fixed-calibration uncertainty, vintage "
        "uncertainty, and possible regime instability remain outside ordinary inference.\n\n"
        "An exploratory sensitivity added after the baseline boundary outcome fixes the "
        "remaining parameters to the full published SW07 posterior-mode columns in "
        "Tables 1a/b. It reestimates the same two parameters without replacing the declared "
        "baseline. calibration_sensitivity.csv and calibration_sensitivity_moments.csv "
        "show both cases. The published calibration used an overlapping original-vintage "
        "sample and is not independent validation. Residuals divided by target SEs are "
        "descriptive standardized discrepancies, not studentized fitted-residual tests.\n\n"
        "Sources: [SW07 model and data definitions](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf), "
        "[Pfeifer model](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod), "
        "[Newey–West HAC](https://www.nber.org/papers/t0055), "
        "[Hansen's moment-estimation asymptotics](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/). "
        "Exact FRED series links, transformations, input hashes, calibration values and "
        "diagnostic reasons are recorded in manifest.json.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, parameter in zip(axes, result.parameter_names):
        profile = study.profiles.loc[study.profiles["parameter"] == parameter]
        ax.plot(profile["value"], profile["delta_objective"], "o-", markersize=3)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(xlabel=parameter, ylabel="Objective minus selected optimum",
               title=f"{parameter}: descriptive profile")
    fig.savefig(output / "profiles.png", dpi=160)
    plt.close(fig)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/empirical_sw07_matching"))
    parser.add_argument("--bandwidth", type=int, default=8)
    parser.add_argument("--profile-points", type=int, default=11)
    parser.add_argument("--allow-conditional-inference", action="store_true",
                        help="opt in to correct-specification local SEs only if all study gates pass")
    args = parser.parse_args()
    study = run_application(args.output, bandwidth=args.bandwidth, profile_points=args.profile_points,
                            allow_conditional_inference=args.allow_conditional_inference)
    print(f"SW07 observed-data study: {study.metadata['status']}; output={args.output}")
    print(study.fit.summary().to_string(index=False))


if __name__ == "__main__":
    main()
