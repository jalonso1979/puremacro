"""Export paired SW07 moment/weight controls and independent validation.

Run calibration first, then validation with --calibration pointing to its
completed evidence directory. Checkpoints preserve interrupted phases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro.structural.sw07_estimator_experiment import (
    SW07EstimatorExperiment, run_sw07_estimator_experiment, compare_sw07_estimator_phases)
from .sw07_finite_sample import _json_value, _markdown_table, _progress

_TABLES = ("summary", "draws", "observed_fits", "paired_differences")


def load_experiment(output: Path) -> SW07EstimatorExperiment:
    """Authenticate a complete exported phase before using it as a reference."""
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    required = {f"{name}.csv" for name in _TABLES} | {"draw_evidence.npz"}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Incomplete experiment artifact manifest")
    for filename, digest in manifest["artifacts"].items():
        if Path(filename).name != filename:
            raise ValueError("Experiment artifact names must be local filenames")
        if hashlib.sha256((output/filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment artifact hash mismatch: {filename}")
    frames = {name: pd.read_csv(output/f"{name}.csv", dtype={"simulation_seed": str}) for name in _TABLES}
    for frame in frames.values():
        for column in ("error", "sample_sha256"):
            if column in frame:
                frame[column] = frame[column].fillna("")
    with np.load(output/"draw_evidence.npz", allow_pickle=False) as arrays:
        moments, covariances = arrays["moments"].copy(), arrays["covariances"].copy()
    count = manifest["replications"]
    if moments.shape != (count, 15) or covariances.shape != (count, 15, 15):
        raise ValueError("Experiment array shapes differ from the declared design")
    return SW07EstimatorExperiment(**frames, draw_moments=moments,
                                  draw_covariances=covariances, metadata=manifest)


def _plot(study, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for variant in study.summary.variant:
        rows = study.draws.loc[(study.draws.variant == variant) & ~study.draws.unresolved]
        values = np.sort(rows.objective.dropna())
        if len(values):
            axes[0].step(values, np.arange(1, len(values)+1)/len(values), where="post", label=variant)
    axes[0].set(xscale="log", xlabel="Minimized criterion (different weights have different scales)",
                ylabel="Cumulative usable fraction", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    table = study.summary
    x = np.arange(len(table))
    low, high = table.naive_rate_lower.to_numpy(), table.naive_rate_upper.to_numpy()
    axes[1].bar(x, low, color="#2864a0")
    axes[1].bar(x, high-low, bottom=low, color="#d99658", label="Unresolved contribution")
    axes[1].axhline(.05, color="#b0472f", label="Nominal 5% reference")
    axes[1].set(xticks=x, xticklabels=table.variant, ylabel="Naive chi-square rejection fraction", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"SW07 paired controls: {study.metadata['phase']}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_application(output: Path, *, replications: int = 399, phase: str = "calibration",
                    seed: int = 20261002, resume: bool = True, progress=None,
                    calibration: Path | None = None):
    if calibration is not None and phase != "validation":
        raise ValueError("A calibration reference is only used by the validation phase")
    reference = None if calibration is None else load_experiment(calibration)
    if reference is not None and (reference.metadata["phase"] != "calibration"
                                  or reference.metadata["seed"] != seed):
        raise ValueError("Reference must be a calibration phase with the requested master seed")
    output = Path(output)
    study = run_sw07_estimator_experiment(replications=replications, phase=phase, seed=seed,
        checkpoint_dir=output/"_checkpoints", resume=resume, progress=progress)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name in _TABLES:
        study_table = getattr(study, name)
        study_table.to_csv(output/f"{name}.csv", index=False)
        artifacts.append(f"{name}.csv")
    np.savez_compressed(output/"draw_evidence.npz", moments=study.draw_moments,
                        covariances=study.draw_covariances)
    artifacts.append("draw_evidence.npz")
    _plot(study, output/"estimator_comparison.png")
    artifacts.append("estimator_comparison.png")
    comparison = None
    if reference is not None:
        comparison = compare_sw07_estimator_phases(reference, study)
        comparison.to_csv(output/"calibrated_validation.csv", index=False)
        artifacts.append("calibrated_validation.csv")
    failures = int(study.draws.unresolved.sum())
    report = (
        f"# SW07 estimator controls: {phase}\n\n"
        f"Completed {replications} paired samples and {4*replications} variant fits; "
        f"{failures} variant fits remain unresolved.\n\n"
        "The full published calibration generates stationary Gaussian data. All four "
        "variants use identical sample moments; finite expectations change the model "
        "map and oracle weights use the exact covariance at known generating parameters. "
        "The oracle is a diagnostic control, not a feasible covariance estimator. "
        "No original estimator is replaced and no parameter confidence intervals are reported.\n\n"
        "## Naive chi-square reference and parameter recovery\n\n"
        + _markdown_table(study.summary) + "\n\n"
        "Rate bounds count unresolved fits both ways. Monte Carlo intervals envelope "
        "Clopper–Pearson limits. Parameter bias/RMSE and criterion quantiles condition "
        "on usable fits. A numerically regular fit is not proof of valid inference.\n\n"
        "## Paired comparisons\n\n" + _markdown_table(study.paired_differences) + "\n\n"
        "Squared-error differences are right minus left on common usable samples only. "
        "The paired Monte Carlo SE measures simulation variation, not parameter uncertainty. "
        "Criteria under different weight matrices have different scales.\n\n"
        "## Observed-data descriptive fits\n\n"
        + _markdown_table(study.observed_fits[["variant", "crr", "em", "objective", "boundary", "unresolved"]])
        + "\n\nThese use revised historical data and a fixed published calibration. "
        "They are not independently validated economic estimates.\n\n"
    )
    if comparison is not None:
        report += ("## Independent validation of calibration cutoffs\n\n" + _markdown_table(comparison)
            + "\n\nCutoffs use calibration order rank ceil(0.95*(B+1)), with strict rejection "
            "above the cutoff. Unresolved calibration criteria lie between zero and infinity, "
            "producing lower/upper cutoffs. Validation rejection bounds additionally retain "
            "unresolved validation fits. Monte Carlo intervals condition on the realized "
            "calibration pool and do not include all cutoff-estimation uncertainty. "
            "Infinite cutoffs mean the reference sample is too small or too incomplete; "
            "JSON null and explicit unbounded flags distinguish this from a finite cutoff. "
            "This is a fixed-DGP experiment, not a composite-null test.\n\n")
    report += ("All starts, seeds, sample hashes, moments and full HAC arrays are retained. "
        "Atomic checkpoints authenticate numerical sources, observed data, settings and "
        "dependency versions. A small replication count is a workflow check only.\n")
    (output/"report.md").write_text(report, encoding="utf-8")
    artifacts.append("report.md")
    manifest = {**study.metadata, "package_version": __version__,
        "comparison": None if comparison is None else comparison.to_dict(orient="records"),
        "calibration_manifest_sha256": None if calibration is None else
            hashlib.sha256((Path(calibration)/"manifest.json").read_bytes()).hexdigest(),
        "artifacts": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in artifacts}}
    (output/"manifest.json").write_text(json.dumps(_json_value(manifest), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--calibration", type=Path, help="completed calibration evidence to authenticate and apply")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    count = args.replications if args.replications is not None else (399 if args.phase == "calibration" else 999)
    study = run_application(args.output, replications=count, phase=args.phase, seed=args.seed,
        resume=not args.no_resume, progress=_progress, calibration=args.calibration)
    print(study.summary.to_string(index=False))
    print(f"Evidence: {args.output}")
    return int(study.draws.unresolved.any())


if __name__ == "__main__":
    raise SystemExit(main())
