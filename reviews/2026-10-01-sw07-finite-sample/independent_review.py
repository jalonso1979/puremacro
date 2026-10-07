"""Recompute the finite-sample report from saved artifacts; no model imports or draws.

Run from any directory with the study's Python environment. The exact Gaussian
reference is read from the saved audit, not independently derived here. Separate
quadratic-form and mapping tests audit that reference. This script independently
checks aggregation and reports joint covariance distortion in its nine fitted
moment directions, not only diagonal variance ratios.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigvalsh
from scipy.stats import beta, chi2


ROOT = Path(__file__).resolve().parent
SCENARIOS = ("baseline_fixed", "published_fixed", "baseline_fitted", "published_fitted")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=label)


def cp(events, n):
    return (0.0 if events == 0 else float(beta.ppf(.025, events, n-events+1)),
            1.0 if events == n else float(beta.ppf(.975, events+1, n-events)))


def matrix(table, estimator, field, labels):
    return (table[table.estimator == estimator]
            .pivot(index="left", columns="right", values=field)
            .reindex(index=labels, columns=labels).to_numpy())


def review_scenario(name):
    folder = ROOT / name
    manifest = json.loads((folder / "manifest.json").read_text())
    hashes = {}
    for filename, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        require(actual == expected, f"Artifact hash mismatch: {name}/{filename}")
        hashes[filename] = actual
    hashes["manifest.json"] = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    summary = pd.read_csv(folder / "summary.csv").iloc[0]
    draws = pd.read_csv(folder / "draws.csv", dtype={"simulation_seed": str})
    moments = pd.read_csv(folder / "moment_audit.csv")
    audit = pd.read_csv(folder / "covariance_audit.csv")
    with np.load(folder / "draw_evidence.npz", allow_pickle=False) as saved:
        values = saved["moments"].copy()
        hac8 = saved["covariances"].copy()
    n = len(draws)
    require(n == 399, f"Unexpected draw count: {name}")
    require(draws.replication.tolist() == list(range(n)), "Missing or reordered draws")
    require(draws.array_row.tolist() == list(range(n)), "NPZ row mapping differs")
    require(draws.simulation_seed.nunique() == n, "Duplicated simulation seeds")
    require(values.shape == (n, 15) and hac8.shape == (n, 15, 15), "Array shape mismatch")
    require(np.isfinite(values).all() and np.isfinite(hac8).all(), "Nonfinite moment evidence")
    good = ~draws.unresolved & np.isfinite(draws.objective)
    regular = good & draws.numerically_regular
    failures = int((~good).sum())
    tail = int((good & (draws.objective >= summary.observed_objective)).sum())
    rejected = int((good & (draws.objective > chi2.ppf(.95, 7))).sum())
    regular_rejected = int((regular & (draws.j_pvalue < .05)).sum())
    counts = {"replications": n, "usable_fits": int(good.sum()),
              "unresolved_draws": failures, "boundary_fits": int((good & draws.boundary).sum()),
              "numerically_regular_fits": int(regular.sum()), "tail_exceedances": tail,
              "naive_chi2_rejections": rejected, "regular_j_rejections": regular_rejected}
    for key, value in counts.items():
        require(value == summary[key], f"Summary count differs: {name}/{key}")
    rates = {"tail_fraction_lower": tail/n, "tail_fraction_upper": (tail+failures)/n,
             "tail_mc_95_lower": cp(tail, n)[0], "tail_mc_95_upper": cp(tail+failures, n)[1],
             "naive_chi2_rate_lower": rejected/n, "naive_chi2_rate_upper": (rejected+failures)/n,
             "naive_chi2_mc_95_lower": cp(rejected, n)[0],
             "naive_chi2_mc_95_upper": cp(rejected+failures, n)[1],
             "regular_j_rate_conditional": regular_rejected/int(regular.sum()),
             "regular_j_mc_95_lower_conditional": cp(regular_rejected, int(regular.sum()))[0],
             "regular_j_mc_95_upper_conditional": cp(regular_rejected, int(regular.sum()))[1]}
    for key, value in rates.items():
        close(value, summary[key], f"Summary rate: {name}/{key}")
    criteria = draws.loc[good, "objective"].to_numpy()
    close(np.median(criteria), summary.criterion_median, "Criterion median")
    close(np.quantile(criteria, .95), summary.criterion_q95, "Criterion q95")
    for parameter in ("crr", "em"):
        errors = draws.loc[good, parameter].to_numpy() - summary[f"true_{parameter}"]
        close(errors.mean(), summary[f"bias_{parameter}_usable"], "Parameter bias")
        close(np.sqrt(np.mean(errors**2)), summary[f"rmse_{parameter}_usable"], "Parameter RMSE")

    labels = moments.label.tolist()
    require(len(labels) == 15 and moments.used_in_fit.tolist() == [True]*9 + [False]*6,
            "Unexpected moment order or fitting subset")
    exact = matrix(audit, "hac8", "exact_demeaned_covariance", labels)
    empirical = np.cov(values, rowvar=False, ddof=1)
    close(empirical, matrix(audit, "empirical_mc", "covariance", labels), "Full empirical covariance")
    close(hac8.mean(axis=0), matrix(audit, "hac8", "covariance", labels), "Full mean HAC8 covariance")
    close(values.mean(axis=0), moments.monte_carlo_mean, "Empirical moment means")
    close(np.diag(exact), moments.exact_sampling_se.to_numpy()**2, "Exact reference diagonal")
    close(np.diag(empirical)/np.diag(exact), moments.empirical_variance_ratio, "Variance ratios")
    close(moments.exact_sample_mean-moments.population, moments.demeaning_bias, "Demeaning bias")
    mc_se = np.sqrt(np.diag(exact)/n)
    close(mc_se, moments.mean_mc_se, "Exact Monte Carlo mean SE")
    mean_errors = (values.mean(axis=0)-moments.exact_sample_mean.to_numpy())/mc_se

    joint = {}
    for estimator in ("empirical_mc", "hac4", "hac8", "hac12"):
        estimated = matrix(audit, estimator, "covariance", labels)
        spectrum = eigvalsh(estimated[:9, :9], exact[:9, :9])
        joint[estimator] = {"generalized_eigenvalues": spectrum.tolist(),
                            "minimum": float(spectrum.min()), "median": float(np.median(spectrum)),
                            "maximum": float(spectrum.max())}
        if estimator.startswith("hac"):
            close(np.diag(estimated)/np.diag(exact), moments[f"{estimator}_variance_ratio"],
                  f"Full covariance and marginal {estimator} ratios")
    return {"scenario": name, "is_plugin": bool(summary.is_plugin), "input_sha256": hashes,
            "all_recomputations_passed": True, "counts": counts, "rates": rates,
            "observed_criterion": float(summary.observed_objective),
            "criterion_median_usable": float(np.median(criteria)),
            "criterion_q95_usable": float(np.quantile(criteria, .95)),
            "fit_moment_labels": labels[:9], "joint_covariance_audit": joint,
            "moment_mean_audit": [{"label": label, "observed": float(moments.observed.iloc[j]),
                "population": float(moments.population.iloc[j]),
                "exact_sample_mean": float(moments.exact_sample_mean.iloc[j]),
                "monte_carlo_mean": float(values[:, j].mean()), "mc_mean_standard_error": float(mc_se[j]),
                "mean_error_in_mc_standard_errors": float(mean_errors[j]),
                "demeaning_bias_in_exact_sampling_sd": float(moments.demeaning_bias.iloc[j]/moments.exact_sampling_se.iloc[j]),
                "empirical_variance_ratio": float(np.diag(empirical)[j]/np.diag(exact)[j]),
                "hac8_variance_ratio": float(moments.hac8_variance_ratio.iloc[j])}
                for j, label in enumerate(labels)],
            "maximum_absolute_mean_error_in_mc_standard_errors": float(np.abs(mean_errors).max())}


def main():
    results = [review_scenario(name) for name in SCENARIOS]
    largest = max(result["maximum_absolute_mean_error_in_mc_standard_errors"] for result in results)
    document = {"review": "Independent aggregation and joint-covariance interpretation audit",
        "scope": "Saved artifacts only; no production imports, estimation, simulation or data changes",
        "exact_reference_scope": "Reads saved exact covariance and expectations; does not independently derive the DSGE oracle",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((ROOT/"protocol.md").read_bytes()).hexdigest(),
        "generalized_eigenvalue_definition": "eigvalsh(mean estimated covariance, exact demeaned covariance) on all nine fitted moments; values below/above one indicate underestimated/overestimated variance in some linear combination; not inverse-weight expectation",
        "mean_mc_se_definition": "sqrt(exact sampling variance / 399); this is a descriptive mean check, not a simultaneous normal test",
        "max_absolute_mean_error_across_60_moments_in_mc_standard_errors": largest,
        "all_recomputations_passed": True, "scenarios": results}
    (ROOT/"independent_review.json").write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"all_recomputations_passed": True, "replications": 1596,
                      "largest_absolute_mc_mean_error_in_standard_errors": largest}, indent=2))


if __name__ == "__main__":
    main()
