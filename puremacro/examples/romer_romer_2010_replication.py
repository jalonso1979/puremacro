"""Original-data Romer–Romer (2010) baseline and independent evidence.

Run ``python -m puremacro.examples.romer_romer_2010_replication --output DIR``.
Source conversion and independent reference regeneration are development steps;
this application runs offline from authenticated package resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro.replication.romer_romer_2010 import (
    estimate_rr2010_baseline, load_rr2010_data, load_rr2010_reference,
)
from puremacro.validation import research_benchmarks, run_research_benchmarks


def run_application(output: str | Path):
    """Write estimates, complete uncertainty, source metadata and comparisons.

    Returns the fitted result and benchmark report. A failed comparison is
    preserved in the dossier; the command-line entry point also exits nonzero.
    """
    result = estimate_rr2010_baseline()
    reference = load_rr2010_reference()
    report = run_research_benchmarks([case for case in research_benchmarks()
                                     if case.id.startswith("rr2010_")])
    comparison_status = {case.id: case.passed for case in report.results}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    load_rr2010_data().to_csv(output / "original_data.csv")
    names = ["intercept", *[f"tax_lag_{lag}" for lag in range(13)]]
    pd.DataFrame({"coefficient": names, "estimate": result.coefficients,
                  "standard_error": np.sqrt(np.diag(result.coefficient_covariance))
                  }).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(result.coefficient_covariance, index=names, columns=names
                 ).to_csv(output / "coefficient_covariance.csv")
    pd.DataFrame({"horizon": np.arange(13), "response": result.irf,
                  "standard_error": result.standard_errors,
                  "t_statistic": result.t_statistics,
                  "independent_reference": reference["irf"]
                  }).to_csv(output / "responses.csv", index=False)
    pd.DataFrame(result.irf_covariance, index=range(13), columns=range(13)
                 ).to_csv(output / "response_covariance.csv")
    report.write(output)

    import matplotlib.pyplot as plt
    horizons = np.arange(13)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axhline(0, color="black", linewidth=.7)
    ax.fill_between(horizons, result.irf-result.standard_errors,
                    result.irf+result.standard_errors, alpha=.18,
                    label="±1 conventional OLS standard error")
    ax.plot(horizons, result.irf, "o-", label="Original-data baseline")
    ax.scatter([10], [-3.08], marker="x", color="black", s=70,
               label="Published rounded trough")
    ax.set(xlabel="Quarters after a tax increase of 1% of GDP",
           ylabel="Real GDP response (log percent)",
           title="Romer–Romer (2010): baseline distributed-lag regression")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output / "tax_response.png", dpi=160)
    plt.close(fig)

    manifest = {"package_version": __version__, "evidence_kind": "published_empirical",
                "passed": report.passed, "comparison_status": comparison_status,
                "nobs": result.nobs, "df_resid": result.df_resid,
                "estimation": result.metadata, "independent_reference": reference["metadata"],
                "published_targets": {"horizon": 10, "response": -3.08, "t_statistic": -3.53,
                                      "absolute_rounding_tolerance": .005},
                "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(output.iterdir())
                                    if path.suffix in {".csv", ".png"}},
                "environment": dict(report.environment), "generated_at": report.generated_at}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n",
                                           encoding="utf-8")
    content = (
        "# Romer–Romer (2010): original baseline replication\n\n"
        f"Independent-software and published-target comparisons: **{'PASS' if report.passed else 'FAIL'}**.\n\n"
        + "\n".join(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
                    for name, passed in comparison_status.items()) + "\n\n"
        f"For a tax increase of one percent of GDP, the horizon-ten GDP response is "
        f"**{result.irf[10]:.8f} log percent**, conventional SE {result.standard_errors[10]:.8f}, "
        f"t statistic **{result.t_statistics[10]:.8f}**. The trough occurs at horizon "
        f"{int(np.argmin(result.irf))}. The printed paper reports -3.08 and -3.53 at horizon ten.\n\n"
        f"The t-statistic rounding comparison has absolute error "
        f"{abs(result.t_statistics[10]+3.53):.10f} against tolerance 0.005. "
        "The bundled original-data result misses that interval by approximately 0.000003925. "
        "This discrepancy remains unresolved; it is not removed by rounding intermediate "
        "values, widening tolerance or omitting the selected statistic. The full software "
        "comparison and each published metric remain separately inspectable.\n\n"
        f"The original-vintage regression has {result.nobs} quarterly observations, "
        f"1950Q1–2007Q4, and {result.df_resid} residual degrees of freedom. It regresses "
        "100 times the change in log real GDP on an intercept and lags zero through twelve "
        "of 100*(DEFIC+LONGR)/NOMGDP. Cumulative tax coefficients give the level response; "
        "the full conventional residual-df coefficient covariance is cumulated with them. "
        "Positive tax shocks mean tax increases.\n\n"
        "The software comparison checks all coefficients, both complete covariance matrices, "
        "responses, standard errors, t statistics and sample sizes against a frozen independent "
        "run. The separate published comparison allows only two-decimal rounding. "
        "The offline application does not execute the authors' RATS program. Source hashes, "
        "reference software and regeneration details are in manifest.json; every numerical "
        "comparison and tolerance is in benchmark_report.json.\n\n"
        "Scope: the original Figure 4 baseline specification, not all paper robustness "
        "exercises or independent validation of narrative tax-shock exogeneity. The displayed "
        "band is plus/minus one conventional OLS standard error, not simultaneous coverage. "
        "The earlier modified local-projection gallery case uses a different sample and "
        "estimator and is not the reference for this replication.\n\n"
        "Sources: [paper and citation](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763), "
        "[author manuscript, printed p. 781](https://emlab.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf), "
        "[original data and RATS archive](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).\n"
    )
    (output / "report.md").write_text(content, encoding="utf-8")
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/romer_romer_2010_replication"))
    args = parser.parse_args()
    result, report = run_application(args.output)
    print(f"RR2010 baseline: {'PASS' if report.passed else 'FAIL'}; "
          f"h10={result.irf[10]:.8f}, t={result.t_statistics[10]:.8f}; output={args.output}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
