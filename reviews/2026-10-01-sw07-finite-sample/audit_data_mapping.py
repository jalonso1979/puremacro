"""Read-only SW07 data/model mapping audit; writes only this review's evidence.

Run from the repository: .venv/bin/python reviews/2026-10-01-sw07-finite-sample/audit_data_mapping.py
Uses the production state-space solution, but independently constructs empirical
moments and exact finite-sample expectations using temporal covariance matrices.
It does not estimate parameters, simulate samples, or invoke the new diagnostic.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.structural.empirical_sw07 import _PUBLISHED_MODE

OUTPUT = Path(__file__).resolve().parent
SERIES = ("gdp_growth", "infl", "ffr")
SPECS = tuple((a, b, 0) for j, a in enumerate(SERIES) for b in SERIES[j:]) + tuple(
    (a, a, h) for h in (1, 2, 4) for a in SERIES)


def model_moments(parameters, n):
    model = make_state_space(parameters)
    z = model.Z[[OBSERVED_VARS.index(x) for x in SERIES]]
    f = model.T
    total_p = solve_discrete_lyapunov(f, model.R @ model.Q @ model.R.T)
    full_p = np.zeros_like(total_p)
    components = {}
    diagnostics = []
    for j, shock in enumerate(SW07_SHOCK_STDS):
        b = model.R[:, j] * np.sqrt(model.Q[j, j])
        p = solve_discrete_lyapunov(f, np.outer(b, b))
        full_p += p
        gamma = []
        lagged_z = z.copy()
        for h in range(n):
            gamma.append(lagged_z @ p @ z.T)
            lagged_z = lagged_z @ f
        components[shock] = np.asarray(gamma)
        diagnostics.append(float(np.max(np.abs(p-f@p@f.T-np.outer(b, b)))))
    assert np.allclose(full_p, total_p, rtol=1e-9, atol=1e-8)
    components["all"] = sum(components.values())
    return components, {
        "transition_spectral_radius": float(max(abs(np.linalg.eigvals(f)))),
        "lyapunov_component_max_residual": max(diagnostics),
        "lyapunov_component_sum_max_discrepancy": float(np.max(abs(full_p-total_p))),
        "measurement_intercepts": dict(zip(OBSERVED_VARS, model.d.tolist())),
        "technology_persistence": parameters["crhoa"],
        "technology_half_life_quarters": float(np.log(.5)/np.log(parameters["crhoa"])),
        "technology_process_stationary_variance": parameters["ea"]**2/(1-parameters["crhoa"]**2),
    }


def exact_centering(gamma, a, b, lag, n, max_lag):
    """E[(a_t-mean(a))(b_t-h-mean(b))] using all N observations for means."""
    indices = np.arange(n)
    differences = indices[:, None] - indices[None, :]
    covariance = np.where(differences >= 0, gamma[abs(differences), a, b],
                          gamma[abs(differences), b, a])
    centered = (covariance-covariance.mean(axis=0, keepdims=True)
                -covariance.mean(axis=1, keepdims=True)+covariance.mean())
    current = indices[max_lag:]
    expectation = centered[current, current-lag].mean()
    return float(expectation), float(covariance.mean())


def main():
    # Analytical scalar control: E[mean((x-mean(x))**2)] = Var(x)-Var(mean(x)).
    control_n, control_rho, control_variance = 37, .9, 2.
    gamma = (control_variance*control_rho**np.arange(control_n))[:, None, None]
    observed_control, mean_variance = exact_centering(gamma, 0, 0, 0, control_n, 0)
    exact_mean_variance = control_variance*(control_n+2*sum(
        (control_n-h)*control_rho**h for h in range(1, control_n)))/control_n**2
    assert abs(mean_variance-exact_mean_variance) < 1e-14
    assert abs(observed_control-(control_variance-exact_mean_variance)) < 1e-14
    payload = (ROOT / "puremacro/dsge/_sw07_data.csv").read_bytes()
    data = pd.read_csv(ROOT / "puremacro/dsge/_sw07_data.csv", comment="#", index_col="date")
    n = len(data)
    assert n == 156 and np.isfinite(data.to_numpy()).all()
    observed = data.loc[:, SERIES].to_numpy()
    centered = observed-observed.mean(axis=0)
    baseline = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    published = {**baseline, **_PUBLISHED_MODE}
    # These descriptive fitted values come from the previous study's frozen
    # review artifacts; this script does not reoptimize or select a calibration.
    calibrations = {
        "declared_pfeifer": baseline,
        "declared_pfeifer_fitted": {**baseline, "crr": .98, "em": .01},
        "pfeifer_technology_rho_only_095": {**baseline, "crhoa": .95},
        "published_sw07_mode": published,
        "published_sw07_mode_fitted": {**published, "crr": .8382576403491104,
                                       "em": .2540855996259026},
    }
    rows, diagnostics = [], {}
    for name, parameters in calibrations.items():
        components, diagnostics[name] = model_moments(parameters, n)
        diagnostics[name]["parameters"] = parameters
        for shock, gamma in components.items():
            for a_name, b_name, h in SPECS:
                a, b = SERIES.index(a_name), SERIES.index(b_name)
                finite, mean_covariance = exact_centering(gamma, a, b, h, n, 4)
                sample = float(np.mean(centered[4:, a]*centered[4-h:n-h, b]))
                population = float(gamma[h, a, b])
                rows.append({"calibration": name, "shock": shock,
                    "moment": f"cov({a_name},{b_name};lag={h})", "left": a_name,
                    "right": b_name, "lag": h, "observed": sample,
                    "stationary_population": population,
                    "expected_sample_centered": finite,
                    "demeaning_change": finite-population,
                    "covariance_of_full_sample_means": mean_covariance})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "mapping_moment_decomposition.csv", index=False)
    source_paths = ["tools/build_sw07_data.py", "puremacro/dsge/sw07_observation.py",
                    "puremacro/dsge/smets_wouters.py", "puremacro/dsge/_sw07_data.csv",
                    "puremacro/dsge/_references/sw07_pfeifer.mod"]
    evidence = {
        "source_hashes": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                          for path in source_paths},
        "source_normalized_sha256": hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
        "source_kind": "revised FRED historical observations, frozen2026-09-30; not original paper data",
        "sample": {"first": data.index[0], "last": data.index[-1], "n": n,
                   "common_window_max_lag": 4, "effective_n": n-4},
        "sample_means": data.mean().to_dict(),
        "sample_standard_deviations_ddof0": data.std(ddof=0).to_dict(),
        "model_diagnostics": diagnostics,
        "method": "Compute each structural-innovation Lyapunov covariance, form exact N-by-N stationary temporal covariance C_ab, then D C_ab D with D=I-11'/N; average entries(t,t-h),t=4..155. No simulation, no asymptotic mean adjustment.",
        "measurement_error": "H numerical ridge excluded, matching the conditional covariance study",
        "analytical_control": "Scalar AR(1) full-sample demeaned variance matched Var(x)-Var(sample mean) to1e-14",
        "independence_scope": "Independent empirical products and finite-sample temporal-covariance algebra; shares package DSGE solution and observation matrices, so not an independent model-solver validation",
        "raw_vintage_limitation": "Nine original raw FRED downloads were not found; transformations were audited against builder source and published definitions, not reconstructed from raw observations",
        "sources": [
            "https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf",
            "https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod",
            "https://fred.stlouisfed.org/series/PRS85006023",
            "https://fred.stlouisfed.org/series/GDPDEF",
            "https://fred.stlouisfed.org/series/CNP16OV",
            "https://fred.stlouisfed.org/series/FEDFUNDS"],
    }
    (OUTPUT / "mapping_audit.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    diagonal = table[(table.shock == "all") & (table.left == table.right) & (table.lag == 0)]
    print(diagonal[["calibration", "left", "observed", "stationary_population", "expected_sample_centered"]].to_string(index=False))


if __name__ == "__main__":
    main()
