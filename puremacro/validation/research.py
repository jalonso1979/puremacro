"""Offline, independent research benchmarks with inspectable evidence dossiers.

These narrowly scoped numerical benchmarks supplement the validation gallery.
Analytical or synthetic exercises are never labelled empirical replications.
The Dynare case uses authenticated, previously exported package resources;
MATLAB, Dynare and a repository checkout are not needed at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import io
from itertools import product
import json
from pathlib import Path
import platform
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["ResearchBenchmark", "BenchmarkResult", "ResearchBenchmarkReport",
           "research_benchmarks", "run_research_benchmarks"]


@dataclass(frozen=True)
class ResearchBenchmark:
    """An independent comparison and the information needed to interpret it.

    ``compute`` and ``reference`` return exactly the same nonempty metric keys.
    Every numeric metric has a unit. The comparison is elementwise with no
    broadcasting: ``abs(observed-reference) <= atol + rtol*abs(reference)``.
    ``evidence_kind`` distinguishes analytical formulas, independent numerical
    oracles, external software exports, official data tables, and published
    empirical replications.
    Published claims use original data and literal published targets; none of
    the synthetic benchmarks makes that claim.
    """

    id: str
    title: str
    evidence_kind: str
    compute: Callable[[], Mapping[str, Any]]
    reference: Callable[[], Mapping[str, Any]]
    sources: tuple[str, ...]
    data: Mapping[str, Any]
    sampling: str
    transformations: str
    units: Mapping[str, str]
    limitations: str
    rtol: float = 1e-7
    atol: float = 1e-9


@dataclass(frozen=True)
class BenchmarkResult:
    """A completed comparison; missing or malformed evidence always fails."""

    id: str
    title: str
    evidence_kind: str
    passed: bool
    metrics: dict[str, Any]
    negative_control_rejected: bool
    provenance: dict[str, Any]
    error: str = ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite values cannot be exported as evidence")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evidence metadata type: {type(value).__name__}")


@dataclass(frozen=True)
class ResearchBenchmarkReport:
    """Evidence, version information and JSON/Markdown exports for one run."""

    results: tuple[BenchmarkResult, ...]
    environment: Mapping[str, str]
    generated_at: str
    schema_version: str = "1.0"

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(case.passed for case in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible evidence, including every compared value."""
        from dataclasses import asdict
        return _jsonable({"schema_version": self.schema_version,
                          "generated_at": self.generated_at,
                          "environment": self.environment, "passed": self.passed,
                          "cases": [asdict(case) for case in self.results]})

    def to_markdown(self) -> str:
        """Render provenance and a compact observed/reference comparison table."""
        lines = ["# Independent research benchmarks", "",
                 f"Overall: **{'PASS' if self.passed else 'FAIL'}**. Generated {self.generated_at}.", "",
                 "These results establish only the claims and domains stated in each case.", "",
                 "Environment: " + ", ".join(f"{k}={v}" for k, v in self.environment.items()) + ".", ""]
        for case in self.results:
            p = case.provenance
            lines.extend([f"## {case.id}: {case.title}", "",
                          f"**{'PASS' if case.passed else 'FAIL'}** — {case.evidence_kind}.", "",
                          "Sources: " + "; ".join(p.get("sources", [])), "",
                          "Data: `" + json.dumps(p.get("data", {}), sort_keys=True) + "`", "",
                          "Sampling: " + p.get("sampling", ""), "",
                          "Transformations: " + p.get("transformations", ""), "",
                          "Limitations: " + p.get("limitations", ""), "",
                          f"Tolerance: rtol={p.get('rtol')}, atol={p.get('atol')}. "
                          f"Perturbed-output negative control rejected: {case.negative_control_rejected}.", ""])
            if case.error:
                lines.extend(["Error: " + case.error, ""])
            if case.metrics:
                lines.extend(["| Metric | Unit | Observed | Reference | Max absolute error | Pass |",
                              "|---|---|---|---|---:|---|"])
                for name, item in case.metrics.items():
                    def brief(v):
                        arr = np.asarray(v)
                        return np.array2string(arr, precision=8, threshold=8).replace("\n", " ").replace("|", "\\|")
                    lines.append(f"| {name} | {item['unit']} | `{brief(item['observed'])}` | "
                                 f"`{brief(item['reference'])}` | {item['max_abs_error']:.6g} | {item['passed']} |")
                lines.extend(["", "Arrays are abbreviated here; benchmark_report.json contains every compared value.", ""])
        return "\n".join(lines)

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write ``benchmark_report.json`` and ``benchmark_report.md`` locally."""
        content = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"json": directory / "benchmark_report.json", "markdown": directory / "benchmark_report.md"}
        paths["json"].write_text(content, encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(), encoding="utf-8")
        return paths


def _compare(observed, reference, units, rtol, atol):
    if not isinstance(observed, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("Observed and reference evidence must be mappings")
    if not observed or not reference or set(observed) != set(reference):
        raise ValueError("Evidence needs identical, nonempty observed/reference metric keys")
    if set(units) != set(reference) or any(not isinstance(u, str) or not u.strip() for u in units.values()):
        raise ValueError("Every metric needs an explicit unit")
    metrics = {}
    for name in reference:
        raw_a, raw_b = np.asarray(observed[name]), np.asarray(reference[name])
        if np.iscomplexobj(raw_a) or np.iscomplexobj(raw_b):
            raise ValueError(f"{name}: complex evidence is not supported")
        a, b = np.asarray(raw_a, dtype=float), np.asarray(raw_b, dtype=float)
        if a.size == 0 or b.size == 0:
            raise ValueError(f"{name}: empty evidence")
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape mismatch {a.shape} versus {b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name}: nonfinite evidence")
        delta = np.abs(a-b)
        allowed = atol + rtol*np.abs(b)
        if not np.isfinite(delta).all() or not np.isfinite(allowed).all():
            raise ValueError(f"{name}: comparison overflow")
        metrics[name] = {"observed": a.tolist(), "reference": b.tolist(),
                         "shape": list(a.shape), "unit": units[name],
                         "max_abs_error": float(delta.max()),
                         "max_tolerance_excess": float(np.max(delta-allowed)),
                         "passed": bool(np.all(delta <= allowed))}
    return metrics


def _run(case):
    provenance = {"sources": case.sources, "data": case.data, "sampling": case.sampling,
                  "transformations": case.transformations, "limitations": case.limitations,
                  "rtol": case.rtol, "atol": case.atol}
    metrics, control, error = {}, False, ""
    try:
        _jsonable(provenance)
        if case.evidence_kind not in {"analytical", "numerical_oracle", "external_software", "official_data", "published_empirical"}:
            raise ValueError("Unknown evidence_kind")
        if (not case.id or not case.title or not case.sources or not case.data
                or not all(isinstance(s, str) and s.strip() for s in case.sources)
                or not all(isinstance(s, str) and s.strip()
                           for s in (case.sampling, case.transformations, case.limitations))):
            raise ValueError("Complete source, data, sampling, transformation and limitation metadata are required")
        if (not np.isfinite(case.rtol) or not np.isfinite(case.atol)
                or case.rtol < 0 or case.atol < 0 or case.rtol >= 1):
            raise ValueError("Tolerances require finite 0 <= rtol < 1 and atol >= 0")
        observed, reference = case.compute(), case.reference()
        metrics = _compare(observed, reference, case.units, case.rtol, case.atol)
        # Test the comparison mechanism, not an alternative puremacro calculation.
        # Starting from the oracle isolates this control from a failed computation.
        bad = {key: np.array(value, dtype=float, copy=True) for key, value in reference.items()}
        first = next(iter(bad))
        b = float(bad[first].flat[0])
        bad[first].flat[0] = b + 10*(case.atol + case.rtol*abs(b) + 1.)
        control = not all(m["passed"] for m in _compare(bad, reference, case.units, case.rtol, case.atol).values())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Invalid metadata must not prevent the report from recording a failure.
        try:
            _jsonable(provenance)
        except (ValueError, TypeError):
            provenance = {"sources": (), "data": {}, "limitations": "Invalid provenance: " + repr(provenance)}
    passed = bool(metrics) and not error and control and all(m["passed"] for m in metrics.values())
    return BenchmarkResult(case.id, case.title, case.evidence_kind, passed, metrics, control, provenance, error)


def run_research_benchmarks(cases: Sequence[ResearchBenchmark] | None = None) -> ResearchBenchmarkReport:
    """Run all eight bundled benchmarks, or an explicit nonempty case selection.

    Computation failures become failed records, never successful skips. An empty
    selection or duplicate IDs raises ``ValueError``. No network calls occur.
    A perturbed-output negative control is required for every successful case.
    """
    import scipy
    import pandas
    from puremacro import __version__

    selected = tuple(research_benchmarks() if cases is None else cases)
    if not selected or any(not isinstance(c, ResearchBenchmark) for c in selected):
        raise ValueError("Select at least one ResearchBenchmark")
    if len({c.id for c in selected}) != len(selected):
        raise ValueError("Benchmark IDs must be unique")
    environment = {"puremacro": __version__, "python": platform.python_version(),
                   "numpy": np.__version__, "scipy": scipy.__version__,
                   "pandas": pandas.__version__, "platform": platform.platform()}
    return ResearchBenchmarkReport(tuple(_run(c) for c in selected), environment,
                                   datetime.now(timezone.utc).isoformat())


def _linear_data():
    design = np.array([[1., 0.], [1., 1.], [1., 2.], [1., 4.]])
    values = np.array([.3, .85, 1.6, 2.75])
    factor = np.array([[.2, 0., 0., 0.], [.08, .3, 0., 0.],
                       [.02, -.04, .25, 0.], [.01, .05, .03, .4]])
    return design, values, factor@factor.T


def _linear_observed():
    from puremacro.structural import MomentTargets, fit_structural
    design, values, covariance = _linear_data()
    labels = ("impact", "h1", "h2", "h4")
    targets = MomentTargets(values, covariance, labels, units=("level",)*4,
                            metadata={"source": "deterministic benchmark", "covariance": "estimator covariance"})
    fit = fit_structural(lambda theta: design@theta, targets, [.1, .2],
                         parameter_names=("intercept", "slope"), jac=lambda theta: design)
    if not fit.success or not fit.inference_valid:
        raise ValueError("Structural fit did not converge with valid inference")
    return {"parameters": fit.theta, "parameter_covariance": fit.covariance,
            "objective": fit.objective}


def _linear_reference():
    design, values, covariance = _linear_data()
    # Independent closed-form GLS: no structural fitting or derivative helper.
    # Linear systems have fixed positive-definite matrices by construction.
    precision_design = np.linalg.solve(covariance, design)
    information = design.T@precision_design
    theta = np.linalg.solve(information, precision_design.T@values)
    residual = design@theta-values
    return {"parameters": theta,
            "parameter_covariance": np.linalg.solve(information, np.eye(2)),
            "objective": residual@np.linalg.solve(covariance, residual)}


def _household_observed():
    from puremacro.trade.household import calibrate_household, compute_household_welfare
    pref = calibrate_household(np.array([[3.], [5.], [2.]]), "stone_geary",
                               expenditure_elasticities=np.array([[.8], [1.04], [1.2]]),
                               supernumerary_share=.6)
    p0, p1 = np.array([[1.04], [.95], [1.02]]), np.array([[1.2], [.9], [1.15]])
    welfare = compute_household_welfare(pref, p1, [10.8], base_prices=p0, base_budget=[10.2])
    return {"ev": welfare.ev, "cv": welfare.cv,
            "base_quantities": pref.evaluate(p0, [10.2], validate=True).quantities[:, 0],
            "new_quantities": pref.evaluate(p1, [10.8], validate=True).quantities[:, 0]}


def _household_reference():
    from scipy.optimize import minimize
    # Raw utility primitives specified independently of HouseholdPreferences.
    gamma, beta = np.array([1.56, 1.88, .56]), np.array([.24, .52, .24])
    p0, p1 = np.array([1.04, .95, 1.02]), np.array([1.2, .9, 1.15])
    bounds = [(float(g+1e-9), None) for g in gamma]

    def logu(q):
        return float(beta@np.log(q-gamma))

    def maximize(p, m):
        start = gamma + (m-p@gamma)/(3*p)
        fit = minimize(lambda q: -logu(q), start, jac=lambda q: -beta/(q-gamma),
                       method="SLSQP", bounds=bounds,
                       constraints={"type": "ineq", "fun": lambda q: m-p@q,
                                    "jac": lambda q: -p}, options={"ftol": 1e-12, "maxiter": 500})
        if not fit.success or abs(p@fit.x-m) > 1e-8:
            raise ValueError("Independent utility maximization failed")
        return fit.x

    def spend(p, q):
        target = logu(q)
        fit = minimize(lambda z: p@z, q, jac=lambda z: p, method="SLSQP", bounds=bounds,
                       constraints={"type": "ineq", "fun": lambda z: logu(z)-target,
                                    "jac": lambda z: beta/(z-gamma)},
                       options={"ftol": 1e-12, "maxiter": 500})
        if not fit.success or abs(logu(fit.x)-target) > 1e-8:
            raise ValueError("Independent expenditure minimization failed")
        return p@fit.x

    q0, q1 = maximize(p0, 10.2), maximize(p1, 10.8)
    return {"ev": [spend(p0, q1)-spend(p0, q0)],
            "cv": [spend(p1, q1)-spend(p1, q0)], "base_quantities": q0, "new_quantities": q1}


def _distribution_observed():
    import pandas as pd
    from puremacro.trade.distributional import prepare_household_groups, compute_distributional_welfare
    sectors, groups = ["food", "other"], ["low", "high"]
    household = prepare_household_groups(pd.DataFrame([[60., 40.], [60., 140.]], index=groups, columns=sectors),
                                         pd.Series([3., 1.], index=groups), monetary_unit="currency",
                                         period="year", provenance={"source": "synthetic benchmark", "is_synthetic": True})
    result = compute_distributional_welfare(household, pd.Series([1., 1.], index=sectors),
                                            pd.Series([1.2, .9], index=sectors), transfer_total=40.,
                                            rule="cobb_douglas")
    ev, cv = result.groups["ev"].to_numpy(), result.groups["cv"].to_numpy()
    return {"ev": ev, "cv": cv, "total_ev": result.aggregate["total_ev"],
            "budget": result.groups["budget"].to_numpy()}


def _distribution_reference():
    prices = np.array([1.2, .9])
    budgets = np.array([100., 200.])
    shares = np.array([[.6, .4], [.3, .7]])
    # Default transfer weights represent equal currency per expanded household.
    new_budget = budgets + 40./4.
    index = np.prod(prices**shares, axis=1)
    ev = new_budget/index-budgets
    cv = new_budget-budgets*index
    return {"ev": ev, "cv": cv, "total_ev": np.array([3., 1.])@ev, "budget": new_budget}


_GROWTH_SOURCE = """var c k a; varexo e;
parameters alpha beta rho; alpha=.36; beta=.96; rho=.8;
model;
1/c=beta/c(+1)*alpha*exp(a(+1))*k^(alpha-1);
k=exp(a)*k(-1)^alpha-c;
a=rho*a(-1)+e;
end;
initval; c=.35; k=.19; a=0; end;
shocks; var e; stderr .01; end;
"""


def _growth_observed():
    from puremacro.dsge import load_mod
    model = load_mod(_GROWTH_SOURCE, order=1, tol=1e-11)
    dr = model.decision_rules()
    return {"steady_state": model.steady_state.loc[["c", "k", "a"]].to_numpy(),
            "lagged_capital_derivative": dr.ghx.loc[["c", "k", "a"], "k"].to_numpy(),
            "technology_derivative": dr.ghu.loc[["c", "k", "a"], "e"].to_numpy()}


def _growth_reference():
    alpha, beta = .36, .96
    capital = (alpha*beta)**(1/(1-alpha))
    consumption = (1-alpha*beta)*capital**alpha
    return {"steady_state": [consumption, capital, 0.],
            "lagged_capital_derivative": [(1-alpha*beta)*alpha*capital**(alpha-1), alpha, 0.],
            "technology_derivative": [consumption, capital, 1.]}


_DYNARE_NAME = "rbc_order2"
_DYNARE_REFERENCE_HASH = "eb7ddf55b223fae31b2664e16ec7970848c816020544b46822eca05e52926d72"
_DYNARE_AXES = {"ghx": "x", "ghu": "u", "ghxx": "xx", "ghxu": "xu", "ghuu": "uu", "ghs2": ""}


def _dynare_fixture():
    root = resources.files("puremacro.dsge").joinpath("_references", "dynare_live")
    manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    info = manifest["cases"][_DYNARE_NAME]
    if info["reference_sha256"] != _DYNARE_REFERENCE_HASH or manifest["run_id"] != "20260920_161322_166":
        raise ValueError("Dynare manifest differs from the benchmark's documented external reference")
    blobs = {}
    for name, hash_key in ((info["model"], "model_sha256"),
                           (_DYNARE_NAME+".npz", "reference_sha256"),
                           ("rbc_innovations.csv", "innovations_sha256")):
        blobs[name] = root.joinpath(name).read_bytes()
        if hashlib.sha256(blobs[name]).hexdigest() != info[hash_key]:
            raise ValueError(f"Dynare fixture checksum mismatch: {name}")
    with np.load(io.BytesIO(blobs[_DYNARE_NAME+".npz"]), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if arrays["path"].shape != (manifest["periods"], len(arrays["variable_names"])):
        raise ValueError("Dynare fixture has an incomplete path")
    return blobs[info["model"]].decode("utf-8"), arrays, manifest


def _sample_moments(path, burn):
    values = path[burn:]
    return {"sample_mean": values.mean(0), "sample_covariance": np.cov(values, rowvar=False),
            "sample_lag1_covariance": (values[1:]-values[1:].mean(0)).T
            @ (values[:-1]-values[:-1].mean(0))/(len(values)-2)}


def _dynare_reference():
    _, arrays, manifest = _dynare_fixture()
    metrics = {name: arrays[name] for name in ("ys", *_DYNARE_AXES)}
    metrics.update(_sample_moments(arrays["path"], manifest["burn"]))
    return metrics


def _dynare_observed():
    import pandas as pd
    from puremacro.dsge import load_mod
    source, arrays, manifest = _dynare_fixture()
    model = load_mod(source, order=2)
    dr = model.decision_rules()
    variables, states, shocks = (list(arrays[key]) for key in ("variable_names", "state_names", "shock_names"))
    rows = [list(dr.variable_names).index(v) for v in variables]
    metrics = {"ys": np.asarray(dr.ys)[rows]}
    for name, axes in _DYNARE_AXES.items():
        cols = list(product(*(states if axis == "x" else shocks for axis in axes)))
        own = list(product(*(list(dr.state_variables) if axis == "x" else list(dr.shock_names) for axis in axes)))
        matrix = np.asarray(getattr(dr, name)).reshape(len(variables), -1)[rows]
        metrics[name] = matrix[:, [own.index(column) for column in cols]]
    innovations = arrays["innovations"][:, [shocks.index(s) for s in model.shock_names]]
    simulation = model.simulate(periods=len(innovations), burn=0, shocks=innovations)
    path = pd.concat([simulation.states, simulation.controls], axis=1)[variables].to_numpy()+np.asarray(dr.ys.loc[variables])
    metrics.update(_sample_moments(path, manifest["burn"]))
    return metrics


def _enigh_observed():
    from puremacro.datasets.enigh import load_enigh2024_deciles
    frame = load_enigh2024_deciles()
    consumption = frame[["food", "clothing", "housing", "household_goods", "health",
                         "transport", "education_recreation", "personal"]].sum(axis=1)
    return {"households": frame["households"].sum(),
            "national_food": frame["households"]@frame["food"],
            "national_monetary_expenditure": frame["households"]@frame["monetary_expenditure"],
            "expenditure_adding_up": (consumption+frame["outward_transfers"]-frame["monetary_expenditure"]).to_numpy()}


def _enigh_reference():
    # Literal published national table entries, not read from the derived CSV
    # or recalculated from the same decile observations used above.
    return {"households": 38830230., "national_food": 698246718791.5,
            "national_monetary_expenditure": 1851206621945.59,
            "expenditure_adding_up": np.zeros(10)}


_RR2010_METRICS = ("coefficients", "coefficient_covariance", "irf",
                   "irf_covariance", "standard_errors", "t_statistics",
                   "nobs", "df_resid")


def _rr2010_observed():
    from puremacro.replication.romer_romer_2010 import estimate_rr2010_baseline
    result = estimate_rr2010_baseline()
    return {name: getattr(result, name) for name in _RR2010_METRICS}


def _rr2010_reference():
    from puremacro.replication.romer_romer_2010 import load_rr2010_reference
    reference = load_rr2010_reference()
    return {name: reference[name] for name in _RR2010_METRICS}


def _rr2010_peak_observed():
    from puremacro.replication.romer_romer_2010 import estimate_rr2010_baseline
    result = estimate_rr2010_baseline()
    return {"response_h10": result.irf[10], "t_h10": result.t_statistics[10],
            "trough_horizon": int(np.argmin(result.irf))}


def _rr2010_peak_reference():
    # Literal printed p. 781 values, not rounded estimator output or software
    # reference values. Horizon is checked independently of the effect size.
    return {"response_h10": -3.08, "t_h10": -3.53, "trough_horizon": 10}


def research_benchmarks() -> tuple[ResearchBenchmark, ...]:
    """Return the eight built-in cases without running estimators or reading data."""
    micro_source = "https://ocw.mit.edu/courses/14-661-labor-economics-i-fall-2024/mit14_661_f24_problem_set_0.pdf"
    rr_sources = ("https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763",
                  "https://eml.berkeley.edu/~dromer/papers/DataSet.zip",
                  "https://emlab.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf")
    rr_data = {"origin": "Romer and Romer (2010) original author archive",
               "archive_sha256": "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c",
               "csv_sha256": "059d7514fb14e5b8a8f2516e23c49b618032236c0b4cf4599082f3637e12fdad",
               "reference_sha256": "3c915d119158db1f7002efa80fa12812e2c4edaf225569960561fef29b915b8b",
               "metadata_sha256": "06cf9cffee4213684b0abd9d05a158e00e880cc6260cfab632cf6a2793ee13d7",
               "author_specification": "EXOGERNR.RAT",
               "reference_software": "statsmodels 0.14.6, Python 3.12.13; QR OLS",
               "reference_loader": "puremacro.replication.load_rr2010_reference"}
    rr_sampling = "1950Q1–2007Q4, 232 quarterly observations; 14 coefficients, 218 residual degrees of freedom."
    rr_transform = ("OLS of 100*delta log(real GDP) on intercept and lags 0–12 of "
                    "100*(DEFIC+LONGR)/NOMGDP; conventional residual-df OLS covariance. "
                    "Cumulate tax coefficients and full covariance to obtain GDP-level responses.")
    return (
        ResearchBenchmark("linear_minimum_distance", "GLS fit and full parameter covariance", "analytical",
                          _linear_observed, _linear_reference,
                          ("https://users.ssc.wisc.edu/~bhansen/econometrics/",),
                          {"origin": "deterministic synthetic four-moment linear design", "observations": 4,
                           "design": _linear_data()[0].tolist(), "values": _linear_data()[1].tolist(),
                           "estimator_covariance": _linear_data()[2].tolist()},
                          "No draws; supplied covariance is Cov(moment estimator), already scaled for its sample.",
                          "Identity; fit intercept and slope with a correlated four-moment target.",
                          {"parameters": "level", "parameter_covariance": "level squared", "objective": "dimensionless"},
                          "Checks linear optimal weighting and covariance. No empirical model identification or finite-sample coverage claim."),
        ResearchBenchmark("household_expenditure", "Stone-Geary welfare versus independent primal optimization", "numerical_oracle",
                          _household_observed, _household_reference, (micro_source,),
                          {"origin": "synthetic three-good household", "baseline_spending": [3., 5., 2.],
                           "gamma": [1.56, 1.88, .56], "beta": [.24, .52, .24],
                           "base_prices": [1.04, .95, 1.02], "prices": [1.2, .9, 1.15],
                           "base_budget": 10.2, "budget": 10.8},
                          "No sample or survey weights. One household; two deterministic states.",
                          "Direct utility maximization and four expenditure minimizations use raw log Stone-Geary utility; no puremacro welfare reference.",
                          {"ev": "currency per period", "cv": "currency per period",
                           "base_quantities": "benchmark-priced composite units", "new_quantities": "benchmark-priced composite units"},
                          "Independent SciPy SLSQP oracle for interior preferences. Not an empirical household calibration.", rtol=2e-6, atol=2e-7),
        ResearchBenchmark("distributional_incidence", "Household incidence, expansion weights and uniform rebates", "analytical",
                          _distribution_observed, _distribution_reference, (micro_source,),
                          {"origin": "synthetic grouped survey", "group_spending": [[60., 40.], [60., 140.]],
                           "expansion_counts": [3., 1.], "price_ratios": [1.2, .9], "transfer_total": 40.},
                          "Two representative groups expand to three low-budget and one high-budget household; no sampling inference.",
                          "Cobb-Douglas exact cost indices, equal transfer per expanded household, and weighted EV totals.",
                          {"ev": "currency per household-year", "cv": "currency per household-year",
                           "total_ev": "currency per year", "budget": "currency per household-year"},
                          "Conditional static incidence with fixed preferences and exogenous prices; no empirical Mexico or general-equilibrium claim."),
        ResearchBenchmark("growth_analytical", "Full-depreciation log-utility growth solution", "analytical",
                          _growth_observed, _growth_reference,
                          ("https://python.quantecon.org/cass_koopmans_1.html",),
                          {"origin": "analytical growth economy", "alpha": .36, "beta": .96, "rho": .8,
                           "depreciation": 1., "model_source": _GROWTH_SOURCE},
                          "No sample. Deterministic steady state and first-order derivatives at zero technology.",
                          "k' = alpha*beta*exp(a)*k**alpha; c = (1-alpha*beta)*exp(a)*k**alpha. Variable order c,k,a.",
                          {"steady_state": "c,k: goods per capita; a: log technology",
                           "lagged_capital_derivative": "output units per unit of lagged capital",
                           "technology_derivative": "output units per unit log-technology innovation"},
                          "Checks this special log-utility, Cobb-Douglas, full-depreciation case and its local derivatives only."),
        ResearchBenchmark("dynare_rbc_order2", "RBC decision rules and common-innovation moments versus Dynare 7", "external_software",
                          _dynare_observed, _dynare_reference,
                          ("https://www.dynare.org/manual/the-model-file.html#stochastic-solution-and-simulation",
                           "Package: puremacro.dsge/_references/DYNARE_FIXTURES.md"),
                          {"origin": "Live Dynare 7.0 in MATLAB R2026a, 2026-09-20", "run_id": "20260920_161322_166",
                           "reference": "puremacro.dsge/_references/dynare_live/rbc_order2.npz",
                           "reference_sha256": _DYNARE_REFERENCE_HASH,
                           "provenance_manifest": "puremacro.dsge/_references/dynare_live/manifest.json"},
                          "2,500 common Gaussian innovations from NumPy default_rng(20260920); discard first 500. Covariance uses ddof=1; lag-1 covariance uses separately centered endpoints and N-2.",
                          "Match variable/state/shock labels; unfold derivative tensors without factorial rescaling; add deterministic steady state to simulated deviations. Validate model, innovations and reference checksums.",
                          {"ys": "model levels (y,c,k,a)",
                           **{name: "level decision-rule derivative in named state/shock coordinates" for name in _DYNARE_AXES},
                           "sample_mean": "model levels", "sample_covariance": "products of model level units",
                           "sample_lag1_covariance": "products of model level units"},
                          "External-software agreement for one small model at order two, conditional on supplied innovations. No live Dynare run, population-moment accuracy or published empirical replication is claimed.",
                          rtol=1e-9, atol=1e-10),
        ResearchBenchmark("enigh2024_official_totals", "Observed ENIGH decile baskets reproduce official national totals", "official_data",
                          _enigh_observed, _enigh_reference,
                          ("https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx",),
                          {"origin": "INEGI ENIGH 2024 national household income deciles; Cuadros 3.2 and 4.2",
                           "source_sha256": "7af7850495255fb1e6a9cb139cf531e5a62c7fd2a9a1de5ba0503f03aa82490b",
                           "accessed": "2026-10-01", "derived_csv": "puremacro.datasets/data/enigh2024_deciles.csv",
                           "reference_origin": "literal national published totals, not the derived CSV"},
                          "Ten national household income deciles; expansion counts include all households. No sampling covariance is available.",
                          "Published thousands of MXN multiplied by 1000; decile totals divided by all-household counts. Eight consumption categories plus outward transfers equal monetary expenditure. CSV decimal rounding motivates the tolerance.",
                          {"households": "expanded households", "national_food": "MXN per quarter",
                           "national_monetary_expenditure": "MXN per quarter", "expenditure_adding_up": "MXN per household-quarter"},
                          "Validates the extraction, weights, units and selected accounting totals of actual survey aggregates. Does not replicate an empirical causal model, identify tariff responses or validate within-decile heterogeneity.",
                          rtol=1e-11, atol=1e-5),
        ResearchBenchmark("rr2010_baseline_software", "Romer–Romer baseline versus independent software", "external_software",
                          _rr2010_observed, _rr2010_reference, rr_sources, rr_data,
                          rr_sampling, rr_transform,
                          {"coefficients": "GDP growth percentage points; intercept or per one-percent-of-GDP tax increase",
                           "coefficient_covariance": "products of coefficient units",
                           "irf": "GDP log-percent per one-percent-of-GDP tax increase",
                           "irf_covariance": "squared GDP log-percent response units",
                           "standard_errors": "GDP log-percent response units", "t_statistics": "dimensionless",
                           "nobs": "quarters", "df_resid": "degrees of freedom"},
                          "Frozen independent software port of the authors' baseline specification; not a live RATS run. "
                          "Numerical agreement does not establish narrative shock exogeneity or reproduce robustness specifications.",
                          rtol=1e-10, atol=1e-11),
        ResearchBenchmark("rr2010_published_peak", "Romer–Romer published Figure 4 baseline trough", "published_empirical",
                          _rr2010_peak_observed, _rr2010_peak_reference, rr_sources,
                          {**rr_data, "printed_page": 781, "reference_origin": "literal published values -3.08, -3.53 and horizon 10"},
                          rr_sampling, rr_transform,
                          {"response_h10": "GDP log-percent per one-percent-of-GDP tax increase",
                           "t_h10": "dimensionless", "trough_horizon": "quarters after the tax increase"},
                          "One original-data baseline result. Absolute tolerance 0.005 reflects two-decimal publication rounding; "
                          "integer horizon must match exactly. The original-data t statistic misses its published rounding interval "
                          "by about 0.000003925; this remains a failure, also present in the authors' RATS databanks. "
                          "It is not a software-parity tolerance or independent causal validation.",
                          rtol=0., atol=.005),
    )
