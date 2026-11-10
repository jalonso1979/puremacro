"""Dated Gaussian news and present-value New Keynesian forecasts.

Run: python -m puremacro.examples.heterogeneous_expectations --output /tmp/news --lang en
Synthetic information-processing experiment; no endogenous market learning or welfare.
"""
from __future__ import annotations

from collections.abc import Mapping

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro.expectations import GaussianSignal, build_gaussian_information
from puremacro.models.information_nk import solve_information_nk, audit_information_nk


def make_information(policy_date: int, regime: str, *, horizon: int | None = None):
    """Two signal-sharing groups; all policy news becomes observed at implementation."""
    T = policy_date+4 if horizon is None else horizon
    groups = []
    for g, noise in enumerate((.5, 2.)):
        signals = []
        if regime == "full_information":
            signals = [GaussianSignal(0, row, f"state{j}") for j, row in enumerate(np.eye(5))]
        elif regime in ("no_advance_news", "heterogeneous", "second_update"):
            if regime != "no_advance_news":
                loading = np.zeros(5)
                loading[0], loading[g+1] = 1., noise
                signals.append(GaussianSignal(0, loading, "initial_news"))
            if regime == "second_update" and policy_date > 4:
                loading = np.zeros(5)
                loading[0], loading[g+3] = 1., .5
                signals.append(GaussianSignal(4, loading, "second_news"))
            signals.append(GaussianSignal(policy_date, [1, 0, 0, 0, 0], "policy_revelation"))
        else:
            raise ValueError("unknown information regime")
        groups.append(tuple(signals))
    return build_gaussian_information(np.eye(5), groups, horizon=T, group_names=("A", "B"),
        state_names=("policy_news", "initial_noise_A", "initial_noise_B", "update_noise_A", "update_noise_B"))


def _json(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False)+"\n", encoding="utf-8")


def _forecast_experiment(output: Path) -> dict:
    info = make_information(8, "second_update")
    seed, count = 88154, 50000
    draws = np.random.default_rng(seed).normal(size=(count, 5))
    rows = []
    for t in (0, 3, 4, 8):
        for g, noise in enumerate((.5, 2.)):
            forecast = draws@info.projectors[t, g, 0]
            errors = draws[:, 0]-forecast
            expected_variance = 0. if t >= 8 else 1/(1+1/noise**2+(4 if t >= 4 else 0))
            mean = float(errors.mean())
            variance = float(np.mean(errors**2))
            orthogonality_z = []
            for signal in info.signals[g]:
                if signal.date <= t:
                    product = errors*(draws@signal.loading)
                    se = float(product.std(ddof=1)/np.sqrt(count))
                    orthogonality_z.append(abs(float(product.mean()))/se if se > 1e-12 else 0.)
            mean_z = abs(mean)/np.sqrt(expected_variance/count) if expected_variance else 0.
            accepted = (abs(variance/expected_variance-1) < .04 and mean_z < 4
                        and max(orthogonality_z) < 4) if expected_variance else np.max(abs(errors)) < 1e-11
            rows.append(dict(date=t, group=info.group_names[g], analytic_variance=expected_variance,
                simulated_mean_error=mean, simulated_mean_squared_error=variance,
                mean_error_in_standard_errors=mean_z, max_observation_orthogonality_z=max(orthogonality_z),
                accepted=bool(accepted)))
    pd.DataFrame(rows).to_csv(output/"forecast-monte-carlo.csv", index=False)
    return dict(seed=seed, draws=count, criterion="mean and orthogonality within 4 MC SE; MSE within 4%; exact revelation <1e-11",
                accepted=all(r["accepted"] for r in rows), results=rows)


def _plot(paths, horizon_frame, output: Path, lang: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spanish = lang == "es"
    labels = (["Información completa", "Sin noticia anticipada", "Señales heterogéneas", "Segunda actualización"]
              if spanish else ["Full information", "No advance news", "Heterogeneous signals", "Second update"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for (name, path), label in zip(paths.items(), labels):
        axes[0, 0].plot(100*path.output, label=label)
        axes[0, 1].plot(10000*path.inflation, label=label)
        row = horizon_frame[horizon_frame.regime == name]
        axes[1, 1].plot(row.policy_date, row.forecast_normalized_ratio, marker="o", label=label)
    beliefs = paths["second_update"].solution.information.beliefs(paths["second_update"].realization)
    for g, label in enumerate(("A", "B")):
        axes[1, 0].plot(beliefs[:, g, 0], marker="o", label=f"Group {label}" if not spanish else f"Grupo {label}")
    titles = (["Brecha del producto, log × 100", "Inflación trimestral (pb)", "Media posterior de la noticia", "Respuesta por noticia / información completa"]
              if spanish else ["Output gap, log × 100", "Quarterly inflation (bp)", "Posterior mean of news", "Response per policy forecast / full information"])
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.axhline(0, color="black", lw=.5)
        ax.grid(alpha=.2)
        ax.set_xlabel("Fecha" if spanish else "Date")
    axes[1, 1].set_xlabel("Fecha de política" if spanish else "Policy date")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend()
    fig.suptitle("Expectativas heterogéneas: experimento sintético" if spanish else "Heterogeneous expectations: synthetic experiment")
    fig.savefig(output/"expectations.png", dpi=160)
    fig.savefig(output/"expectations.pdf")
    plt.close(fig)


def run_example(output: Path, *, lang: str = "en") -> dict:
    if lang not in ("en", "es"):
        raise ValueError("lang must be en or es")
    output.mkdir(parents=True, exist_ok=True)
    regimes = ("full_information", "no_advance_news", "heterogeneous", "second_update")
    realization = np.array([1., 0., 0., 0., 0.])
    paths, records = {}, {}
    for regime in regimes:
        info = make_information(8, regime)
        R = np.zeros((info.horizon, 5))
        R[8, 0] = -.0025
        solution = solve_information_nk(info, R)
        path = solution.evaluate(realization)
        audit = audit_information_nk(solution)
        destination = output/regime
        destination.mkdir(exist_ok=True)
        path.to_frame().to_csv(destination/"path.csv")
        path.forecast_frame().to_csv(destination/"forecasts.csv")
        info.belief_frame(realization).to_csv(destination/"beliefs.csv")
        solution.to_frame().to_csv(destination/"coefficients.csv")
        pd.DataFrame(audit["period_errors"], columns=("demand", "pricing")).to_csv(destination/"audit.csv", index_label="date")
        arrays: dict[str, Any] = dict(covariance=info.covariance, projectors=info.projectors,
            posterior_covariances=info.posterior_covariances, ranks=info.ranks, weights=info.weights,
            rate_loadings=R, output_loadings=solution.output_loadings, inflation_loadings=solution.inflation_loadings,
            realization=realization, firm_weights=solution.firm_weights)
        np.savez_compressed(destination/"arrays.npz", **arrays)
        record = dict(information_fingerprint=info.fingerprint, model_fingerprint=solution.fingerprint,
            parameters=vars(solution.parameters), information_events=info.diagnostics["information_events"],
            signals=[[dict(date=s.date, loading=s.loading.tolist(), label=s.label) for s in group] for group in info.signals],
            diagnostics=dict(solution.diagnostics), initial_output_pct=float(path.output[0]*100),
            initial_inflation_bp=float(path.inflation[0]*10000))
        _write(destination/"record.json", record)
        records[regime], paths[regime] = record, path
    horizon_rows = []
    for date in (0, 2, 4, 8, 12, 16):
        for regime in regimes:
            info = make_information(date, regime)
            R = np.zeros((info.horizon, 5))
            R[date, 0] = -.0025
            solution = solve_information_nk(info, R)
            output_path = solution.output_loadings@realization
            inflation_path = solution.inflation_loadings@realization
            expected_rate = float(R[date]@info.average_projectors[0]@realization)
            horizon_rows.append(dict(policy_date=date, regime=regime,
                initial_output_pct=float(output_path[0]*100),
                initial_inflation_bp=float(inflation_path[0]*10000),
                initial_expected_policy_rate=expected_rate,
                policy_forecast_fraction=expected_rate/(-.0025),
                large_linear_response=bool(max(np.max(abs(output_path)), np.max(abs(inflation_path))) > .05),
                max_coefficient_error=solution.diagnostics["max_error"]))
    horizon_frame = pd.DataFrame(horizon_rows)
    full_impacts = horizon_frame[horizon_frame.regime == "full_information"].set_index("policy_date").initial_output_pct
    horizon_frame["full_information_ratio"] = horizon_frame.initial_output_pct/horizon_frame.policy_date.map(full_impacts)
    horizon_frame["forecast_normalized_ratio"] = horizon_frame.full_information_ratio/horizon_frame.policy_forecast_fraction.replace(0., np.nan)
    horizon_frame.to_csv(output/"policy-horizons.csv", index=False)
    simulation = _forecast_experiment(output)
    _plot(paths, horizon_frame, output, lang)
    sources = {}
    for module_name in ("puremacro.expectations.gaussian", "puremacro.models.information_nk", "puremacro.examples.heterogeneous_expectations"):
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        sources[module_name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    report = dict(scope="synthetic present-value NK forecast system; only listed signals processed; no endogenous market learning or welfare",
        language=lang, policy=dict(date=8, quarterly_rate_deviation=-.0025), realization=realization,
        realization_scope="policy news 1 and zero group noise; not a population average",
        linear_response_scope="output is log deviation times 100; large_linear_response flags >0.05 absolute log/rate deviations, not an accuracy certificate",
        comparison_scope="forecast_normalized_ratio divides the response ratio by the first-order policy-forecast ratio; undefined without advance news",
        cases=records, forecast_monte_carlo=simulation, source_hashes=sources,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))
    _write(output/"summary.json", report)
    print(horizon_frame.to_string(index=False))
    print(f"Forecast Monte Carlo accepted: {simulation['accepted']}")
    print(f"Records: {output}")
    if not simulation["accepted"]:
        raise RuntimeError("forecast experiment failed its fixed criteria; inspect retained results")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("heterogeneous-expectations"))
    parser.add_argument("--lang", choices=("en", "es"), default="en")
    args = parser.parse_args()
    run_example(args.output, lang=args.lang)


if __name__ == "__main__":
    main()
