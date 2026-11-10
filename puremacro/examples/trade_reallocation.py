"""Illustrative joint trade transition with three real adjustment costs.

Run: python -m puremacro.examples.trade_reallocation --output /tmp/trade --lang en
All inputs are synthetic assumptions. This is not an empirical policy estimate.
"""
from __future__ import annotations

from collections.abc import Mapping

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy

from puremacro.trade import (
    ReallocationEconomy, audit_reallocation, reallocation_stability,
    run_reallocation_horizon_ladder, solve_reallocation_steady_state,
)


def tariff_path(horizon: int) -> np.ndarray:
    """Date-zero announcement: AAA taxes S01 imports 10% at dates 2,...,7."""
    rates = np.zeros((horizon, 2, 2))
    rates[2:8, 0, 0] = .1
    return rates


def _json(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unserializable {type(value)}")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False)+"\n", encoding="utf-8")


def _plot(runs, destination: Path, lang: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spanish = lang == "es"
    labels = (["Completo", "Sin costo de capital", "Sin costo laboral", "Sin costo de abastecimiento"]
              if spanish else ["Full model", "Zero capital cost", "Zero labor cost", "Zero sourcing cost"])
    titles = (["Consumo de AAA (%)", "Capital de AAA, S01 (%)", "Empleo de AAA, S01 (%)",
               "Cuota importada de AAA, S01 (pp)", "Salario real de AAA, S01 (%)", "Consumo de BBB (%)"]
              if spanish else ["AAA consumption (%)", "AAA S01 capital (%)", "AAA S01 employment (%)",
                              "AAA S01 import share (pp)", "AAA S01 real wage (%)", "BBB consumption (%)"])
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, layout="constrained")
    for (name, run), label in zip(runs.items(), labels):
        v, initial = run.paths, run.initial.values
        series = [100*(v["consumption"][:, 0]/initial["consumption"][0]-1),
                  100*(v["capital"][:, 0, 0]/initial["capital"][0, 0]-1),
                  100*(v["labor"][:, 0, 0]/initial["labor"][0, 0]-1),
                  100*(v["import_shares"][:, 0, 0]-initial["import_shares"][0, 0]),
                  100*((v["wages"][:, 0, 0]/v["price_index"][:, 0])/
                       (initial["wages"][0, 0]/initial["price_index"][0])-1),
                  100*(v["consumption"][:, 1]/initial["consumption"][1]-1)]
        for ax, y, title in zip(axes.flat, series, titles):
            ax.plot(np.arange(min(32, len(y))), y[:32], label=label)
            ax.set_title(title)
    for ax in axes.flat:
        ax.axhline(0, color="black", lw=.5)
        ax.axvspan(2, 7, color="grey", alpha=.12)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Fecha" if spanish else "Date")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Arancel temporal: ejemplo sintético" if spanish else "Temporary tariff: synthetic illustration")
    fig.savefig(destination/"paths.png", dpi=160)
    fig.savefig(destination/"paths.pdf")
    plt.close(fig)


def run_example(output: Path, *, lang: str = "en") -> dict:
    if lang not in ("en", "es"):
        raise ValueError("lang must be en or es")
    output.mkdir(parents=True, exist_ok=True)
    model = ReallocationEconomy()
    limits: dict[str, Any] = dict(horizons=(40, 80, 160, 320), window=20, tolerance=1e-8,
                  path_tolerance=1e-6, welfare_tolerance=1e-4, terminal_tolerance=1e-5,
                  max_evaluations=200)
    models = {"full": model, "zero_capital_cost": replace(model, capital_cost=0.),
              "zero_labor_cost": replace(model, labor_cost=0.),
              "zero_sourcing_cost": replace(model, sourcing_cost=0.)}
    report: dict[str, Any] = dict(scope="synthetic two-country/two-sector experiment; pooled national income; approximate terminal value",
        language=lang, policy=dict(importer="AAA", sector="S01", rate=.1, first_date=2, last_date=7,
                                  information="all dates announced at date zero"),
        thresholds=limits, cases={},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         pandas=pd.__version__, platform=platform.platform()),
        source_hashes={})
    for module_name in ("puremacro.trade.reallocation", "puremacro.trade._reallocation_audit",
                        "puremacro.trade.dynamic.welfare", "puremacro.examples.trade_reallocation"):
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        report["source_hashes"][module_name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    # This originally considered calibration is deliberately retained as rejected.
    rejected = replace(model, sourcing_cost=1.)
    report["rejected_initial_calibration"] = dict(sourcing_cost=1.,
        terminal_stability=reallocation_stability(solve_reallocation_steady_state(rejected)),
        reason="failed local terminal stable-graph check; default amended to 0.1 after the first pilot",
        scope="no universal upper bound on sourcing costs is inferred")
    _write_json(output/"design.json", report)
    accepted = {}
    welfare_rows = []
    for name, economy in models.items():
        destination = output/name
        destination.mkdir(exist_ok=True)
        started = time.perf_counter()
        ladder = run_reallocation_horizon_ladder(economy, tariff_path, **limits)
        record: dict[str, Any] = dict(parameters=vars(economy), fingerprint=economy.fingerprint,
                      status=ladder.status, attempts=ladder.attempts,
                      seconds=time.perf_counter()-started, runs=[])
        for run in ladder.runs:
            T = len(run.tariffs)
            certificate = audit_reallocation(run)
            arrays: dict[str, Any] = dict(tariffs=run.tariffs, coordinates=run.coordinates,
                **{f"path__{k}": v for k, v in run.paths.items()},
                **{f"initial__{k}": v for k, v in run.initial.values.items()},
                **{f"terminal__{k}": v for k, v in run.terminal.values.items()})
            np.savez_compressed(destination/f"path-{T}.npz", **arrays)
            run.to_frame().to_csv(destination/f"countries-{T}.csv")
            run.sector_frame().to_csv(destination/f"sectors-{T}.csv")
            run.welfare.to_csv(destination/f"welfare-{T}.csv")
            pd.DataFrame(certificate["period_errors"], columns=certificate["check_names"]).to_csv(
                destination/f"audit-{T}.csv", index_label="date")
            record["runs"].append(dict(horizon=T, diagnostics=dict(run.diagnostics),
                                       audit={k: v for k, v in certificate.items() if k != "period_errors"},
                                       welfare=run.welfare.reset_index().to_dict(orient="records")))
        ladder.to_frame().drop(columns=["diagnostics"], errors="ignore").to_csv(destination/"horizons.csv", index=False)
        _write_json(destination/"record.json", record)
        report["cases"][name] = record
        if ladder.accepted is not None:
            accepted[name] = ladder.accepted
            frame = ladder.accepted.welfare.reset_index()
            frame.insert(0, "case", name)
            frame["horizon"] = len(ladder.accepted.tariffs)
            welfare_rows.append(frame)
        print(f"{name}: {ladder.status}", flush=True)
        print(ladder.to_frame().drop(columns=["diagnostics"], errors="ignore").to_string(index=False), flush=True)
    report["all_cases_verified"] = len(accepted) == len(models)
    _write_json(output/"summary.json", report)
    if welfare_rows:
        pd.concat(welfare_rows, ignore_index=True).to_csv(output/"welfare.csv", index=False)
    if report["all_cases_verified"]:
        _plot(accepted, output, lang)
    title = "Transición comercial ilustrativa" if lang == "es" else "Illustrative trade transition"
    scope = ("Supuestos sintéticos; ingreso nacional compartido. La prueba de horizonte no demuestra unicidad global."
             if lang == "es" else "Synthetic assumptions; pooled national income. Horizon refinement does not prove global uniqueness.")
    (output/"README.md").write_text(f"# {title}\n\n{scope}\n\n"
        "`summary.json`: parameters, thresholds, failed/accepted attempts, source hashes and environment.\n"
        "`*/path-*.npz`: complete retained arrays; `*/audit-*.csv`: independently rebuilt period accounts/FOCs.\n"
        "`welfare.csv`: consumption equivalents only for cases passing horizon refinement.\n"
        "The initial sourcing_cost=1 calibration remains rejected; the example uses sourcing_cost=0.1.\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("trade-reallocation"))
    parser.add_argument("--lang", choices=("en", "es"), default="en")
    args = parser.parse_args()
    report = run_example(args.output, lang=args.lang)
    print(f"Records: {args.output}")
    if not report["all_cases_verified"]:
        raise SystemExit("one or more horizon ladders unresolved; inspect retained attempts")


if __name__ == "__main__":
    main()
