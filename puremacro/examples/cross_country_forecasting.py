"""Offline annual GDP-growth forecast comparison on a frozen WDI snapshot.

Run ``python -m puremacro.examples.cross_country_forecasting --output DIR``.
This is an expanding-window, latest-vintage backtest, not real-time forecasting:
the historical observations were revised after the forecast origins.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
from io import BytesIO
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from puremacro import __version__, pocket
from puremacro.reports import df_to_markdown
from puremacro.var import fit_var


COUNTRIES = ("USA", "MEX", "BRA")
MODELS = ("zero_growth", "historical_mean", "ar1")
DESIGN = {
    "countries": list(COUNTRIES),
    "level_start_year": 1960, "level_end_year": 2024,
    "evaluation_start_year": 2000, "evaluation_end_year": 2024,
    "horizon_years": 1, "minimum_training_growth_observations": 20,
    "window": "expanding", "transformation": "100 * diff(log(gdp_real))",
    "models": list(MODELS), "ar_lags": 1, "ar_intercept": True,
    "model_selection": "none; fixed design before outcomes were inspected",
    "preprocessing": "no imputation, scaling, seasonal adjustment or tuning",
    "error_convention": "actual minus forecast",
    "information_set": "growth observations through the preceding calendar year",
    "data_vintage": "frozen latest-vintage snapshot; not historical real-time vintages",
}
LIMITATIONS = [
    "Historical data are revised latest-vintage observations, not observations available at each origin.",
    "Calendar-year origins assume the preceding year's GDP is observed; publication lags are not modeled.",
    "Three selected countries and 25 target years do not establish general forecasting superiority.",
    "RMSE, MAE and mean error are descriptive; no significance test, predictive intervals or causal claims are made.",
    "The 2020 contraction and recovery remain in the predeclared evaluation sample.",
    "Constant-LCU GDP levels are not compared across countries; only within-country log growth is forecast.",
]


def _validated_levels(data: pd.DataFrame) -> pd.DataFrame:
    """Reject missing years rather than turning multi-year changes into annual growth."""
    required = ["code", "date", "gdp_real"]
    if not isinstance(data, pd.DataFrame) or list(data.columns) != required or data.empty:
        raise ValueError("GDP input must be a nonempty frame with columns code,date,gdp_real")
    frame = data.copy(deep=True)
    if frame.code.isna().any() or not frame.code.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Country codes must be nonempty strings")
    try:
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        values = pd.to_numeric(frame.gdp_real, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("GDP dates and levels must be valid annual observations") from exc
    dates = frame.date
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.to_period("Y").dt.start_time).all():
        raise ValueError("Annual GDP dates must be timezone-naive January 1 dates")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("GDP levels must be finite and strictly positive")
    frame["gdp_real"] = values
    if frame.duplicated(["code", "date"]).any():
        raise ValueError("Duplicate country/year observations are not allowed")
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)
    for code, country in frame.groupby("code", sort=True):
        if not np.all(np.diff(country.date.dt.year) == 1):
            raise ValueError(f"{code}: missing calendar years; annual growth cannot bridge gaps")
    return frame


def load_data() -> pd.DataFrame:
    """Authenticate the packaged snapshot and its complete declared country/year grid."""
    directory = files("puremacro.datasets").joinpath("data")
    metadata = json.loads(directory.joinpath("cross_country_gdp_metadata.json").read_text(encoding="utf-8"))
    raw = directory.joinpath("cross_country_gdp.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("Bundled GDP CSV checksum differs from its provenance record")
    if (metadata.get("schema_version") != 1 or metadata.get("countries") != list(COUNTRIES)
            or metadata.get("start_year") != 1960 or metadata.get("end_year") != 2024):
        raise ValueError("Bundled GDP metadata differ from the frozen study design")
    frame = _validated_levels(pd.read_csv(BytesIO(raw), float_precision="round_trip"))
    if set(frame.code) != set(COUNTRIES):
        raise ValueError("Bundled GDP countries differ from the frozen study design")
    for code, country in frame.groupby("code"):
        if country.date.dt.year.tolist() != list(range(1960, 2025)):
            raise ValueError(f"{code}: bundled GDP must contain every year 1960–2024")
    frame.attrs.update(metadata)
    return frame


def evaluate_forecasts(data: pd.DataFrame, *, evaluation_start: int = 2000,
                       evaluation_end: int = 2024, min_train: int = 20) -> dict:
    """Compute fixed one-step baselines using only each origin's training observations.

    The configurable dates support numerical tests and explicitly different
    studies. The CLI always runs the frozen default design. Invalid inputs and
    unidentified AR designs raise rather than silently changing model samples.
    """
    for name, value in (("evaluation_start", evaluation_start), ("evaluation_end", evaluation_end), ("min_train", min_train)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if evaluation_start > evaluation_end or min_train < 4:
        raise ValueError("Evaluation years must be ordered and min_train must be at least 4")
    levels = _validated_levels(data)
    rows, coverage, exclusions, growth_rows = [], [], [], []
    expected_years = list(range(evaluation_start, evaluation_end + 1))
    for code, country in levels.groupby("code", sort=True):
        years = country.date.dt.year.to_numpy()
        growth = 100 * np.diff(np.log(country.gdp_real.to_numpy()))
        growth_years = years[1:]
        if not set(expected_years).issubset(set(growth_years)):
            raise ValueError(f"{code}: complete evaluation years are required; countries are not silently dropped")
        if np.count_nonzero(growth_years < evaluation_start) < min_train:
            raise ValueError(f"{code}: fewer than {min_train} growth observations precede evaluation")
        growth_rows.extend({"code": code, "year": int(year), "growth": float(value)}
                           for year, value in zip(growth_years, growth))
        for year in years:
            if year == years[0]:
                reason = "first_level_has_no_previous_year"
            elif year < evaluation_start:
                reason = "before_fixed_evaluation_window"
            elif year > evaluation_end:
                reason = "after_fixed_evaluation_window"
            else:
                continue
            exclusions.append({"code": code, "year": int(year), "reason": reason})
        coverage.append({
            "code": code, "first_level_year": int(years[0]), "last_level_year": int(years[-1]),
            "level_observations": len(years), "growth_observations": len(growth),
            "first_training_growth_year": int(growth_years[0]),
            "initial_training_growth_observations": int(np.count_nonzero(growth_years < evaluation_start)),
            "evaluation_start_year": evaluation_start, "evaluation_end_year": evaluation_end,
            "evaluation_observations_per_model": len(expected_years), "excluded_evaluation_observations": 0,
        })
        for target in expected_years:
            train = growth[growth_years < target]
            actual = float(growth[growth_years == target][0])
            design = np.column_stack([np.ones(len(train) - 1), train[:-1]])
            if np.linalg.matrix_rank(design) < 2:
                raise np.linalg.LinAlgError(f"{code}, origin {target - 1}: AR(1) design is rank deficient")
            fit = fit_var(train[:, None], lags=1)
            intercept, slope = float(fit.c[0]), float(fit.A_list[0][0, 0])
            predictions = (0., float(train.mean()), intercept + slope * train[-1])
            if not np.isfinite(predictions).all():
                raise np.linalg.LinAlgError(f"{code}, origin {target - 1}: non-finite forecast")
            for model, prediction in zip(MODELS, predictions):
                rows.append({
                    "code": code, "model": model, "origin_year": target - 1, "target_year": target,
                    "horizon_years": 1, "training_start_year": int(growth_years[0]),
                    "training_end_year": target - 1, "training_growth_observations": len(train),
                    "actual": actual, "forecast": float(prediction), "error": actual - prediction,
                })
    forecasts = pd.DataFrame(rows)
    accuracy_rows = []
    for (code, model), group in forecasts.groupby(["code", "model"], sort=True):
        errors = group.error.to_numpy()
        accuracy_rows.append({"code": code, "model": model, "n_forecasts": len(errors),
                              "rmse": float(np.sqrt(np.mean(errors ** 2))),
                              "mae": float(np.mean(np.abs(errors))), "mean_error": float(errors.mean())})
    accuracy = pd.DataFrame(accuracy_rows)
    # Equal country weighting, distinct from pooling GDP levels or weighting by size.
    pooled_rows = []
    for model, group in accuracy.groupby("model", sort=True):
        pooled_rows.append({"model": model, "n_countries": len(group),
                            "n_forecasts": int(group.n_forecasts.sum()),
                            "rmse": float(np.sqrt(np.mean(group.rmse ** 2))),
                            "mae": float(group.mae.mean()), "mean_error": float(group.mean_error.mean())})
    return {"levels": levels, "growth": pd.DataFrame(growth_rows), "forecasts": forecasts,
            "accuracy": accuracy, "pooled_accuracy": pd.DataFrame(pooled_rows),
            "coverage": pd.DataFrame(coverage),
            "exclusions": pd.DataFrame(exclusions, columns=["code", "year", "reason"])}


def run_application(output: Path) -> dict:
    """Export the predeclared offline study, a portable cartridge and its evidence."""
    data = load_data()
    result = evaluate_forecasts(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = []
    for name, frame in result.items():
        filename = f"{name}.csv"
        frame.to_csv(output / filename, index=False, float_format="%.17g", date_format="%Y-%m-%d")
        artifact_names.append(filename)
    metadata_name = "input_metadata.json"
    (output / metadata_name).write_text(json.dumps(data.attrs, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    artifact_names.append(metadata_name)
    cartridge_name = "cross_country_forecasting.pmz"
    pocket.pack(result, output / cartridge_name,
                source="World Bank WDI frozen GDP snapshot",
                notes="Latest-vintage backtest. Fetch date and upstream update date are distinct; see levels.attrs. "
                      "No historical real-time vintages or publication-lag model.",
                call="python -m puremacro.examples.cross_country_forecasting --output OUTPUT")
    artifact_names.append(cartridge_name)
    _plot(output, result)
    artifact_names.append("forecast_accuracy.png")
    manifest = {
        "study": "cross_country_gdp_forecasting", "package_version": __version__,
        "evidence_kind": "latest_vintage_expanding_window_forecast_evaluation",
        "is_real_time": False, "is_causal": False, "design": DESIGN,
        "observed_data": data.attrs, "limitations": LIMITATIONS,
        "aggregation": "Equal country weight: pooled RMSE is sqrt(mean(country MSE)); other losses are mean country losses.",
        "numerical_reproducibility": "Numerical tables are deterministic; cartridge packing records a new creation timestamp.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    _report(output, result, manifest)
    artifact_names.append("report.md")
    package = files("puremacro")
    source_names = ["examples/cross_country_forecasting.py", "var/estimate.py", "runtime/store.py", "pocket/_cartridge.py"]
    manifest["source_sha256"] = {name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest()
                                 for name in source_names}
    manifest["artifacts"] = {name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                                     "bytes": (output / name).stat().st_size} for name in artifact_names}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result["manifest"] = manifest
    return result


def _plot(output, result):
    # The explicit canvas is headless without changing the caller's global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 4.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    accuracy = result["accuracy"]
    x = np.arange(len(COUNTRIES))
    for j, model in enumerate(MODELS):
        values = accuracy[accuracy.model == model].set_index("code").loc[list(COUNTRIES), "rmse"]
        axis.bar(x + (j - 1) * .24, values, width=.24, label=model.replace("_", " "))
    axis.set_xticks(x, COUNTRIES)
    axis.set_ylabel("RMSE (annual log-growth percentage points)")
    axis.set_title("GDP growth, 2000–2024: latest-vintage backtest")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "forecast_accuracy.png", dpi=160, metadata={"Software": "puremacro"})


def _report(output, result, manifest):
    text = (
        "# Cross-country GDP-growth forecasting\n\n"
        "A frozen latest-vintage World Bank WDI snapshot for USA, Mexico and Brazil. "
        "This backtest does not reconstruct information available in real time.\n\n"
        "The fixed target is annual GDP log growth, 100 × Δlog(real GDP). "
        "Expanding training windows begin in 1961; one-year-ahead forecasts target 2000–2024. "
        "At each origin only growth through the preceding year enters estimation. "
        "Zero growth means unchanged GDP level; the historical mean and AR(1) with intercept "
        "are fitted independently for each country. No lag, model or sample tuning is performed.\n\n"
        "## Accuracy\n\n"
        + df_to_markdown(result["accuracy"].set_index(["code", "model"]))
        + "\n\nErrors are actual minus forecast, in annual log-growth percentage points. "
        "Lower RMSE and MAE indicate smaller losses in this fixed sample. "
        "The equal-country summary takes the square root of mean country MSE for RMSE.\n\n"
        + df_to_markdown(result["pooled_accuracy"].set_index("model"))
        + "\n\n## Coverage and interpretation\n\n"
        + df_to_markdown(result["coverage"].set_index("code"))
        + "\n\nAll models use the same target years, including 2020 and the recovery. "
        "The exclusions file records years outside the fixed target window and the first level "
        "without a growth observation. Invalid levels, annual gaps and rank-deficient AR fits raise.\n\n"
        + "\n".join(f"- {item}" for item in LIMITATIONS)
        + "\n\n## Provenance and replay\n\n"
        f"Snapshot fetched at: `{manifest['observed_data'].get('fetched_at')}`. "
        f"Upstream last-updated field: `{manifest['observed_data'].get('source_lastupdated')}`. "
        "Neither identifies historical vintages available at forecast origins.\n\n"
        "`input_metadata.json` retains source URLs, units, transformations and checksums. "
        "The portable `.pmz` retains the input metadata in the levels frame's attrs and every result table. "
        "`manifest.json` authenticates exported artifacts and numerical sources.\n"
    )
    (output / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research_output/cross_country_forecasting"))
    args = parser.parse_args()
    result = run_application(args.output)
    print(result["pooled_accuracy"].to_string(index=False))
    print(f"Latest-vintage forecast evidence written to {args.output}")


if __name__ == "__main__":
    main()
