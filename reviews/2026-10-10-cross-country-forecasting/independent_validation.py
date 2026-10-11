"""Independently reproduce the frozen study from raw WDI JSON, offline.

Run ``.venv/bin/python reviews/2026-10-10-cross-country-forecasting/independent_validation.py``.
No puremacro code, panel builder, or forecast helper is imported. GDP growth is
computed as log level ratios, and AR coefficients use SciPy's SVD least squares.
The study being checked must already exist. Assertions fail without writing a
passing validation report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import lstsq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COUNTRIES = ("BRA", "MEX", "USA")
MODELS = ("zero_growth", "historical_mean", "ar1")
SOURCE_HASHES = {
    "wdi_countries.json": "c9acf3932785191a649cbc04c3056af4cf64a236e5c3b49d49be25ddb0440bce",
    "wdi_gdp.json": "13aae503fd11572c2c886fe43613c23c6a7ef78882032233c1609e90cda616c7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def raw_levels(source_dir: Path) -> dict[tuple[str, int], float]:
    """Authenticate the archived responses and extract unscaled GDP directly."""
    for filename, digest in SOURCE_HASHES.items():
        require(sha256(source_dir / filename) == digest, f"Raw source changed: {filename}")
    country_header, countries = json.loads((source_dir / "wdi_countries.json").read_text())
    require(int(country_header["pages"]) == 1, "Country response is incomplete")
    country_lookup = {row["id"]: row for row in countries}
    require(all(code in country_lookup for code in COUNTRIES), "Unknown country codes")
    header, records = json.loads((source_dir / "wdi_gdp.json").read_text())
    require(header["sourceid"] == "2", "Wrong WDI source")
    require(int(header["pages"]) == 1 and int(header["total"]) == len(records) == 195,
            "Raw response is incomplete")
    output = {}
    for row in records:
        code, year = row["countryiso3code"], int(row["date"])
        require(code in COUNTRIES and 1960 <= year <= 2024, "Unexpected country/year")
        require(row["country"]["id"] == country_lookup[code]["iso2Code"], "ISO codes differ")
        require(row["indicator"]["id"] == "NY.GDP.MKTP.KN", "Wrong GDP indicator")
        require((code, year) not in output, "Duplicate observation")
        value = float(row["value"])
        require(math.isfinite(value) and value > 0, "Missing or nonpositive GDP")
        output[code, year] = value
    require(set(output) == {(code, year) for code in COUNTRIES for year in range(1960, 2025)},
            "Expected complete 3-country, 65-year grid")
    return output


def independent_tables(raw: dict[tuple[str, int], float]) -> dict[str, pd.DataFrame]:
    """Each forecast is independently fitted from its raw data prefix only."""
    level_rows, growth_rows, forecast_rows, coverage_rows, exclusion_rows = [], [], [], [], []
    for code in COUNTRIES:
        for year in range(1960, 2025):
            level_rows.append({"code": code, "date": f"{year}-01-01", "gdp_real": raw[code, year] / 1e6})
        growth = {year: 100 * math.log(raw[code, year] / raw[code, year - 1])
                  for year in range(1961, 2025)}
        growth_rows.extend({"code": code, "year": year, "growth": value} for year, value in growth.items())
        for target in range(2000, 2025):
            # Rebuild the training series from the raw prefix for every origin;
            # neither target nor future observations enter either fit.
            train = np.array([100 * math.log(raw[code, year] / raw[code, year - 1])
                              for year in range(1961, target)])
            x = np.stack((np.ones(len(train) - 1), train[:-1]), axis=1)
            beta, _, rank, _ = lstsq(x, train[1:], lapack_driver="gelss")
            require(rank == 2, f"Unidentified independent AR: {code} {target}")
            predictions = {"zero_growth": 0.0,
                           "historical_mean": math.fsum(train) / len(train),
                           "ar1": float(beta[0] + beta[1] * train[-1])}
            for model in MODELS:
                forecast_rows.append({
                    "code": code, "model": model, "origin_year": target - 1,
                    "target_year": target, "horizon_years": 1, "training_start_year": 1961,
                    "training_end_year": target - 1, "training_growth_observations": target - 1961,
                    "actual": growth[target], "forecast": predictions[model],
                    "error": growth[target] - predictions[model],
                })
        coverage_rows.append({
            "code": code, "first_level_year": 1960, "last_level_year": 2024,
            "level_observations": 65, "growth_observations": 64,
            "first_training_growth_year": 1961, "initial_training_growth_observations": 39,
            "evaluation_start_year": 2000, "evaluation_end_year": 2024,
            "evaluation_observations_per_model": 25, "excluded_evaluation_observations": 0,
        })
        exclusion_rows.extend({"code": code, "year": year,
                               "reason": "first_level_has_no_previous_year" if year == 1960
                               else "before_fixed_evaluation_window"}
                              for year in range(1960, 2000))
    forecasts = pd.DataFrame(forecast_rows)
    accuracy_rows, pooled_rows = [], []
    for code in COUNTRIES:
        for model in MODELS:
            errors = [row["error"] for row in forecast_rows if row["code"] == code and row["model"] == model]
            accuracy_rows.append({"code": code, "model": model, "n_forecasts": len(errors),
                                  **losses(errors)})
    for model in MODELS:
        # With 25 observations per country, pooled individual errors equal
        # equal-country means of MSE, MAE, and mean error.
        errors = [row["error"] for row in forecast_rows if row["model"] == model]
        pooled_rows.append({"model": model, "n_countries": 3, "n_forecasts": len(errors), **losses(errors)})
    return {"levels": pd.DataFrame(level_rows), "growth": pd.DataFrame(growth_rows),
            "forecasts": forecasts, "accuracy": pd.DataFrame(accuracy_rows),
            "pooled_accuracy": pd.DataFrame(pooled_rows), "coverage": pd.DataFrame(coverage_rows),
            "exclusions": pd.DataFrame(exclusion_rows)}


def losses(errors: list[float]) -> dict[str, float]:
    n = len(errors)
    return {"rmse": math.sqrt(math.fsum(error * error for error in errors) / n),
            "mae": math.fsum(abs(error) for error in errors) / n,
            "mean_error": math.fsum(errors) / n}


def compare(observed: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> dict:
    require(set(observed.columns) == set(expected.columns), "Table columns differ")
    require(not observed.duplicated(keys).any(), "Duplicate table keys")
    observed = observed.sort_values(keys).reset_index(drop=True)[expected.columns]
    expected = expected.sort_values(keys).reset_index(drop=True)
    require(len(observed) == len(expected), "Table row counts differ")
    differences = {}
    for column in expected:
        if expected[column].dtype.kind == "f":
            actual, wanted = observed[column].to_numpy(), expected[column].to_numpy()
            require(np.isfinite(actual).all(), f"Nonfinite values in {column}")
            # Log ratios versus differences of logged levels differ by rounding.
            np.testing.assert_allclose(actual, wanted, rtol=2e-13, atol=5e-11, err_msg=column)
            differences[column] = float(np.max(np.abs(actual - wanted)))
        else:
            require(observed[column].tolist() == expected[column].tolist(), f"Column differs: {column}")
    return {"rows": len(expected), "maximum_absolute_differences": differences, "passed": True}


def validate(study_dir: Path, source_dir: Path, dataset_dir: Path) -> dict:
    raw = raw_levels(source_dir)
    tables = independent_tables(raw)
    keys = {"levels": ["code", "date"], "growth": ["code", "year"],
            "forecasts": ["code", "model", "target_year"], "accuracy": ["code", "model"],
            "pooled_accuracy": ["model"], "coverage": ["code"], "exclusions": ["code", "year"]}
    comparisons = {name: compare(pd.read_csv(study_dir / f"{name}.csv", float_precision="round_trip"), frame, keys[name])
                   for name, frame in tables.items()}
    packaged = pd.read_csv(dataset_dir / "cross_country_gdp.csv", float_precision="round_trip")
    comparisons["packaged_levels"] = compare(packaged, tables["levels"], keys["levels"])
    # Require exact binary64 scaled levels as well as tolerance-based comparison.
    np.testing.assert_array_equal(packaged.sort_values(keys["levels"]).gdp_real,
                                  tables["levels"].sort_values(keys["levels"]).gdp_real)

    metadata = json.loads((dataset_dir / "cross_country_gdp_metadata.json").read_text())
    manifest = json.loads((study_dir / "manifest.json").read_text())
    require(sha256(dataset_dir / "cross_country_gdp.csv") == metadata["csv_sha256"], "Packaged CSV hash differs")
    require(metadata == json.loads((study_dir / "input_metadata.json").read_text()) == manifest["observed_data"],
            "Provenance differs across packaged input and study exports")
    require({row["file"]: row["sha256"] for row in metadata["raw_sources"]} == SOURCE_HASHES,
            "Metadata raw source hashes differ")
    require(metadata["source"]["indicator"] == "NY.GDP.MKTP.KN", "Metadata indicator differs")
    require(metadata["source"]["units"] == "millions of constant LCU, country-specific base/reference year",
            "Metadata scale/units differ")
    source_header = json.loads((source_dir / "wdi_gdp.json").read_text())[0]
    require(metadata["source_lastupdated"] == source_header["lastupdated"], "Upstream vintage field differs")
    require(metadata["panel_attrs"]["missing"] == [] and metadata["panel_attrs"]["complete"] is True,
            "Metadata claims an incomplete panel")
    require(manifest["is_real_time"] is False and manifest["is_causal"] is False, "Statistical labels differ")
    design = manifest["design"]
    require(set(design["countries"]) == set(COUNTRIES) and tuple(design["models"]) == MODELS,
            "Study countries/models differ")
    require((design["level_start_year"], design["level_end_year"], design["evaluation_start_year"],
             design["evaluation_end_year"], design["horizon_years"], design["ar_lags"],
             design["minimum_training_growth_observations"]) == (1960, 2024, 2000, 2024, 1, 1, 20),
            "Numerical design differs")
    require(design["ar_intercept"] is True and design["window"] == "expanding", "AR/window design differs")
    require(design["transformation"] == "100 * diff(log(gdp_real))", "Growth definition differs")
    require(design["error_convention"] == "actual minus forecast", "Error sign differs")
    report = (study_dir / "report.md").read_text()
    for wording in ("latest-vintage", "does not reconstruct information available in real time",
                    "publication lags", "descriptive", "no significance test", "causal claims",
                    "not establish general forecasting superiority", "log-growth percentage points"):
        require(wording in report, f"Missing interpretation limit: {wording}")
    require(set(manifest["artifacts"]) == {f"{name}.csv" for name in tables} |
            {"input_metadata.json", "cross_country_forecasting.pmz", "forecast_accuracy.png", "report.md"},
            "Manifest artifact inventory differs")
    for filename, record in manifest["artifacts"].items():
        path = study_dir / filename
        require(sha256(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                f"Artifact authentication failed: {filename}")
    for filename, digest in manifest["source_sha256"].items():
        require(sha256(ROOT / "puremacro" / filename) == digest, f"Study source changed: {filename}")

    return {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_directory": str(study_dir.resolve()), "implementation": "No puremacro imports; direct raw JSON extraction, log ratios, scipy.linalg.lstsq(lapack_driver='gelss'), math.fsum losses",
        "independence_scope": "Separate numerical implementation within development; not independent external adoption or peer review",
        "counts": {"countries": 3, "level_observations": 195, "growth_observations": 192,
                   "country_target_pairs": 75, "forecasts": 225, "country_model_accuracy_rows": 9},
        "table_comparisons": comparisons,
        "no_look_ahead": {
            "status": "all 225 exported forecasts match independently fitted origin-specific prefixes",
            "latest_training_growth_year": "target year minus one",
            "first_training_growth_year": 1961,
            "initial_training_growth_observations": 39,
            "scope": "Numerical equality on frozen data plus source review; package tests separately exercise future-data perturbation and prefix invariance",
        },
        "statistical_labels": {
            "latest_vintage": True, "real_time": False, "publication_lags_modeled": False,
            "causal": False, "significance_tests_claimed": False,
            "loss_units": "annual log-growth percentage points",
            "equal_country_weighting_verified": True,
            "all_predeclared_target_years_and_models_reported": True,
        },
        "review_limits": [
            "Source hashes authenticate the archived bytes, not their truth or historical availability.",
            "Source inspection found prefix-only fitting with fixed model specifications and no full-sample preprocessing.",
            "The design document states it preceded fitting; this numerical check cannot independently establish that timing.",
            "This script authenticates the cartridge bytes but leaves cartridge deserialization and installed-wheel replay to separate checks.",
        ],
        "source_sha256": SOURCE_HASHES,
        "evidence_sha256": {"manifest.json": sha256(study_dir / "manifest.json"),
                            "DESIGN.md": sha256(HERE / "DESIGN.md"),
                            "builder": sha256(ROOT / "tools/build_cross_country_forecasting_data.py"),
                            "validator": sha256(Path(__file__))},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=HERE / "study")
    parser.add_argument("--source-dir", type=Path, default=HERE / "source")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "puremacro/datasets/data")
    parser.add_argument("--output", type=Path, default=HERE / "validation.json")
    args = parser.parse_args()
    result = validate(args.study_dir, args.source_dir, args.dataset_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Passed: 195 raw levels and 225 independent forecasts; evidence: {args.output}")


if __name__ == "__main__":
    main()
