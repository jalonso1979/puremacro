"""Rebuild the frozen GDP panel by replaying authenticated World Bank responses.

Run from the repository root with ``PYTHONPATH=.``. This development tool is
offline: it replaces only the HTTP transport, then runs the public WDI panel
builder. A separate, direct extraction checks every returned observation.
Revisions require reviewing new raw responses and deliberately updating hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from puremacro.fetch.wdi import wdi_panel

COUNTRIES = ("USA", "MEX", "BRA")
FETCHED_AT = "2026-10-11T02:25:05+00:00"
SOURCES = (
    {
        "file": "wdi_countries.json",
        "url": "https://api.worldbank.org/v2/country?format=json&per_page=400&page=1",
        "sha256": "c9acf3932785191a649cbc04c3056af4cf64a236e5c3b49d49be25ddb0440bce",
    },
    {
        "file": "wdi_gdp.json",
        "url": "https://api.worldbank.org/v2/country/BRA;MEX;USA/indicator/NY.GDP.MKTP.KN?format=json&date=1960:2024&per_page=32767&page=1",
        "sha256": "13aae503fd11572c2c886fe43613c23c6a7ef78882032233c1609e90cda616c7",
    },
)


def extract(source_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Replay the reviewed source bytes and require a complete positive panel."""
    responses = {}
    for source in SOURCES:
        raw = (source_dir / source["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Unreviewed source SHA256: {source['file']}")
        responses[source["url"]] = raw

    def replay(url, *args, **kwargs):
        if url not in responses:
            raise AssertionError(f"Unrecorded HTTP request: {url}")
        return responses[url]

    # Replay at the network boundary; parsing, units, country selection, and
    # all panel-builder checks execute unchanged, with no live fallback.
    with patch("puremacro._http.safe_get_bytes_cached", side_effect=replay):
        panel = wdi_panel(["gdp_real"], COUNTRIES, start=1960, end=2024, pause=0)
    if panel.attrs.get("missing"):
        raise ValueError(f"Incomplete WDI fetch: {panel.attrs['missing']}")
    dates = pd.date_range("1960-01-01", "2024-01-01", freq="YS")
    expected = pd.MultiIndex.from_product([sorted(COUNTRIES), dates], names=["code", "date"])
    if not panel.index.equals(expected) or list(panel.columns) != ["gdp_real"]:
        raise ValueError("Expected exactly three countries and 65 annual observations each")
    if not np.isfinite(panel.to_numpy()).all() or not (panel.to_numpy() > 0).all():
        raise ValueError("GDP must be finite and strictly positive; no interpolation is permitted")

    header, records = json.loads(responses[SOURCES[1]["url"]])
    if header["sourceid"] != "2" or int(header["total"]) != 195 or int(header["pages"]) != 1:
        raise ValueError("Unexpected WDI source or pagination")
    direct = {}
    for row in records:
        if row["indicator"]["id"] != "NY.GDP.MKTP.KN":
            raise ValueError("Unexpected GDP indicator")
        key = (row["countryiso3code"], pd.Timestamp(int(row["date"]), 1, 1))
        if key in direct:
            raise ValueError(f"Duplicate source observation: {key}")
        direct[key] = float(row["value"]) / 1_000_000
    oracle = np.array([direct[key] for key in panel.index])
    np.testing.assert_allclose(panel.gdp_real.to_numpy(), oracle, rtol=1e-15, atol=0)
    panel.attrs["fetched_at"] = FETCHED_AT
    panel.attrs["missing"] = ()
    panel.attrs["complete"] = True
    panel.attrs["splice_seams"] = []
    panel.attrs["splice_policy"] = "One WDI indicator; no provider splicing or gap filling"
    panel.attrs["vintage_semantics"] = "Latest revised WDI edition captured at fetched_at; not historical information sets"
    metadata = {
        "schema_version": 1,
        "countries": list(COUNTRIES),
        "start_year": 1960,
        "end_year": 2024,
        "source": {
            "name": "World Bank World Development Indicators (source 2)",
            "url": SOURCES[1]["url"],
            "indicator": "NY.GDP.MKTP.KN",
            "license": "CC BY 4.0",
            "license_url": "https://datacatalog.worldbank.org/public-licenses#cc-by",
            "units": "millions of constant LCU, country-specific base/reference year",
        },
        "fetched_at": FETCHED_AT,
        "source_lastupdated": header["lastupdated"],
        "panel_attrs": panel.attrs,
        "raw_sources": list(SOURCES),
        "transformations": [
            "Raw constant-LCU levels divided by 1,000,000 by wdi_panel",
            "Annual GDP growth = 100 * (log GDP_t - log GDP_(t-1))",
            "No interpolation, seasonal adjustment, cross-country level comparison, or source splicing",
        ],
        "evaluation_design": {
            "target": "annual percent log GDP growth",
            "first_target_year": 2000,
            "last_target_year": 2024,
            "horizon_years": 1,
            "training": "expanding window, observations through target year minus one only",
            "models": ["zero_growth", "historical_mean", "ar1"],
            "minimum_training_growth_observations": 20,
            "model_selection": "none; AR lag order fixed at one before evaluation",
            "interpretation": "Latest-vintage historical forecast exercise, not real-time or causal evidence",
        },
        "independent_data_check": "Every scaled level compared with a direct extraction of the authenticated raw JSON",
    }
    return panel, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("reviews/2026-10-10-cross-country-forecasting/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("puremacro/datasets/data"))
    args = parser.parse_args()
    panel, metadata = extract(args.source_dir)
    csv = panel.reset_index().to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n").encode("utf-8")
    metadata["csv_sha256"] = hashlib.sha256(csv).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_country_gdp.csv").write_bytes(csv)
    (args.output_dir / "cross_country_gdp_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(panel)} observations; CSV SHA256 {metadata['csv_sha256']}")


if __name__ == "__main__":
    main()
