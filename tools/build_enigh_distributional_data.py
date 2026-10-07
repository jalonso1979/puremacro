"""Rebuild the small ENIGH 2024 distributional application dataset.

Download the official workbook named in SOURCE_URL, then run::

    python tools/build_enigh_distributional_data.py --workbook /path/to/file.xlsx

Only this build step needs openpyxl. The installed loader reads CSV and JSON.
Unknown workbook revisions are rejected; review the source before changing the
digest. Amounts are converted from thousands of pesos to pesos per *all*
households in the income decile, not per household reporting a positive expense.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/"
    "enigh2024_ns_basicos_tabulados.xlsx"
)
SOURCE_SHA256 = "7af7850495255fb1e6a9cb139cf531e5a62c7fd2a9a1de5ba0503f03aa82490b"
# Zero-based heading rows; each heading is followed by HOGARES and GASTO.
CATEGORIES = {
    "food": (8, "ALIMENTOS, BEBIDAS Y TABACO"),
    "clothing": (62, "VESTIDO Y CALZADO"),
    "housing": (83, "VIVIENDA Y SERVICIOS DE CONSERVACIÓN"),
    "household_goods": (98, "ARTÍCULOS Y SERVICIOS PARA LA LIMPIEZA"),
    "health": (110, "CUIDADOS DE LA SALUD"),
    "transport": (137, "TRANSPORTE; ADQUISICIÓN"),
    "education_recreation": (155, "SERVICIOS DE EDUCACIÓN"),
    "personal": (167, "CUIDADOS PERSONALES"),
}


def extract(workbook: Path) -> tuple[pd.DataFrame, dict]:
    """Extract authenticated source tables and verify their accounting totals."""
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Unrecognized ENIGH workbook SHA256; review the revision before rebuilding")
    expense = pd.read_excel(workbook, sheet_name="Cuadro 4.2", header=None)
    income = pd.read_excel(workbook, sheet_name="Cuadro 3.2", header=None)
    deciles = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for table in (expense, income):
        if table.iloc[4, 3:13].tolist() != deciles or "Miles de pesos" not in table.iloc[2, 0]:
            raise ValueError("ENIGH table labels or units differ from the reviewed source")
    population = income.iloc[6, 3:13].to_numpy(dtype=float)
    if not np.isclose(population.sum(), float(income.iloc[6, 2]), rtol=0, atol=0):
        raise ValueError("Income-decile household counts do not aggregate to the national total")
    data = {"decile": deciles, "households": population.astype(int)}
    for key, (row, heading) in CATEGORIES.items():
        if not str(expense.iloc[row, 0]).startswith(heading):
            raise ValueError(f"ENIGH expenditure heading changed: {key}")
        totals = expense.iloc[row + 2, 3:13].to_numpy(dtype=float)
        if not np.isclose(totals.sum(), float(expense.iloc[row + 2, 2]), atol=1e-3, rtol=0):
            raise ValueError(f"ENIGH decile amounts do not aggregate: {key}")
        data[key] = 1000 * totals / population
    data["outward_transfers"] = 1000 * expense.iloc[181, 3:13].to_numpy(dtype=float) / population
    data["monetary_expenditure"] = 1000 * expense.iloc[7, 3:13].to_numpy(dtype=float) / population
    # Monetary remuneration only: exclude in-kind remuneration (row 34),
    # business income, imputed rent and transfers. These are exposures, not a
    # decomposition of the monetary-consumption budget.
    wage_rows = [16, 19, 22, 25, 28, 31]
    data["cash_wages"] = 1000 * income.iloc[wage_rows, 3:13].to_numpy(dtype=float).sum(axis=0) / population
    result = pd.DataFrame(data)
    if not np.isfinite(result.select_dtypes("number")).all().all() or (result.select_dtypes("number") < 0).any().any():
        raise ValueError("Nonfinite or negative ENIGH amounts")
    consumption = result[list(CATEGORIES)].sum(axis=1)
    if not np.allclose(consumption + result.outward_transfers, result.monetary_expenditure, atol=1e-6, rtol=0):
        raise ValueError("Consumption categories plus outward transfers do not recover monetary spending")
    metadata = {
        "source": "INEGI ENIGH 2024, Tabulados basicos, Cuadros 3.2 and 4.2",
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "accessed": "2026-10-01",
        "survey_year": 2024,
        "monetary_unit": "MXN",
        "period": "quarter",
        "population_unit": "households",
        "value_semantics": "MXN per household per quarter, averaged over ALL households in each income decile",
        "decile_definition": "National household deciles ordered by total current quarterly income",
        "population_source": "Cuadro 3.2, HOGARES under INGRESO CORRIENTE (38,830,230 households)",
        "national_monetary_expenditure_mxn": float(expense.iloc[7, 2]) * 1000,
        "national_food_expenditure_mxn": float(expense.iloc[10, 2]) * 1000,
        "categories": {k: heading for k, (_, heading) in CATEGORIES.items()},
        "source_rows_1based": {k: row + 3 for k, (row, _) in CATEGORIES.items()},
        "cash_wage_rows_1based": [r + 1 for r in wage_rows],
        "transformations": [
            "Multiply published totals in thousands of MXN by 1000, divide by all-household decile expansion counts",
            "Exclude outward expenditure transfers from the eight-category consumption basket; retain separately",
            "Sum six monetary subordinate-employment remuneration categories for cash_wages; exclude in-kind remuneration",
        ],
        "limitations": [
            "Group means do not recover within-decile heterogeneity or survey sampling covariance",
            "Broad consumption categories are not an industry concordance or measured import shares",
            "Monetary consumption excludes imputed rent, own production and other nonmonetary consumption",
            "No trade elasticity, pass-through, policy shock, or factor-price response is identified by these tables",
            "Applying income changes to consumption assumes the user-specified spending response; the example spends all incremental cash wages",
        ],
        "is_synthetic": False,
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "puremacro/datasets/data")
    args = parser.parse_args()
    data, metadata = extract(args.workbook)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "enigh2024_deciles.csv"
    data.to_csv(csv_path, index=False, float_format="%.12g")
    metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (args.output / "enigh2024_deciles_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(data)} income deciles and source metadata to {args.output}")


if __name__ == "__main__":
    main()
