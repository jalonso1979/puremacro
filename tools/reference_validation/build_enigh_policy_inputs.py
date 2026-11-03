"""Freeze selected public ENIGH columns without fabricating household records.

Run from the repository root with a downloaded INEGI archive as the argument.
The raw archive stays outside the package. Its hash and the exact retained
variables are recorded. All source rows, including zero-spending households,
are retained; the study declares its positive-spending analysis domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from puremacro.fetch.micro.inegi import read_table, enigh_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("puremacro/trade/_datafiles"))
    args = parser.parse_args()
    payload = args.archive.read_bytes()
    data = read_table(payload)
    variables = ["factor", "ing_cor", "gasto_mon", "alimentos", "tot_integ"]
    values = data[variables].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("source contains missing, nonfinite or negative values")
    if np.any(data.factor <= 0) or np.any(data.tot_integ <= 0) or np.any(data.alimentos > data.gasto_mon):
        raise ValueError("source violates weight, household-size or spending identities")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "enigh2022_household_policy.npz"
    # No object arrays or pickle; source row position identifies every record.
    np.savez_compressed(path, **{name: data[name].to_numpy(dtype=float) for name in variables},
                        est_dis=data.est_dis.to_numpy(dtype=str), upm=data.upm.to_numpy(dtype=str))
    with zipfile.ZipFile(args.archive) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError("record exact CSV selection before accepting a different archive")
        csv_hash = hashlib.sha256(z.read(csvs[0])).hexdigest()
    w = data.factor.to_numpy()
    domain = data.gasto_mon.to_numpy() > 0
    record = {
        "schema": 1, "source": "INEGI ENIGH 2022 nueva serie, concentradohogar",
        "source_url": enigh_url(2022), "source_archive_sha256": hashlib.sha256(payload).hexdigest(),
        "source_csv_sha256": csv_hash, "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "variables": variables + ["est_dis", "upm"], "source_rows": len(data),
        "population_households": float(w.sum()), "analysis_rows": int(domain.sum()),
        "excluded_zero_spending_rows": int((~domain).sum()),
        "excluded_zero_spending_population": float(w[~domain].sum()),
        "weighted_means_all_rows": {name: float(np.average(data[name], weights=w)) for name in variables[1:4]},
        "monetary_unit": "MXN at August 2022 prices", "period": "quarter",
        "population_unit": "households", "weight": "factor", "stratum": "est_dis", "psu": "upm",
        "reference": "https://www.inegi.org.mx/programas/enigh/nc/2022/",
        "terms": "https://www.inegi.org.mx/inegi/terminos.html",
        "attribution": "Fuente: INEGI. ENIGH 2022 nueva serie. Transformaciones: PureMacro.",
        "transformation": "retain listed columns in original CSV row order; no imputation, rescaling or sampling",
        "is_synthetic": False,
    }
    (args.output / "enigh2022_household_policy.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["source_rows", "analysis_rows", "excluded_zero_spending_population", "payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
