#!/usr/bin/env python3
"""Datos del laboratorio «Filtros»: PIB real trimestral, 100·log, por país.

EE. UU. usa GDPC1 (BEA vía FRED, desde 1947). El resto sale de los bloques de
cuentas nacionales trimestrales de la OCDE congelados en data_curso/qna_gasto.csv,
como volumen = 100 × nominal / deflactor (misma definición que _datos.volumen).

    python tools/datos_filtros.py [--datos curso/notebooks/data_curso]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PAISES = {  # código: (español, inglés)
    "USA": ("Estados Unidos", "United States"), "MEX": ("México", "Mexico"), "CAN": ("Canadá", "Canada"),
    "BRA": ("Brasil", "Brazil"), "CHL": ("Chile", "Chile"), "COL": ("Colombia", "Colombia"),
    "ESP": ("España", "Spain"), "DEU": ("Alemania", "Germany"), "GBR": ("Reino Unido", "United Kingdom"),
    "JPN": ("Japón", "Japan"), "KOR": ("Corea del Sur", "South Korea"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=AQUI.parent / "notebooks" / "data_curso")
    ap.add_argument("--salida", type=Path, default=AQUI.parent / "site" / "data" / "filtros.json")
    a = ap.parse_args()

    out = {"fuente": {
        "es": "EE. UU.: BEA, PIB real (GDPC1, vía FRED). Resto: OCDE, cuentas nacionales trimestrales "
              "desestacionalizadas, volumen = 100 × nominal / deflactor.",
        "en": "United States: BEA real GDP (GDPC1, via FRED). Others: OECD quarterly national accounts, "
              "seasonally adjusted, volume = 100 × nominal / deflator."}, "paises": {}}

    us = pd.read_csv(a.datos / "GDPC1.csv", parse_dates=["observation_date"]).dropna()
    series = {"USA": (us["observation_date"], us["GDPC1"])}
    q = pd.read_csv(a.datos / "qna_gasto.csv", parse_dates=["date"])
    for code in PAISES:
        if code == "USA":
            continue
        d = q[q["code"] == code].sort_values("date")
        vol = (100 * d["gdp"] / d["gdp_defl"]).where(lambda s: s > 0)
        keep = vol.notna()
        series[code] = (d.loc[keep, "date"], vol[keep])

    for code, (dates, level) in series.items():
        y = 100 * np.log(level.to_numpy(dtype=float))
        out["paises"][code] = {
            "nombre": {"es": PAISES[code][0], "en": PAISES[code][1]},
            "fechas": [f"{t:%Y-%m}" for t in dates],
            "y": [round(v, 4) for v in y],
        }
        print(f"{code}: {len(y)} trimestres, {dates.iloc[0]:%Y-%m} a {dates.iloc[-1]:%Y-%m}")
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"→ {a.salida} ({a.salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
