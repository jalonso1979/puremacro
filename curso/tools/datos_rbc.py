#!/usr/bin/env python3
"""Datos del laboratorio «RBC»: momentos del ciclo de EE. UU. a la Kydland–Prescott.

Cinco series trimestrales de FRED congeladas en data_curso/: PIB real (GDPC1), consumo
real (PCECC96), inversión privada bruta real (GPDIC1), horas del sector de negocios no
agrícola (HOANBS) y producto por hora del mismo sector (OPHNFB). Para cada muestra se
recorta ANTES de filtrar, se toma 100·log y se aplica HP con λ = 1600. Se guardan
σ_x (%), σ_x/σ_y, corr(x, y), la autocorrelación de orden 1 y corr(horas, productividad).

    python tools/datos_rbc.py [--datos curso/notebooks/data_curso]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
SERIES = {"y": "GDPC1", "c": "PCECC96", "i": "GPDIC1", "h": "HOANBS", "p": "OPHNFB"}
MUESTRAS = {  # clave: (inicio, fin, etiqueta es, etiqueta en)
    "1948-2019": ("1948-01-01", "2019-10-01", "1948T1–2019T4", "1948Q1–2019Q4"),
    "1984-2019": ("1984-01-01", "2019-10-01", "1984T1–2019T4", "1984Q1–2019Q4"),
}


def hp_ciclo(y, lam=1600.0):
    """Componente cíclico de Hodrick–Prescott: y − τ con (I + λ K'K) τ = y."""
    n = len(y)
    K = np.zeros((n - 2, n))
    for t in range(n - 2):
        K[t, t:t + 3] = (1.0, -2.0, 1.0)
    tau = np.linalg.solve(np.eye(n) + lam * K.T @ K, y)
    return y - tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=AQUI.parent / "notebooks" / "data_curso")
    ap.add_argument("--salida", type=Path, default=AQUI.parent / "site" / "data" / "rbc.json")
    a = ap.parse_args()

    niveles = {}
    for clave, sid in SERIES.items():
        d = pd.read_csv(a.datos / f"{sid}.csv", parse_dates=["observation_date"]).dropna()
        niveles[clave] = d.set_index("observation_date")[sid].astype(float)
    df = pd.DataFrame(niveles).dropna()

    out = {"fuente": {
        "es": "FRED: PIB real (GDPC1, BEA), consumo personal real (PCECC96, BEA), inversión privada bruta real "
              "(GPDIC1, BEA), horas del sector de negocios no agrícola (HOANBS, BLS) y producto por hora del mismo "
              "sector (OPHNFB, BLS). Componente cíclico HP (λ = 1600) de 100 × log, recortando la muestra antes de filtrar.",
        "en": "FRED: real GDP (GDPC1, BEA), real personal consumption (PCECC96, BEA), real gross private domestic "
              "investment (GPDIC1, BEA), nonfarm business hours (HOANBS, BLS) and output per hour in the same sector "
              "(OPHNFB, BLS). HP cyclical component (λ = 1600) of 100 × log, trimming the sample before filtering."},
        "muestras": {}}

    for clave, (ini, fin, et_es, et_en) in MUESTRAS.items():
        sub = df.loc[ini:fin]
        cyc = pd.DataFrame({v: hp_ciclo(100 * np.log(sub[v].to_numpy())) for v in SERIES}, index=sub.index)
        sd = cyc.std(ddof=1)
        mom = {}
        for v in SERIES:
            mom[v] = {
                "sd": round(float(sd[v]), 4),
                "rel": round(float(sd[v] / sd["y"]), 4),
                "corr": round(float(cyc[v].corr(cyc["y"])), 4),
                "ac1": round(float(cyc[v].autocorr(1)), 4),
            }
        out["muestras"][clave] = {
            "etiqueta": {"es": et_es, "en": et_en},
            "n": int(len(sub)),
            "momentos": mom,
            "corr_h_p": round(float(cyc["h"].corr(cyc["p"])), 4),
        }
        print(f"{clave}: {len(sub)} trimestres, {sub.index[0]:%Y-%m} a {sub.index[-1]:%Y-%m}")
        for v in SERIES:
            m = mom[v]
            print(f"  {v}: sd={m['sd']:.2f}  rel={m['rel']:.2f}  corr={m['corr']:.2f}  ac1={m['ac1']:.2f}")
        print(f"  corr(h, y/h) = {out['muestras'][clave]['corr_h_p']:.2f}")

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"→ {a.salida} ({a.salida.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
