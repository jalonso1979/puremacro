#!/usr/bin/env python3
"""Datos del laboratorio «Desigualdad»: referencias contra las que se lee el Aiyagari.

El modelo se resuelve en el navegador; aquí solo se congelan tres referencias:

* Participaciones en la riqueza neta de los hogares de EUA, último trimestre de las
  Distributional Financial Accounts de la Reserva Federal (cuatro series de FRED ya
  guardadas en data_curso/), con el Gini de la poligonal de Lorenz de cuatro grupos:
  una cota INFERIOR del Gini verdadero, porque supone riqueza uniforme dentro de cada grupo.
* La solución del curso por grillas endógenas (data_curso/a6_aiyagari_egm.npz, 1000 nodos,
  la de figuras_curso.aiyagari_a6 y del mazo A6), para que la página diga contra qué se valida.
* El Gini de la riqueza de la SCF 2013 que reportan Kuhn y Ríos-Rull (2016).

    python tools/datos_desigualdad.py [--datos curso/notebooks/data_curso]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DFA = {"b50": "WFRBSB50215", "p50_90": "WFRBSN40188", "p90_99": "WFRBSN09161", "top1": "WFRBST01134"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=AQUI.parent / "notebooks" / "data_curso")
    ap.add_argument("--salida", type=Path, default=AQUI.parent / "site" / "data" / "desigualdad.json")
    a = ap.parse_args()

    shares, fechas = {}, set()
    for k, sid in DFA.items():
        s = pd.read_csv(a.datos / f"{sid}.csv", parse_dates=["observation_date"]).dropna()
        shares[k] = float(s[sid].iloc[-1])
        fechas.add(s["observation_date"].iloc[-1])
    assert len(fechas) == 1, f"las cuatro series DFA no terminan en la misma fecha: {fechas}"
    fecha = fechas.pop()
    assert abs(sum(shares.values()) - 100) < 0.5, shares
    pop = np.array([0.0, 0.50, 0.90, 0.99, 1.0])
    val = np.cumsum([0.0, shares["b50"], shares["p50_90"], shares["p90_99"], shares["top1"]]) / 100
    gini_min = 1.0 - float(np.sum(np.diff(pop) * (val[1:] + val[:-1])))

    egm = np.load(a.datos / "a6_aiyagari_egm.npz")
    out = {
        "fuente": {
            "es": f"Datos de EUA: Distributional Financial Accounts de la Reserva Federal (vía FRED), {fecha.year}T{(fecha.month - 1) // 3 + 1}, "
                  "y el Gini de la SCF 2013 según Kuhn y Ríos-Rull (2016). El Gini de las DFA es una cota inferior: supone riqueza "
                  "uniforme dentro de cada grupo. Modelo: calibración del mazo A6 del curso.",
            "en": f"US data: Federal Reserve Distributional Financial Accounts (via FRED), {fecha.year}Q{(fecha.month - 1) // 3 + 1}, "
                  "and the 2013 SCF Gini as reported by Kuhn and Ríos-Rull (2016). The DFA Gini is a lower bound: it assumes equal "
                  "wealth within each group. Model: the course's deck A6 calibration.",
        },
        "dfa": {"fecha": f"{fecha:%Y-%m}", **{k: round(v, 2) for k, v in shares.items()},
                "lorenz_pop": pop.tolist(), "lorenz_val": [round(v, 4) for v in val], "gini_min": round(gini_min, 3)},
        "scf": {"gini": 0.85, "anio": 2013},
        "curso": {"n_a": int(egm["a"].size), "r": round(float(egm["r"]), 6), "K": round(float(egm["K"]), 4),
                  "KY": round(float(egm["K"] / egm["Y"]), 4), "gini": round(float(egm["gini"]), 4)},
    }
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"DFA {fecha:%Y-%m}: {shares}, Gini ≥ {gini_min:.3f}")
    print(f"curso: r = {out['curso']['r']:.4%}, K/Y = {out['curso']['KY']}, Gini = {out['curso']['gini']}")
    print(f"→ {a.salida} ({a.salida.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
