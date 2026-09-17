#!/usr/bin/env python3
"""Datos del laboratorio «Mercado laboral»: Beveridge, flujos de la CPS y la ENOE en cuatro estados.

Tres bloques, todos desde data_curso/ (congelados, sin red):

* EE. UU., mensual: v = 100·JTSJOL/CLF16OV y u = UNRATE desde dic. 2000 (la definición de
  figuras_curso.fig_beveridge), y las probabilidades de transición de la CPS p_UE (f) y p_EU (s)
  de cps_transiciones_mensuales.csv, alineadas con UNRATE.
* Europa, trimestral: tasa de desempleo y tasa de vacantes de Eurostat de beveridge_uv.csv (29 países;
  se omite la fila USA, que es la tasa de aperturas de JOLTS y ya está arriba en versión mensual).
* México, ENOE: las 16 probabilidades trimestrales p_od (F, I, U, N) de
  enoe_transitions_quarterly_observed.parquet, que vienen por mes de referencia, promediadas por
  trimestre calendario; y las participaciones de F, I, U y N en la población en edad de trabajar de
  enoe_stocks_monthly.csv, también promediadas por trimestre. Se descartan los meses sin matriz
  (feb.–jun. 2020, suspensión de la ENOE) y los de panel incompleto con probabilidades exactamente
  cero (oct. 2024, el último mes del panel rotatorio).

    python tools/datos_mercado_laboral.py [--datos curso/notebooks/data_curso]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PAISES = {  # código: (español, inglés)
    "AUT": ("Austria", "Austria"), "BEL": ("Bélgica", "Belgium"), "BGR": ("Bulgaria", "Bulgaria"),
    "CHE": ("Suiza", "Switzerland"), "CYP": ("Chipre", "Cyprus"), "CZE": ("Chequia", "Czechia"),
    "DEU": ("Alemania", "Germany"), "ESP": ("España", "Spain"), "EST": ("Estonia", "Estonia"),
    "FIN": ("Finlandia", "Finland"), "FRA": ("Francia", "France"), "GBR": ("Reino Unido", "United Kingdom"),
    "GRC": ("Grecia", "Greece"), "HRV": ("Croacia", "Croatia"), "HUN": ("Hungría", "Hungary"),
    "IRL": ("Irlanda", "Ireland"), "ITA": ("Italia", "Italy"), "LTU": ("Lituania", "Lithuania"),
    "LUX": ("Luxemburgo", "Luxembourg"), "LVA": ("Letonia", "Latvia"), "MLT": ("Malta", "Malta"),
    "NLD": ("Países Bajos", "Netherlands"), "NOR": ("Noruega", "Norway"), "POL": ("Polonia", "Poland"),
    "PRT": ("Portugal", "Portugal"), "ROU": ("Rumania", "Romania"), "SVK": ("Eslovaquia", "Slovakia"),
    "SVN": ("Eslovenia", "Slovenia"), "SWE": ("Suecia", "Sweden"),
}
ESTADOS = "FIUN"
COLS = [f"p_{o}{d}" for o in ESTADOS for d in ESTADOS]


def fred(datos, sid):
    df = pd.read_csv(datos / f"{sid}.csv", parse_dates=["observation_date"])
    return df.set_index("observation_date")[sid].astype(float)


def r(x, d):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=AQUI.parent / "notebooks" / "data_curso")
    ap.add_argument("--salida", type=Path, default=AQUI.parent / "site" / "data" / "mercado_laboral.json")
    a = ap.parse_args()

    out = {"fuente": {
        "es": "EE. UU.: BLS, vacantes JOLTS (JTSJOL), fuerza laboral (CLF16OV) y tasa de desempleo (UNRATE), vía FRED; "
              "probabilidades de transición de los flujos brutos de la CPS. Europa: Eurostat, tasa de vacantes (jvs_q_nace2, "
              "secciones B–S) y tasa de desempleo de la EPA. México: INEGI, ENOE, panel rotatorio procesado con "
              "puremacro.labor_flows_enoe (2005–2024). Las vacantes de Eurostat y de JOLTS no tienen el mismo denominador: "
              "compara formas y desplazamientos, no niveles.",
        "en": "United States: BLS, JOLTS job openings (JTSJOL), labor force (CLF16OV) and unemployment rate (UNRATE), via FRED; "
              "transition probabilities from CPS gross flows. Europe: Eurostat job vacancy rate (jvs_q_nace2, sections B–S) and "
              "LFS unemployment rate. Mexico: INEGI, ENOE, rotating panel processed with puremacro.labor_flows_enoe "
              "(2005–2024). Eurostat and JOLTS vacancies use different denominators: compare shapes and shifts, not levels."}}

    # ---- EE. UU.: Beveridge mensual
    bev = pd.concat({"v": 100.0 * fred(a.datos, "JTSJOL") / fred(a.datos, "CLF16OV"),
                     "u": fred(a.datos, "UNRATE")}, axis=1).dropna().loc["2000-12-01":]
    # ---- EE. UU.: flujos de la CPS y desempleo observado
    cps = pd.read_csv(a.datos / "cps_transiciones_mensuales.csv", parse_dates=["date"]).set_index("date")
    fl = cps.join(fred(a.datos, "UNRATE").rename("u"), how="inner").dropna()
    out["eua"] = {
        "nombre": {"es": "Estados Unidos (JOLTS, mensual)", "en": "United States (JOLTS, monthly)"},
        "bev": {"fechas": [f"{t:%Y-%m}" for t in bev.index],
                "u": [r(x, 3) for x in bev["u"]], "v": [r(x, 9) for x in bev["v"]]},
        "flujos": {"fechas": [f"{t:%Y-%m}" for t in fl.index],
                   "f": [r(x, 9) for x in fl["p_UE"]], "s": [r(x, 10) for x in fl["p_EU"]],
                   "u": [r(x, 3) for x in fl["u"]]},
    }
    print(f"EE. UU.: Beveridge {len(bev)} meses ({bev.index[0]:%Y-%m} a {bev.index[-1]:%Y-%m}); "
          f"flujos {len(fl)} meses ({fl.index[0]:%Y-%m} a {fl.index[-1]:%Y-%m})")

    # ---- Europa: Beveridge trimestral
    eu = pd.read_csv(a.datos / "beveridge_uv.csv", parse_dates=["date"])
    eu = eu[(eu["code"] != "USA") & (eu["urate"] > 0) & (eu["vrate"] > 0)].sort_values(["code", "date"])
    out["paises"] = {}
    for code, g in eu.groupby("code"):
        out["paises"][code] = {"nombre": {"es": PAISES[code][0], "en": PAISES[code][1]},
                               "fechas": [f"{t:%Y-%m}" for t in g["date"]],
                               "u": [r(x, 8) for x in g["urate"]], "v": [r(x, 8) for x in g["vrate"]]}
    print(f"Europa: {len(out['paises'])} países, {len(eu)} trimestres-país")

    # ---- México: matrices trimestrales y participaciones
    tp = pd.read_parquet(a.datos / "enoe_transitions_quarterly_observed.parquet").sort_index()[COLS].dropna()
    cero = (tp <= 0).any(axis=1)
    if cero.any():
        print("ENOE: se descartan meses con probabilidades exactamente cero:", [f"{t:%Y-%m}" for t in tp.index[cero]])
    tp = tp[~cero]
    q = tp.groupby(tp.index.to_period("Q"))
    Pq, nq = q.mean(), q.size()
    st = pd.read_csv(a.datos / "enoe_stocks_monthly.csv", parse_dates=["ref_date"]).set_index("ref_date")[list(ESTADOS)]
    sh = st.div(st.sum(axis=1), axis=0)
    shq = sh.groupby(sh.index.to_period("Q")).mean()
    qs = pd.period_range(min(Pq.index.min(), shq.index.min()), max(Pq.index.max(), shq.index.max()), freq="Q")
    mats, shares, counts = [], [], []
    for p in qs:
        if p in Pq.index:
            M = Pq.loc[p].to_numpy().reshape(4, 4)
            M = M / M.sum(axis=1, keepdims=True)
            mats.append([r(x, 9) for x in M.ravel()]); counts.append(int(nq.loc[p]))
        else:
            mats.append(None); counts.append(0)
        shares.append([r(x, 9) for x in shq.loc[p].to_numpy()] if p in shq.index else None)
    out["mex"] = {"trimestres": [f"{p.start_time:%Y-%m}" for p in qs], "estados": list(ESTADOS),
                  "P": mats, "meses": counts, "acervos": shares}
    print(f"México: {len(qs)} trimestres ({qs[0]} a {qs[-1]}), {sum(m is not None for m in mats)} con matriz, "
          f"{sum(s is not None for s in shares)} con acervos")

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"→ {a.salida} ({a.salida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
