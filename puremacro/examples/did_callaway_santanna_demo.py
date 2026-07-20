"""Callaway-Sant'Anna + Sun-Abraham staggered DiD demo.

Simulates a 50-unit × 12-period panel with three staggered treatment
cohorts (treated at t=4, t=7, never-treated) and a known constant
treatment effect tau = 1.0. Both Callaway-Sant'Anna 2021 and
Sun-Abraham 2021 estimators should recover tau within sampling error.
The ``main()`` printout reports both overall ATTs plus the first five
horizons of the Callaway-Sant'Anna event study.

Run:
    python -m puremacro.examples.did_callaway_santanna_demo

Español
-------
Demostración de diferencias en diferencias escalonadas con
Callaway-Sant'Anna + Sun-Abraham.

Simula un panel de 50 unidades × 12 períodos con tres cohortes de
tratamiento escalonado (tratadas en t=4, t=7, nunca tratadas) y un
efecto de tratamiento constante conocido tau = 1.0. Tanto el estimador
de Callaway-Sant'Anna 2021 como el de Sun-Abraham 2021 deben recuperar
tau dentro del error de muestreo. La salida de ``main()`` reporta ambos
ATT globales más los primeros cinco horizontes del estudio de eventos de
Callaway-Sant'Anna.

Run:
    python -m puremacro.examples.did_callaway_santanna_demo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..did import callaway_santanna, sun_abraham


def _simulate(
    n_units: int = 50,
    T: int = 12,
    cohorts: tuple[float, ...] = (4, 7, np.nan),
    tau: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a balanced staggered-treatment panel.

    Each unit is assigned one of the cohorts at random (uniform). For
    treated units, outcome y_{i,t} = unit_FE + time_FE + tau * 1[t >= g_i] + eps.
    Never-treated units (NaN cohort) have no treatment kick.
    """
    rng = np.random.default_rng(seed)
    assign = rng.choice(len(cohorts), size=n_units, replace=True)
    unit_fe = rng.standard_normal(n_units)
    time_fe = rng.standard_normal(T)
    rows = []
    for i in range(n_units):
        g = cohorts[assign[i]]
        for t in range(1, T + 1):
            treated = 1.0 if (not pd.isna(g)) and t >= g else 0.0
            y = unit_fe[i] + time_fe[t - 1] + tau * treated + 0.3 * rng.standard_normal()
            rows.append({"unit": i, "time": t, "y": y, "treat_time": g})
    return pd.DataFrame(rows)


def run_demo(seed: int = 0) -> dict:
    """Build the panel and run both CS and SA estimators on it.

    Returns a dict with keys ``panel`` (pd.DataFrame), ``cs_result``
    (CallawaySantannaResult), and ``sa_result`` (SunAbrahamResult).
    """
    df = _simulate(seed=seed)
    cs = callaway_santanna(df, unit="unit", time="time", outcome="y",
                            treat_time="treat_time", n_boot=200, seed=seed)
    sa = sun_abraham(df, unit="unit", time="time", outcome="y",
                      treat_time="treat_time", n_boot=200, seed=seed)
    return {"panel": df, "cs_result": cs, "sa_result": sa}


def main() -> None:
    out = run_demo(seed=0)
    cs = out["cs_result"]
    sa = out["sa_result"]
    print("Callaway-Sant'Anna + Sun-Abraham staggered DiD demo")
    print("  Planted treatment effect tau = +1.000")
    print()
    print(f"  CS  ATT (overall): {cs.att_overall:+.3f}")
    print(f"  SA  ATT (overall): {sa.att_overall:+.3f}")
    print()
    print("  CS event-study (first 5 horizons):")
    es = cs.att_event_study.sort_values("event_time").head(5)
    for _, r in es.iterrows():
        print(f"    h={int(r['event_time']):+d}  ATT={r['att']:+.3f}")


if __name__ == "__main__":
    main()
