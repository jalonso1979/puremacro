"""Labor-force transitions demo using the Shimer 2-state fallback.

Without CPS LNS17 caches, the toolkit falls back to Shimer (2005)
job-finding and separation rates inferred from the UNRATE level
dynamics. This demo simulates a 240-month UNRATE series with two
regimes (high vs low separation), runs ``transitions_from_shimer``,
and then exercises ``supply_demand_decomposition``.

If ``notebooks/data_cache/cps_flows_monthly.parquet`` is present,
prefers the full 3-state CPS path automatically.

Run:
    python -m puremacro.examples.labor_flows_demo

Español
-------
Demo de transiciones en la fuerza laboral usando la alternativa de dos estados
de Shimer.

Sin cachés de CPS LNS17, el toolkit recurre a las tasas de búsqueda de empleo
y de separación de Shimer (2005), inferidas a partir de la dinámica del nivel
de `UNRATE`. Esta demo simula una serie `UNRATE` de 240 meses con dos
regímenes (separación alta frente a baja), ejecuta ``transitions_from_shimer``
y luego ejerce ``supply_demand_decomposition``.

Si ``notebooks/data_cache/cps_flows_monthly.parquet`` está disponible,
el script prefiere automáticamente la ruta completa de tres estados del CPS.

Ejecución:
    python -m puremacro.examples.labor_flows_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..labor_flows import (
    transitions_from_cps_flows,
    transitions_from_shimer,
    supply_demand_decomposition,
)


def _simulate_unrate(T: int = 240, seed: int = 0) -> pd.Series:
    """Two-regime AR(1) UNRATE — first half centred at 5%, second half at 7%."""
    rng = np.random.default_rng(seed)
    target = np.concatenate([np.full(T // 2, 5.0), np.full(T - T // 2, 7.0)])
    u = np.empty(T)
    u[0] = target[0]
    for t in range(1, T):
        u[t] = 0.85 * u[t - 1] + 0.15 * target[t] + rng.standard_normal() * 0.4
    u = np.clip(u, 3.0, 12.0)
    idx = pd.date_range("2000-01-01", periods=T, freq="MS")
    return pd.Series(u, index=idx, name="urate")


def run_demo(seed: int = 0) -> dict:
    """Build a TransitionPanel and supply-demand decomposition.

    Returns a dict with keys ``panel`` (TransitionPanel), ``decomp``
    (pd.DataFrame), and ``path`` ('cps_3state' or 'shimer_2state')
    indicating which input path was taken.
    """
    cache = Path("notebooks/data_cache/cps_flows_monthly.parquet")
    stocks_cache = Path("notebooks/data_cache/cps_stocks_monthly.parquet")
    if cache.exists() and stocks_cache.exists():
        flows = pd.read_parquet(cache)
        stocks = pd.read_parquet(stocks_cache)
        panel = transitions_from_cps_flows(flows, stocks)
        path = "cps_3state"
    else:
        urate = _simulate_unrate(seed=seed)
        panel = transitions_from_shimer(urate)
        path = "shimer_2state"
    decomp = supply_demand_decomposition(panel)
    return {"panel": panel, "decomp": decomp, "path": path}


def main() -> None:
    out = run_demo(seed=0)
    print("Labor-force transitions demo")
    print(f"  Data path: {out['path']}")
    print(f"  Monthly panel rows: {len(out['panel'].monthly)}")
    print()
    m = out["panel"].monthly
    print("  Average monthly transition rates (Shimer 2-state path):")
    if "p_UE" in m.columns:
        print(f"    job-finding rate  p_UE = {m['p_UE'].mean():.3f}")
    if "p_EU" in m.columns:
        print(f"    separation rate   p_EU = {m['p_EU'].mean():.3f}")
    if "p_UN" in m.columns:
        print(f"    p_UN (exit-LF)    = {m['p_UN'].mean():.3f}")
    print()
    d = out["decomp"]
    print("  Supply-demand decomposition (first 3 rows):")
    print(d.head(3).to_string())


if __name__ == "__main__":
    main()
