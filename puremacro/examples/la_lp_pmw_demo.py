"""Lag-augmented LP (Plagborg-Møller-Wolf 2021) vs Newey-West Jordà LP.

PMW 2021 show that augmenting the LP with extra lags of x and y makes
the *standard* heteroskedasticity-robust SE valid uniformly in horizon —
no HAC, no horizon-dependent bandwidth. The cost is one extra lag per
horizon; the benefit is that long-horizon coverage stops degrading.

This demo simulates an AR(1) x process and a true IRF that decays
geometrically, then compares LA-LP and Newey-West LP at horizons
0..20. We expect:
  - similar point estimates (both consistent);
  - SE that DON'T blow up at long horizons under LA-LP, because no
    horizon-dependent HAC bandwidth is needed (PMW 2021).
The point isn't that LA-LP is always *narrower*; it is that LA-LP
gives uniformly valid coverage without ad-hoc bandwidth choices.

Run:
    python -m puremacro.examples.la_lp_pmw_demo

Español
-------
Proyección local aumentada con rezagos (Plagborg-Møller-Wolf 2021)
frente a LP de Jordà con Newey-West.

PMW 2021 demuestran que aumentar la proyección local con rezagos
adicionales de x e y hace que los errores estándar *estándar* robustos
a la heterocedasticidad sean válidos de manera uniforme en el horizonte —
sin HAC, sin ancho de banda dependiente del horizonte. El costo es un
rezago adicional por horizonte; el beneficio es que la cobertura en
horizontes largos deja de deteriorarse.

Esta demostración simula un proceso AR(1) para x y una función de
impulso-respuesta verdadera que decae geométricamente, y luego compara
LA-LP y LP con Newey-West en los horizontes 0..20. Se espera:
  - estimaciones puntuales similares (ambos son consistentes);
  - errores estándar que NO explotan en horizontes largos bajo LA-LP,
    porque no se necesita un ancho de banda HAC dependiente del horizonte
    (PMW 2021).
El punto no es que LA-LP sea siempre *más estrecho*; es que LA-LP
ofrece cobertura uniformemente válida sin elecciones de ancho de banda
ad hoc.

Run:
    python -m puremacro.examples.la_lp_pmw_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..lp.jorda import lp_hac
from ..lp.la_lp import la_lp


def _simulate(T: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rho_x = 0.7
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_x * x[t - 1] + rng.standard_normal() * 0.6
    rho_y = 0.5
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = rho_y * y[t - 1] + 0.5 * x[t] + rng.standard_normal() * 0.5
    idx = pd.date_range("1980", periods=T, freq="QE")
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def run_demo(H: int = 20, n_lags: int = 4) -> dict:
    df = _simulate()
    horizons = range(0, H + 1)
    nw = lp_hac(df, y="y", x="x", horizons=horizons, n_lags=n_lags)
    la = la_lp(df, y="y", x="x", horizons=horizons, n_lags=n_lags)
    return {"data": df, "newey_west": nw, "la_lp": la}


def main() -> None:
    out = run_demo()
    nw = out["newey_west"].set_index("h")
    la = out["la_lp"].set_index("h")
    print("Lag-augmented LP (PMW 2021) vs Newey-West LP")
    print(f"  T = {len(out['data'])},  horizons 0..{nw.index.max()}")
    print()
    print("    h    beta_NW    SE_NW    beta_LA    SE_LA   SE ratio (LA/NW)")
    for h in nw.index:
        b_nw = nw.loc[h, "beta"]; s_nw = nw.loc[h, "se"]
        b_la = la.loc[h, "beta"]; s_la = la.loc[h, "se"]
        ratio = s_la / s_nw if s_nw > 0 else float("nan")
        print(f"  {h:>3d}   {b_nw:+.3f}    {s_nw:.3f}    "
              f"{b_la:+.3f}    {s_la:.3f}    {ratio:.2f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        h_arr = nw.index.values
        ax.plot(h_arr, nw["beta"], color="0.4", lw=1.0, label="Newey-West LP")
        ax.fill_between(h_arr, nw["lo"], nw["hi"], color="0.4", alpha=0.15)
        ax.plot(h_arr, la["beta"], color="0.0", lw=1.0,
                label="Lag-augmented LP (PMW)")
        ax.fill_between(h_arr, la["lo"], la["hi"], color="0.0", alpha=0.15)
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("β_h")
        ax.set_title("LP impulse response: NW vs lag-augmented (PMW)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "la_lp_pmw.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'la_lp_pmw.png'}")


if __name__ == "__main__":
    main()
