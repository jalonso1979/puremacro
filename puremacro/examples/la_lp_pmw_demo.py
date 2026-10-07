"""Lag-augmented LP (Montiel Olea & Plagborg-Møller 2021) vs Newey-West Jordà LP.

Montiel Olea and Plagborg-Møller (2021, *Econometrica* 89(4), 1789-1823)
show that controlling for one more lag of every series than the local
projection needs makes the *standard* heteroskedasticity-robust
(Eicker-Huber-White) SE valid: the regression scores become serially
uncorrelated, so no HAC correction and no horizon-dependent bandwidth are
needed, uniformly over the persistence of the data. The augmentation is a
single lag, the same at every horizon (``extra_lags=1``, the default of
``la_lp``); with ``n_lags=4`` each regression controls for 5 lags of x and y.

This demo simulates an AR(1) shock x_t = 0.7 x_{t-1} + 0.6 e_t and an
outcome y_t = 0.5 y_{t-1} + 0.5 x_t + 0.5 u_t, whose true impulse response
to a unit x innovation is 0.5 (0.7^{h+1} - 0.5^{h+1}) / 0.2, and compares
LA-LP and Newey-West LP at horizons 0..20. We expect:
  - similar point estimates (both consistent);
  - LA-LP standard errors that need no horizon-dependent HAC bandwidth.
The point isn't that LA-LP is always *narrower*; it is that its EHW
intervals are valid without ad-hoc bandwidth choices.

(The file keeps its historical name; "pmw" is not a citation.)

Run:
    python -m puremacro.examples.la_lp_pmw_demo

Español
-------
Proyección local aumentada con rezagos (Montiel Olea y Plagborg-Møller 2021)
frente a LP de Jordà con Newey-West.

Montiel Olea y Plagborg-Møller (2021, *Econometrica* 89(4), 1789-1823)
demuestran que controlar por un rezago más de cada serie de los que necesita
la proyección local hace válido el error estándar *estándar* robusto a la
heterocedasticidad (Eicker-Huber-White): los *scores* de la regresión dejan
de estar serialmente correlacionados, de modo que no hace falta corrección
HAC ni ancho de banda dependiente del horizonte, uniformemente en la
persistencia de los datos. El aumento es un único rezago, el mismo en todos
los horizontes (``extra_lags=1``, el valor por defecto de ``la_lp``); con
``n_lags=4`` cada regresión controla por 5 rezagos de x e y.

Esta demostración simula un choque AR(1) x_t = 0.7 x_{t-1} + 0.6 e_t y un
resultado y_t = 0.5 y_{t-1} + 0.5 x_t + 0.5 u_t, cuya respuesta verdadera a
una innovación unitaria de x es 0.5 (0.7^{h+1} - 0.5^{h+1}) / 0.2, y compara
LA-LP y LP con Newey-West en los horizontes 0..20. Se espera:
  - estimaciones puntuales similares (ambos son consistentes);
  - errores estándar de LA-LP que no necesitan un ancho de banda HAC
    dependiente del horizonte.
El punto no es que LA-LP sea siempre *más estrecho*; es que sus intervalos
EHW son válidos sin elecciones de ancho de banda ad hoc.

(El archivo conserva su nombre histórico; "pmw" no es una cita.)

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


def true_irf(h: int) -> float:
    """Response of y_{t+h} to a unit innovation in x_t in :func:`_simulate`."""
    return 0.5 * (0.7 ** (h + 1) - 0.5 ** (h + 1)) / (0.7 - 0.5)


def run_demo(H: int = 20, n_lags: int = 4) -> dict:
    df = _simulate()
    horizons = range(0, H + 1)
    nw = lp_hac(df, y="y", x="x", horizons=horizons, n_lags=n_lags)
    # Default extra_lags=1: Montiel Olea & Plagborg-Møller's single extra lag.
    la = la_lp(df, y="y", x="x", horizons=horizons, n_lags=n_lags)
    return {"data": df, "newey_west": nw, "la_lp": la}


def main() -> None:
    out = run_demo()
    nw = out["newey_west"].set_index("h")
    la = out["la_lp"].set_index("h")
    spec = out["la_lp"].attrs
    print("Lag-augmented LP (Montiel Olea & Plagborg-Møller 2021) vs Newey-West LP")
    print(f"  T = {len(out['data'])},  horizons 0..{nw.index.max()},  "
          f"LA-LP lags: n_lags={spec['n_lags']} + extra_lags={spec['extra_lags']} "
          f"= {spec['p_aug']}")
    print()
    print("    h    true    beta_NW    SE_NW    beta_LA    SE_LA   SE ratio (LA/NW)")
    for h in nw.index:
        b_nw = nw.loc[h, "beta"]; s_nw = nw.loc[h, "se"]
        b_la = la.loc[h, "beta"]; s_la = la.loc[h, "se"]
        ratio = s_la / s_nw if s_nw > 0 else float("nan")
        print(f"  {h:>3d}   {true_irf(int(h)):+.3f}   {b_nw:+.3f}    {s_nw:.3f}    "
              f"{b_la:+.3f}    {s_la:.3f}    {ratio:.2f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        h_arr = nw.index.values
        ax.plot(h_arr, nw["beta"], color="0.4", lw=1.0, label="Newey-West LP")
        ax.fill_between(h_arr, nw["lo"], nw["hi"], color="0.4", alpha=0.15)
        ax.plot(h_arr, la["beta"], color="0.0", lw=1.0,
                label="Lag-augmented LP (MOPM)")
        ax.fill_between(h_arr, la["lo"], la["hi"], color="0.0", alpha=0.15)
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("β_h")
        ax.plot(h_arr, [true_irf(int(h)) for h in h_arr], color="0.0", lw=0.8,
                ls="--", label="True IRF")
        ax.set_title("LP impulse response: NW vs lag-augmented (MOPM 2021)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "la_lp_pmw.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'la_lp_pmw.png'}")


if __name__ == "__main__":
    main()
