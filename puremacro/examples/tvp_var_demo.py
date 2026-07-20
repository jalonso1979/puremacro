"""Time-varying-parameter VAR (Primiceri-style FFBS Gibbs).

Simulate a 2-variable VAR(1) whose autoregressive matrix shifts at the
midpoint of the sample, then fit a TVP-VAR-CT (constant Σ) and recover
the time-varying coefficient path. The recovered β̂_t should track the
true switch (smoothed by the random-walk prior on β).

Run:
    python -m puremacro.examples.tvp_var_demo

Español
-------
VAR con parámetros variables en el tiempo (Gibbs FFBS al estilo Primiceri).

Se simula un VAR(1) de 2 variables cuya matriz autorregresiva cambia en el
punto medio de la muestra; luego se estima un TVP-VAR-CT (Σ constante) y se
recupera la trayectoria de coeficientes variables en el tiempo. El β̂_t
recuperado debería rastrear el cambio verdadero (suavizado por la
distribución a priori de caminata aleatoria sobre β).

Ejecución:
    python -m puremacro.examples.tvp_var_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.tvp import tvp_var_fit


def _simulate(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Coefficient sequence: equation 0 has [const, A_00, A_01], same for eq 1.
    # Pre: [0, 0.5, 0.0]; Post: [0, 0.2, 0.1] for eq 0.
    Y = np.zeros((T, 2))
    A_pre = np.array([[0.5, 0.0], [0.05, 0.4]])
    A_post = np.array([[0.2, 0.1], [0.0, 0.7]])
    cut = T // 2
    for t in range(1, T):
        A = A_pre if t < cut else A_post
        Y[t] = A @ Y[t - 1] + rng.standard_normal(2) * 0.4
    return Y


def run_demo() -> dict:
    Y = _simulate()
    res = tvp_var_fit(Y, p=1, n_draws=150, burn=150,
                       rng=np.random.default_rng(0))
    beta_mean = res.beta_draws.mean(axis=0)         # (T_eff, k_state)
    beta_lo = np.percentile(res.beta_draws, 5, axis=0)
    beta_hi = np.percentile(res.beta_draws, 95, axis=0)
    return {"Y": Y, "res": res,
            "beta_mean": beta_mean, "beta_lo": beta_lo, "beta_hi": beta_hi}


def main() -> None:
    out = run_demo()
    res = out["res"]; bm = out["beta_mean"]
    # Coefficient layout: per equation [const, A_*0, A_*1] → 3 entries.
    # eq 0: indices 0..2; eq 1: indices 3..5.
    pre, post = 30, len(bm) - 30
    print("TVP-VAR(1) Gibbs (constant Σ) on a coefficient-switching DGP")
    print(f"  Posterior draws: {res.beta_draws.shape[0]}, "
          f"T_eff = {res.T_eff}")
    print()
    print("Equation 0 lag coefficients [A_00, A_01]:")
    print(f"  early (t≈{pre}):   "
          f"{np.round(bm[pre, [1, 2]], 3).tolist()}  (true [0.5, 0.0])")
    print(f"  late  (t≈{post}):  "
          f"{np.round(bm[post, [1, 2]], 3).tolist()}  (true [0.2, 0.1])")
    print()
    print("Equation 1 lag coefficients [A_10, A_11]:")
    print(f"  early (t≈{pre}):   "
          f"{np.round(bm[pre, [4, 5]], 3).tolist()}  (true [0.05, 0.4])")
    print(f"  late  (t≈{post}):  "
          f"{np.round(bm[post, [4, 5]], 3).tolist()}  (true [0.0, 0.7])")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
        idx = np.arange(len(out["beta_mean"]))
        # Equation 0: A_00 (index 1) and A_01 (index 2).
        axes[0].plot(idx, out["beta_mean"][:, 1], color="0.0", lw=1.0,
                     label="β̂_{0,0} (A_00)")
        axes[0].fill_between(idx, out["beta_lo"][:, 1], out["beta_hi"][:, 1],
                              color="0.0", alpha=0.15)
        axes[0].plot(idx, out["beta_mean"][:, 2], color="0.5", lw=1.0,
                     label="β̂_{0,1} (A_01)")
        axes[0].fill_between(idx, out["beta_lo"][:, 2], out["beta_hi"][:, 2],
                              color="0.5", alpha=0.20)
        axes[0].axvline(len(out["beta_mean"]) // 2, color="0.7", lw=0.6, ls="--")
        axes[0].axhline(0.5, color="0.0", lw=0.4, ls=":")
        axes[0].axhline(0.2, color="0.0", lw=0.4, ls=":")
        axes[0].set_title("Equation 0: time-varying lag coefficients (median + 90%)")
        axes[0].legend(frameon=False, loc="best")

        axes[1].plot(idx, out["beta_mean"][:, 4], color="0.0", lw=1.0,
                     label="β̂_{1,0} (A_10)")
        axes[1].fill_between(idx, out["beta_lo"][:, 4], out["beta_hi"][:, 4],
                              color="0.0", alpha=0.15)
        axes[1].plot(idx, out["beta_mean"][:, 5], color="0.5", lw=1.0,
                     label="β̂_{1,1} (A_11)")
        axes[1].fill_between(idx, out["beta_lo"][:, 5], out["beta_hi"][:, 5],
                              color="0.5", alpha=0.20)
        axes[1].axvline(len(out["beta_mean"]) // 2, color="0.7", lw=0.6, ls="--")
        axes[1].axhline(0.4, color="0.0", lw=0.4, ls=":")
        axes[1].axhline(0.7, color="0.0", lw=0.4, ls=":")
        axes[1].set_title("Equation 1: time-varying lag coefficients")
        axes[1].set_xlabel("t")
        axes[1].legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "tvp_var.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'tvp_var.png'}")


if __name__ == "__main__":
    main()
