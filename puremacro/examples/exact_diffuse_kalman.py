"""Exact-diffuse Kalman initialisation (Koopman-Durbin 2003).

When a state has no proper prior — for instance the level of a
random-walk state or the trend in a UC decomposition — a finite
``P0`` is wrong and a "very large" P0 (the standard
``diffuse_scale=1e6`` approximation) introduces bias in the early
log-likelihood contributions. Koopman-Durbin's exact-diffuse filter
processes the diffuse stage in closed form, then hands off to the
standard recursion.

This example fits a local-level model

    α_{t+1} = α_t + η_t,    y_t = α_t + ε_t

three ways:
  (a) finite ``P0 = 1.0 * I`` — wrong, the level is *not* close to 0.
  (b) ``diffuse_scale = 1e6`` — the usual approximation.
  (c) ``diffuse_states = [0]`` — the exact algorithm.

The smoothed level paths from (b) and (c) should be visually
indistinguishable; the log-likelihood from (c) drops the early
diffuse contribution and is the reference value to use for
likelihood-based estimation.

Run:
    python -m puremacro.examples.exact_diffuse_kalman
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..state_space import StateSpaceModel, kalman_smoother


def _simulate(T: int = 150, level0: float = 50.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(T)
    alpha[0] = level0
    for t in range(1, T):
        alpha[t] = alpha[t - 1] + rng.standard_normal() * 0.5
    y = alpha + rng.standard_normal(T) * 1.0
    return y, alpha


def run_demo() -> dict:
    y, alpha_true = _simulate()
    model = StateSpaceModel(T=np.array([[1.0]]), Z=np.array([[1.0]]),
                             Q=np.array([[0.25]]), H=np.array([[1.0]]))
    out_finite = kalman_smoother(y, model, P0=1.0 * np.eye(1))
    out_approx = kalman_smoother(y, model)               # diffuse_scale=1e6
    out_exact = kalman_smoother(y, model, diffuse_states=[0])
    return {"y": y, "alpha_true": alpha_true,
            "finite": out_finite, "approx": out_approx, "exact": out_exact}


def main() -> None:
    out = run_demo()
    print("Exact-diffuse Kalman initialisation demo")
    print(f"  T = {len(out['y'])} obs, true level starts at "
          f"{out['alpha_true'][0]:.1f}, drifts as a random walk.")
    print()
    print("Log-likelihood under each initialisation:")
    print(f"  finite P0 = 1·I       : {out['finite']['loglik']:+.3f}  "
          f"(wrong — biases early)")
    print(f"  diffuse_scale = 1e6   : {out['approx']['loglik']:+.3f}  "
          f"(approximate diffuse)")
    print(f"  exact-diffuse (KD03)  : {out['exact']['loglik']:+.3f}  "
          f"(reference for ML)")
    print()
    rmse = lambda x: np.sqrt(np.mean((x - out["alpha_true"]) ** 2))
    print(f"  Smoothed RMSE vs truth (finite):    "
          f"{rmse(out['finite']['a_smooth'].ravel()):.3f}")
    print(f"  Smoothed RMSE vs truth (approx):    "
          f"{rmse(out['approx']['a_smooth'].ravel()):.3f}")
    print(f"  Smoothed RMSE vs truth (exact):     "
          f"{rmse(out['exact']['a_smooth'].ravel()):.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(out["alpha_true"], color="0.7", lw=0.7, label="true α")
        ax.plot(out["finite"]["a_smooth"], color="0.6", lw=0.6,
                ls="-.", label="smoothed (finite P0)")
        ax.plot(out["approx"]["a_smooth"], color="0.0", lw=1.0, ls="--",
                label="smoothed (1e6 approx.)")
        ax.plot(out["exact"]["a_smooth"], color="0.0", lw=1.0,
                label="smoothed (exact diffuse)")
        ax.set_xlabel("t"); ax.set_ylabel("level")
        ax.set_title("Local-level model — three initialisations of the Kalman")
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "exact_diffuse_kalman.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'exact_diffuse_kalman.png'}")


if __name__ == "__main__":
    main()
