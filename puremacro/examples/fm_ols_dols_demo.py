"""Phillips-Hansen FM-OLS and Stock-Watson DOLS vs naive OLS on
a known cointegrating DGP.

DGP:
    x_t = x_{t-1} + nu_t,    nu_t ~ N(0, 1)        (I(1) regressor)
    y_t = beta * x_t + u_t,  u_t correlated with nu_t (endogenous)

OLS on a cointegrating regression is super-consistent but biased in
finite samples whenever Δx and u are correlated. FM-OLS and DOLS
remove that bias by either (FM) correcting with a long-run covariance
estimator or (DOLS) augmenting the regression with leads/lags of Δx.

Run:
    python -m puremacro.examples.fm_ols_dols_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cointegration_modern import fm_ols, dols, phillips_ouliaris


def _simulate(T: int = 300, beta: float = 2.0, rho: float = 0.5,
              seed: int = 0):
    """Endogeneity: u_t = rho * nu_t + xi_t with iid xi_t."""
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(T)
    xi = rng.standard_normal(T) * 0.7
    u = rho * nu + xi
    x = np.cumsum(nu)
    y = beta * x + u
    return y, x, beta


def run_demo(reps: int = 100, T: int = 300) -> dict:
    beta_true = 2.0
    ols_betas = []
    fm_betas = []
    dols_betas = []
    pos_z_t = []
    for r in range(reps):
        y, x, _ = _simulate(T=T, beta=beta_true, rho=0.5, seed=r)
        Z = np.column_stack([np.ones(T), x])
        b_ols = float(np.linalg.lstsq(Z, y, rcond=None)[0][1])
        b_fm = float(fm_ols(y, x).beta[0])
        b_dols = float(dols(y, x, leads=2, lags=2).beta[0])
        ols_betas.append(b_ols)
        fm_betas.append(b_fm)
        dols_betas.append(b_dols)
        pos_z_t.append(phillips_ouliaris(y, x).z_t)
    return {
        "beta_true": beta_true,
        "ols": np.array(ols_betas),
        "fm": np.array(fm_betas),
        "dols": np.array(dols_betas),
        "po_z_t": np.array(pos_z_t),
        "reps": reps,
    }


def main() -> None:
    out = run_demo()
    bt = out["beta_true"]
    print(f"Cointegrating regression (true β = {bt}), {out['reps']} Monte Carlo replications")
    print(f"  OLS:    mean = {out['ols'].mean():.4f}, "
          f"bias = {out['ols'].mean() - bt:+.4f}, "
          f"sd = {out['ols'].std(ddof=0):.4f}")
    print(f"  FM-OLS: mean = {out['fm'].mean():.4f}, "
          f"bias = {out['fm'].mean() - bt:+.4f}, "
          f"sd = {out['fm'].std(ddof=0):.4f}")
    print(f"  DOLS:   mean = {out['dols'].mean():.4f}, "
          f"bias = {out['dols'].mean() - bt:+.4f}, "
          f"sd = {out['dols'].std(ddof=0):.4f}")
    print()
    print(f"  Phillips-Ouliaris Z_t: mean {out['po_z_t'].mean():.2f}  "
          f"(more negative ⇒ stronger evidence of cointegration; "
          f"5% CV ≈ −3.37 for k=1)")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        bins_arr = np.linspace(min(out["ols"].min(), out["fm"].min(), out["dols"].min()),
                               max(out["ols"].max(), out["fm"].max(), out["dols"].max()),
                               25)
        bins: list[float] = bins_arr.tolist()
        ax.hist(out["ols"], bins=bins, alpha=0.55, color="0.55",
                label="OLS")
        ax.hist(out["fm"], bins=bins, alpha=0.55, color="0.20",
                label="FM-OLS")
        ax.hist(out["dols"], bins=bins, alpha=0.55, color="0.0",
                label="DOLS")
        ax.axvline(bt, color="0.0", lw=0.8, ls="--", label=f"true β={bt}")
        ax.set_xlabel("Estimated cointegrating coefficient")
        ax.set_ylabel("Frequency")
        ax.set_title("OLS / FM-OLS / DOLS Monte Carlo distribution")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fm_ols_dols.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'fm_ols_dols.png'}")


if __name__ == "__main__":
    main()
