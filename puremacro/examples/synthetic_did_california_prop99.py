"""Synthetic Difference-in-Differences causal policy evaluation (Arkhangelsky et al. 2021).

Applies SDID to evaluate the causal impact of a policy intervention on a state/firm
panel, illustrating the combination of unit weights omega and time weights lambda
with affine intercept adjustment.

Run:
    python -m puremacro.examples.synthetic_did_california_prop99

Español
-------
Evaluación causal de políticas mediante Diferencias en Diferencias Sintéticas (Arkhangelsky et al. 2021).

Aplica SDID para evaluar el impacto causal de una intervención de política en un
panel de estados o empresas, ilustrando la combinación de ponderaciones de unidades
omega y ponderaciones temporales lambda con ajuste de intercepto.

Ejecución:
    python -m puremacro.examples.synthetic_did_california_prop99
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..did import synthetic_did


def run_sdid_simulation(N_donors: int = 15, T_periods: int = 20, T_treat: int = 12, seed: int = 2021) -> dict:
    rng = np.random.default_rng(seed)
    N_units = N_donors + 1

    time_trend = np.linspace(10, 25, T_periods)
    macro_factor = np.sin(np.linspace(0, 3 * np.pi, T_periods)) * 3.0

    unit_fixed_effects = rng.uniform(5, 20, size=N_units)
    factor_loadings = rng.uniform(0.5, 2.0, size=N_units)
    factor_loadings[0] = 1.4

    tau_true = -4.5

    records = []
    for i in range(N_units):
        for t in range(T_periods):
            y_it = unit_fixed_effects[i] + time_trend[t] + factor_loadings[i] * macro_factor[t] + rng.normal(0, 0.4)
            if (i == 0) and (t >= T_treat):
                y_it += tau_true
            records.append({
                "unit": f"State_{i:02d}",
                "time": t,
                "y": y_it,
                "treat_time": T_treat if i == 0 else np.nan,
            })

    df_panel = pd.DataFrame(records)
    res = synthetic_did(
        df=df_panel,
        unit="unit",
        time="time",
        outcome="y",
        treat_time="treat_time",
        n_boot=100,
        seed=seed,
    )

    return {
        "df": df_panel,
        "res": res,
        "tau_true": tau_true,
        "T_treat": T_treat,
        "T_periods": T_periods,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Synthetic Difference-in-Differences Demo...")
    out = run_sdid_simulation()
    res = out["res"]

    print(f"  SDID Estimated Tau: {res.tau:.4f} (True: {out['tau_true']:.4f})")
    print(f"  Bootstrap SE:       {res.se:.4f}")
    print(f"  90% CI:             [{res.lo:.4f}, {res.hi:.4f}]")

    piv = out["df"].pivot(index="time", columns="unit", values="y")
    treated_traj = piv["State_00"].values
    donor_matrix = piv.drop(columns=["State_00"]).values

    synthetic_path = donor_matrix @ res.omega
    naive_path = donor_matrix.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [2.2, 1.2]})

    # Panel 1: Trajectories
    ax = axes[0]
    time_axis = np.arange(out["T_periods"])
    ax.plot(time_axis, treated_traj, color="0.00", lw=2.0, label="Treated State")
    ax.plot(time_axis, synthetic_path, color="0.40", ls="--", lw=1.8, label=r"SDID Synthetic Control ($\hat{\omega}$)")
    ax.plot(time_axis, naive_path, color="0.75", ls=":", lw=1.5, label="Naive DiD Average")
    ax.axvline(out["T_treat"] - 0.5, color="0.60", ls="-.", lw=1.0, label="Intervention")
    ax.set_title("(a) State Outcome Trajectories", loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Outcome")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Panel 2: Donor weights
    ax_w = axes[1]
    top_donors = res.omega.nlargest(6).iloc[::-1]
    ax_w.barh(top_donors.index, top_donors.values, color="0.30", edgecolor="0.00")
    ax_w.set_title(r"(b) Top Donor Weights $\hat{\omega}_i$", loc="left", fontsize=9.5, fontweight="bold")
    ax_w.set_xlabel("Weight")

    fig.tight_layout()
    fig_path = out_dir / "synthetic_did_prop99.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
