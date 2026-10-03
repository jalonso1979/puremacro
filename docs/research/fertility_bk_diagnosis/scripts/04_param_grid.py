"""04 — Where does n_stable = 12 hold ROBUSTLY?

Robustness at a parameter point means all of:
  (i)   companion-QZ n_stable == 12 (the package's own rule),
  (ii)  min | |mu| - 1 | over the finite spectrum > 1e-3,
  (iii) classification identical for central-diff h = 1e-6 (package
        default) and complex-step (exact) Jacobians.

Baseline calibration: the Bayesian .mod + parameters.mat values with an
exact steady state re-solved at every grid point (Dynare's `steady;`
semantics) — this is the calibration the port is supposed to replicate.

Outputs
-------
output/04_sensitivity_1d.csv     per-parameter 1-D sweeps
output/04_grid_beta_omega.csv    the 2-D grid of the two most influential
figures/04_sweep_1d.pdf
figures/04_heatmap.pdf
output/04_port_rescue.csv        can any single dial rescue the port point?
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as cm

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "output"
FIG = HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

MOD2_SS = {"c": 0.266225, "k": 1.03382, "i": 0.114395, "u": 1.13744,
           "n": 1.06629, "b": 0.0672904, "l_o": 0.474782, "l_w": 0.26552,
           "y": 0.481514}


def z_from_ss(params, ss):
    z = np.empty(cm.N_VARS)
    full = dict(ss)
    full["a"] = params["bara"]; full["mun"] = params["barn"]; full["ph"] = params["barp"]
    for i, name in enumerate(cm.VAR_NAMES):
        z[i] = full[name]
    return z


def evaluate(overrides: dict, z_warm: np.ndarray | None) -> dict:
    """Exact SS -> Jacobians (cs + central h=1e-6) -> spectrum + robustness."""
    p = cm.build_params({**cm.MOD2_PARAMS, **overrides})
    guess = z_warm if z_warm is not None else z_from_ss(p, MOD2_SS)
    try:
        z = cm.exact_steady_state(p, guess)
    except RuntimeError:
        try:
            z = cm.exact_steady_state(p, z_from_ss(p, MOD2_SS))
        except RuntimeError as e:
            return {"ok": False, "err": str(e)[:50], "z": None}
    A, B, C, _ = cm.build_ABCD(p, z, method="cs")
    qz_cs = cm.companion_qz(A, B, C)
    roots, _ = cm.det_roots(A, B, C)
    parts = cm.split_roots(roots, p)
    nz = np.abs(roots[np.abs(roots) > 1e-7])
    margin = float(np.min(np.abs(nz - 1.0)))
    Ah, Bh, Ch, _ = cm.build_ABCD(p, z, method="central", h=1e-6)
    qz_h = cm.companion_qz(Ah, Bh, Ch)
    endo = np.abs(parts["endogenous"])
    return {
        "ok": True, "z": z,
        "n_stable_cs": qz_cs["n_stable"], "n_stable_h6": qz_h["n_stable"],
        "margin": margin,
        "robust": (qz_cs["n_stable"] == 12 and qz_h["n_stable"] == 12
                   and margin > 1e-3),
        "endo": endo,
    }


def main() -> None:
    # ---------------- 1-D sweeps at the healthy calibration ----------------
    sweeps = {
        "beta":    np.linspace(0.85, 0.995, 16),
        "psik":    np.linspace(0.25, 10.0, 16),
        "psin":    np.linspace(0.25, 10.0, 16),
        "omega":   np.linspace(1.15, 3.5, 16),
        "delta_k": np.linspace(0.06, 0.30, 16),
        "rhoa":    np.linspace(0.05, 0.99, 16),
        "rhon":    np.linspace(0.05, 0.99, 16),
        "rhop":    np.linspace(0.05, 0.99, 16),
    }
    rows = []
    influence = {}
    for pname, grid in sweeps.items():
        z_warm = None
        margins, bounds = [], []
        for val in grid:
            r = evaluate({pname: float(val)}, z_warm)
            if r["ok"]:
                z_warm = r["z"]
                # boundary root: the endo root closest to 1
                broot = float(r["endo"][np.argmin(np.abs(r["endo"] - 1))])
                rows.append({"param": pname, "value": float(val), "ok": True,
                             "n_stable": r["n_stable_cs"], "margin": r["margin"],
                             "robust": r["robust"], "boundary_root": broot})
                margins.append(r["margin"]); bounds.append(broot)
            else:
                rows.append({"param": pname, "value": float(val), "ok": False,
                             "n_stable": np.nan, "margin": np.nan,
                             "robust": False, "boundary_root": np.nan})
        if bounds:
            influence[pname] = float(np.nanmax(bounds) - np.nanmin(bounds))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "04_sensitivity_1d.csv", index=False)

    print("influence ranking (range of the boundary-root modulus over sweep):")
    for k, v in sorted(influence.items(), key=lambda kv: -kv[1]):
        sub = df[(df.param == k) & df.ok]
        n12 = int((sub.n_stable == 12).sum())
        print(f"  {k:8s} boundary-root range = {v:.4f}   "
              f"n_stable==12 at {n12}/{len(sub)} points")
    top2 = [k for k, _ in sorted(influence.items(), key=lambda kv: -kv[1])[:2]]
    print(f"\ntwo most influential: {top2}")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for ax, pname in zip(axes.ravel(), sweeps):
        sub = df[(df.param == pname) & df.ok]
        ax.plot(sub.value, sub.boundary_root, "o-", ms=3, lw=1)
        ax.axhline(1.0, color="#cc3311", ls="--", lw=0.8)
        ax.set_title(pname, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Boundary-root modulus, 1-D sweeps around the healthy "
                 "(Bayesian .mod) calibration, exact SS", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "04_sweep_1d.pdf")
    plt.close(fig)

    # ---------------- 2-D heat map over the two most influential -----------
    # (fixed to beta x omega if they win, else whatever ranked top-2)
    p1, p2 = top2
    g1 = np.linspace(sweeps[p1][0], sweeps[p1][-1], 21)
    g2 = np.linspace(sweeps[p2][0], sweeps[p2][-1], 21)
    grid_rows = []
    Zmask = np.full((len(g2), len(g1)), np.nan)
    Zmargin = np.full((len(g2), len(g1)), np.nan)
    z_warm = None
    for jj, v2 in enumerate(g2):
        z_row_start = None
        for ii, v1 in enumerate(g1):
            r = evaluate({p1: float(v1), p2: float(v2)}, z_warm)
            if r["ok"]:
                z_warm = r["z"]
                if ii == 0:
                    z_row_start = r["z"]
                Zmask[jj, ii] = 1.0 if r["robust"] else 0.0
                Zmargin[jj, ii] = r["margin"] if r["n_stable_cs"] == 12 else -r["margin"]
                grid_rows.append({p1: v1, p2: v2, "ok": True,
                                  "n_stable": r["n_stable_cs"],
                                  "margin": r["margin"], "robust": r["robust"]})
            else:
                grid_rows.append({p1: v1, p2: v2, "ok": False,
                                  "n_stable": np.nan, "margin": np.nan,
                                  "robust": False})
        z_warm = z_row_start  # warm-start next row from this row's first col
    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(OUT / f"04_grid_{p1}_{p2}.csv", index=False)
    n_rob = int(gdf.robust.sum())
    print(f"\n2-D grid {p1} x {p2}: {n_rob}/{len(gdf)} points ROBUSTLY "
          f"determinate (n_stable=12, margin>1e-3, h-invariant)")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    pc = ax.pcolormesh(g1, g2, Zmargin, shading="nearest", cmap="RdYlGn",
                       vmin=-np.nanmax(np.abs(Zmargin)),
                       vmax=np.nanmax(np.abs(Zmargin)))
    fig.colorbar(pc, ax=ax, label="signed margin  (+ : n_stable=12, - : BK fails)")
    bad = gdf[~gdf.ok]
    if len(bad):
        ax.scatter(bad[p1], bad[p2], marker="x", s=24, color="k", label="no SS found")
        ax.legend(fontsize=8, loc="upper left")
    ax.scatter([cm.MOD2_PARAMS[p1]], [cm.MOD2_PARAMS[p2]], marker="*", s=180,
               color="#004488", edgecolor="w", zorder=5)
    ax.annotate("Bayesian .mod calibration", (cm.MOD2_PARAMS[p1], cm.MOD2_PARAMS[p2]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel(p1); ax.set_ylabel(p2)
    ax.set_title(f"BK determinacy map ({p1} x {p2}), exact SS at every point\n"
                 "green: unique stable solution; red: BK fails; margin = "
                 "min | |mu|-1 |", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "04_heatmap.pdf")
    fig.savefig(FIG / "04_heatmap.png", dpi=160)
    plt.close(fig)

    # ---------------- can a single dial rescue the PORT point? -------------
    # (dynamics-only overrides at the port's frozen LM linearisation point,
    # exactly what solve_fertility(params={...}) can reach today)
    print("\nport-point rescue sweeps (no SS repair possible there):")
    port_params = cm.build_params()
    z_port = cm.z_from_params(port_params)
    rescue_rows = []
    for pname, grid in [("beta", np.linspace(0.80, 0.99, 20)),
                        ("omega", np.linspace(1.2, 3.5, 20)),
                        ("psik", np.linspace(0.25, 10, 20)),
                        ("psin", np.linspace(0.25, 10, 20))]:
        n12 = 0
        for val in grid:
            p = cm.build_params({pname: float(val)})
            A, B, C, _ = cm.build_ABCD(p, z_port, method="cs")
            qz = cm.companion_qz(A, B, C)
            rescue_rows.append({"param": pname, "value": float(val),
                                "n_stable": qz["n_stable"]})
            n12 += int(qz["n_stable"] == 12)
        print(f"  {pname:6s}: n_stable==12 at {n12}/20 grid points")
    pd.DataFrame(rescue_rows).to_csv(OUT / "04_port_rescue.csv", index=False)


if __name__ == "__main__":
    main()
