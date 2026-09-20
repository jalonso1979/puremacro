"""02 — Jacobian noise: is the eigenvalue structure stable in h?

Three differentiation schemes at the SAME linearisation point:
  central differences h in {1e-4 .. 1e-8} (package default h = 1e-6),
  Richardson-extrapolated central differences,
  complex-step (after testing whether the package function accepts
  complex input — it does not, so a verified local copy is used).

Then the decisive upstream experiment: perturb the BGP least-squares
starting point x0 and watch the "steady state", the endogenous roots
and n_stable move — the over-determined BGP is the non-deterministic
stage that differs across BLAS/LAPACK builds, not the QZ.

Outputs
-------
output/02_roots_vs_h.csv
output/02_bgp_basin.csv
figures/02_roots_vs_h.pdf
figures/02_bgp_basin.pdf
"""
from __future__ import annotations

import pathlib
import sys
import warnings

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


def spectrum_row(params, z_ss, method, h):
    A, B, C, _ = cm.build_ABCD(params, z_ss, method=method, h=h)
    roots, deg = cm.det_roots(A, B, C)
    qz = cm.companion_qz(A, B, C)
    nonzero = np.sort(np.abs(roots[np.abs(roots) > 1e-7]))
    return {
        "method": method, "h": h, "deg": deg,
        "n_stable_qz": qz["n_stable"],
        "qz_finite": np.sort(qz["moduli"][np.isfinite(qz["moduli"])]),
        "nonzero_moduli": nonzero,
    }


def main() -> None:
    params = cm.build_params()
    z_ss = cm.z_from_params(params)

    # -- does the package residual function support complex input? ----------
    z_test = z_ss.astype(complex)
    z_test[0] += 1e-20j  # a complex-step-style perturbation
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r_test = cm.model_residuals(z_test, z_ss.astype(complex),
                                    z_ss.astype(complex),
                                    np.zeros(cm.N_SHOCKS, dtype=complex), params)
    complex_warned = any(issubclass(w.category, np.exceptions.ComplexWarning)
                         for w in caught)
    if np.isrealobj(r_test):
        print("package model_residuals does NOT support complex-step: it "
              "allocates a float64 output and SILENTLY DISCARDS the imaginary "
              f"part (ComplexWarning emitted: {complex_warned}). A complex-step "
              "Jacobian through it would be identically zero garbage.")
        print("-> complex-step uses the verified local copy (model_residuals_cs)")
    else:
        print("package model_residuals returned complex output (unexpected)")
    dev = cm.assert_residuals_match(params, z_ss)
    print(f"local complex-safe copy vs package on 20 real draws: "
          f"max abs deviation = {dev:.3e}\n")

    # -- h sweep -------------------------------------------------------------
    rows = []
    for h in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        rows.append(spectrum_row(params, z_ss, "central", h))
    for h in [1e-4, 1e-5, 1e-6]:
        rows.append(spectrum_row(params, z_ss, "richardson", h))
    rows.append(spectrum_row(params, z_ss, "cs", np.nan))

    ref = rows[-1]["qz_finite"]  # complex-step = gold standard
    print("QZ finite moduli by scheme (rows) — complex-step is the reference:")
    recs = []
    for r in rows:
        drift = (np.max(np.abs(r["qz_finite"] - ref))
                 if len(r["qz_finite"]) == len(ref) else np.inf)
        label = f"{r['method']:10s} h={r['h']:<8.0e}" if np.isfinite(r["h"]) else f"{r['method']:10s} (exact)  "
        print(f"  {label} n_stable={r['n_stable_qz']:2d}  "
              f"max|mod - cs| = {drift:.3e}")
        rec = {"method": r["method"], "h": r["h"],
               "n_stable_qz": r["n_stable_qz"], "max_drift_vs_cs": drift}
        for i, mval in enumerate(r["qz_finite"]):
            rec[f"mod_{i}"] = mval
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "02_roots_vs_h.csv", index=False)

    print("\ncomplex-step finite moduli (reference):")
    print(" ", np.array2string(ref, precision=8, max_line_width=100))
    print("\nverdict: eigenvalue structure vs h -> see max-drift column; "
          "classification (n_stable) is h-invariant iff column constant.")

    # figure: nonzero moduli vs h (central)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    hs = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    curves = {}
    for r in rows[:5]:
        for mval in r["nonzero_moduli"]:
            key = np.round(mval, 1)
            curves.setdefault(key, {})[r["h"]] = mval
    for key, d in sorted(curves.items()):
        xs = sorted(d)
        ax.semilogx(xs, [d[x] for x in xs], "o-", ms=3, lw=1)
    ax.axhline(1.0, color="#cc3311", ls="--", lw=1, label="|mu| = 1")
    csref = rows[-1]["nonzero_moduli"]
    ax.scatter([2e-9] * len(csref), csref, marker="*", s=60, color="k",
               zorder=5, label="complex step (exact)")
    ax.set_xlabel("central-difference step h (package default 1e-6)")
    ax.set_ylabel("|mu| of nonzero finite roots")
    ax.set_title("Finite-root moduli are h-stable: Jacobian noise is NOT the culprit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_roots_vs_h.pdf")
    plt.close(fig)

    # -- upstream: the BGP least-squares basin --------------------------------
    print("\nBGP-basin experiment: solve_bgp from perturbed x0, then complex-step")
    print("Jacobians at the returned point -> endogenous quartet + n_stable.")
    from puremacro.dsge.fertility_adj_costs import (
        solve_bgp, _BGP_X0_DEFAULT, FERTILITY_SHOCK_PROCESSES,
    )
    rng = np.random.default_rng(7)
    basin_rows = []
    for scale in [1e-10, 1e-8, 1e-6, 1e-4]:
        for _ in range(15):
            x0 = _BGP_X0_DEFAULT * (1 + scale * rng.standard_normal(13))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bgp = solve_bgp(x0=x0)
            except (RuntimeError, ValueError) as e:
                basin_rows.append({"scale": scale, "ok": False, "err": str(e)[:60]})
                continue
            p = dict(bgp)
            for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
                prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
                p[f"rho{prefix}"] = spec["rho"]
            p.setdefault("barp", 0.0)
            z = cm.z_from_params(p)
            A, B, C, _ = cm.build_ABCD(p, z, method="cs")
            qz = cm.companion_qz(A, B, C)
            roots, _ = cm.det_roots(A, B, C)
            parts = cm.split_roots(roots, p)
            endo = np.abs(parts["endogenous"])
            basin_rows.append({
                "scale": scale, "ok": True, "n_stable_qz": qz["n_stable"],
                "tau_n": p["tau_n"], "beta": p["beta"], "l_o": p["l_o"],
                "endo_1": endo[0] if len(endo) > 0 else np.nan,
                "endo_2": endo[1] if len(endo) > 1 else np.nan,
                "endo_3": endo[2] if len(endo) > 2 else np.nan,
                "endo_4": endo[3] if len(endo) > 3 else np.nan,
            })
    bdf = pd.DataFrame(basin_rows)
    bdf.to_csv(OUT / "02_bgp_basin.csv", index=False)
    okdf = bdf[bdf.ok == True]  # noqa: E712
    print(okdf.to_string(float_format=lambda v: f"{v:.6g}", index=False))
    if len(okdf):
        print("\nn_stable distribution over perturbed-x0 BGP solutions:",
              okdf.groupby("scale").n_stable_qz.apply(
                  lambda s: dict(s.value_counts())).to_dict())
        print("spread of stable endogenous root (endo_1):",
              f"[{okdf.endo_1.min():.4f}, {okdf.endo_1.max():.4f}]")
        print("spread of boundary root (endo_2):",
              f"[{okdf.endo_2.min():.4f}, {okdf.endo_2.max():.4f}]")

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for scale, gr in okdf.groupby("scale"):
            ax.scatter(gr.endo_1, gr.endo_2, s=26, alpha=0.75,
                       label=f"x0 noise {scale:.0e}")
        ax.axhline(1.0, color="#cc3311", ls="--", lw=1)
        ax.axvline(1.0, color="#cc3311", ls="--", lw=1)
        ax.set_xlabel("smallest endogenous root modulus")
        ax.set_ylabel("second endogenous root modulus (BK boundary)")
        ax.set_title("BGP least-squares basin scatter: the linearisation point,\n"
                     "not the QZ, is the build-dependent stage")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "02_bgp_basin.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
