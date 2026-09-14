"""01 — Reproduce the BK failure; dissect the companion-pencil structure.

Outputs
-------
output/01_pencil_eigs.csv      per-eigenvalue (|alpha|, |beta|, modulus, class)
output/01_det_vs_qz.csv        det-interpolation roots vs QZ finite moduli
output/01_singular_values.csv  SVDs of A and of C's lag block
figures/01_roots_plane.pdf     finite roots vs unit circle
figures/01_beta_clusters.pdf   log10 |beta| clusters vs the 1e-12 threshold
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


def main() -> None:
    params = cm.build_params()
    z_ss = cm.z_from_params(params)

    # -- how far is the linearisation point from an actual steady state? --
    resid = cm.model_residuals(z_ss, z_ss, z_ss, np.zeros(cm.N_SHOCKS), params)
    print("residuals at the port's linearisation point (should be 0 at a true SS):")
    for name, r in zip(
        ["production", "invest LoM", "resource", "children LoM", "time",
         "leisure FOC", "capital eff", "cons Euler", "fert Euler",
         "AR a", "AR mun", "AR ph"], resid,
    ):
        print(f"  {name:13s} {r: .6e}")
    print(f"  ||resid||_2 = {np.linalg.norm(resid):.6e}\n")

    # -- Jacobians exactly as the package computes them --------------------
    A, B, C, D = cm.build_ABCD(params, z_ss, method="central", h=1e-6)

    sA = np.linalg.svd(A, compute_uv=False)
    sC = np.linalg.svd(C, compute_uv=False)
    sB = np.linalg.svd(B, compute_uv=False)
    print("singular values of A (lead Jacobian):")
    print(" ", np.array2string(sA, precision=3, max_line_width=100))
    rankA = int(np.sum(sA > 1e-8 * sA[0]))
    rankC = int(np.sum(sC > 1e-8 * sC[0]))
    print(f"numerical rank(A) = {rankA}  (nonzero rows: cons Euler, fert Euler)")
    print("singular values of C (lag Jacobian):")
    print(" ", np.array2string(sC, precision=3, max_line_width=100))
    print(f"numerical rank(C) = {rankC}  (lag columns: a, mun, ph, k, n)")
    print(f"prediction: deg det(A mu^2+B mu+C) = n + rank(A) = {cm.N_VARS + rankA}, "
          f"zeros at origin = n - rank(C) = {cm.N_VARS - rankC}, "
          f"infinite companion eigenvalues = {2 * cm.N_VARS - (cm.N_VARS + rankA)}\n")
    pd.DataFrame({"sv_A": sA, "sv_B": sB, "sv_C": sC}).to_csv(
        OUT / "01_singular_values.csv", index=False)

    # -- companion QZ, package classification rule -------------------------
    qz = cm.companion_qz(A, B, C)
    aa, bb, mods = np.abs(qz["alpha"]), np.abs(qz["beta"]), qz["moduli"]
    order = np.argsort(mods)
    df = pd.DataFrame({
        "abs_alpha": aa[order], "abs_beta": bb[order], "modulus": mods[order],
        "class": np.where(mods[order] < 1, "stable",
                          np.where(np.isinf(mods[order]), "infinite", "unstable")),
    })
    df.to_csv(OUT / "01_pencil_eigs.csv", index=False)
    print("companion QZ (package rule, |beta| > 1e-12 <=> finite):")
    print(df.to_string(float_format=lambda v: f"{v:.6g}"))
    print(f"\nn_stable = {qz['n_stable']} (package expects {cm.N_VARS}) -> "
          f"{'BK FAILS' if qz['n_stable'] != cm.N_VARS else 'BK passes'}")

    fin = bb > cm.BETA_THRESHOLD
    print(f"\n|beta| clusters: finite cluster min = {bb[fin].min():.3e}, "
          f"infinite cluster max = {bb[~fin].max():.3e}, "
          f"threshold = {cm.BETA_THRESHOLD:.0e}")
    print(f"gap ratio (finite min / infinite max) = {bb[fin].min() / max(bb[~fin].max(), 1e-300):.3e}")
    # any pair whose classification could flip with build-dependent rounding?
    borderline = [(a_, b_) for a_, b_ in zip(aa, bb)
                  if 1e-16 < b_ < 1e-8]
    print(f"borderline pairs with 1e-16 < |beta| < 1e-8: {len(borderline)}")
    for a_, b_ in borderline:
        print(f"  |alpha|={a_:.3e}  |beta|={b_:.3e}  |alpha/beta|={a_/b_:.3e}")

    # spurious-stable emulation: would a tiny random perturbation of the
    # pencil (different LAPACK arithmetic) ever produce a 12th stable root?
    rng = np.random.default_rng(42)
    flips = []
    for trial in range(200):
        E1 = rng.standard_normal(qz["M1"].shape) * 1e-13 * np.linalg.norm(qz["M1"], 2)
        E0 = rng.standard_normal(qz["M0"].shape) * 1e-13 * np.linalg.norm(qz["M0"], 2)
        import scipy.linalg as sla
        _, _, al, be, _, _ = sla.ordqz(qz["M1"] + E1, qz["M0"] + E0,
                                       sort=lambda a, b: abs(a) < abs(b))
        with np.errstate(divide="ignore", invalid="ignore"):
            ev = np.where(np.abs(be) > cm.BETA_THRESHOLD, np.abs(al / be), np.inf)
        flips.append(int(np.sum(ev < 1.0)))
    flips = np.array(flips)
    print(f"\nn_stable under 200 random 1e-13-relative pencil perturbations "
          f"(emulating build-dependent arithmetic):")
    print("  counts:", {int(v): int(np.sum(flips == v)) for v in np.unique(flips)})

    # -- determinant-interpolation cross-check ------------------------------
    roots, deg = cm.det_roots(A, B, C)
    parts = cm.split_roots(roots, params)
    print(f"\ndeterminant interpolation: effective degree = {deg} "
          f"(predicted {cm.N_VARS + rankA})")
    print("finite roots (moduli):",
          np.array2string(np.abs(roots), precision=6, max_line_width=100))
    print("  zeros           :", len(parts["zeros"]))
    print("  shock-rho roots :",
          np.array2string(np.abs(parts["rho_roots"]), precision=6))
    print("  endogenous      :",
          np.array2string(np.abs(parts["endogenous"]), precision=6))
    qz_fin = np.sort(mods[np.isfinite(mods)])
    det_fin = np.sort(np.abs(roots))
    k = min(len(qz_fin), len(det_fin))
    cmp_df = pd.DataFrame({"qz_modulus": qz_fin[:k], "det_modulus": det_fin[:k]})
    cmp_df["abs_diff"] = np.abs(cmp_df.qz_modulus - cmp_df.det_modulus)
    cmp_df.to_csv(OUT / "01_det_vs_qz.csv", index=False)
    print("\nQZ vs det-interpolation (sorted moduli):")
    print(cmp_df.to_string(float_format=lambda v: f"{v:.8f}"))

    # -- eigenvalue sensitivities of the companion pencil -------------------
    sens = cm.pencil_eig_conditions(qz["M1"], qz["M0"])
    sens_df = pd.DataFrame(sens).sort_values("modulus")
    sens_fin = sens_df[np.isfinite(sens_df.modulus)]
    print("\ncompanion-pencil per-eigenvalue sensitivity "
          "(|dlam| per unit relative pencil perturbation):")
    print(sens_fin.to_string(float_format=lambda v: f"{v:.4g}", index=False))

    # -- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), lw=0.8, color="#999999")
    endo = parts["endogenous"]
    other = np.concatenate([parts["zeros"], parts["rho_roots"]])
    ax.scatter(other.real, other.imag, s=28, color="#4477aa", label="zeros + shock AR roots")
    ax.scatter(endo.real, endo.imag, s=48, color="#cc3311", marker="D",
               label="endogenous quartet")
    for r in endo:
        ax.annotate(f"{abs(r):.3f}", (r.real, r.imag), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    ax.axhline(0, lw=0.4, color="#cccccc"); ax.axvline(0, lw=0.4, color="#cccccc")
    ax.set_aspect("equal")
    ax.set_title("Fertility DSGE: finite pencil roots (port default calibration)")
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_roots_plane.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    vals = np.sort(np.log10(np.maximum(bb, 1e-300)))
    ax.stem(np.arange(len(vals)), vals)
    ax.axhline(np.log10(cm.BETA_THRESHOLD), color="#cc3311", ls="--",
               label="package threshold |beta| = 1e-12")
    ax.set_xlabel("eigenvalue index (sorted by |beta|)")
    ax.set_ylabel(r"$\log_{10}|\beta|$")
    ax.set_title("QZ |beta| clusters: finite vs infinite eigenvalues")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_beta_clusters.pdf")
    plt.close(fig)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
