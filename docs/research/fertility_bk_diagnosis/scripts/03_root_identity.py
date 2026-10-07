"""03 — Which root is missing, and what is it economically?

(a) Loadings (quadratic eigenvectors) of every nonzero finite root at the
    port default calibration -> map roots to model blocks.
(b) Root sensitivity to psik / psin -> capital pair vs fertility pair.
(c) Validation against the original Dynare runs: reproduce the logged
    `check` eigenvalues of fertility_adj_costs.mod and of the Bayesian
    variant from the ported equations + exact steady state.
(d) Ablation from the BK-passing Bayesian calibration toward the port's
    conventions (barp = 0, omega = 2.5, g = 0.017, no exact SS) — which
    single change kills the 12th stable root?
(e) Flatness of the BGP least-squares problem at the port's LM endpoint
    (spectrum of J'J), explaining the cross-build nondeterminism found
    in 02.

Outputs: output/03_loadings.csv, output/03_dynare_replication.csv,
         output/03_ablation.csv
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as cm

HERE = pathlib.Path(__file__).resolve().parent.parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

MOD1_SS = {"c": 0.237904, "k": 0.841843, "i": 0.142513, "u": 1.45004,
           "n": 0.526141, "b": 0.0348831, "l_o": 0.550914, "l_w": 0.337637,
           "y": 0.564566}
MOD2_SS = {"c": 0.266225, "k": 1.03382, "i": 0.114395, "u": 1.13744,
           "n": 1.06629, "b": 0.0672904, "l_o": 0.474782, "l_w": 0.26552,
           "y": 0.481514}


def z_from_ss(params: dict, ss: dict) -> np.ndarray:
    z = np.empty(cm.N_VARS)
    full = dict(ss)
    full["a"] = params["bara"]; full["mun"] = params["barn"]; full["ph"] = params["barp"]
    for i, name in enumerate(cm.VAR_NAMES):
        z[i] = full[name]
    return z


def quartet(params: dict, z_ss: np.ndarray) -> dict:
    A, B, C, _ = cm.build_ABCD(params, z_ss, method="cs")
    roots, _ = cm.det_roots(A, B, C)
    parts = cm.split_roots(roots, params)
    qz = cm.companion_qz(A, B, C)
    return {"A": A, "B": B, "C": C, "roots": roots, "parts": parts,
            "n_stable": qz["n_stable"],
            "endo": np.abs(parts["endogenous"]),
            "finite_nonzero": np.sort(np.abs(roots[np.abs(roots) > 1e-7]))}


def main() -> None:
    params = cm.build_params()
    z_ss = cm.z_from_params(params)
    base = quartet(params, z_ss)
    A, B, C = base["A"], base["B"], base["C"]

    # (a) loadings ---------------------------------------------------------
    print("=== (a) eigenvector loadings of the nonzero finite roots "
          "(port default calibration) ===")
    rows = []
    for mu in sorted(base["roots"], key=abs):
        if abs(mu) < 1e-7:
            continue
        load, smin = cm.quadratic_eigvec(A, B, C, mu)
        rec = {"modulus": abs(mu), "smin": smin}
        rec.update({v: load[i] for i, v in enumerate(cm.VAR_NAMES)})
        rows.append(rec)
    ldf = pd.DataFrame(rows)
    ldf.to_csv(OUT / "03_loadings.csv", index=False)
    print(ldf.to_string(float_format=lambda v: f"{v:.3f}", index=False))
    print("\n(reading: 0.5/0.7807/0.9 load exclusively on mun/a/ph = the AR(1)")
    print(" shock block; the remaining four are the endogenous (k,n)-system")
    print(" roots, loading on c,y,k,n,l_w,u,i,b,l_o)\n")

    # (b) block assignment via psik / psin ----------------------------------
    print("=== (b) quartet sensitivity to adjustment costs ===")
    recs = []
    for pname, delta in [("psik", 0.25), ("psin", 0.32)]:
        for sgn in (+1, -1):
            p2 = cm.build_params({pname: params[pname] + sgn * delta})
            q2 = quartet(p2, z_ss)
            recs.append({"param": pname, "value": p2[pname],
                         **{f"endo_{i+1}": m for i, m in enumerate(q2["endo"])}})
    bdf = pd.DataFrame(recs)
    print(bdf.to_string(float_format=lambda v: f"{v:.5f}", index=False))
    print("base quartet:", np.array2string(base["endo"], precision=5))
    print("(roots that move with psik = capital-adjustment pair; with psin ="
          " fertility-adjustment pair)\n")

    # (c) Dynare replication -------------------------------------------------
    print("=== (c) replication of the original Dynare `check` eigenvalues ===")
    rep_rows = []
    for tag, prm, ssgz, dyn in [
        ("fertility_adj_costs.mod", cm.MOD1_PARAMS, MOD1_SS, cm.MOD1_DYNARE_FINITE),
        ("bayesian_estimation.mod", cm.MOD2_PARAMS, MOD2_SS, cm.MOD2_DYNARE_FINITE),
    ]:
        p = cm.build_params(prm)          # port BGP values, overridden by .mod params
        z_guess = z_from_ss(p, ssgz)
        z_exact = cm.exact_steady_state(p, z_guess)
        resid = cm.model_residuals(z_exact, z_exact, z_exact,
                                   np.zeros(cm.N_SHOCKS), p)
        q = quartet(p, z_exact)
        ours = q["finite_nonzero"]
        print(f"\n{tag}")
        print(f"  exact-SS max|resid| = {np.max(np.abs(resid)):.2e}")
        print(f"  SS (ours) : ", {n: round(float(z_exact[i]), 6)
                                  for i, n in enumerate(cm.VAR_NAMES)})
        print(f"  Dynare    : {np.array2string(np.sort(dyn), precision=4)}")
        print(f"  this port : {np.array2string(ours, precision=4)}")
        k = min(len(dyn), len(ours))
        err = np.max(np.abs(np.sort(dyn)[:k] - ours[:k]))
        print(f"  max |diff| = {err:.2e}  -> "
              f"{'EQUATIONS VALIDATED' if err < 5e-4 else 'MISMATCH'}")
        print(f"  n_stable = {q['n_stable']} (12 = BK passes)")
        rep_rows.append({"model": tag, "max_diff": err,
                         "n_stable": q["n_stable"],
                         **{f"mod_{i}": m for i, m in enumerate(ours)}})
    pd.DataFrame(rep_rows).to_csv(OUT / "03_dynare_replication.csv", index=False)

    # (d) ablation: BK-passing Bayesian calibration -> port conventions ------
    print("\n=== (d) ablation from the Bayesian calibration (BK OK) toward "
          "the port's conventions ===")
    abl_rows = []

    def run_case(name, overrides, use_exact_ss=True, ss_guess=None):
        p = cm.build_params({**cm.MOD2_PARAMS, **overrides})
        if use_exact_ss:
            zg = z_from_ss(p, ss_guess or MOD2_SS)
            try:
                z = cm.exact_steady_state(p, zg)
            except RuntimeError as e:
                print(f"  {name:42s} SS solve FAILED: {e}")
                abl_rows.append({"case": name, "n_stable": np.nan})
                return
        else:
            z = z_from_ss(p, ss_guess or MOD2_SS)
        q = quartet(p, z)
        margin = float(np.min(np.abs(q["finite_nonzero"] - 1.0)))
        print(f"  {name:42s} n_stable={q['n_stable']:2d}  "
              f"quartet={np.array2string(q['endo'], precision=4)}  "
              f"margin={margin:.4f}")
        abl_rows.append({"case": name, "n_stable": q["n_stable"],
                         "margin": margin,
                         **{f"endo_{i+1}": m for i, m in enumerate(q["endo"])}})

    run_case("Bayesian .mod baseline (exact SS)", {})
    run_case("+ barp: log(1/1.03) -> 0", {"barp": 0.0})
    run_case("+ omega: 2.0 -> 2.5", {"omega": 2.5})
    run_case("+ g: 0.0175 -> 0.017", {"g": 0.017})
    run_case("+ all three port constants", {"barp": 0.0, "omega": 2.5, "g": 0.017})
    # the port's real sin: no exact SS at all — linearise at parameters.mat point
    run_case("Bayesian params, NO exact SS (initval pt)",
             {}, use_exact_ss=False,
             ss_guess={"c": 0.360443, "k": 1.835291, "u": 0.999997,
                       "n": 1.384615, "b": 0.087379, "l_w": 0.329987,
                       "i": (cm.MOD2_PARAMS["g"] + 0.144 * 1.0 ** 2 / 2) * 1.835291,
                       "l_o": 1 - 0.213943 * 1.384615 - 0.329987 - 0.469184 * 0.087379,
                       "y": (1.0 * 1.835291) ** 0.4 * 0.329987 ** 0.6})
    # port's own LM params, repaired with an exact SS (the port's z is in a
    # bad basin, so also try the healthy Bayesian SS as the starting guess)
    p_port = cm.build_params()
    try:
        try:
            z_port_exact = cm.exact_steady_state(p_port)
        except RuntimeError:
            z_port_exact = cm.exact_steady_state(p_port, z_from_ss(p_port, MOD2_SS))
        q = quartet(p_port, z_port_exact)
        margin = float(np.min(np.abs(q["finite_nonzero"] - 1.0)))
        print(f"  {'port LM params + exact SS repair':42s} n_stable={q['n_stable']:2d}  "
              f"quartet={np.array2string(q['endo'], precision=4)}  margin={margin:.4f}")
        abl_rows.append({"case": "port LM params + exact SS repair",
                         "n_stable": q["n_stable"], "margin": margin,
                         **{f"endo_{i+1}": m for i, m in enumerate(q["endo"])}})
    except RuntimeError as e:
        print(f"  port LM params + exact SS repair: SS solve FAILED: {e}")
    pd.DataFrame(abl_rows).to_csv(OUT / "03_ablation.csv", index=False)

    # (e) flatness of the BGP least-squares problem --------------------------
    print("\n=== (e) BGP least-squares flatness at the port's LM endpoint ===")
    from puremacro.dsge.fertility_adj_costs import (
        _bgp_system, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS,
    )
    bgp = cm.port_bgp()
    x_hat = np.array([bgp["barn"], bgp["mu_l"], bgp["beta"], bgp["tau_n"],
                      bgp["tau_b"], bgp["p_n"], bgp["delta_k"], bgp["c"],
                      bgp["b"], bgp["l_w"], bgp["u"], bgp["k"], bgp["n"]])
    F0 = _bgp_system(x_hat, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS)
    J = cm.jac_central(
        lambda x: _bgp_system(x, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS),
        x_hat, 1e-7)
    s = np.linalg.svd(J, compute_uv=False)
    print(f"  residual norm at LM endpoint: {np.linalg.norm(F0):.6f} "
          f"(a true root would be ~0)")
    print(f"  singular values of the 13x13 BGP Jacobian at the endpoint:")
    print("   ", np.array2string(s, precision=4, max_line_width=100))
    print(f"  condition number = {s[0] / s[-1]:.3e}; "
          f"{int(np.sum(s < 1e-6 * s[0]))} singular value(s) < 1e-6 * s_max")
    print("  -> the LM endpoint sits in a near-flat valley of an inconsistent")
    print("     system: the returned 'BGP' is arithmetic-dependent, which is")
    print("     exactly what 02's basin experiment shows.")


if __name__ == "__main__":
    main()
