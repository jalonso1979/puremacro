"""Per-country budget identity of the legacy trade system and the wedges that pin each country.

For every country j the legacy residuals ``F`` of ``compute_equilibrium_residuals`` satisfy,
for ANY state x (not only at a root),

    lambda_j(x)' F(x) = W1_j + W2_j + W3_j + W4_j,

    lambda_j = p_sj on the goods equations ff0 of j's sectors, y_sj on its price equations ff1,
               w_j on ff2, r_j on ff3, 1 on ff5 and -1 on ff4 (j's foreign-balance equation),

    W1  value-added cost minus factor payments: sum_s (v_sj y_sj - w_j xl_sj - r_j xk_sj); nonzero
        only because the MATLAB precedence ``w/(1-a)^(1-a)`` makes v homogeneous of degree 1+a in
        (r, w) (``replicate_matlab_precedence=False`` sets W1 = 0);
    W2  production tax charged in prices minus tax collected: sum_s t_sj (pp_sj - 1) y_sj
        (the fiscal block collects t*y, not t*p*y);
    W3  tariffs paid on j's imports minus the tariff revenue credited to j (``tauf = 0`` in the
        MATLAB scenarios credits none);
    W4  legacy trade balance minus the producer-price trade balance: final-demand flows are valued
        at the importer's composite price ppfd instead of the exporter's price p.

With consistent accounting all four vanish, so lambda_j' J = 0 for every j: each country's
foreign-balance equation is implied by its other equations, and its price level relative to the
rest of the world and its foreign balance are left undetermined. In the legacy model only the
wedges determine them. This script verifies the identity and reports, for the weakest directions
of the row/column-equilibrated Jacobian, how each wedge responds along them.

    PYTHONPATH=. python tools/reference_validation/trade_clean_table/wedge_identity.py [oecd2020|legacy] [rate]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from puremacro.trade import calibrate_trade_model, compute_equilibrium_residuals, solve_trade_equilibrium
from puremacro.trade.data import load_icio_data
from puremacro.trade.equilibrium import _evaluate_equilibrium
from puremacro.trade.solver import _fd_jacobian_dense, _ruiz_equilibrate

NS, NC, NFD, USA = 11, 77, 3, 73


def us_reciprocal_tariff(rate: float, chn_hkg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Multiplier tensors of the Main77c_11s.m scenarios: a US tariff ``rate`` on every partner."""
    tau = np.ones((NS * NC, NS, NC)); taufd = np.ones((NS * NC, NFD, NC))
    rt = np.full(NC, rate); rt[USA] = 0.0
    if chn_hkg is not None:
        rt[12] = rt[28] = chn_hkg
    for k in range(NC):
        tau[k * NS:(k + 1) * NS, :, USA] = 1.0 + rt[k]
        taufd[k * NS:(k + 1) * NS, :, USA] = 1.0 + rt[k]
    return tau, taufd


def wedges(x, calib, tau, taufd, precedence: bool = True) -> dict[str, np.ndarray]:
    """lambda_j' F(x) and W1..W4 for every country (legacy accounting, lump-sum closure)."""
    d = _evaluate_equilibrium(x, calib, tau_a=tau, taufd_a=taufd, tauf=np.zeros(NC),
                              tauf_fd=np.zeros(NC), replicate_matlab_precedence=precedence)
    F = d["residuals"]; m = NS * NC
    ff0 = F[:m].reshape(NC, NS); ff1 = F[m:2 * m].reshape(NC, NS)
    ff2 = F[2 * m:2 * m + NC]; ff3 = F[2 * m + NC:2 * m + 2 * NC]
    ff4 = np.append(F[2 * m + 2 * NC:2 * m + 3 * NC - 1], 0.0)
    ff5 = F[2 * m + 3 * NC - 1:]
    p = d["p"][0].T; y = d["ytot"][0].T; pp = d["pp"][0].T
    r = d["r"].ravel(); w = d["w"].ravel()
    lhs = (p * ff0).sum(1) + (y * ff1).sum(1) + w * ff2 + r * ff3 + ff5 - ff4

    al = calib.alpha[0].T; be = calib.beta[0].T; tax = calib.tax[0].T
    term_w = w[:, None] / ((1 - al) ** (1 - al)) if precedence else (w[:, None] / (1 - al)) ** (1 - al)
    va = (r[:, None] / al) ** al * term_w / be
    W1 = (va * y - w[:, None] * d["xl"][0].T - r[:, None] * d["xk"][0].T).sum(1)
    W2 = (tax * (pp - 1.0) * y).sum(1)
    pv, xm, xc = d["p_vec"], d["x_mat"], d["xc"]
    W3 = (np.einsum("i,isj,isj->j", pv, tau - 1.0, xm) + np.einsum("i,ikj,ikj->j", pv, taufd - 1.0, xc)
          - d["Tarifs_Totals"])
    flows = np.concatenate([xm.reshape(m, m, order="F"), xc.reshape(m, NFD * NC, order="F")], 1)
    orig = np.repeat(np.arange(NC), NS)
    dest = np.concatenate([np.arange(m) // NS, np.arange(NFD * NC) // NFD])
    ppfd = d["ppfd"].reshape(NFD * NC, order="F")
    val_leg = np.concatenate([d["pp"].reshape(m, order="F")[:, None] * flows[:, :m],
                              flows[:, m:] * ppfd[None, :]], 1)
    val_true = pv[:, None] * flows

    def balance(val):
        M = np.zeros((NC, NC))
        np.add.at(M, (np.broadcast_to(orig[:, None], val.shape), np.broadcast_to(dest[None, :], val.shape)), val)
        np.fill_diagonal(M, 0.0)
        return M.sum(1) - M.sum(0)

    W4 = balance(val_leg) - balance(val_true)
    return {"lhs": lhs, "W1": W1, "W2": W2, "W3": W3, "W4": W4}


def owner_of_unknowns() -> np.ndarray:
    return np.concatenate([np.tile(np.repeat(np.arange(NC), NS), 2), np.tile(np.arange(NC), 3), np.arange(NC - 1)])


def weak_direction_pins(calib, x, tau, taufd, n_show: int = 10):
    """For the n_show weakest equilibrated directions: country, singular value, wedge derivatives."""
    f = lambda v: compute_equilibrium_residuals(v, calib, tau=tau, tau_fd=taufd, tauf=np.zeros(NC), tauf_fd=np.zeros(NC))
    J = _fd_jacobian_dense(f, x, f(x), 1e-6, np.ones(x.size))
    A, _, dc = _ruiz_equilibrate(J)
    _, s, vt = np.linalg.svd(A)
    own = owner_of_unknowns(); out = []
    for k in range(1, n_show + 1):
        share = np.bincount(own, weights=vt[-k] ** 2, minlength=NC); j = int(np.argmax(share))
        v = dc * vt[-k]; eps = 1e-6 / np.abs(v).max()
        up, dn = wedges(x + eps * v, calib, tau, taufd), wedges(x - eps * v, calib, tau, taufd)
        sign = np.sign(v[j * NS:(j + 1) * NS].mean()) or 1.0       # direction in which j's prices rise
        d = {t: sign * (up[t][j] - dn[t][j]) / (2 * eps) for t in ("W1", "W2", "W3", "W4")}
        out.append((calib.country_codes[j], float(share[j]), float(s[-k] / s[0]), d))
    return out


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    source = sys.argv[1] if len(sys.argv) > 1 else "oecd2020"
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    calib = calibrate_trade_model(load_icio_data(source=source), ns=NS, nc=NC, nfd=NFD, validate=True)
    tau, taufd = us_reciprocal_tariff(rate)
    x = np.asarray(solve_trade_equilibrium(calib, method="newton", tol=2.5e-3).x_sol, float)
    if rate > 0:
        x = np.asarray(solve_trade_equilibrium(calib, tau=tau, tau_fd=taufd, tauf=np.zeros(NC), tauf_fd=np.zeros(NC),
                                               x0=x, method="equilibrated_newton").x_sol, float)
    o = wedges(x, calib, tau, taufd)
    total = o["W1"] + o["W2"] + o["W3"] + o["W4"]
    print(f"{source}, US tariff {rate:.3f}: identity error {np.abs(o['lhs'][:-1] - total[:-1]).max():.2e}")
    for c, share, sv, d in weak_direction_pins(calib, x, tau, taufd):
        net = sum(d.values())
        print(f"  {c} (share {share:.2f}) s={sv:.2e}  dW1+dW2={d['W1'] + d['W2']: .3e}  dW3={d['W3']: .2e}  "
              f"dW4={d['W4']: .3e}  net={net: .3e}")
