"""Follow the clean-table US tariff path through its fold, parametrised by Costa Rica's price level.

Natural continuation in the tariff rate cannot pass a turning point, so the path is followed in
q = mean log p_CRI - its base value instead: unknowns (x, rate), equations F(x; rate) = 0 and
mean log p_CRI - base = q, solved by the equilibrated Newton of puremacro.trade.solver. The
bordered system stays regular at the fold because Costa Rica's prices load on the null vector of
F_x there. Printed per step: the tariff rate, the smallest singular value of the equilibrated F_x
and Costa Rica's foreign balance. The rate rising to a maximum and falling back is the fold.

    PYTHONPATH=. python tools/reference_validation/trade_clean_table/fold_trace.py [n_steps] [dq]
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from puremacro.trade import calibrate_trade_model, compute_equilibrium_residuals, solve_trade_equilibrium
from puremacro.trade.data import load_icio_data
from puremacro.trade.solver import _equilibrated_newton_solve, _fd_jacobian_dense, _ruiz_equilibrate

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wedge_identity import NC, NFD, NS, us_reciprocal_tariff  # noqa: E402


def main(n_steps: int = 16, dq: float = -0.004) -> None:
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    calib = calibrate_trade_model(load_icio_data(source="oecd2020"), ns=NS, nc=NC, nfd=NFD, validate=True)
    cri = calib.country_codes.index("CRI")
    idx = np.arange(cri * NS, (cri + 1) * NS)
    zeros = np.zeros(NC)

    def F(x, rate):
        tau, taufd = us_reciprocal_tariff(rate)
        return compute_equilibrium_residuals(x, calib, tau=tau, tau_fd=taufd, tauf=zeros, tauf_fd=zeros)

    xb = np.asarray(solve_trade_equilibrium(calib, method="newton", tol=2.5e-3).x_sol, float)
    n = xb.size
    q_of = lambda x: x[idx].mean() - xb[idx].mean()
    z_prev2, z_prev, q = None, np.r_[xb, 0.0], 0.0
    scale = np.r_[np.ones(n), 1e-3]
    for step in range(n_steps):
        q_target = q + dq
        z0 = z_prev if z_prev2 is None else 2 * z_prev - z_prev2
        f_aug = lambda z: np.r_[F(z[:n], z[n]), 1e4 * (q_of(z[:n]) - q_target)]
        z, conv, it, *_ = _equilibrated_newton_solve(f_aug, z0, tol=2.5e-3, max_iter=15, fd_scale=scale)
        x, rate = z[:n], float(z[n])
        J = _fd_jacobian_dense(lambda v: F(v, rate), x, F(x, rate), 1e-6, np.ones(n))
        s = np.linalg.svd(_ruiz_equilibrate(J)[0], compute_uv=False)
        print(f"[{time.perf_counter() - t0:5.0f}s] q={q_target:+.4f} converged={conv} it={it} "
              f"max|F|={np.abs(F(x, rate)).max():.1e} rate={rate:.6f} smin={s[-1] / s[0]:.2e} "
              f"XN_CRI={x[2 * NS * NC + 3 * NC + cri]:.1f}", flush=True)
        if not conv:
            break
        z_prev2, z_prev, q = z_prev, z, q_target


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else 16, float(args[1]) if len(args) > 1 else -0.004)
