"""Damped Newton with Ruiz row/column equilibration for the legacy trade model on the clean table."""
import time, warnings, sys, numpy as np, scipy.io as sio
warnings.simplefilter("ignore")
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_equilibrium_residuals
from puremacro.trade.data import load_icio_data
S = sys.argv[1]; rate_target = float(sys.argv[2]); tag = sys.argv[3]
nc, ns, nfd = 77, 11, 3
import os; SRC = os.environ.get("PM_TABLE", "oecd2020"); calib = calibrate_trade_model(load_icio_data(source=SRC), ns=ns, nc=nc, nfd=nfd, validate=True)
base = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)
x = np.asarray(base.x_sol, float); n = x.size
def tensors(rate, chn_hkg=None):
    tau_a = np.ones((ns*nc, ns, nc)); taufd_a = np.ones((ns*nc, nfd, nc))
    rt = np.full(nc, rate); rt[73] = 0.0
    if chn_hkg is not None: rt[12] = chn_hkg; rt[28] = chn_hkg
    for ik in range(nc):
        tau_a[ik*ns:(ik+1)*ns, :, 73] = 1.0 + rt[ik]; taufd_a[ik*ns:(ik+1)*ns, :, 73] = 1.0 + rt[ik]
    return tau_a, taufd_a
chn = float(tag.split("_")[1]) / 100.0 if tag.startswith("t10_") else None
tau_a, taufd_a = tensors(rate_target, chn); tauf = np.zeros(nc); tauf_fd = np.zeros(nc)
f = lambda v: compute_equilibrium_residuals(v, calib, tau=tau_a, tau_fd=taufd_a, tauf=tauf, tauf_fd=tauf_fd)
def jac(v, fv):
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-6 * max(1.0, abs(v[j])); vp = v.copy(); vp[j] += h
        J[:, j] = (f(vp) - fv) / h
    return J
def equilibrate(J):
    Dr = np.ones(n); Dc = np.ones(n); A = J.copy()
    for _ in range(10):
        r = np.sqrt(np.maximum(np.abs(A).max(axis=1), 1e-300)); A /= r[:, None]; Dr /= r
        c = np.sqrt(np.maximum(np.abs(A).max(axis=0), 1e-300)); A /= c[None, :]; Dc /= c
    return A, Dr, Dc
t0 = time.perf_counter(); fx = f(x)
print(f"{tag}: start max|f|={np.abs(fx).max():.3e} L1={np.abs(fx).sum():.3e}", flush=True)
for it in range(1, 41):
    J = jac(x, fx); A, Dr, Dc = equilibrate(J)
    g = Dr * fx                                   # scaled residual
    dz = np.linalg.solve(A, -g)                   # scaled step
    dx = Dc * dz
    # backtracking on the scaled residual norm
    gn0 = np.linalg.norm(g); lam = 1.0; accepted = False
    for _ in range(25):
        xn = x + lam * dx; fn = f(xn)
        if np.all(np.isfinite(fn)) and np.linalg.norm(Dr * fn) < (1 - 1e-4 * lam) * gn0:
            accepted = True; break
        lam *= 0.5
    if not accepted:
        print(f"  iter {it}: no descent (lam={lam:.1e}); stopping", flush=True); break
    x, fx = xn, fn
    print(f"  iter {it}: lam={lam:.3g} max|f|={np.abs(fx).max():.3e} L1={np.abs(fx).sum():.3e} cond={np.linalg.cond(A):.2e} [{time.perf_counter()-t0:.0f}s]", flush=True)
    if np.abs(fx).sum() <= 2.5e-3 * 0.5: break
ok = bool(np.abs(fx).max() < 2.5e-3)
print(f"{tag}: done converged(max)={ok} L1={np.abs(fx).sum():.3e} max|f|={np.abs(fx).max():.3e} [{time.perf_counter()-t0:.0f}s]", flush=True)
sio.savemat(f"{S}/matlab/puremacro_{tag}_{SRC}_scaled.mat", {"xx_pm": x.reshape(-1, 1), "tau_a": tau_a, "taufd_a": taufd_a, "tauf": tauf.reshape(1, -1), "tauf_fd": tauf_fd.reshape(1, -1), "max_resid": np.abs(fx).max(), "l1_resid": np.abs(fx).sum()})
print("saved", flush=True)
