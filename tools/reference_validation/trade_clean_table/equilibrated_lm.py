"""Levenberg-Marquardt on the row/column-equilibrated legacy trade system (clean table)."""
import time, warnings, sys, os, numpy as np, scipy.io as sio
from scipy.optimize import least_squares
warnings.simplefilter("ignore")
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_equilibrium_residuals
from puremacro.trade.data import load_icio_data
S = sys.argv[1]; rate = float(sys.argv[2]); tag = sys.argv[3]; warm = sys.argv[4] if len(sys.argv) > 4 else None
nc, ns, nfd = 77, 11, 3
calib = calibrate_trade_model(load_icio_data(source="oecd2020"), ns=ns, nc=nc, nfd=nfd, validate=True)
base = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3); x0 = np.asarray(base.x_sol, float); n = x0.size
tau_a = np.ones((ns*nc, ns, nc)); taufd_a = np.ones((ns*nc, nfd, nc)); rt = np.full(nc, rate); rt[73] = 0.0
chn = float(tag.split("_")[1]) / 100.0 if tag.startswith("t10_") else None
if chn is not None: rt[12] = chn; rt[28] = chn
for ik in range(nc):
    tau_a[ik*ns:(ik+1)*ns, :, 73] = 1.0 + rt[ik]; taufd_a[ik*ns:(ik+1)*ns, :, 73] = 1.0 + rt[ik]
tauf = np.zeros(nc); tauf_fd = np.zeros(nc)
f = lambda v: compute_equilibrium_residuals(v, calib, tau=tau_a, tau_fd=taufd_a, tauf=tauf, tauf_fd=tauf_fd)
def jac_raw(v, fv):
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-6*max(1.0, abs(v[j])); vp = v.copy(); vp[j] += h; J[:, j] = (f(vp) - fv)/h
    return J
# fixed equilibration from the base-point Jacobian
J0 = jac_raw(x0, f(x0)); Dr = np.ones(n); Dc = np.ones(n); A = J0.copy()
for _ in range(10):
    r = np.sqrt(np.maximum(np.abs(A).max(axis=1), 1e-300)); A /= r[:, None]; Dr /= r
    c = np.sqrt(np.maximum(np.abs(A).max(axis=0), 1e-300)); A /= c[None, :]; Dc /= c
xs = x0 if warm is None else sio.loadmat(warm)["xx_pm"].ravel()
z0 = xs / Dc
g = lambda z: Dr * f(Dc * z)
def jac_s(z):
    v = Dc * z; return (Dr[:, None] * jac_raw(v, f(v))) * Dc[None, :]
t0 = time.perf_counter(); it = [0]
def cb(z):
    it[0] += 1
def gg(z):
    val = g(z); print(f"  eval: ||g||={np.linalg.norm(val):.3e} max|f|={np.abs(val/Dr).max():.3e} L1={np.abs(val/Dr).sum():.3e} [{time.perf_counter()-t0:.0f}s]", flush=True); return val
res = least_squares(gg, z0, jac=jac_s, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=120)
x = Dc * res.x; fx = f(x)
print(f"{tag}: LM status={res.status} ({res.message}) nfev={res.nfev} njev={res.njev} | max|f|={np.abs(fx).max():.3e} L1={np.abs(fx).sum():.3e} [{time.perf_counter()-t0:.0f}s]", flush=True)
sio.savemat(f"{S}/matlab/puremacro_{tag}_lm.mat", {"xx_pm": x.reshape(-1, 1), "tau_a": tau_a, "taufd_a": taufd_a, "tauf": tauf.reshape(1, -1), "tauf_fd": tauf_fd.reshape(1, -1), "max_resid": np.abs(fx).max(), "l1_resid": np.abs(fx).sum()})
print("saved", flush=True)
