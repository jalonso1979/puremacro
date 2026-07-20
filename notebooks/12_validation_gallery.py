# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Validation gallery: puremacro vs trusted references
#
# Every headline estimator in `puremacro` is checked against an **independent**
# reference — `statsmodels` / `linearmodels` / `arch` / `scipy` where one exists,
# a closed-form solution, a published number, or an internal cross-method identity.
# The checks live in `puremacro.validation` as a declarative case registry, so
# **anyone can re-verify the library** — including in the browser:
#
# ```python
# from puremacro.validation import scorecard
# scorecard()            # puremacro vs reference, one row per case
# ```
#
# This notebook renders that scorecard and a few representative side-by-side
# overlays. It imports only the pyodide-safe stack (numpy/scipy/pandas/matplotlib
# + `puremacro`), so it runs unchanged in the JupyterLite playground — the
# package-comparison numbers are read from frozen *golden* values, while the
# closed-form / scipy references are recomputed live.

# %% [markdown]
# ## Why validate?
#
# A pure-Python macro library is only worth using if you can trust its numbers.
# Reimplementing a VAR, a Kalman filter, or a GARCH estimator from scratch in
# numpy is exactly where silent bugs hide — an off-by-one lag, a transposed
# matrix, a wrong normalization — and none of them announce themselves.
#
# **Intuition.** Trust here is *earned per estimator, against something
# independent*. Each case in `puremacro.validation` pairs a puremacro estimator
# with a reference the puremacro code did not produce, declares **how** that
# reference is sourced (its *mechanism*) and **how close** the two must agree
# (its *tolerance tier*), then reports pass/fail. The reference's independence is
# the whole argument — comparing the library to itself would prove nothing.
#
# - **Mechanism** — `PACKAGE` (statsmodels / linearmodels / arch, captured once
#   as a frozen *golden* so the library never imports the heavy package at run
#   time), `SCIPY` (scipy/numpy, recomputed live), `ANALYTICAL` (a closed-form
#   solution), `PUBLISHED` (a number from a paper, with citation), `INTERNAL` (a
#   cross-method identity, e.g. EGM = VFI, or a simulate-then-recover check).
# - **Tolerance tier** — `EXACT` (rtol $10^{-10}$), `TIGHT` ($10^{-6}$),
#   `NUMERIC` ($10^{-2}$), `COARSE` (10%), `QUALITATIVE` (a sign, an ordering, or
#   a lower-bound threshold). The tier is chosen to match what the reference can
#   honestly certify — machine precision for an algebraic identity, a looser band
#   for a simulate-and-recover estimate.
#
# Keeping the `PACKAGE` references as frozen goldens (re-guarded against the live
# packages in CI) is what lets the *same* `scorecard()` run in the browser with
# none of statsmodels/linearmodels/arch installed.

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = __import__("pathlib").Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.validation import run_all, scorecard

# %% [markdown]
# ## 1. The scorecard
# One row per validation case: which subsystem, the reference *mechanism*
# (`package` = vs statsmodels/linearmodels/arch via a frozen golden; `scipy`,
# `analytical`, `published`, `internal`), the tolerance tier, whether it passes,
# and the safety margin (`max_margin <= 0` means inside tolerance). The
# `assert` below is the library verifying itself.

# %%
df = scorecard()
n, n_pass = len(df), int(df["passed"].sum())
print(f"{n_pass}/{n} validation cases pass across {df['subsystem'].nunique()} subsystems")
assert df["passed"].all(), "a validation case failed"
print("by mechanism:", df["mechanism"].value_counts().to_dict())
print("by tolerance:", df["tol"].value_counts().to_dict())

# Per-subsystem summary: how many cases, which mechanisms, worst (largest) margin.
agg = (
    df.groupby("subsystem")
    .agg(
        n_cases=("id", "size"),
        mechanisms=("mechanism", lambda s: "+".join(sorted(s.unique()))),
        all_pass=("passed", "all"),
        worst_margin=("max_margin", "max"),
    )
    .sort_values("n_cases", ascending=False)
)
agg

# %% [markdown]
# ### Hero figure — coverage at a glance
# Every subsystem, every case green. The reference mechanism mix shows the
# gallery spans live-package cross-checks, scipy, closed-form, published, and
# internal-consistency anchors.

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 4.0))

a = agg.sort_values("n_cases")
axL.barh(a.index, a["n_cases"], color="0.30")
for i, v in enumerate(a["n_cases"]):
    axL.text(v + 0.08, i, str(int(v)), va="center", fontsize=9)
axL.set_xlabel("validation cases (all pass)")
axL.set_title(f"{n_pass}/{n} cases pass, {df['subsystem'].nunique()} subsystems")
axL.grid(axis="y", visible=False)

mech = df["mechanism"].value_counts()
axR.bar(mech.index, mech.values, color="0.30")
for i, v in enumerate(mech.values):
    axR.text(i, v + 0.4, str(int(v)), ha="center", fontsize=9)
axR.set_ylabel("cases")
axR.set_title("reference mechanism")
axR.tick_params(axis="x", labelrotation=30)
axR.grid(axis="x", visible=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Read the output.** The printed line is the headline: *every* case passes, and
# the `assert df["passed"].all()` is the library certifying itself — if any
# estimator had drifted, importing this notebook would have raised. The
# per-subsystem table shows the gallery is broad, not deep in one place: roughly a
# dozen subsystems each carry a handful of cases, and `worst_margin` (the largest
# `max_margin` in the subsystem) stays at or below zero, so even the *tightest*
# case in each block clears its tolerance with room to spare. The left bar reads
# this as coverage — no subsystem is left unchecked — while the right bar shows
# the references are genuinely mixed: live-package cross-checks, scipy, closed
# form, published, and internal-identity anchors. A wall of green built on a
# single mechanism would be weak; a wall of green spanning five independent kinds
# of reference is the point.

# %% [markdown]
# ## 2. `var` — Cholesky IRF vs statsmodels
# A `package` case: puremacro's pure-numpy structural IRF against statsmodels'
# `orth_irfs`, captured as a frozen golden (so this renders with no statsmodels
# installed). They agree to machine precision.

# %%
from puremacro.validation._fixtures import var_demo_data
from puremacro.validation._goldens import load_golden
from puremacro.var.identify import cholesky

d = var_demo_data()
pm = np.asarray(cholesky(d["Y"], p=d["p"], horizon=d["horizon"], n_boot=10, seed=0).irf_point)
ref = np.asarray(load_golden("var:cholesky_irf_vs_statsmodels")["irf"])
hz = np.arange(pm.shape[0])
max_diff = float(np.abs(pm - ref).max())

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
for resp, ax in zip((0, 1), axes):
    ax.plot(hz, ref[:, resp, 0], color="0.58", lw=2.6, ls=(0, (4, 2)),
            label="statsmodels (golden)")
    ax.plot(hz, pm[:, resp, 0], color="0.0", lw=1.2, label="puremacro")
    ax.set_title(f"response of $y_{resp}$ to shock 0")
    ax.set_xlabel("horizon")
axes[0].set_ylabel("impulse response")
axes[0].legend(loc="best")
fig.suptitle(f"Cholesky IRF: puremacro vs statsmodels orth_irfs  (max |Δ| = {max_diff:.1e})")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. `vfi` — stochastic-growth policy vs Brock–Mirman closed form
# An `analytical` case: with log utility and full depreciation the optimal
# capital policy is known in closed form, $k' = \alpha\beta\,e^{z}k^{\alpha}$.
# The grid value-function-iteration policy lands on it (interior to the grid).

# %%
from puremacro.vfi.discretize import rouwenhorst
from puremacro.vfi.problem import VFIProblem

alpha, beta = 0.30, 0.95
z_log, P = rouwenhorst(5, 0.6, 0.10)
z_lev = np.exp(z_log)
k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
k_grid = np.linspace(0.3 * k_ss, 2.5 * k_ss, 600)

def _return_fn(kp, k, z, xp=np):
    c = xp.exp(z) * k**alpha - kp                 # full depreciation: c = y - k'
    return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

sol = VFIProblem(
    a_grid=k_grid, z_grid=z_log, P_z=P, return_fn=_return_fn, beta=beta,
    options=dict(tol=1e-10, n_howard=50, max_iter=20000),
).solve("numpy")
k_pol = k_grid[sol.policy_aprime]                 # (n_k, n_z)
k_closed = alpha * beta * z_lev[None, :] * k_grid[:, None] ** alpha

fig, ax = plt.subplots(figsize=(6.6, 4.1))
cols = _nbstyle.palette(3)
for col, zi, lab in zip(cols, (0, 2, 4), ("low $z$", "mid $z$", "high $z$")):
    ax.plot(k_grid, k_closed[:, zi], color=col, lw=2.6, ls=(0, (4, 2)))
    ax.plot(k_grid, k_pol[:, zi], color=col, lw=1.1)
    ax.text(k_grid[-1], k_pol[-1, zi], f"  {lab}", color=col, va="center", fontsize=9)
ax.plot([], [], color="0.0", lw=2.6, ls=(0, (4, 2)), label="Brock–Mirman closed form")
ax.plot([], [], color="0.0", lw=1.1, label="puremacro VFI")
ax.set_xlabel("capital $k$")
ax.set_ylabel("next-period capital $k'$")
ax.set_title(r"Stochastic-growth policy: VFI vs $k'=\alpha\beta e^{z}k^{\alpha}$")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## 4. `forecast` — Gaussian CRPS vs closed form
# An `analytical` case over a grid of standardized errors: `crps_gaussian`
# against the Gneiting–Raftery (2007) closed form
# $\sigma[z(2\Phi(z)-1)+2\phi(z)-1/\sqrt{\pi}]$.

# %%
import math
from puremacro.forecast.density import crps_gaussian

z = np.linspace(-3.0, 3.0, 121)
pm_crps = np.array([crps_gaussian([zz], [0.0], [1.0])[0] for zz in z])
Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
cf = z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi)   # sigma = 1

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(z, cf, color="0.58", lw=2.6, ls=(0, (4, 2)), label="closed form (Gneiting–Raftery)")
ax.plot(z, pm_crps, color="0.0", lw=1.2, label="puremacro crps_gaussian")
ax.set_xlabel(r"standardized error $z=(y-\mu)/\sigma$")
ax.set_ylabel("CRPS")
ax.set_title(f"Gaussian CRPS: puremacro vs closed form  (max |Δ| = {np.abs(pm_crps - cf).max():.1e})")
ax.legend(loc="upper center")
plt.show()

# %% [markdown]
# ## 5. `spectral` — Welch PSD vs `scipy.signal`
# A `scipy` case: puremacro's pure-`numpy.fft` Welch estimator against
# `scipy.signal.welch` configured to the same estimator (symmetric Hann window,
# constant detrend, density scaling, matched overlap). puremacro returns the
# two-sided density on the one-sided grid, so we apply the standard one-sided
# fold (×2 except DC/Nyquist) to line them up. The demo series carries a
# 16-period business-cycle component — the spectral peak both methods recover.

# %%
from scipy.signal import welch as scipy_welch
from puremacro.spectral import welch_psd
from puremacro.validation.cases_spectral import spectral_demo_data

s = spectral_demo_data()
f, Pxx = welch_psd(s["x"], fs=s["fs"], n_seg=s["n_seg"], overlap=s["overlap"])
fold = np.full(len(f), 2.0); fold[0] = 1.0; fold[-1] = 1.0     # one-sided fold (even segment)
Pxx = np.asarray(Pxx, dtype=float) * fold
f_sp, P_sp = scipy_welch(
    s["x"], fs=s["fs"], window=np.hanning(s["n_seg"]), nperseg=s["n_seg"],
    noverlap=s["noverlap"], detrend="constant", scaling="density",
)

fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.semilogy(f_sp, P_sp, color="0.58", lw=2.6, ls=(0, (4, 2)), label="scipy.signal.welch")
ax.semilogy(f, Pxx, color="0.0", lw=1.2, label="puremacro welch_psd")
ax.axvline(1.0 / 16.0, color="0.85", lw=0.8)
ax.set_xlabel("frequency (cycles / period)")
ax.set_ylabel("power spectral density")
ax.set_title(f"Welch PSD: puremacro vs scipy.signal  (max |Δ| = {np.abs(Pxx - P_sp).max():.1e})")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## Your turn — audit one subsystem
#
# The scorecard `df` (computed once in section 1) is the whole gallery in a tidy
# table, so you can interrogate it without recomputing anything. Pick a subsystem
# and confirm the claim that matters: it carries at least one case, and *all* of
# its cases pass. Change `my_subsystem` below to any block — the available names
# are printed for you (`var`, `lp`, `garch`, `inference`, `state_space`,
# `spectral`, `forecast`, `vfi`, `dsge`, `dynpanel`, `narrative`).

# %%
print("available subsystems:", sorted(df["subsystem"].unique()))

my_subsystem = "spectral"          # ← change this to any subsystem above
sub = df[df["subsystem"] == my_subsystem]
n_sub, n_sub_pass = len(sub), int(sub["passed"].sum())
worst = float(sub["max_margin"].max())   # largest margin; <= 0 means all inside tolerance
print(f"{my_subsystem}: {n_sub_pass}/{n_sub} cases pass | worst margin = {worst:.1e}")

assert n_sub >= 1, f"no validation cases for subsystem {my_subsystem!r}"
assert sub["passed"].all(), f"a {my_subsystem} case failed"   # the subsystem certifies itself

# %% [markdown]
# **Prompts.** (1) Swap `my_subsystem` for `"narrative"` (the largest block, 7
# cases) or `"var"` (the smallest, 3) and re-run — the assert holds for every
# subsystem in the gallery. (2) Filter by *mechanism* instead:
# `df[df["mechanism"] == "scipy"]` isolates the live-recomputed scipy checks;
# confirm `.passed.all()` and inspect their `tol`. (3) Find the single case with
# the largest `max_margin` (`df.loc[df["max_margin"].idxmax()]`) — the largest reported
# margin (margins aren't directly comparable across tolerance tiers) — and read its
# `citation` to see what it is checked against.

# %% [markdown]
# **How comprehensive is this?** The gallery spans every major subsystem of
# `puremacro` — `var` / `lp` / `garch` / `inference` / `state_space` / `spectral`
# / `forecast` / `vfi` / `dsge` / `dynpanel` / `narrative` — and grows by simply
# dropping a `cases_<subsystem>.py` module into `puremacro/validation/`, which
# `run_all()` discovers automatically. `docs/VALIDATION.md` lists every case, its
# reference, and its citation, and documents the *honest scope*: where no sound
# independent reference exists (e.g. the heavy DSGE estimation routines, the
# narrative LLM path), a case is skipped with a stated reason rather than
# rubber-stamped with a circular check.

# %% [markdown]
# **What this shows about `puremacro.validation`:** the package ships a
# declarative registry of validation cases — `run_all()` runs them all and
# `scorecard()` tabulates puremacro against an independent reference for each.
# Package cross-checks (statsmodels / linearmodels / arch) are frozen as golden
# values and re-guarded against the live packages in CI (`pytest -m reference`),
# so `run_all()` itself stays pure and browser-runnable, while closed-form,
# scipy, and internal-identity references are recomputed live here.
# **Try it:** `from puremacro.validation import scorecard; scorecard()` — or add a
# case by dropping a `cases_<subsystem>.py` file into the package.
