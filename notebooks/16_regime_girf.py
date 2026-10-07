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
# # State-dependent transmission done right
#
# **How does the macroeconomic transmission of structural shocks vary across regimes, such as calm and stressed financial conditions?** Regime models — threshold VARs, Markov-switching VARs, threshold
# VECMs — exist to answer exactly that. But fitting one and then pushing its
# per-regime coefficients through a *linear* IRF routine quietly answers a
# different question: "what if the economy were locked in that regime
# forever?" For a long time that frozen-regime shortcut was the only impulse
# response the regime fits in this package could offer. This notebook builds
# the correct object — the **generalized IRF** of Koop, Pesaran and Potter
# (1996, *J. Econometrics*) — and uses it to recover a planted asymmetry,
# test it with a bootstrap band, and expose size/sign nonlinearity in the
# style of Kilian and Vigfusson (2011).

# %% [markdown]
# ## The method in math: from linear IRF to generalized IRF
#
# A two-regime threshold VAR switches its dynamics on a lagged threshold
# variable $z_{t-d}$:
# $$ y_t = c^{(r_t)} + A^{(r_t)} y_{t-1} + \varepsilon_t^{(r_t)}, \qquad
#    r_t = \mathbb{1}\{z_{t-d} > c\}. $$
# In a linear VAR the IRF is one deterministic path. Here the response
# depends on the **history** $\omega_{t-1}$ (which regime you start in) and
# on the shock's **size and sign** (a big shock can push $z$ across the
# threshold mid-flight), so KPP define the IRF as a conditional expectation
# and integrate the future out by Monte Carlo:
# $$ GI(h, \delta, \omega_{t-1}) =
#    \mathbb{E}\!\left[y_{t+h} \mid \varepsilon_{j,t}+\delta,\ \omega_{t-1}\right]
#  - \mathbb{E}\!\left[y_{t+h} \mid \omega_{t-1}\right]. $$
# The estimator simulates **paired paths**: a baseline with innovations
# $\varepsilon \sim N(0, I)$ mapped through the *current regime's* Cholesky
# factor, and a shocked twin that adds $\delta$ to identified shock $j$ at
# $h=0$ — same draws otherwise. Along both paths the regime is re-evaluated
# from the simulated $z$ at every step: **switching is endogenous**. Averaging
# over histories drawn from the sample, conditional on the starting regime,
# gives $GI_{\text{calm}}(h)$, $GI_{\text{stress}}(h)$, and the
# transmission-asymmetry statistic
# $$ \Delta(h) = GI_{\text{stress}}(h) - GI_{\text{calm}}(h), $$
# banded by bootstrapping over the drawn histories. Kilian-Vigfusson's check
# completes the toolkit: in a linear model $GI(2\delta) = 2\,GI(\delta)$ and
# $GI(-\delta) = -GI(\delta)$, so plotting $GI(\delta)/\delta$ across sizes
# and signs measures how nonlinear the transmission really is.
#
# ### Baseline Simulation Parameters
#
# | Symbol | Parameter Description | Baseline Value | Units / Accounting Convention |
# | :--- | :--- | :--- | :--- |
# | $c^*$ | True threshold parameter separating regimes | $0.00$ | Units of the simulated FCI (innovation sd = 1) |
# | $d$ | Threshold delay parameter | $1$ | Lag period ($z_{t-1}$) |
# | $T$ | Total simulation length | $600$ | Quarters ($150$ years) |
# | $R$ | Monte Carlo paired paths per history (main GIRF) | $80$ | Paths per history; $40$ histories per starting regime |
# | $A_{\text{calm}}$ | Autoregressive matrix in calm regime | $[[0.5, 0], [-0.1, 0.4]]$ | State transition parameters |
# | $A_{\text{stress}}$ | Autoregressive matrix in stress regime | $[[0.8, 0], [-0.5, 0.4]]$ | Elevated persistence and drag |

# %% [markdown]
# **Intuition.** A linear IRF is a single number per horizon because a linear
# model treats every quarter and every shock size identically. A regime model
# does not: transmission depends on where the economy *is* — and, crucially,
# on where the shock *sends* it. If a financial-stress shock drags the
# economy across the threshold, the shock changes the very dynamics it will
# propagate through; freezing the regime assumes away precisely that
# feedback. The GIRF keeps it: histories come from the sample (so "starting
# in stress" means the stress quarters the data actually contains), paths
# switch regimes endogenously, and the IRF becomes an *average of futures*
# rather than a mechanical recursion. The price is simulation noise; the
# payoff is an object that can honestly differ across regimes, sizes, and
# signs — and a difference band that tells you whether it does.
#
# ### Key References
#
# - **Koop, G., Pesaran, M. H., & Potter, S. M. (1996).** *Impulse response analysis in nonlinear multivariate models.* Journal of Econometrics, 74(1), 119–147.
# - **Kilian, L., & Vigfusson, R. J. (2011).** *Are the responses of the U.S. economy asymmetric in energy price increases and decreases?* Quantitative Economics, 2(3), 419–453.
# - **Tsay, R. S. (1998).** *Testing and modeling multivariate threshold models.* Journal of the American Statistical Association, 93(443), 1188–1202.
# - **Hansen, B. E. (1999).** *Threshold effects in non-dynamic panels: Estimation, testing, and inference.* Journal of Econometrics, 93(2), 345–368.
# - **Krolzig, H.-M. (1997).** *Markov-Switching Vector Autoregressions.* Lecture Notes in Economics and Mathematical Systems. Springer. doi:10.1007/978-3-642-51684-9
# - **Caldara, D., Fuentes-Albero, C., Gilchrist, S., & Zakrajšek, E. (2016).** *The macroeconomic impact of financial and uncertainty shocks.* European Economic Review, 88, 185–207.

# %% [markdown]
# ## Setup — a simulated economy with calm and stress regimes
#
# Two variables: a financial-conditions index ($z_t$, column 0 — the
# threshold variable) and output growth (column 1). We **plant** the state
# dependence: in the stress regime ($z_{t-1} > 0$) the FCI is more
# persistent (0.8 vs 0.5) and drags growth down five times harder (−0.5 vs
# −0.1). The innovation covariance is the *identity in both regimes*, so
# every asymmetry below is transmission, not impact volatility.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf as var_irf
from puremacro.var.regime import girf, tvar_fit
from puremacro.var.regime.threshold import TVARResult

A_CALM = np.array([[0.5, 0.0],      # FCI moderately persistent
                   [-0.1, 0.4]])    # ...and a mild drag on growth
A_STRESS = np.array([[0.8, 0.0],    # FCI very persistent under stress
                     [-0.5, 0.4]])  # ...and a 5x stronger drag on growth

def simulate_economy(T=600, seed=20):
    """Self-exciting TVAR(1): regime decided by last quarter's FCI."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        A = A_STRESS if Y[t - 1, 0] > 0.0 else A_CALM
        Y[t] = A @ Y[t - 1] + rng.standard_normal(2)
    return Y

Y = simulate_economy()
stress_share = float((Y[:, 0] > 0.0).mean())
print(f"T = {len(Y)} quarters | share of stress quarters = {stress_share:.3f}")
assert 0.40 < stress_share < 0.70   # both regimes well populated

fig, ax = _nbstyle.figura(ancho=7.2, alto=3.2)
ax.plot(Y[:, 0], color=_nbstyle.TINTA, linewidth=0.9)
ax.axhline(0.0, color=_nbstyle.SPINE, linewidth=0.9, linestyle="--")
ax.fill_between(np.arange(len(Y)), Y[:, 0].min(), Y[:, 0].max(),
                where=Y[:, 0] > 0.0, color=_nbstyle.NOTA, alpha=0.25, zorder=0)
ax.set_xlabel("Quarter")
ax.set_ylabel("Financial conditions index $z_t$")
ax.set_title("Simulated FCI — shaded spans are stress quarters ($z_t > 0$)")

# %% [markdown]
# ## Fit the threshold VAR
#
# `tvar_fit` grid-searches the threshold $c$ and delay $d$ (Tsay 1998) and
# runs OLS per regime. Nothing tells it the true split is at zero — it has
# to find that, and the planted coefficient gap, on its own. One catch:
# `tvar_fit` only tries delays $d \le p$ and skips larger ones without a
# warning, so at the true lag order $p=1$ the delay is fixed at 1 and
# cannot be "found". We keep the $p=1$ fit for the rest of the notebook and
# run the delay search as a separate check at $p=2$, where $d=2$ is a real
# candidate. A second check is independent of `puremacro`: plain OLS on the
# *true* split ($z_{t-1} > 0$), which says how much of any estimation error
# is sampling noise in this one sample rather than the estimated threshold.

# %%
fit = tvar_fit(Y, threshold_var_idx=0, p=1, delay_grid=(1,), n_threshold_grid=40)
print(f"threshold c = {fit.threshold:+.3f} (true 0.0) | delay d = {fit.delay} (only d <= p = 1 is tried)")
print(f"regime split: {fit.n_low} calm / {fit.n_high} stress quarters")
print("A_low  (calm)  =", np.round(fit.A_low, 3).tolist())
print("A_high (stress)=", np.round(fit.A_high, 3).tolist())

# a delay search that can fail: at p = 2 both d = 1 and d = 2 are admissible
fit_p2 = tvar_fit(Y, threshold_var_idx=0, p=2, delay_grid=(1, 2), n_threshold_grid=40)
print(f"delay search at p = 2 over d in (1, 2): d = {fit_p2.delay} (true 1)")

# independent check: OLS with intercept on the true stress quarters (z_{t-1} > 0)
X_true = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
in_stress = Y[:-1, 0] > 0.0
B_true = np.linalg.lstsq(X_true[in_stress], Y[1:][in_stress], rcond=None)[0]
print(f"OLS on the true split: stress FCI persistence = {B_true[1, 0]:.3f}, "
      f"growth drag = {B_true[1, 1]:.3f} (true 0.8 and -0.5)")
assert fit_p2.delay == 1                              # the true delay wins when d = 2 is allowed
assert abs(fit.threshold) < 0.5                       # near the true zero
assert fit.A_high[1, 0] < fit.A_low[1, 0] - 0.25      # planted cross-effect gap

# %% [markdown]
# **Read the output.** The threshold lands at $+0.236$ (true 0), and when a
# second delay is admissible ($p=2$) the search picks $d=1$, the true delay.
# The estimated stress regime carries both planted fingerprints: FCI
# persistence 0.691 against 0.526 in calm (true 0.8 and 0.5), and a growth
# drag of −0.472 against −0.11 (true −0.5 and −0.1). Why 0.691 and not
# 0.8? About half of the gap is sampling noise in this one sample of 600
# quarters: OLS on the true split gives 0.744. The rest comes from the
# estimated threshold. With $c = +0.236$ the stress regime keeps only
# quarters with $z_{t-1} > 0.236$, so it drops true stress quarters rather
# than adding calm ones. So far this is just estimation; the question is
# what these two coefficient blocks *mean* for a shock — and that is not
# answerable by feeding either block to a linear IRF.

# %% [markdown]
# ## The GIRF: histories, endogenous switching, paired simulation
#
# One call runs the full KPP construction: draw 40 histories per starting
# regime from the sample, simulate 80 paired paths per history with the
# impulse added to the identified FCI shock at $h=0$ (Cholesky within
# regime), let the regime switch endogenously along every path, and average.
# `shock_size=[1, 2, -1, -2]` adds the Kilian-Vigfusson size/sign menu, and
# the between-regime difference comes with a 90% bootstrap band.

# %%
res = girf(fit, Y, shock=0, horizon=16, n_hist=40, n_sim=80,
           shock_size=[1.0, 2.0, -1.0, -2.0], n_boot=300, rng=16)
print(res.summary())

h = np.arange(res.horizon + 1)
g_calm = res.girf_by_regime[0, 0]     # (+1 sd shock) x (H+1, n)
g_strs = res.girf_by_regime[1, 0]
d_gr = res.difference[0, :, 1]
d_lo = res.difference_lo[0, :, 1]
d_hi = res.difference_hi[0, :, 1]
print(f"growth response at h=2: calm {g_calm[2, 1]:+.3f} | stress {g_strs[2, 1]:+.3f} "
      f"(ratio {g_strs[2, 1] / g_calm[2, 1]:.2f})")
print(f"difference (stress-calm) at h=2: {d_gr[2]:+.3f}, 90% band "
      f"[{d_lo[2]:+.3f}, {d_hi[2]:+.3f}]")
below = np.flatnonzero(d_hi < 0.0)
print(f"band entirely below zero at {below.size} of {h.size} horizons: h = {below.tolist()}")
print(f"FCI's own response at h=1: calm {g_calm[1, 0]:+.3f} | stress {g_strs[1, 0]:+.3f}")
assert g_strs[2, 1] < g_calm[2, 1] - 0.10   # stress transmission is stronger...
assert d_hi[2] < -0.10                      # ...and the band excludes zero

# the naive frozen-regime IRF: pretend the economy never leaves stress
naive = var_irf([fit.A_high], np.linalg.cholesky(fit.Sigma_high), 16)[:, :, 0]
cum_naive, cum_girf = naive[:, 1].sum(), g_strs[:, 1].sum()
print(f"cumulative growth loss in stress: frozen-regime {cum_naive:+.2f} "
      f"vs GIRF {cum_girf:+.2f}")
assert cum_naive < cum_girf - 0.30          # freezing the regime overstates it

cols2 = _nbstyle.palette(3)
fig, axes = _nbstyle.figura(ancho=9.4, alto=3.8, ncols=2)
axes[0].axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
axes[0].plot(h, g_calm[:, 1], color=cols2[0], linewidth=1.8, label="start in calm")
axes[0].plot(h, g_strs[:, 1], color=cols2[1], linewidth=1.8, linestyle="--",
             label="start in stress")
axes[0].plot(h, res.girf_pooled[0, :, 1], color=cols2[2], linewidth=1.0,
             linestyle=":", label="pooled")
axes[0].plot(h, naive[:, 1], color=_nbstyle.NOTA, linewidth=1.0, linestyle="-.",
             label="frozen-stress (naive)")
axes[0].set_xlabel("Horizon (quarters)")
axes[0].set_ylabel("Growth response to a +1 sd FCI shock")
axes[0].set_title("Generalized IRF by starting regime")
axes[0].legend(fontsize=8)
axes[1].axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
axes[1].fill_between(h, d_lo, d_hi, color=_nbstyle.NOTA, alpha=0.25,
                     label="90% bootstrap band")
axes[1].plot(h, d_gr, color=_nbstyle.TINTA, linewidth=1.8, label="stress $-$ calm")
axes[1].set_xlabel("Horizon (quarters)")
axes[1].set_ylabel("Difference in growth response")
axes[1].set_title("Regime-dependent transmission test")
axes[1].legend(fontsize=8)
for a in axes:
    a.set_xticks(h[::4])

# %% [markdown]
# **Read the output.** Left panel: the same +1 sd financial shock costs
# **−0.418** growth at $h=2$ when it lands in stress against **−0.238** in
# calm — 1.75 times the damage, purely from where it lands. The naive
# frozen-stress line overstates the stress response (cumulative loss −2.56
# against −1.93 for the GIRF): real paths *escape* to the calm regime as the
# FCI mean-reverts, and the GIRF averages over those exits while the frozen
# recursion forbids them. Right panel: the stress-minus-calm difference at
# $h=2$ is −0.179 with a 90% band of [−0.199, −0.163], and the band lies
# entirely below zero at horizons 0 to 14. That recovers the asymmetry we
# planted. Read the band for what it is: it resamples histories and
# simulation draws with the fitted coefficients held fixed, so it carries
# no estimation uncertainty and is narrower than a full test of
# regime-dependent transmission would be (the stretch prompt below shows a
# case where it collapses to a line). One subtlety worth savoring: the FCI's
# *own* response at $h=1$ is +0.680 starting in calm and +0.688 starting in
# stress, much closer than the fitted persistence (0.526 against 0.691)
# suggests. A positive shock in calm pushes the FCI across the threshold,
# and it then propagates under stress dynamics anyway — endogenous
# switching at work.

# %% [markdown]
# ## Size and sign asymmetry, Kilian-Vigfusson style
#
# In a linear model $GI(\delta)/\delta$ is one curve, whatever $\delta$. In
# a threshold model it is not: a +2 sd shock spends more of its life in the
# stress regime than a +1 sd shock, and a −1 sd shock actively *rescues*
# the economy from stress. Dividing each GIRF by its own $\delta$ makes
# those deviations visible.

# %%
sc = res.scaled()                     # girf_pooled / delta, shape (S, H+1, n)
labels = [f"$\\delta = {d:+.0f}$ sd" for d in res.shock_sizes]
cols = _nbstyle.palette(4)
stys = _nbstyle.styles(4)
dev_size = np.abs(sc[1, :, 1] - sc[0, :, 1]).max()   # |GI(2d)/2 - GI(d)| gap
dev_sign = np.abs(sc[2, :, 1] - sc[0, :, 1]).max()   # |GI(-d)/(-d) - GI(d)| gap
print(f"max size deviation |GI(2)/2 - GI(1)|  (growth): {dev_size:.3f}")
print(f"max sign deviation |GI(-1)/-1 - GI(1)| (growth): {dev_sign:.3f}")
print("per-sd growth response at h=2: " + " | ".join(
    f"{d:+.0f} sd {sc[s, 2, 1]:+.3f}" for s, d in enumerate(res.shock_sizes)))
assert np.abs(sc[:, 0, :] - sc[0, 0, :]).max() < 1e-12  # impact IS proportional
assert dev_size > 0.02 and dev_sign > 0.05              # dynamics are NOT

fig, ax = _nbstyle.figura(ancho=7.0, alto=3.8)
ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
for s in range(4):
    ax.plot(h, sc[s, :, 1], color=cols[s], linestyle=stys[s], linewidth=1.6,
            label=labels[s])
ax.set_xlabel("Horizon (quarters)")
ax.set_ylabel("Scaled growth response  $GI(\\delta)/\\delta$")
ax.set_title("Kilian-Vigfusson size/sign check: one curve iff linear")
ax.legend(fontsize=8)

# %% [markdown]
# **Read the output.** At $h=0$ all four curves coincide *exactly* — the
# impact response is $\mathrm{chol}(\Sigma_r)e_j\delta$, proportional by
# construction — so any spread at $h \geq 1$ is pure transmission
# nonlinearity. And it spreads. At $h=2$ the growth cost per sd is −0.375
# for a +2 sd shock, −0.330 for +1 sd, −0.232 for −1 sd and −0.195 for
# −2 sd. Positive shocks, which recruit the stress regime, transmit more
# damage per sd than negative ones, which pull paths out of it (largest
# sign gap 0.097). Doubling the shock widens the gap further in both
# directions (largest size gap 0.045), because larger impulses relocate
# more paths across the threshold. If someone hands you one IRF for a
# regime model with no $\delta$ on the label, this plot is the question to
# ask.

# %% [markdown]
# ## The linear-limit sanity check — validation you can rerun
#
# How do we know the simulator is right? Force the model to be linear and
# demand the closed form back. Build a `TVARResult` whose two regimes are
# *identical copies* of a fitted linear VAR: the threshold still "switches"
# regimes along every path, but both regimes carry the same coefficients,
# so the GIRF must collapse to `var.irf` — and because the impulse is
# *added* under common random numbers, the paired difference is
# deterministic and the match is to machine precision, not up to Monte
# Carlo noise. This is the same oracle the test suite pins.

# %%
est = estimate_var(Y, 1)
A_hat = np.hstack(est.A_list)
fit_lin = TVARResult(
    A_low=A_hat, intercept_low=est.c, Sigma_low=est.Sigma,
    A_high=A_hat.copy(), intercept_high=est.c.copy(), Sigma_high=est.Sigma.copy(),
    threshold=float(np.median(Y[:, 0])), delay=1, threshold_var_idx=0,
    n_low=0, n_high=0, rss_total=0.0, grid={},
)
chk = girf(fit_lin, Y, shock=0, horizon=16, n_hist=20, n_sim=30, n_boot=100, rng=1)
closed_form = var_irf(est.A_list, np.linalg.cholesky(est.Sigma), 16)[:, :, 0]
gap = np.abs(chk.girf_pooled[0] - closed_form).max()
print(f"max |GIRF - linear IRF| in the linear limit: {gap:.2e}")
print(f"max |regime difference| in the linear limit: {np.abs(chk.difference).max():.2e}")
assert gap < 1e-10
assert np.abs(chk.difference).max() < 1e-12

fig, ax = _nbstyle.figura(ancho=7.0, alto=3.6)
ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
for j, (name, sty) in enumerate([("FCI", "-"), ("growth", "--")]):
    ax.plot(h, closed_form[:, j], color=_nbstyle.TINTA, linestyle=sty, linewidth=1.6,
            label=f"linear IRF — {name}")
    ax.plot(h[::2], chk.girf_pooled[0, ::2, j], "o", color=_nbstyle.NOTA, markersize=4,
            label="GIRF (both series, every 2nd h)" if j == 0 else None)
ax.set_xlabel("Horizon (quarters)")
ax.set_ylabel("Response to a +1 sd FCI shock")
ax.set_title("Linear limit: the GIRF collapses onto the closed form")
ax.legend(fontsize=8)

# %% [markdown]
# **The validation moment.** This is the discipline the notebook wants you
# to steal: every simulation-based estimator should ship with a limit in
# which its answer is known *exactly*. For the KPP GIRF that limit is
# "regimes that do not differ", and the agreement above is at machine
# precision (order $10^{-16}$, not $10^{-2}$). That is only possible
# because the impulse is added to the identified shock under common random
# numbers — a design choice made *for* testability. Be clear about what
# kind of oracle this is: both sides are `puremacro` code (`girf` against
# `var.irf`), so it is an internal consistency check of the simulator, not
# independent evidence. The independent anchors in this notebook are the
# planted data-generating process and the plain OLS on the true split.
# When the linear limit holds and the planted asymmetry is recovered with
# the right sign, the difference band inherits that credibility — within
# the limits of what the band measures.

# %% [markdown]
# ## Your turn — how big a shock breaks proportionality?
#
# The fill-in below recomputes the GIRF for a shock size of your choosing
# and compares it, per sd, against the +1 sd benchmark. Impact is
# proportional by construction, so the test is the *drift* at
# business-cycle horizons: the per-sd growth response minus the +1 sd one,
# averaged over $h = 1,\dots,8$. **Predict its sign before you run.** A
# shock larger than +1 sd pushes more paths into stress, where the growth
# drag is five times stronger, so per sd it should cost *more* growth
# (drift < 0). A shock smaller than +1 sd, or a negative one, recruits
# fewer stress paths or pulls paths out, so per sd it should cost *less*
# (drift > 0). The assert grades that prediction; it holds for every
# $\delta$ in the advertised range and fails when the model has no
# threshold nonlinearity in its dynamics.

# %%
DELTA_TRY = 2.0   # ← change this: shock size in sd, -5 <= delta <= 5 with |delta - 1| >= 0.5 and delta != 0 (try 0.5, 3.0, -2.0, 5.0)
assert -5.0 <= DELTA_TRY <= 5.0 and abs(DELTA_TRY - 1.0) >= 0.5 and DELTA_TRY != 0.0, \
    "outside the advertised range"
res_try = girf(fit, Y, shock=0, horizon=16, n_hist=30, n_sim=60,
               shock_size=[1.0, DELTA_TRY], n_boot=100, rng=2)
sc_try = res_try.scaled()
drift = sc_try[1, :, 1] - sc_try[0, :, 1]    # per-sd growth response minus the +1 sd benchmark
mean_drift = drift[1:9].mean()              # the first two years after impact
print(f"delta = {DELTA_TRY:+.1f} sd -> mean per-sd growth drift vs +1 sd over h = 1..8: "
      f"{mean_drift:+.4f} (largest |drift| {np.abs(drift).max():.3f} at h = {int(np.abs(drift).argmax())})")
# the prediction: drift has the sign of (1 - delta), and is not zero
assert abs(mean_drift) > 1e-3 and np.sign(mean_drift) == np.sign(1.0 - DELTA_TRY), \
    "prediction failed: explain why"

# %% [markdown]
# **Prompts.** (1) *Basic*: before running, predict the sign of the drift
# for `DELTA_TRY = -2.0` and for `0.5`, then run both. Which regime do
# negative FCI shocks recruit, and why does that make them *weaker* per
# sd? (2) *Intermediate*: force the wrong delay with
# `fit_wrong = tvar_fit(Y, threshold_var_idx=0, p=2, delay_grid=(2,), n_threshold_grid=40)`
# (a delay of 2 needs $p \ge 2$, because `tvar_fit` skips $d > p$). Pass
# `fit_wrong` to the GIRF above with the main settings (`n_hist=40,
# n_sim=80, n_boot=300, rng=16`) and to the fill-in cell. Does the
# stress-calm difference at $h=2$ move toward zero? Does the sign
# prediction survive, and what does that say about which conclusions
# depend on getting $d$ right? (3) *Stretch*: fit
# `fit_ms = ms_var_fit(Y, K=2, p=1)` (import it from `puremacro.var.regime`).
# This is an MSIH(2)-VAR(1) in
# Krolzig's (1997) notation: regime-specific intercepts $\mu_k$ and
# covariances $\Sigma_k$, one shared $A$, estimated by EM with the
# Hamilton filter. It is not Hamilton's (1989) model, which switches the
# mean of an AR process with a constant variance. Pass `fit_ms` to the
# *same* `girf` in the fill-in cell and predict, before running, why the
# assert now fails for every $\delta$. Then check that
# `girf(fit_ms, ...).difference[0]` (regime 1 minus regime 0) equals
# $A^h\big(\mathrm{chol}(\Sigma_1) - \mathrm{chol}(\Sigma_0)\big)e_0$ at
# every horizon, from `fit_ms.A` and `fit_ms.Sigma`, and that its band has
# zero width. The data were simulated with identity covariance in both
# regimes: what is this "regime difference" picking up? Does either fitted
# regime persist (inspect `fit_ms.P`)? What would the model need in order
# to generate transmission asymmetry?
#
# ## Why this matters for regime-uncertainty research
#
# The empirical regime-uncertainty literature lives on exactly this object.
# Caldara, Fuentes-Albero, Gilchrist and Zakrajšek (2016, *EER*) argue that
# uncertainty shocks transmit mainly when financial conditions are tight —
# a claim *about* $\Delta(h)$, testable only with a construction that lets
# the economy move between tight and loose states mid-response, as here.
# On the structural side, this package's regime-dependent
# Diamond-Mortensen-Pissarides model
# (`puremacro.models.dmp_regime_dependent`, `dmp_irf(..., regime="H")` vs
# `regime="L"`) produces the same phenomenon from theory — volatility
# regimes changing the surplus calculus of job creation — and the GIRF is
# the reduced-form lens you would point at data generated by such a model.
# The honest workflow the two share: state the regime model, simulate it
# forward *with switching intact*, and report the regime difference with a
# band, never a frozen-regime IRF.
#
# **How comprehensive is this?** `girf` dispatches on all three regime fits
# in `puremacro.var.regime` — `tvar_fit` (used here), `tvecm_fit`
# (threshold cointegration, responses in levels), and `ms_var_fit` (an
# MSIH VAR with a shared $A$, fitted by EM; regime paths drawn from the
# fitted transition matrix, so its GIRF differs across regimes only through
# $\mathrm{chol}(\Sigma_k)$). Single-equation
# alternatives live in `puremacro.lp` (`lp_state_dep` for
# Auerbach-Gorodnichenko-style state-dependent local projections);
# `puremacro.uncertainty.regimes` dates the regimes themselves
# (Bai-Perron breaks, calendar regimes); and notebook 06 covers the
# identification conventions the Cholesky-within-regime choice inherits.
