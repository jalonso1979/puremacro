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
# # Growth-at-Risk: the vulnerable left tail
#
# How bad could growth get? Not the average outlook, but the *downside*: the
# Adrian-Boyarchenko-Giannone (2019) result is that tight financial conditions
# shift the **lower** quantiles of future GDP growth far more than the median —
# downside risk is the whole story. We build a financial-conditions index, run a
# quantile autoregression, and fit a skew-t conditional density with
# `puremacro.gar`, all in the browser on synthetic data.

# %% [markdown]
# ## The method in math
#
# **Quantile regression.** Ordinary least squares targets the conditional *mean*;
# to trace a conditional *quantile* we replace the squared loss with the asymmetric
# **check (pinball) loss**
# $$ \rho_\tau(u) = u\,\bigl(\tau - \mathbb{1}\{u<0\}\bigr),\qquad \tau\in(0,1). $$
# Minimizing $\sum_t \rho_\tau\!\bigl(y_{t+h}-x_t'\beta\bigr)$ over $\beta$ returns the
# coefficients of the conditional quantile
# $$ Q_\tau\!\left(y_{t+h}\mid x_t\right) = x_t'\beta(\tau). $$
# Under-prediction is penalized with weight $\tau$ and over-prediction with weight
# $1-\tau$; for $\tau=\tfrac12$ this is least-absolute-deviations (the median), and
# small $\tau$ tilts the fit toward the lower tail. Run one regression per $\tau$ and
# the slopes $\beta(\tau)$ are allowed to *differ* across the distribution — that is
# exactly what we test.
#
# **Growth-at-Risk** is the lower-tail conditional quantile,
# $$ \text{GaR}_{0.05} \;=\; Q_{0.05}\!\left(y_{t+h}\mid x_t\right), $$
# the level of growth that is exceeded with 95% probability given today's conditions.
# ABG's empirical finding is asymmetry: with a financial-conditions index (FCI) as the
# conditioner $x_t$, the slope $\partial Q_\tau/\partial\text{FCI}$ is sharply negative
# at $\tau=0.05$ but near zero at the median — tightness eats the left tail, not the center.
#
# **Skew-t density.** A handful of estimated quantiles $\{Q_\tau\}$ already sketch a
# distribution; ABG recover the *whole* conditional density by fitting Azzalini's
# **skew-t** $f(\cdot\mid\mu,\sigma,\alpha,\nu)$ by quantile matching,
# $$ (\hat\mu,\hat\sigma,\hat\alpha,\hat\nu)=\arg\min \sum_\tau\bigl(F^{-1}(\tau;\mu,\sigma,\alpha,\nu)-Q_\tau\bigr)^2, $$
# which gives a location $\mu$, scale $\sigma$, **skew** $\alpha$, and tail-thickness
# (degrees of freedom) $\nu$ — four shape parameters where a Gaussian has only two.
#
# **What this notebook computes.** A synthetic FCI from a five-indicator panel; a
# quantile autoregression of GDP growth on two own lags plus that FCI, at the grid
# $\tau\in\{0.05,0.50,0.95\}$ (and $\{0.05,0.25,0.50,0.75,0.95\}$ for the density); and
# a skew-t fit to the predicted quantiles at a loose vs. a tight FCI.

# %% [markdown]
# **Intuition.** The conditional *median* can look perfectly healthy while the *left
# tail* quietly deteriorates — an economy can have a respectable central forecast and
# still be one shock away from a sharp contraction. Growth-at-Risk reads that left tail
# directly, so it moves when *downside* risk builds even if the point forecast does not.
# Why a skew-t rather than a Gaussian? Recessions are not mirror images of booms: the
# distribution of growth is **left-skewed** (the bad tail is longer) and **fat-tailed**
# (extreme contractions are more likely than a bell curve allows). The skew-t's $\alpha$
# captures the asymmetry and its $\nu$ the heavy tails, so it tracks the conditional
# distribution of macro growth where a two-parameter Gaussian cannot.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.gar import qar, fci, fit_skewt_to_quantiles

rng = np.random.default_rng(20240529)

# %% [markdown]
# ## 1. A financial-conditions index from a synthetic indicator panel
# A persistent latent "tightness" factor drives five financial indicators
# (credit spreads and volatility load positively; asset prices negatively).
# `fci` extracts the first principal component and sign-normalizes it so that
# **higher = tighter conditions**.

# %%
T = 320  # quarters
f = np.zeros(T)
rho_f = 0.85
for t in range(1, T):
    f[t] = rho_f * f[t - 1] + 0.6 * rng.standard_normal()
load = np.array([1.0, 0.8, 1.2, -0.9, 0.7])             # equity loads negative
fin = f[:, None] * load + 0.4 * rng.standard_normal((T, load.size))
fin_df = pd.DataFrame(fin, columns=["spread", "vol", "credit", "equity", "term"])

fci_out = fci(fin_df, tightening_columns=["spread", "vol", "credit"])
FCI = fci_out["index"].values                           # standardized; higher = tighter
var_exp = fci_out["var_explained"]
print(f"FCI variance explained by first PC = {var_exp:.3f}")
assert 0.3 < var_exp < 1.0, var_exp

# %% [markdown]
# ## 2. Synthetic GDP growth with a vulnerable left tail
# Tighter conditions shave the conditional mean *and* fatten the lower tail
# (the downside shock scale rises with `FCI`, the upside scale stays flat) —
# the Adrian-Boyarchenko-Giannone "vulnerable growth" mechanism.

# %%
g = np.zeros(T)
phi, mu0, beta_mean = 0.30, 2.0, -0.35
for t in range(1, T):
    tight = FCI[t - 1]
    scale_dn = 1.0 + 1.6 * np.maximum(tight, 0.0)       # fat left tail when tight
    e = rng.standard_normal()
    shock = scale_dn * min(e, 0.0) + 1.0 * max(e, 0.0)
    g[t] = mu0 + phi * (g[t - 1] - mu0) + beta_mean * tight + shock

# %% [markdown]
# ## 3. Quantile autoregression with FCI as a control
# `qar` fits one row per `(h, tau)`; with `p=2` lags plus one control the
# coefficient layout is `beta0` (intercept), `beta1, beta2` (AR lags), `beta3`
# (the FCI slope). The asymmetry test: the 5th-percentile FCI slope is far more
# negative than the median's.

# %%
taus = (0.05, 0.50, 0.95)
out = qar(g, quantiles=taus, horizons=(1,), p=2,
          controls=FCI, n_boot=200, alpha=0.10, seed=7)
fci_col = "beta3"                                        # p=2 lags -> FCI is beta3
slopes = out.set_index("tau")[fci_col]
qpred = out.set_index("tau")["q_pred"]
print(out[["tau", "q_pred", "lo", "hi", fci_col]].to_string(index=False))

slope_05, slope_50, slope_95 = slopes.loc[0.05], slopes.loc[0.50], slopes.loc[0.95]
print(f"\nFCI slope:  tau=0.05 -> {slope_05:.3f}   tau=0.50 -> {slope_50:.3f}"
      f"   tau=0.95 -> {slope_95:.3f}")
assert slope_05 < slope_50, (slope_05, slope_50)                 # left tail more exposed
assert abs(slope_05) > abs(slope_50), (slope_05, slope_50)       # ABG asymmetry
assert qpred.loc[0.05] < qpred.loc[0.50] < qpred.loc[0.95], qpred.to_dict()

# %% [markdown]
# **Read the output.** The three `beta3` numbers are the FCI loadings of the three
# quantiles, and they are the heart of the result. The median slope is small — a
# tighter FCI barely moves the central forecast — while the 5th-percentile slope is
# large and negative: tightening drags the *bottom* of the distribution down hard.
# That gap (`|slope τ=0.05| ≫ |slope τ=0.50|`) is the ABG asymmetry, and it is why the
# `q_pred` column already fans out, with the 5th percentile far below the median and
# the 95th. The `lo`/`hi` bootstrap band around each `q_pred` confirms the lower
# quantile is estimated, not an artifact of one draw.

# %% [markdown]
# ### Hero figure — the Growth-at-Risk fan
# Holding the AR lags at their mean, sweep `FCI` from loose to tight. The median
# barely moves while the 5th-percentile (the Growth-at-Risk line) plunges: the
# 5-95 band is fanning open downward.

# %%
gbar = g.mean()
fci_grid = np.linspace(np.percentile(FCI, 2), np.percentile(FCI, 98), 60)
B = out.set_index("tau")
def qhat(tau, x):
    r = B.loc[tau]
    return r["beta0"] + (r["beta1"] + r["beta2"]) * gbar + r[fci_col] * x
fan = {tau: np.array([qhat(tau, x) for x in fci_grid]) for tau in taus}

cols = _nbstyle.palette(3)
sty = _nbstyle.styles(3)
fig, ax = plt.subplots()
ax.fill_between(fci_grid, fan[0.05], fan[0.95], color="0.88",
                label="5-95% conditional band")
ax.plot(fci_grid, fan[0.50], color=cols[0], linestyle=sty[0], label="median")
ax.plot(fci_grid, fan[0.05], color=cols[1], linestyle=sty[1], label="5th pct (GaR)")
ax.plot(fci_grid, fan[0.95], color=cols[2], linestyle=sty[2], label="95th pct")
ax.set_xlabel("Financial conditions index (higher = tighter)")
ax.set_ylabel("Conditional GDP growth (%)")
ax.set_title("Growth-at-Risk fan: the left tail widens as conditions tighten")
ax.legend(loc="lower left", fontsize=8)
plt.show()

# %% [markdown]
# ### Supporting — skew-t conditional densities, loose vs tight
# Following ABG 2019, fit Azzalini's skew-t to the full set of predicted
# quantiles at a loose (10th-pct) and a tight (90th-pct) FCI. The tight density
# is left-skewed (`alpha < 0`) with a 5% Growth-at-Risk far below the loose case.

# %%
taus_full = (0.05, 0.25, 0.50, 0.75, 0.95)
out_full = qar(g, quantiles=taus_full, horizons=(1,), p=2,
               controls=FCI, n_boot=50, alpha=0.10, seed=11)
Bf = out_full.set_index("tau")
def qdict_at(x):
    return {tau: float(Bf.loc[tau, "beta0"] + (Bf.loc[tau, "beta1"]
                       + Bf.loc[tau, "beta2"]) * gbar + Bf.loc[tau, fci_col] * x)
            for tau in taus_full}

loose_val, tight_val = np.percentile(FCI, 10), np.percentile(FCI, 90)
fit_loose = fit_skewt_to_quantiles(qdict_at(loose_val))
fit_tight = fit_skewt_to_quantiles(qdict_at(tight_val))
gar_loose, gar_tight = fit_loose.downside_quantile(0.05), fit_tight.downside_quantile(0.05)
print(f"loose: alpha={fit_loose.alpha:+.2f}  5% GaR={gar_loose:.2f}")
print(f"tight: alpha={fit_tight.alpha:+.2f}  5% GaR={gar_tight:.2f}")
assert gar_tight < gar_loose, (gar_tight, gar_loose)             # GaR worse when tight
assert fit_tight.alpha < 0.0, fit_tight.alpha                    # left-skewed when tight

xs = np.linspace(-8, 8, 400)
c2 = _nbstyle.palette(2)
fig, ax = plt.subplots()
ax.plot(xs, fit_loose.pdf(xs), color=c2[0], label=f"loose FCI  (5% GaR {gar_loose:.1f})")
ax.plot(xs, fit_tight.pdf(xs), color=c2[1], linestyle=(0, (4, 2)),
        label=f"tight FCI  (5% GaR {gar_tight:.1f})")
ax.axvline(gar_loose, color=c2[0], linewidth=0.7, linestyle=":")
ax.axvline(gar_tight, color=c2[1], linewidth=0.7, linestyle=":")
ax.set_xlabel("GDP growth (%)"); ax.set_ylabel("Conditional density")
ax.set_title("Skew-t densities: tight conditions push mass into the left tail")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# **Read the output.** Across the **fan** (hero figure) the median line is nearly
# flat in the FCI while the 5th-percentile line dives — the 5-95 band opens *downward*
# as conditions tighten, the visual signature of vulnerable growth. The **skew-t
# densities** make the mechanism explicit: the loose density is roughly symmetric, but
# the tight density is visibly left-skewed (`alpha < 0`) with extra mass pushed into
# the lower tail. The headline number is the 5% Growth-at-Risk — the dotted vertical
# line: it sits at `gar_loose` under loose conditions and falls to a much lower
# `gar_tight` when conditions tighten (`gar_tight < gar_loose`). The same shift that
# left the median almost untouched moves the downside scenario by several points of
# growth — which is the entire point of looking at the tail rather than the center.

# %% [markdown]
# ## Your turn — pick where on the FCI to read the conditional distribution
#
# The conditional growth distribution depends on *where* financial conditions sit.
# Below we evaluate the full skew-t fit at a chosen FCI percentile (a working default
# of the 90th — fairly tight) and read off its 5% Growth-at-Risk and median. Change
# `fci_pct` and re-run. The downstream `assert`s check three *robust* facts that hold
# anywhere on the FCI: the predicted quantiles are monotone in $\tau$, the 5% GaR never
# exceeds the median, and the fitted skew-t puts ≈5% of its mass below its own 5%
# quantile. (We deliberately do **not** assert on the fit's `success` flag — the
# Nelder-Mead quantile-match often reports `False` even when the fitted quantiles match
# to three decimals, so it is not a reliable convergence signal.)

# %%
# ← change this: the FCI percentile to evaluate the distribution at
#   (0 = loosest conditions in-sample, 100 = tightest).
fci_pct = 90
x_eval = np.percentile(FCI, fci_pct)
qd_you = qdict_at(x_eval)                                # {tau: predicted Q_tau}
fit_you = fit_skewt_to_quantiles(qd_you)
gar5 = fit_you.downside_quantile(0.05)                   # the 5% Growth-at-Risk
med = fit_you.downside_quantile(0.50)                    # the conditional median
cdf_at_gar = float(fit_you.cdf(np.array([gar5]))[0])     # skew-t mass below its own 5% quantile
print(f"FCI pct {fci_pct} (value {x_eval:+.2f}):  5% GaR = {gar5:.2f}   median = {med:.2f}")
print(f"skew-t CDF at the fitted 5% quantile = {cdf_at_gar:.3f}  (target 0.05)")

q_sorted = [qd_you[t] for t in sorted(qd_you)]
assert all(b >= a - 1e-9 for a, b in zip(q_sorted, q_sorted[1:]))   # lower tau -> weakly lower quantile
assert gar5 <= med + 1e-9                                           # 5% GaR never above the median
assert abs(cdf_at_gar - 0.05) < 0.02                                # skew-t cdf recovers the 5% level

# %% [markdown]
# **Prompts.** (1) *Basic:* set `fci_pct = 10` (loose) and re-run — the 5% GaR jumps up
# (less downside risk) while all three assertions still hold; the gap between the GaR
# and the median shrinks because the loose-conditions density is closer to symmetric.
# (2) *Intermediate:* compute the GaR at the 10th and 90th percentiles and take the
# difference — this *Growth-at-Risk spread* is the single number ABG use to summarize
# how much downside risk financial conditions are pricing. (3) *Stretch:* raise the
# downside-scale coefficient `1.6` in §2 (e.g. to `2.5`), re-run the whole notebook, and
# watch the tight-conditions `alpha` turn more negative and the GaR spread widen — a
# more nonlinear left tail in the data-generating process shows up as stronger skew in
# the fit.
#
# **How comprehensive is this?** `puremacro.gar` is the full vulnerable-growth toolkit:
# `qar` (quantile autoregression, used here), `lp_quantile` (the Adrian-Boyarchenko-
# Giannone quantile *local projection* for a panel, when a second variable is at hand),
# the skew-t family (`fit_skewt_to_quantiles` / `SkewTFit` with `skewt_pdf`/`cdf`/`ppf`,
# plus `expected_shortfall` and `downside_entropy`), and `fci` / `fci_rolling` for the
# conditioning index. The volatility side of "how risky is the world" lives in
# `puremacro.garch` — GARCH(1,1) and DCC conditional volatility (Notebook 08) — and the
# two meet in **Notebook 13**, which uses `gar.fci` as one kernel of a build-your-own
# uncertainty index.
