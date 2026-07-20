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
# # A life over time: consumption, saving, demographics
#
# A finite-horizon household earns a hump-shaped income, saves against income
# risk, then runs assets down toward the end of life. We solve the life-cycle
# problem, look at the cohort wealth distribution by age, and weight cohorts by a
# demographic profile (with and without mortality) to see how an aging population
# reshapes aggregate capital. All with `puremacro.vfi`.

# %% [markdown]
# ## The method in math
#
# A household lives ages $j = 0, 1, \dots, J-1$ (here $J = 40$). Unlike the
# infinite-horizon Aiyagari problem, this is **not** a fixed point: with a known
# terminal value it is solved by *backward induction* — exactly $J$ applications of
# the one-step Bellman operator, from the last age to the first. At age $j$, with
# assets $a$ and idiosyncratic productivity $z$,
# $$ V_j(a,z) = \max_{a' \ge 0}\; u(c) + \beta\, s_j\, \mathbb{E}\!\left[V_{j+1}(a',z') \mid z\right],
# \qquad c = \kappa_j\, e^{z} + (1+r)\,a - a', $$
# with CRRA utility $u(c) = (c^{1-\gamma}-1)/(1-\gamma)$ ($\gamma = 2$ here) and the
# **terminal condition** $V_{J} \equiv 0$ — the household values nothing after the
# last age, so it leaves **no bequest** and runs assets down. Income is the product
# of a *deterministic hump* $\kappa_j = \exp(0.1\,j - 0.0025\,j^2)$ and an AR(1) risk
# $e^{z}$; the rate $r$ is fixed (partial equilibrium) and borrowing is ruled out by
# $a' \ge 0$. The factor $s_j \in (0,1]$ is the conditional probability of surviving
# age $j$, so the *effective* discount from one age to the next is $\beta\, s_j$.
#
# A cohort is born at age $0$ holding (almost) no assets and ages
# deterministically: $\mu_{j+1} = \Pi^{\!\top}_z\, T_{j}\,\mu_j$, where $T_j$ is age
# $j$'s saving policy and $\Pi_z$ the productivity transition. To turn these cohorts
# into a *cross-section* we weight age $j$ by its **population mass** $\ell_j$, with
# $\ell_0 = 1$ and $\ell_{j+1} = \ell_j\, s_j / (1+n)$ (then normalized to sum to one).
# With no mortality ($s_j \equiv 1$) and no population growth ($n = 0$) every cohort
# carries equal weight $1/J$; mortality makes the old fewer, so their (high) asset
# holdings count for less in aggregate capital $K = \sum_j \ell_j\, \mathbb{E}_j[a]$.
#
# **Intuition.** This is the permanent-income / life-cycle logic. A young household
# *knows* its earnings will rise toward the middle-age peak of $\kappa_j$, so it
# would like to borrow against that future income to smooth consumption — and is
# blocked only by $a' \ge 0$, which is why the young hold near-zero assets. Through
# working life, saving against falling later-life earnings and self-insurance against the AR(1) shock
# build a buffer of wealth that **peaks near the end of working life**; with a
# zero terminal value and no bequest motive, the household then *decumulates*,
# spending the buffer down before the last age. Mortality weighting matters for the
# cross-section because the asset-rich are concentrated among the old — precisely
# the cohorts a declining survival schedule thins out.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import life_cycle_profile, stationary_age_weights

# %% [markdown]
# ## 1. The life-cycle profile
# `life_cycle_profile` solves a 40-age consumption-saving problem (hump-shaped
# deterministic earnings × AR(1) risk, no borrowing, no bequest) and returns mean
# assets and consumption by age plus the age-indexed cohort distribution.

# %%
lc = life_cycle_profile(horizon=40, n_z=5, n_a=120, a_max=40.0)
assets_by_age = lc["assets_by_age"]
cons_by_age = lc["consumption_by_age"]
dist = lc["life_cycle_dist"]                               # (J, n_a, n_z)
J = assets_by_age.shape[0]
age = np.arange(J)
print(f"peak mean assets {assets_by_age.max():.2f} at age {assets_by_age.argmax()}")
assert assets_by_age.max() > assets_by_age[0]             # assets build up...
assert assets_by_age[-1] < assets_by_age.max()           # ...then draw down
assert np.all(cons_by_age > 0)

# %% [markdown]
# **Read the output.** Mean assets start at essentially zero — newborns enter with
# nothing and, blocked by the no-borrowing constraint, cannot front-load the
# earnings they expect later. They then climb steadily and peak in the interior
# (the printed peak age is well before the final age $J-1 = 39$), the textbook
# life-cycle hump: this is saving against falling later-life earnings plus a precautionary buffer against
# the AR(1) shock. After the peak the profile turns down — with $V_J \equiv 0$ and
# no bequest the household optimally spends its wealth down toward the end of life,
# though not all the way to zero given the survival/income mix. Consumption stays
# strictly positive throughout and is visibly *smoother* than the income hump that
# drives it: that smoothing is exactly what the saving policy buys.

# %% [markdown]
# ### Hero figure — the hump
# Assets accumulate over working life and decumulate toward the end; consumption
# is much smoother than the underlying income hump.

# %%
cols = _nbstyle.palette(2)
fig, ax = plt.subplots()
ax.plot(age, assets_by_age, color=cols[0], label="mean assets")
ax.plot(age, cons_by_age, color=cols[1], linestyle="--", label="mean consumption")
ax.set_xlabel("Age (model period)"); ax.set_ylabel("Level")
ax.set_title("Life-cycle assets and consumption"); ax.legend()
plt.show()

# %% [markdown]
# ### Supporting — the cohort wealth distribution by age
# Each column is one age's wealth distribution (marginal over income); the mass
# fans out into higher assets through working life, then contracts.

# %%
dist_a = dist.sum(axis=2)                                 # (J, n_a) marginal over z
a_grid = np.linspace(0.0, 40.0, dist_a.shape[1])
fig, ax = plt.subplots(figsize=(7.0, 3.6))
im = ax.imshow(dist_a.T, origin="lower", aspect="auto", cmap="Greys",
               extent=[0, J - 1, a_grid[0], a_grid[-1]])
ax.set_ylim(0, 25)
ax.set_xlabel("Age"); ax.set_ylabel("Assets a")
ax.set_title("Cohort wealth distribution by age")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mass")
plt.show()

# %% [markdown]
# ## 2. Demographics: who is alive, and how much capital they hold
# Without mortality every cohort has equal mass (`1/J`). Add a declining survival
# schedule and the old are fewer — so their (high) asset holdings get less weight
# in the aggregate. Aggregate capital per capita = Σ_age weight × mean assets.

# %%
surv = np.clip(1.0 - 0.0015 * age, 0.85, 1.0)   # gentle decline; oldest cohort ~1/3 the youngest
w_uniform = stationary_age_weights(J)                     # no mortality → 1/J
w_mortality = stationary_age_weights(J, survival=surv)
K_uniform = float(np.sum(w_uniform * assets_by_age))
K_mortality = float(np.sum(w_mortality * assets_by_age))
print(f"aggregate capital per capita: uniform {K_uniform:.2f}, "
      f"with mortality {K_mortality:.2f}")
assert np.isclose(w_uniform.sum(), 1.0) and np.isclose(w_mortality.sum(), 1.0)
assert K_mortality < K_uniform                            # fewer asset-rich elderly

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
a1.plot(age, w_uniform, color=cols[0], label="no mortality")
a1.plot(age, w_mortality, color=cols[1], linestyle="--", label="with mortality")
a1.set_xlabel("Age"); a1.set_ylabel("Population mass"); a1.set_title("Demographic weights")
a1.legend()
a2.plot(age, w_uniform * assets_by_age, color=cols[0], label="no mortality")
a2.plot(age, w_mortality * assets_by_age, color=cols[1], linestyle="--", label="with mortality")
a2.set_xlabel("Age"); a2.set_ylabel("Weighted assets"); a2.set_title("Capital contribution by age")
a2.legend()
plt.show()

# %% [markdown]
# ## Your turn — does the hump survive a different discount factor?
#
# The baseline solved a $\beta = 0.96$ household. Patience is the central lever on
# saving: a more patient household values the future more, so it accumulates a
# *taller* buffer; an impatient one saves less. Re-solving the life-cycle problem
# is cheap (it is just $J$ backward Bellman steps, no equilibrium loop), so change
# `beta_you` below and watch the peak move — while the *shape* stays a hump.
#
# The assert checks a **robust structural fact**, not a fragile magnitude: for any
# admissible $\beta$ the peak of mean assets lands strictly in the interior (after
# age 0, before the last age), because the no-borrowing constraint pins the young
# near zero and the zero terminal value forces decumulation at the end.

# %%
# ← Change this discount factor (patience). Keep it strictly in (0, 1).
beta_you = 0.98
lc_you = life_cycle_profile(horizon=40, n_z=5, n_a=120, a_max=40.0, beta=beta_you)
assets_you = lc_you["assets_by_age"]
peak_age_you = int(assets_you.argmax())
print(f"beta={beta_you}: peak mean assets {assets_you.max():.2f} at age {peak_age_you} "
      f"(baseline beta=0.96 → {assets_by_age.max():.2f} at age {assets_by_age.argmax()})")
# Hump-shaped: the peak is interior (not the first or the last age) for any valid beta.
assert 0 < peak_age_you < J - 1, peak_age_you
assert assets_you[0] < assets_you.max() and assets_you[-1] < assets_you.max()
# More patient → at least as much peak wealth as the (less patient) 0.96 baseline.
assert (beta_you < 0.96) or (assets_you.max() >= assets_by_age.max() - 1e-9)

# %% [markdown]
# **Prompts.** (1) Set `beta_you = 0.92` (impatient): the peak shrinks and shifts
# slightly later, but it is still interior — the hump is structural, not a knife-edge.
# (2) Push `beta_you = 0.99` (very patient) and compare peak wealth to the baseline;
# the household over-saves but still decumulates by the last age. (3) *Stretch:* steepen
# the demographic `surv` schedule above (e.g. `0.0030 * age`, still clipped into `(0, 1]`),
# recompute `w_mortality` and aggregate capital, and confirm `K_mortality` falls further as
# the asset-rich elderly are thinned out. (For mortality acting *inside* the solve — the
# effective discount $\beta s_j$ — build a `FiniteHorizonProblem` directly with its
# `survival=` argument.)
#
# **How comprehensive is this?** `life_cycle_profile` drives the finite-horizon stack
# (`FiniteHorizonProblem` → `life_cycle_distribution` → `age_profile`) and
# `stationary_age_weights` adds demographics; `puremacro.vfi` closes the loop with full
# **OLG general equilibrium and endogenous labor** (`olg_stationary_equilibrium`). The
# same solve → distribution → market-clearing machinery powers the other showcase
# notebooks: infinite-horizon **Aiyagari / Huggett** wealth inequality (NB01),
# **Krusell-Smith** aggregate risk and transition paths (NB02), **Hopenhayn** firm
# entry and exit (NB04), and **two-asset** portfolios with EGM and Epstein-Zin
# preferences (NB05).
