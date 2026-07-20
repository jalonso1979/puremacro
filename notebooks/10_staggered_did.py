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
# # Staggered difference-in-differences
#
# Did a policy work, when different units adopted it in different years? When
# treatment timing is *staggered*, the textbook two-way fixed-effects (TWFE)
# regression silently uses already-treated units as controls and returns a
# contaminated number. We plant a known effect in a synthetic staggered panel
# and recover it cleanly with the modern heterogeneity-robust estimators in
# `puremacro.did` — **Callaway-Sant'Anna** group-time ATTs and the
# **Sun-Abraham** interaction-weighted event study. Everything runs in the
# browser on synthetic data.

# %% [markdown]
# ## The method in math
#
# **Setting.** Unit $i$ is first treated at calendar time $G_i = g$ (its *cohort*); units that
# are never treated have $G_i=\infty$. Potential outcomes are $Y_{it}(0)$ (untreated path) and
# $Y_{it}(g)$ (treated-at-$g$ path); we only ever see one of them.
#
# **The TWFE pitfall.** The default applied specification regresses
# $$ Y_{it} = \alpha_i + \lambda_t + \beta\,D_{it} + \varepsilon_{it}, \qquad
# D_{it}=\mathbb{1}\{t \ge G_i\}, $$
# and reads $\hat\beta$ as "the" treatment effect. **Goodman-Bacon (2021)** shows $\hat\beta$ is
# a weighted average of *all* possible $2\times2$ DiD comparisons — including **forbidden** ones
# that use *already-treated* units as the control group. When effects are heterogeneous and
# *time-varying*, those comparisons subtract a moving treated trend, so some weights go
# **negative**: $\hat\beta$ can have the wrong sign even when every unit's true effect is positive.
#
# **Callaway-Sant'Anna (2021).** Sidestep the regression. Define a clean **group-time ATT** for
# cohort $g$ at time $t$, differenced off the cohort's last pre-treatment period $g-1$ and a
# *never-treated* (or not-yet-treated) control set $C$:
# $$ \text{ATT}(g,t)=\mathbb{E}\!\left[Y_t-Y_{g-1}\mid G=g\right]
# -\mathbb{E}\!\left[Y_t-Y_{g-1}\mid C\right]. $$
# The first term is the cohort's own change from base period $g-1$ to $t$; the second nets out the
# control group's change over the same window, leaving the causal effect.
# Aggregating the $\text{ATT}(g,t)$ by elapsed time $e=t-g$ gives the **event-study profile**
# $\text{ATT}(e)$; averaging the post-treatment cells gives one **overall ATT**.
#
# **Sun-Abraham (2021).** Repair the *dynamic* TWFE event study instead: estimate
# **interaction-weighted** cohort$\times$relative-time coefficients $\text{ATT}(e,g)$ and combine
# them with cohort-size shares $w_g = n_g/\sum_{g'} n_{g'}$,
# $\text{ATT}_{\text{SA}}(e)=\sum_g w_g\,\text{ATT}(e,g)$ — so cohort $g$'s profile is never
# contaminated by other cohorts' effects. This notebook computes **both**: CS (an *unweighted*
# cohort-mean aggregation) and SA (the *cohort-share-weighted* aggregation) of the **same**
# $\text{ATT}(g,t)$ matrix.
#
# **Intuition.** The bias is about *forbidden comparisons*. A late-adopting cohort, differenced
# against an early-adopting cohort that is *already* treated, attributes the early cohort's
# ongoing dynamic response to the late cohort — with a sign that can flip the headline. CS and SA
# restrict every comparison to a *clean* control (never-treated or not-yet-treated) and a *fixed*
# pre-period $g-1$, then aggregate transparently. The event-study profile they report is the
# dynamic ATT by horizon $e$ *since* treatment: how the effect builds (or fades) after switch-on.

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

from puremacro.did import (
    callaway_santanna, sun_abraham,
    CallawaySantannaResult, SunAbrahamResult,
)

# %% [markdown]
# ## 1. A synthetic staggered-adoption panel with a known ATT
# Three cohorts are first treated at t = 5, 8, 11; a fourth group is never
# treated. The DGP is two-way fixed effects with parallel pre-trends and a
# *constant* treatment kick `tau = +1.0`, so the true event-study profile is
# flat at +1 from event time 0 onward.

# %%
rng = np.random.default_rng(20240529)

N_UNITS, T = 90, 14
COHORTS = (5, 8, 11, np.nan)          # three treated cohorts + never-treated
TRUE_ATT, SIGMA = 1.0, 0.3

assign = rng.choice(len(COHORTS), size=N_UNITS, replace=True)
unit_fe = rng.standard_normal(N_UNITS)
time_fe = rng.standard_normal(T) * 0.3

rows = []
for i in range(N_UNITS):
    g = COHORTS[assign[i]]
    for t in range(1, T + 1):
        treated = 1.0 if (not pd.isna(g)) and t >= g else 0.0
        y = unit_fe[i] + time_fe[t - 1] + TRUE_ATT * treated + SIGMA * rng.standard_normal()
        rows.append({"unit": i, "time": t, "y": y, "treat_time": g})
panel = pd.DataFrame(rows)

n_never = int(panel.groupby("unit")["treat_time"].first().isna().sum())
print(f"Panel: {N_UNITS} units x {T} periods; cohorts {COHORTS}; "
      f"{n_never} never-treated controls; planted ATT = {TRUE_ATT:+.1f}")

# %% [markdown]
# **Intuition.** This is the cleanest possible world for the estimators: parallel pre-trends
# (the only thing separating units before treatment is the fixed effect `unit_fe`), a genuine
# never-treated control group, and a homogeneous effect. If an estimator cannot recover `+1.0`
# *here*, it has no business being trusted on messy data. The never-treated group is what lets
# Callaway-Sant'Anna avoid the forbidden already-treated comparisons entirely.

# %% [markdown]
# ## 2. Two heterogeneity-robust estimators on the same panel
# `callaway_santanna` builds the group-time ATT(g,t) matrix off never-treated
# controls with a panel bootstrap; `sun_abraham` re-aggregates the *same* matrix
# with cohort-size shares. A naive TWFE regression would be contaminated by
# already-treated units used as controls — these estimators are not.

# %%
cs = callaway_santanna(panel, unit="unit", time="time", outcome="y",
                       treat_time="treat_time", n_boot=400, alpha=0.10, seed=0)
sa = sun_abraham(panel, unit="unit", time="time", outcome="y",
                 treat_time="treat_time", n_boot=400, alpha=0.10, seed=0)

assert isinstance(cs, CallawaySantannaResult)
assert isinstance(sa, SunAbrahamResult)

print(cs.summary())
print(sa.summary())

# %% [markdown]
# ## 3. Recovery checks (asserts on headline numbers)
# The aggregate ATT recovers the truth and lies inside its bootstrap CI; the
# base period (event time -1) is the normalization point (exactly 0); and every
# post-treatment horizon tracks the constant +1.0.

# %%
es_cs = cs.att_event_study.sort_values("event_time").reset_index(drop=True)

assert abs(cs.att_overall - TRUE_ATT) < 0.25, cs.att_overall
assert abs(sa.att_overall - TRUE_ATT) < 0.25, sa.att_overall

post = es_cs[es_cs["event_time"] >= 0]
lo_overall, hi_overall = float(post["lo"].mean()), float(post["hi"].mean())
assert lo_overall <= TRUE_ATT <= hi_overall, (lo_overall, hi_overall)

# Base period (event_time == -1) is the normalization point: exactly 0.
# (This CS implementation skips t < g-1, so -1 is the only pre-coefficient;
#  its being exactly 0 confirms correct base-period normalization.)
base = es_cs[es_cs["event_time"] == -1]
assert len(base) == 1 and abs(float(base["att"].iloc[0])) < 1e-12

# Every post-treatment horizon tracks the constant true effect.
for _, r in post.iterrows():
    assert abs(r["att"] - TRUE_ATT) < 0.40, (r["event_time"], r["att"])

print(f"CS overall ATT = {cs.att_overall:+.3f}   (CI [{lo_overall:+.3f}, {hi_overall:+.3f}])")
print(f"SA overall ATT = {sa.att_overall:+.3f}   true = {TRUE_ATT:+.3f}")

# %% [markdown]
# **Read the output.** Both estimators land on an overall ATT near the planted `+1.0`, and the
# true value sits inside the 90% bootstrap band — the staggered design is recovered, not the
# contaminated TWFE average. Two things to read honestly. First, the **post-treatment profile**
# is flat: each `event_time >= 0` cell hugs `+1.0`, exactly as planted (a constant kick, no
# build-up or decay). Second, the **`event_time = -1` coefficient is exactly 0** *by
# construction*, not because a pre-trend test passed: this CS implementation differences off the
# base period $g-1$ and skips all earlier periods ($t<g-1$), so $-1$ is the *only* pre-treatment
# point and it is mechanically normalized to zero. There are **no genuine pre-trend leads** here
# to inspect — a clean base normalization is all the event_time$=-1$ point certifies. CS and SA
# nearly coincide because the cohorts are balanced and the effect is homogeneous; the next
# section shows when they would pull apart.

# %% [markdown]
# ### Hero figure — the event study
# Estimates jump from the normalized 0 at the base period to the true +1.0 line
# and stay there; the shaded band is the 90% panel-bootstrap CI.

# %%
cols = _nbstyle.palette(2)
es = es_cs
fig, ax = plt.subplots()
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.axvline(-0.5, color="0.6", linewidth=0.8, linestyle="--")
ax.fill_between(es["event_time"], es["lo"], es["hi"], color="0.85",
                step="mid", label="90% CI")
ax.plot(es["event_time"], es["att"], color=cols[0], marker="o", markersize=4,
        label="CS estimate")
post_e = es[es["event_time"] >= 0]
ax.plot(post_e["event_time"], [TRUE_ATT] * len(post_e), color=cols[1],
        linestyle="--", label=f"true ATT = {TRUE_ATT:+.1f}")
ax.set_xlabel("Event time (periods since treatment)")
ax.set_ylabel("ATT")
ax.set_title("Staggered DiD event study (Callaway-Sant'Anna)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### Supporting — CS vs Sun-Abraham aggregation
# Same group-time effects, two aggregation rules (unweighted cohort-mean vs
# cohort-size-share). With balanced cohort sizes and a homogeneous effect they
# essentially coincide.

# %%
fig, ax = plt.subplots()
es_sa = sa.att_event_study.sort_values("event_time")
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.plot(es["event_time"], es["att"], color=cols[0], marker="o", markersize=4,
        label=f"Callaway-Sant'Anna (overall {cs.att_overall:+.2f})")
ax.plot(es_sa["event_time"], es_sa["att"], color=cols[1], marker="s",
        markersize=4, linestyle="--",
        label=f"Sun-Abraham (overall {sa.att_overall:+.2f})")
ax.set_xlabel("Event time")
ax.set_ylabel("ATT")
ax.set_title("CS vs Sun-Abraham aggregation")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### Supporting — the underlying group-time effects
# Each cohort's ATT(g, t) hovers around the common +1.0 (dashed); the event-study
# numbers above are just averages of these cohort curves at each event time.

# %%
fig, ax = plt.subplots()
gt = cs.att_gt
cohorts = sorted(gt["g"].unique())
ccols = _nbstyle.palette(len(cohorts))
for c, color in zip(cohorts, ccols):
    sub = gt[gt["g"] == c].sort_values("event_time")
    ax.plot(sub["event_time"], sub["att"], marker="o", markersize=3,
            color=color, label=f"cohort g={int(c)}")
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.axhline(TRUE_ATT, color="0.4", linewidth=0.8, linestyle="--")
ax.set_xlabel("Event time")
ax.set_ylabel("ATT(g, t)")
ax.set_title("Group-time effects by cohort")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# **Read the output.** This panel exposes the machinery: every cohort's $\text{ATT}(g,t)$ curve
# orbits the dashed `+1.0` line, and the event-study point at each horizon is simply the average
# of these curves across cohorts at that elapsed time. Because the planted effect is identical
# across cohorts, the curves overlap — there is nothing for the cohort-mean (CS) and the
# share-weighted mean (SA) to disagree about. If you made cohort 5 weak and cohort 11 strong,
# these curves would fan out and the two aggregations would visibly separate (see *Your turn*).

# %% [markdown]
# ## Your turn — recover a different planted effect
#
# The recovery above used one planted effect, `TRUE_ATT = +1.0`. Re-simulate the panel with a
# *different* true kick and confirm Callaway-Sant'Anna still recovers it — sign and magnitude.
# Because we re-use the same seed, the only thing that changes is the treatment effect itself, so
# the comparison is apples-to-apples. Change `TRUE_ATT_you` below.
#
# Note the deliberately *robust* asserts: the overall ATT recovers the planted sign and
# approximate magnitude, and the `event_time = -1` base stays exactly 0. We do **not** assert any
# nonzero pre-period dynamic — this CS implementation produces none (the base period is its only
# pre-treatment coefficient, normalized to 0 by construction).

# %%
def _simulate_panel(true_att):
    """Re-draw the staggered panel with a different planted effect (same seed,
    so the only thing that changes is the treatment kick)."""
    rng = np.random.default_rng(20240529)
    assign = rng.choice(len(COHORTS), size=N_UNITS, replace=True)
    unit_fe = rng.standard_normal(N_UNITS)
    time_fe = rng.standard_normal(T) * 0.3
    rows = []
    for i in range(N_UNITS):
        g = COHORTS[assign[i]]
        for t in range(1, T + 1):
            treated = 1.0 if (not pd.isna(g)) and t >= g else 0.0
            y = unit_fe[i] + time_fe[t - 1] + true_att * treated + SIGMA * rng.standard_normal()
            rows.append({"unit": i, "time": t, "y": y, "treat_time": g})
    return pd.DataFrame(rows)

# ← Change this planted treatment effect (try +2.0, -0.5, or 0.0 for a placebo).
TRUE_ATT_you = -1.5

panel_you = _simulate_panel(TRUE_ATT_you)
cs_you = callaway_santanna(panel_you, unit="unit", time="time", outcome="y",
                           treat_time="treat_time", n_boot=200, alpha=0.10, seed=0)
es_you = cs_you.att_event_study.sort_values("event_time").reset_index(drop=True)
base_you = float(es_you.loc[es_you["event_time"] == -1, "att"].iloc[0])

print(f"planted ATT = {TRUE_ATT_you:+.2f}  ->  CS overall ATT = {cs_you.att_overall:+.3f}"
      f"   (baseline planted {TRUE_ATT:+.1f} -> {cs.att_overall:+.3f})")
print(f"event_time = -1 base coefficient = {base_you:+.1e}")

# Robust fact 1: the overall ATT recovers the planted effect to within sampling noise...
assert abs(cs_you.att_overall - TRUE_ATT_you) < 0.30, cs_you.att_overall
# ...and reproduces its sign whenever the effect is non-zero.
if TRUE_ATT_you != 0.0:
    assert np.sign(cs_you.att_overall) == np.sign(TRUE_ATT_you), cs_you.att_overall
# Robust fact 2: the base period (event_time == -1) is the normalization point —
# exactly 0 regardless of the planted effect. This CS implementation emits no
# genuine pre-trend leads, so there is no nonzero pre-period dynamic to assert.
assert abs(base_you) < 1e-12, base_you

# %% [markdown]
# **Prompts.** (1) *Basic*: set `TRUE_ATT_you = +2.0` and confirm the recovered overall ATT
# roughly doubles while the `event_time = -1` base stays pinned at 0. (2) *Intermediate*: set
# `TRUE_ATT_you = 0.0` (a placebo) — the overall ATT falls to within sampling noise of 0 (here ≈ +0.13) and the assert still
# passes (the sign check is skipped for a zero effect). (3) *Stretch*: make the effect
# **heterogeneous across cohorts** by editing the DGP in section 1 (e.g. cohort 5 → +0.5, cohort
# 11 → +2.0 via a per-`g` lookup) and re-run sections 2-3; watch the CS *unweighted* cohort-mean
# and the SA *share-weighted* mean diverge in the "CS vs Sun-Abraham" figure.
#
# **How comprehensive is this?** `puremacro.did` is a full modern staggered-DiD toolkit:
# alongside `callaway_santanna` and `sun_abraham` it ships the `borusyak_jaravel_spiess` (2024)
# imputation estimator, `cdh_did` (de Chaisemartin-D'Haultfoeuille 2020) switchers estimator,
# and `synthetic_did` / `sdid_multi_cohort` (Arkhangelsky et al. 2021) — all on the same
# `(unit, time, outcome, treat_time)` long panel. For a *single* treated unit with many donors,
# `puremacro.synthetic_control` builds the counterfactual by weighting controls. And when the
# treatment is a *time-series* shock rather than a panel intervention, the causal machinery lives
# in the local-projection and SVAR notebooks (NB07 local projections, NB06 SVAR identification).
