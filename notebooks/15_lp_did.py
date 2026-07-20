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
# # DiD meets local projections: LP-DiD
#
# A policy rolls out across states in waves, its effect *builds over time*, and
# you want the whole dynamic response — not one contaminated number. Notebook 10
# showed why the textbook two-way fixed-effects (TWFE) event study breaks under
# staggered adoption and how Callaway-Sant'Anna repairs it. This sequel shows the
# *local-projections* repair: **LP-DiD** (Dube, Girardi, Jordà & Taylor 2023),
# which keeps the familiar Jordà regression-per-horizon workflow but restricts
# every comparison to newly-treated units versus *clean* controls. We plant a
# known dynamic effect, watch naive TWFE mangle it, recover it with
# `puremacro.lp.lp_did`, and confirm that LP-DiD, Callaway-Sant'Anna and
# Sun-Abraham — three different machines — agree because they share one
# principle. Everything runs in the browser on synthetic data.

# %% [markdown]
# ## The method in math
#
# **Setting.** Unit $i$ adopts an absorbing treatment at time $G_i$ (never, for
# controls); $D_{it}=\mathbb{1}\{t \ge G_i\}$ and $\Delta D_{it}=1$ marks the
# *switch* quarter. Effects may be dynamic and heterogeneous across cohorts.
#
# **The TWFE failure.** The dynamic TWFE event study regresses $y_{it}$ on
# relative-time dummies plus unit and time effects. Under staggered adoption with
# heterogeneous effects, each coefficient is a weighted sum of effects at *other*
# relative times and cohorts — with possibly negative weights — so even the
# *leads* pick up treatment effects (Sun-Abraham 2021; Goodman-Bacon 2021
# for the static case).
#
# **LP-DiD.** Estimate one cross-section regression per horizon $h$:
# $$ y_{i,t+h} - y_{i,t-1} \;=\; \beta_h\,\Delta D_{it} \;+\; \delta_t^h \;+\; e_{it}^h, $$
# keeping only (i) **newly-treated** observations ($\Delta D_{it}=1$) and (ii)
# **clean controls**: units with $D_{i,t+h}=0$, i.e. still untreated through
# $t+h$ (never-treated or later-treated). The long difference kills the unit
# effect; the period effect $\delta_t^h$ does the differencing across groups; the
# sample restriction outlaws every forbidden already-treated comparison.
# Pre-trend coefficients come **for free**: the same regression at leads
# $h = -K,\dots,-2$ (with controls untreated through $t$) should give
# $\beta_h \approx 0$ under parallel trends; $h=-1$ is the base period.
#
# **Weights.** Plain OLS ('vw') returns a *variance-weighted* average of the
# period-specific clean DiDs — weights $\propto n_t\,p_t(1-p_t)$, always convex,
# the well-behaved analogue of the TWFE weighting. Reweighting each newly-treated
# observation to count equally ('equal', weight $n_t^{tr}/n_t^{co}$ on controls)
# returns the **equally-weighted ATT** across treatment events. Both are exposed
# in `lp_did`, plus the per-period weight diagnostics.

# %% [markdown]
# **Intuition.** LP-DiD is "Callaway-Sant'Anna, but you never leave the LP
# world". Every trick you know from Jordà local projections — horizon-by-horizon
# regressions, cluster-robust bands, IV variants, state dependence — carries
# over, because the estimator *is* an LP: the only DiD-specific ingredients are
# the long-differenced outcome and the clean-control sample rule. That rule is
# the whole ballgame: TWFE breaks not because fixed effects are wrong but because
# OLS silently manufactures comparisons in which an *already-treated* unit —
# whose outcome still carries its own treatment dynamics — serves as the
# "control". Ban those comparisons and any sensible aggregation of what remains
# is consistent; CS, Sun-Abraham and LP-DiD are just three aggregation rules on
# the same clean building blocks.

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

from puremacro.lp import lp_did, LPDiDResult
from puremacro.did import callaway_santanna, sun_abraham

# %% [markdown]
# ## 1. The pitfall, planted: TWFE with heterogeneous dynamics
#
# Two cohorts adopt at t = 7 and t = 13 (30 units each; 40 never-treated). The
# *early* cohort's effect **grows** with time since adoption
# ($\tau_E(e)=0.4+0.35e$); the *late* cohort's is flat ($\tau_L = 1.0$). We make
# this first panel **noiseless**, so every deviation you see below is pure
# specification bias — not sampling error. Both naive regressions are fit inline
# in a few lines (exact two-way demeaning on a balanced panel).

# %%
def simulate(cohort_sizes, T, effect, sigma, seed, pre_slope=0.0):
    """Staggered-adoption panel with unit/time FE and known dynamic ATTs."""
    rng = np.random.default_rng(seed)
    lam = rng.standard_normal(T) * 0.3
    rows, uid = [], 0
    for g, m in cohort_sizes.items():
        for _ in range(m):
            a = rng.standard_normal()
            for t in range(1, T + 1):
                on = (g is not None) and (t >= g)
                tau = effect(g, t - g) if on else 0.0
                drift = pre_slope * t if g is not None else 0.0  # trend violation knob
                rows.append({"unit": uid, "time": t,
                             "y": a + lam[t - 1] + drift + tau
                                  + sigma * rng.standard_normal(),
                             "D": float(on),
                             "treat_time": np.nan if g is None else float(g)})
            uid += 1
    return pd.DataFrame(rows)

def two_way_demean(df, cols):
    """Exact TWFE projection for a balanced panel (unit and time means)."""
    out = {}
    for c in cols:
        v = df[c].astype(float)
        out[c] = (v - df.groupby("unit")[c].transform("mean")
                    - df.groupby("time")[c].transform("mean") + v.mean()).to_numpy()
    return out

def naive_event_study(df, leads=range(-4, -1), lags=range(0, 14)):
    """Fully saturated dynamic TWFE event study (no binning tricks needed)."""
    d = df.copy()
    d["e"] = d["time"] - d["treat_time"]
    evts = list(leads) + list(lags)                   # e = -1 omitted (base)
    for e in evts:
        d[f"e{e}"] = (d["e"] == e).astype(float)   # NaN event time compares False
    dm = two_way_demean(d, ["y"] + [f"e{e}" for e in evts])
    X = np.column_stack([dm[f"e{e}"] for e in evts])
    coef, *_ = np.linalg.lstsq(X, dm["y"], rcond=None)
    return dict(zip(evts, coef))

TRUE = lambda g, e: (0.4 + 0.35 * e) if g == 7 else 1.0
SIZES = {7: 30, 13: 30, None: 40}
T = 20
true_path = np.array([(30 * TRUE(7, h) + 30 * TRUE(13, h)) / 60 for h in range(9)])

demo = simulate(SIZES, T, TRUE, sigma=0.0, seed=0)
dm = two_way_demean(demo, ["y", "D"])
beta_twfe = float(dm["D"] @ dm["y"] / (dm["D"] @ dm["D"]))
cell_avg = float(np.mean([TRUE(g, t - g) for g, m in {7: 30, 13: 30}.items()
                          for t in range(g, T + 1) for _ in range(m)]))
naive = naive_event_study(demo)

print(f"static TWFE 'the effect'   = {beta_twfe:+.3f}   "
      f"(true avg over treated cells = {cell_avg:+.3f})")
print(f"naive event-study leads    = " +
      ", ".join(f"e={e}: {naive[e]:+.3f}" for e in (-4, -3, -2)) +
      "   (truth: all exactly 0)")
print(f"naive event-study at e=0   = {naive[0]:+.3f}   (truth {true_path[0]:+.3f})")

assert beta_twfe < cell_avg - 0.6                 # static TWFE badly downward-biased
assert abs(naive[-2]) > 0.10                      # spurious pre-trend from nothing
assert abs(naive[0] - true_path[0]) > 0.20        # impact effect distorted too

# %% [markdown]
# **Read the output.** With *zero* noise, static TWFE reports +1.25 for a policy
# whose true average effect on the treated is +2.07 — a 40% understatement
# manufactured entirely by forbidden comparisons (early-treated units, still on
# their rising effect path, get used as "controls" for the late cohort, and
# their growth is subtracted off). The dynamic event study is no rescue: it is
# *fully saturated* in relative time, yet the leads come out at −0.10 to −0.17
# when the truth is exactly zero — heterogeneity across cohorts leaks treatment
# effects into pre-treatment coefficients (Sun-Abraham's contamination result),
# the impact effect is off by a third, and a referee reading these leads would
# reject parallel trends *that hold by construction*.

# %% [markdown]
# ## 2. LP-DiD: the fix that stays in the LP world
#
# Same design, now with realistic noise ($\sigma = 0.35$). `lp_did` runs one
# regression per horizon on newly-treated vs clean controls, reports pre-trends
# for $h=-4..-2$ "for free", counts the treated/clean sample per horizon, and
# clusters standard errors by unit. The default `weights='equal'` targets the
# equally-weighted ATT across the 60 adoption events.

# %%
panel = simulate(SIZES, T, TRUE, sigma=0.35, seed=42)
res = lp_did(panel, "y", "D", "unit", "time", horizons=range(0, 7), pre_window=4)
assert isinstance(res, LPDiDResult)
print(res.summary())
print(res.estimates.round(3).to_string(index=False))

est = res.post.set_index("h")["beta"]
err = np.abs(est.to_numpy() - true_path[:7])
pre_t = res.pretrend["t"].abs().max()
print(f"\nmax |error| vs planted dynamic ATT = {err.max():.3f}")
print(f"max |t| across pre-trends          = {pre_t:.2f}")

assert err.max() < 0.25                            # dynamic path recovered
assert pre_t < 2.5                                 # no false pre-trend alarm
# Clean-control discipline: at h = 6 the late cohort (t=13) would need
# controls untreated through t+6 = 19 -- only the never-treated remain.
assert res.estimates.set_index("h").loc[6, "n_clean"] < \
       res.estimates.set_index("h").loc[0, "n_clean"]

# %% [markdown]
# **Read the output.** The estimated path climbs from ≈ +0.6 at impact to ≈ +1.8
# at $h=6$, tracking the planted cohort-average profile within two standard
# errors everywhere, and the three genuine leads hover around zero (max |t| <
# 2) — the same panel on which the saturated TWFE event study just failed. Watch
# the bookkeeping columns: `n_clean` drops from 110 to 80 at $h = 6$, because a
# clean control must stay untreated through $t+h$, and the later-treated cohort
# stops qualifying for the early cohort's long horizons. That visible attrition
# *is* the identification: you can read exactly which comparisons produced each
# coefficient — no black box.

# %% [markdown]
# ### Hero figure — one panel, two event studies
#
# Naive TWFE (hollow squares) versus LP-DiD with its 90% cluster band, against
# the planted truth. The naive leads dip below zero and its impact estimate is
# too low; LP-DiD sits on the truth at every horizon.

# %%
cols = _nbstyle.palette(3)
ev = res.estimates
naive_n = naive_event_study(panel)
e_grid = [e for e in sorted(naive_n) if -4 <= e <= 6] + [-1]
naive_y = [0.0 if e == -1 else naive_n[e] for e in e_grid]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.axvline(-0.5, color="0.6", linewidth=0.8, linestyle="--")
ax.fill_between(ev["h"], ev["lo"], ev["hi"], color="0.85", label="90% CI (LP-DiD)")
ax.plot(ev["h"], ev["beta"], color=cols[0], marker="o", markersize=4,
        label="LP-DiD (equal weights)")
order = np.argsort(e_grid)
ax.plot(np.array(e_grid)[order], np.array(naive_y)[order], color=cols[1],
        marker="s", markersize=4, markerfacecolor="none", linestyle="--",
        label="naive TWFE event study")
hh = np.arange(0, 7)
ax.plot(hh, true_path[:7], color=cols[2], linestyle="-.",
        label="planted dynamic ATT")
ax.plot(np.arange(-4, 0), np.zeros(4), color=cols[2], linestyle="-.")
ax.set_xlabel("Event time $h$ (base period $h=-1$)")
ax.set_ylabel("Effect on $y$")
ax.set_title("Staggered adoption: naive TWFE vs LP-DiD")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# ## 3. Three machines, one principle
#
# Callaway-Sant'Anna builds group-time ATTs from clean 2×2 blocks and averages
# them; Sun-Abraham reweights the same blocks by cohort shares; LP-DiD gets
# there by regression, one horizon at a time. If the clean-comparison principle
# is what matters — and not the machinery — the three event studies should lie
# on top of each other. That is exactly the lesson.

# %%
cs = callaway_santanna(panel, unit="unit", time="time", outcome="y",
                       treat_time="treat_time", n_boot=200, alpha=0.10, seed=0)
sa = sun_abraham(panel, unit="unit", time="time", outcome="y",
                 treat_time="treat_time", n_boot=200, alpha=0.10, seed=0)

es_cs = cs.att_event_study.set_index("event_time")
es_sa = sa.att_event_study.set_index("event_time")
gap_cs = max(abs(est.loc[h] - es_cs.loc[h, "att"]) for h in range(7))
gap_sa = max(abs(est.loc[h] - es_sa.loc[h, "att"]) for h in range(7))
print(f"max |LP-DiD - CS| over h=0..6 = {gap_cs:.3f}")
print(f"max |LP-DiD - SA| over h=0..6 = {gap_sa:.3f}")
assert gap_cs < 0.15 and gap_sa < 0.15             # three estimators, same answer

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.plot(est.index, est.to_numpy(), color=cols[0], marker="o", markersize=4,
        label="LP-DiD (equal)")
ax.plot(es_cs.index[es_cs.index >= 0], es_cs.loc[es_cs.index >= 0, "att"],
        color=cols[1], marker="s", markersize=4, linestyle="--",
        label="Callaway-Sant'Anna")
ax.plot(es_sa.index[es_sa.index >= 0], es_sa.loc[es_sa.index >= 0, "att"],
        color=cols[2], marker="^", markersize=4, linestyle=":",
        label="Sun-Abraham")
ax.plot(hh, true_path[:7], color="0.5", linewidth=0.9, linestyle="-.",
        label="planted ATT")
ax.set_xlabel("Event time $h$")
ax.set_ylabel("ATT")
ax.set_title("LP-DiD vs Callaway-Sant'Anna vs Sun-Abraham")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# **Read the output.** The three paths differ by at most ≈ 0.03 — less than half
# a standard error — despite entirely different code paths: no step of `lp_did`
# calls the DiD module. Their small remaining daylight is exactly the announced
# aggregation difference (CS averages cohorts equally, SA and LP-DiD weight by
# cohort size, and LP-DiD pools not-yet-treated units into the controls while
# this CS implementation uses only the never-treated). When someone asks "which
# staggered-DiD estimator should I use?", this figure is the answer: any of
# them, *as long as the comparisons are clean* — pick the one whose outputs you
# need. LP-DiD's edge is everything that comes free in the LP world: horizon
# regressions you can hand to `cum_irf_block_bootstrap`, IV versions, state
# dependence, Driscoll-Kraay bands.

# %% [markdown]
# ## 4. Pre-trends as a first-class output
#
# Because the leads are just more LP horizons, pre-trend testing is not a
# separate procedure — the same call returns them, with the same clustered
# standard errors. We re-simulate the panel giving ever-treated units a
# differential drift of +0.18 per period (a genuine parallel-trends violation)
# and compare the lead coefficients with the clean panel's.

# %%
vio = simulate({7: 30, None: 40}, 18, lambda g, e: 1.0, sigma=0.35, seed=9,
               pre_slope=0.18)
cln = simulate({7: 30, None: 40}, 18, lambda g, e: 1.0, sigma=0.35, seed=9,
               pre_slope=0.0)
r_vio = lp_did(vio, "y", "D", "unit", "time", range(0, 5), pre_window=4)
r_cln = lp_did(cln, "y", "D", "unit", "time", range(0, 5), pre_window=4)

pre_v = r_vio.pretrend.set_index("h")
pre_c = r_cln.pretrend.set_index("h")
print("violated panel  :",
      ", ".join(f"h={h}: {pre_v.loc[h, 'beta']:+.2f} (t {pre_v.loc[h, 't']:+.1f})"
                for h in (-4, -3, -2)))
print("clean panel     :",
      ", ".join(f"h={h}: {pre_c.loc[h, 'beta']:+.2f} (t {pre_c.loc[h, 't']:+.1f})"
                for h in (-4, -3, -2)))

assert pre_v.loc[-4, "beta"] < -0.25 and pre_v.loc[-4, "t"] < -3.0
assert pre_c["t"].abs().max() < 2.0

fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.axvline(-0.5, color="0.6", linewidth=0.8, linestyle="--")
for r, c, lbl, mk in [(r_cln, cols[0], "parallel trends hold", "o"),
                      (r_vio, cols[1], "planted violation (+0.18/period)", "s")]:
    e = r.estimates
    ax.fill_between(e["h"], e["lo"], e["hi"], color=c, alpha=0.12)
    ax.plot(e["h"], e["beta"], color=c, marker=mk, markersize=4, label=lbl)
ax.set_xlabel("Event time $h$")
ax.set_ylabel("Coefficient")
ax.set_title("The leads catch the violation before you believe the lags")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# **Read the output.** On the violated panel the leads fan out downward —
# $\beta_{-4} \approx -0.43$ with $t \approx -3.6$, close to the mechanical
# prediction $-0.18 \times 3 = -0.54$ (each extra lead accumulates one more
# period of differential drift; the sign is negative because the treated group's
# *base period* $t-1$ is higher, not lower, than its past). The clean panel's
# leads stay flat with |t| < 1. And note what the violation does to the "effect":
# the post coefficients on the violated panel are inflated by the same drift —
# a pre-trend rejection is not a formality, it is telling you the lags are
# absorbing trend, not treatment.

# %% [markdown]
# **Where you would use this.** LP-DiD's natural habitat is exactly the panels
# this package targets. State policy adoptions: minimum-wage changes,
# unemployment-insurance extensions, right-to-work laws — staggered by
# construction, with effects that phase in (the setting of the minimum-wage
# reanalyses that motivated the clean-control literature). Uncertainty event
# studies at the state level: treat a state as "switching" when a plant-closure
# wave, a WARN-notice spike, or a trade-policy shock first hits it, and trace
# employment and participation over the following quarters — the kind of
# local labor-market analysis this package's fetchers and panels support,
# where the same clean-control discipline separates a state's own shock
# response from the dynamics of states hit earlier. And because LP-DiD *is* a local projection,
# the extensions you would reach for in that setting — cumulative multipliers
# with entity-bootstrap bands, instrumenting the switch, state-dependent
# responses — are the LP tools you already have, not new estimators.

# %% [markdown]
# ## Your turn — watch the two weightings disagree
#
# `weights='vw'` (plain OLS) weights each adoption period by $n_t p_t(1-p_t)$ —
# it loves periods with a balanced treated/control mix. `weights='equal'` gives
# every adoption *event* the same say. With one big early cohort (30 units,
# $\tau = 0.5$) and one small late cohort (5 units, $\tau$ = your choice), the
# two answers must split whenever the cohorts truly differ. The panel below is
# noiseless, so the estimates are exact and the gap is pure weighting.

# %%
# ← Change this: the small late cohort's treatment effect (try 0.5, -1.0, 4.0).
TAU_SMALL_YOU = 2.0

TAU_BIG = 0.5
mini = simulate({6: 30, 12: 5, None: 10}, 20,
                lambda g, e: TAU_BIG if g == 6 else TAU_SMALL_YOU,
                sigma=0.0, seed=1)
r_eq = lp_did(mini, "y", "D", "unit", "time", range(0, 5), pre_window=0)
r_vw = lp_did(mini, "y", "D", "unit", "time", range(0, 5), pre_window=0,
              weights="vw")
eq0 = float(r_eq.post.loc[r_eq.post["h"] == 0, "beta"].iloc[0])
vw0 = float(r_vw.post.loc[r_vw.post["h"] == 0, "beta"].iloc[0])
print(f"equal-weighted ATT(0) = {eq0:+.3f}   variance-weighted ATT(0) = {vw0:+.3f}")
print(r_eq.group_weights[r_eq.group_weights["h"] == 0].round(3).to_string(index=False))

# The event shares are 30/35 vs 5/35, but the vw period weights are 0.75/0.25:
# vw tilts TOWARD the small cohort here (its period has the better mix), so the
# gap's sign follows the small cohort's effect relative to the big one's.
if abs(TAU_SMALL_YOU - TAU_BIG) > 1e-9:
    assert np.sign(vw0 - eq0) == np.sign(TAU_SMALL_YOU - TAU_BIG), (eq0, vw0)
else:
    assert abs(vw0 - eq0) < 1e-8

# %% [markdown]
# **Prompts.** (1) *Basic*: set `TAU_SMALL_YOU = 0.5` (homogeneous effects) and
# confirm the two weightings collapse onto the same number — weighting only
# matters when there is heterogeneity to weight. (2) *Intermediate*: set it to
# `-1.0` and predict, before running, which estimator reports the more negative
# ATT; verify against the printed `w_vw` / `w_equal` shares. (3) *Stretch*:
# delete the never-treated group from `SIZES` in section 2 (use
# `{7: 30, 13: 30}`) and re-run `lp_did` with `horizons=range(0, 9)` — read the
# `n_clean` column as later horizons run out of clean controls and the longest
# horizons go `NaN`; then explain why Section 1's naive TWFE *never* warns you
# about this.
#
# **How comprehensive is this?** `puremacro.lp.lp_did` returns a frozen
# `LPDiDResult` with the estimates, per-horizon treated/clean counts, and the
# DGJT period-weight diagnostics (`group_weights`). Its DiD cousins live in
# `puremacro.did` — `callaway_santanna`, `sun_abraham`, the
# `borusyak_jaravel_spiess` imputation estimator, `cdh_did`, and synthetic DiD —
# all sharing the `(unit, time, outcome, treat_time)` panel format used here
# (notebook 10 is the prequel). On the LP side, `panel_lp` / `panel_lp_dk` are
# the continuous-shock work-horses, `lp_iv` instruments the shock, and
# `puremacro.inference.lp_block_bootstrap` turns any per-horizon path into
# cumulative-IRF bands via the entity bootstrap (notebook 07 covers the LP
# fundamentals).
