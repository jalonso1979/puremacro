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
# # Synthetic Difference-in-Differences — Unifying Synthetic Controls and TWFE
#
# **How can researchers evaluate a policy that hit a single unit when parallel trends fail and the control donors differ from the treated unit?**
# Two of the most used causal evaluation designs in empirical economics are:
# 1. **Difference-in-Differences (DiD)**: assumes untreated potential outcomes follow parallel trends over time, with additive unit and time fixed effects.
# 2. **Synthetic Control (SC)**: drops parallel trends and finds non-negative donor weights $\omega$ that reproduce the treated unit's pre-treatment path. SC has no intercept, so it must match the treated unit's *level* from inside the donors' convex hull, and adding a constant to one unit's outcome changes its answer.
#
# Dmitry Arkhangelsky, Susan Athey, David Hirshberg, Guido Imbens, and Stefan Wager (2021, *American Economic Review*) combine the two in **Synthetic Difference-in-Differences (SDID)**: unit weights as in SC, time weights that do the same job across periods, and DiD's fixed effects.
#
# This notebook uses **simulated data**: one treated unit, 15 donors, 20 periods, a planted effect of −4.5, and a common factor to which units are exposed with different loadings, so parallel trends fail. We fit `puremacro.did.synthetic_did`, read its placebo standard error and interval, and check in *Your turn* whether the 90% interval covers the planted effect 90% of the time.

# %% [markdown]
# ## The method in math
#
# Let $Y_{it}$ be the outcome of unit $i$ in period $t$. One unit is treated from period $T_0$ on
# ($T_{pre}$ periods before, $T_{post}$ after); the $N_{co}$ donors never are. Write $Y_{tr,t}$ for
# the treated path and $\bar Y_{i,\text{post}}$ for a unit's post-period mean.
#
# **Unit weights** (paper eq. 2.1): a synthetic control with an intercept $\omega_0$ and a ridge
# penalty, $\zeta = (N_{tr}T_{post})^{1/4}\hat\sigma$ with $\hat\sigma$ the s.d. of the donors'
# one-period changes before treatment (eq. 2.2):
# $$ (\hat\omega_0,\hat\omega) = \arg\min_{\omega_0,\ \omega\in\Delta} \sum_{t<T_0}\Big(\omega_0 + \sum_{i\in co}\omega_i Y_{it} - Y_{tr,t}\Big)^2 + \zeta^2 T_{pre}\,\|\omega\|_2^2 .$$
# **Time weights** (eq. 2.3): the same problem across periods, fitted on the donors only,
# $$ (\hat\lambda_0,\hat\lambda) = \arg\min_{\lambda_0,\ \lambda\in\Delta} \sum_{i\in co}\Big(\lambda_0 + \sum_{t<T_0}\lambda_t Y_{it} - \bar Y_{i,\text{post}}\Big)^2 .$$
# **Estimate**:
# $$ \hat\tau = \Big(\bar Y_{tr,\text{post}} - \sum_i \hat\omega_i \bar Y_{i,\text{post}}\Big) - \sum_{t<T_0}\hat\lambda_t\Big(Y_{tr,t} - \sum_i\hat\omega_i Y_{it}\Big). $$
# Uniform $\omega$ and $\lambda$ give the $2\times2$ DiD; dropping the intercept and the pre-period
# term gives SC (p. 4). Because of the intercepts, the unit weights only need to make the donors'
# path *parallel* to the treated one, and $\hat\tau$ does not change when a constant is added to
# any unit's or period's outcomes, as for DiD and unlike SC (p. 20). Using both sets of weights
# gives what the authors call "a type of double robustness": if either the unit weights or the
# time weights balance the latent factor structure, its bias is approximately removed (p. 20).
#
# **Inference with one treated unit** (Section 5). The **placebo** estimator (Algorithm 4)
# assigns the treatment to a donor drawn at random, re-estimates SDID on the donors only, repeats,
# and takes the variance $\hat V$ of those placebo estimates; the 90% interval is
# $\hat\tau \pm z_{0.95}\sqrt{\hat V}$ (eq. 5.1). It assumes the treated unit's noise looks like the
# donors' (homoskedasticity across units), which the paper calls effectively necessary for any
# inference with one treated unit (p. 29). The unit bootstrap (Algorithm 2) and the jackknife
# (Algorithm 3) are not used here: the paper reports neither for $N_{tr}=1$ "because the estimators
# are not well-defined" (Table 4 note, p. 30). With one treated unit, every bootstrap sample
# contains that same unit, so resampling varies only the donors and the treated unit's own noise
# never enters the standard error.
#
# **Intuition.** The factor in this simulation is a business cycle that hits units with different
# strength. DiD compares the treated unit with the average donor and so assumes equal exposure; SC
# looks for donors with the same exposure *and* the same level. SDID only asks for the same
# exposure (the intercept absorbs levels), and its time weights pick the pre-treatment periods in
# which the cycle stood where it stands, on average, after treatment. Differences taken from those
# periods cancel the part of the cycle the unit weights did not match.
#
# Page and equation numbers refer to arXiv:1812.09970v4 (July 2021), the version of the AER paper
# read for this notebook.

# %% [markdown]
# ## Setup — a simulated policy intervention with non-parallel donors
#
# $y_{it} = a_i + \text{trend}_t + \ell_i f_t + \varepsilon_{it}$, with $f_t$ a sine-shaped common
# factor, loadings $\ell_i \sim U(0.5, 2)$ for the donors and $\ell_0 = 1.4$ for the treated unit,
# $\varepsilon_{it}\sim N(0, 0.4^2)$, and $\tau = -4.5$ added to the treated unit from $t = 12$ on.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.did import synthetic_did
from puremacro.synthetic_control import synthetic_control

N_donors = 15
N_units = N_donors + 1  # Unit 0 is treated
T_periods = 20
T_treat = 12

# Common latent factors
time_trend = np.linspace(10, 25, T_periods)
macro_factor = np.sin(np.linspace(0, 3 * np.pi, T_periods)) * 3.0

tau_true = -4.5      # planted treatment effect after T_treat
NOISE_SD = 0.4


def simulate(seed, load0=1.4):
    """Simulated panel and loadings; unit 0 is treated and has factor loading load0."""
    rng = np.random.default_rng(seed)
    unit_fixed_effects = rng.uniform(5, 20, size=N_units)
    factor_loadings = rng.uniform(0.5, 2.0, size=N_units)
    factor_loadings[0] = load0
    records = []
    for i in range(N_units):
        for t in range(T_periods):
            y_it = unit_fixed_effects[i] + time_trend[t] + factor_loadings[i] * macro_factor[t]
            y_it += rng.normal(0, NOISE_SD)
            if (i == 0) and (t >= T_treat):
                y_it += tau_true
            records.append({
                "unit": f"unit_{i:02d}",
                "time": t,
                "y": y_it,
                "treat_time": T_treat if i == 0 else np.nan,
            })
    return pd.DataFrame(records), factor_loadings


df_panel, factor_loadings = simulate(2021)
print(f"Simulated panel: {N_units} units ({N_donors} donors) x {T_periods} periods; "
      f"treated from t = {T_treat}; planted tau = {tau_true:+.1f}")
print(f"factor loadings: treated {factor_loadings[0]:.2f}; donors "
      f"{factor_loadings[1:].min():.2f} to {factor_loadings[1:].max():.2f} "
      f"(mean {factor_loadings[1:].mean():.2f})")

# %% [markdown]
# ## Estimating Synthetic DiD with puremacro
#
# With one treated unit, `se_method="auto"` selects the placebo variance estimator; `n_boot` is
# the number of placebo replications. We also compute the plain $2\times2$ DiD and the Abadie
# synthetic control on the same panel.

# %%
res_sdid = synthetic_did(
    df=df_panel,
    unit="unit",
    time="time",
    outcome="y",
    treat_time="treat_time",
    n_boot=100,
    seed=42,
)
assert res_sdid.se_method == "placebo"   # Algorithm 4: the choice for one treated unit
print(res_sdid.summary())

covers = res_sdid.lo <= tau_true <= res_sdid.hi
print(f"planted tau = {tau_true:+.3f}; error {res_sdid.tau - tau_true:+.3f} "
      f"= {(res_sdid.tau - tau_true) / res_sdid.se:+.2f} placebo s.e.; "
      f"does the 90% CI contain it? {covers}")

assert abs(res_sdid.tau - tau_true) < 1.0
assert np.isclose(res_sdid.omega.sum(), 1.0) and (res_sdid.omega >= 0).all()
assert np.isclose(res_sdid.lambda_w.sum(), 1.0) and (res_sdid.lambda_w >= 0).all()

# Units invariance: measuring the outcome in units 1000 times smaller multiplies tau-hat by 1000.
# (Up to puremacro 4.3.0 the weight solver fell back to uniform weights, i.e. plain DiD,
# for outcomes in large units.)
tau_x1000 = synthetic_did(df_panel.assign(y=1000 * df_panel["y"]), n_boot=0).tau
print(f"outcome x 1000: tau-hat / 1000 = {tau_x1000 / 1000:.4f}")
assert np.isclose(tau_x1000, 1000 * res_sdid.tau, rtol=1e-6)

# The same panel through the two parent designs.
piv = df_panel.pivot(index="time", columns="unit", values="y")
treated_traj = piv["unit_00"].to_numpy()
donor_matrix = piv.drop(columns=["unit_00"]).to_numpy()
unweighted_control_path = donor_matrix.mean(axis=1)
gap_did = treated_traj - unweighted_control_path
did = gap_did[T_treat:].mean() - gap_did[:T_treat].mean()          # plain 2x2 DiD
sc = synthetic_control(piv, "unit_00", list(range(T_treat)),
                       list(range(T_treat, T_periods)), placebo=False).treatment_effect.mean()
print(pd.DataFrame({"estimate": [did, sc, res_sdid.tau]},
                   index=["DiD (2x2)", "SC (Abadie)", "SDID"])
      .assign(error=lambda d: d["estimate"] - tau_true).round(3).to_string())

# %% [markdown]
# ## Hero figure — the SDID counterfactual
#
# The $\hat\omega$-weighted donor path is only asked to run *parallel* to the treated unit before
# treatment. Adding the $\hat\lambda$-weighted pre-treatment gap turns it into SDID's counterfactual,
# and $\hat\tau$ is the average post-treatment gap between the treated path and that counterfactual
# (asserted below, an internal consistency check of the plotted lines). The DiD counterfactual does
# the same with uniform weights. Panel (b) shows the time weights.

# %%
omega = res_sdid.omega.to_numpy()
lam = res_sdid.lambda_w.to_numpy()
synthetic_path = donor_matrix @ omega
sdid_cf = synthetic_path + lam @ (treated_traj - synthetic_path)[:T_treat]
did_cf = unweighted_control_path + gap_did[:T_treat].mean()
assert np.isclose((treated_traj - sdid_cf)[T_treat:].mean(), res_sdid.tau)
assert np.isclose((treated_traj - did_cf)[T_treat:].mean(), did)

fig, axes = _nbstyle.figura(ancho=7.6, alto=3.6, ncols=2, gridspec_kw={"width_ratios": [2.2, 1.2]})

# Panel 1: treated path and the two counterfactuals
ax = axes[0]
time_axis = np.arange(T_periods)
ax.plot(time_axis, treated_traj, **_nbstyle.S1, label="Treated unit")
ax.plot(time_axis, sdid_cf, **_nbstyle.S2, label=r"SDID counterfactual ($\hat\omega$, $\hat\lambda$)")
ax.plot(time_axis, did_cf, **_nbstyle.S4, label="DiD counterfactual (uniform)")
ax.axvline(T_treat - 0.5, color=_nbstyle.SPINE, ls="-.", lw=1.0)
ax.text(T_treat - 0.3, ax.get_ylim()[0] + 0.5, "treatment", color=_nbstyle.NOTA, fontsize=8)
ax.set_title("(a) Treated path and counterfactuals", loc="left", fontsize=9.5, fontweight="bold")
ax.set_xlabel("Period t")
ax.set_ylabel("Outcome $y$ (simulated units)")
ax.legend(loc="upper left", fontsize=8)

# Panel 2: time weights
ax_w = axes[1]
pre_times = np.arange(T_treat)
ax_w.bar(pre_times, lam, color=_nbstyle.S2["color"], width=0.6)
ax_w.axhline(1.0 / T_treat, color=_nbstyle.SPINE, ls="--", lw=1.0)
ax_w.text(T_treat - 0.4, 0.2, f"dashed line:\nuniform 1/{T_treat} (DiD)", ha="right",
          va="center", color=_nbstyle.TEXTO, fontsize=8)
ax_w.set_title(r"(b) Time weights $\hat{\lambda}_t$", loc="left", fontsize=9.5, fontweight="bold")
ax_w.set_xlabel("Pre-treatment period t")
ax_w.set_ylabel(r"Weight $\hat{\lambda}_t$")

# %% [markdown]
# ## What the time weights do
#
# Up to noise, the treated unit's untreated path differs from any donor combination by a level
# and by (loading gap) $\times f_t$. Differencing off the $\hat\lambda$-weighted pre-period leaves a
# factor bias of (loading gap) $\times(\bar f_{post} - \hat\lambda' f_{pre})$; DiD's uniform weights
# leave (loading gap) $\times(\bar f_{post} - \bar f_{pre})$. The check below uses the planted
# factor, which the estimator never sees.

# %%
f_pre, f_post = macro_factor[:T_treat], macro_factor[T_treat:].mean()
gap_sdid = factor_loadings[0] - omega @ factor_loadings[1:]
gap_mean = factor_loadings[0] - factor_loadings[1:].mean()
print(f"common factor f_t: post-period mean {f_post:+.3f}; lambda-weighted pre-period "
      f"{lam @ f_pre:+.3f}; plain pre-period mean {f_pre.mean():+.3f}")
print("largest time weights:", {int(t): round(float(w), 3) for t, w in res_sdid.lambda_w.nlargest(3).items()})
print("largest donor weights:", res_sdid.omega.nlargest(3).round(3).to_dict())
print(f"factor bias: DiD {gap_mean * (f_post - f_pre.mean()):+.3f} "
      f"(loading gap {gap_mean:+.2f}); SDID {gap_sdid * (f_post - lam @ f_pre):+.4f} "
      f"(loading gap {gap_sdid:+.2f})")

assert abs(lam @ f_pre - f_post) < 0.05        # the time weights balance the factor...
assert abs(f_pre.mean() - f_post) > 1.0        # ...which uniform pre-period weights do not

# %% [markdown]
# **Read the output.** SDID estimates $\hat\tau = -4.0827$ against the planted $-4.5$, an error of
# $+0.417$, or $+1.61$ placebo standard errors ($\hat{\text{se}} = 0.2599$ from 100 placebo
# replications). The 90% interval $[-4.5102, -3.6552]$ contains the truth, but only just: this draw
# is unlucky, and one interval says nothing about coverage (see *Your turn*). Rescaling the outcome
# by 1000 returns the same $-4.0827$ once divided back: the weights do not depend on units. DiD
# misses by $+0.583$ and SC by $+0.316$; at this treated loading SC happens to land closest.
#
# The printed factor biases explain DiD's miss. The treated loading (1.40) is above the donors' mean
# (1.13), so the factor enters the treated unit more strongly, and DiD, which uses all twelve
# pre-periods equally, compares the post-period cycle ($+1.380$) with a pre-period average near zero
# ($+0.067$): with a loading gap of $+0.27$, a factor bias of $+0.355$. SDID's time weights put
# their largest weights on periods 2, 1 and 0 and none on the last three pre-periods (panel b):
# early in the sample the cycle stood where it stands, on average, after treatment. So
# $\hat\lambda' f_{pre} = +1.377$ and the factor bias is $+0.0002$, even though the unit weights
# leave a loading gap of $+0.07$. What is left of SDID's error is noise, above all the treated
# unit's own, which is what the placebo s.e. is built to measure. In panel (a) both
# counterfactuals follow the treated path before $t = 12$, the DiD one less closely at the
# cycle's turning points, and the treated path falls below both afterwards.

# %% [markdown]
# ## Your turn — is the 90% interval a 90% interval?
#
# One interval cannot tell you its coverage. Re-simulate the panel `R_YOU` times (seeds
# `0, 1, …`), fit SDID on each, and record $\hat\tau$ and two intervals: the placebo one (the
# default) and the unit bootstrap, which `synthetic_did` warns about with one treated unit. The
# warning is silenced inside the loop only because we already know it. Each panel uses 50
# replications for each method.
#
# **Predict first:** which interval covers the planted $-4.5$ more often, and which standard error
# is closer to the Monte Carlo standard deviation of $\hat\tau$?

# %%
def coverage(R, load0=1.4, n_boot=50):
    """Fit SDID on R simulated panels; placebo and bootstrap s.e. and 90% coverage."""
    rows = []
    for s in range(R):
        d, _ = simulate(s, load0)
        rp = synthetic_did(d, n_boot=n_boot, seed=42)                        # placebo (auto)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                                   # N_tr = 1 warning
            rb = synthetic_did(d, n_boot=n_boot, seed=42, se_method="bootstrap")
        rows.append({"tau": rp.tau, "se_placebo": rp.se, "se_boot": rb.se,
                     "cover_placebo": rp.lo <= tau_true <= rp.hi,
                     "cover_boot": rb.lo <= tau_true <= rb.hi})
    return pd.DataFrame(rows)


# ← Change this: number of simulated panels (try 40, 100 or 200; range 40 to 200, about 0.5 s each).
R_YOU = 40

mc = coverage(R_YOU)
sd_tau = mc["tau"].std(ddof=1)
mc_se = np.sqrt(0.90 * 0.10 / R_YOU)          # Monte Carlo s.e. of a 90% coverage rate
print(f"R = {R_YOU} panels: mean tau-hat {mc['tau'].mean():+.3f} (planted {tau_true:+.1f}); "
      f"Monte Carlo sd of tau-hat {sd_tau:.3f}")
print(pd.DataFrame({"mean se": [mc["se_placebo"].mean(), mc["se_boot"].mean()],
                    "coverage of 90% CI": [mc["cover_placebo"].mean(), mc["cover_boot"].mean()]},
                   index=["placebo (Algorithm 4)", "unit bootstrap"]).round(3).to_string())
print(f"Monte Carlo s.e. of a 90% coverage rate with R = {R_YOU}: {mc_se:.3f}")

# 1. tau-hat is centred on the planted effect.
assert abs(mc["tau"].mean() - tau_true) < 3 * sd_tau / np.sqrt(R_YOU)
# 2. The placebo interval covers at its nominal rate, up to Monte Carlo error.
assert abs(mc["cover_placebo"].mean() - 0.90) < 3 * mc_se
# 3. The bootstrap se omits the treated unit's own noise: too small, and it covers less.
assert mc["se_boot"].mean() < 0.75 * sd_tau
assert mc["cover_boot"].mean() < mc["cover_placebo"].mean()

# %% [markdown]
# **Prompts.**
#
# 1. *Basic — a treated unit outside the donors' range.* Build `df3, _ = simulate(2021, load0=3.0)`
#    and compute the three estimates of the worked code on it (pivot, 2×2 DiD, `synthetic_control`,
#    `synthetic_did(df3, n_boot=0).tau`). Rank them by distance to the planted effect and check
#    `abs(sdid3 - tau_true) < abs(sc3 - tau_true) < abs(did3 - tau_true)`. Name the assumption that
#    fails for DiD (equal exposure, i.e. parallel trends) and for SC (matching the treated level and
#    path from inside the donors' convex hull, with no intercept). Print the three largest
#    $\hat\omega_i$: why can no simplex weighting reproduce a loading of 3.0? At the baseline loading
#    SC was closest (table above), so the ordering is a property of this design, not a law.
# 2. *Intermediate — where $\hat\lambda$ looks.* Compare `synthetic_did(df3, n_boot=0).lambda_w`
#    with `res_sdid.lambda_w`. They are identical: why? (The time weights are fitted on donors
#    only.) Raising the treated loading by $d\ell$ adds $d\ell\,f_t$ to the treated path. Write
#    `did_at(l)`, the 2×2 DiD on `treated_traj + (l - factor_loadings[0]) * macro_factor`, and
#    `tau_lam_at(l)`, which uses uniform donor weights with the SDID time weights
#    (`gap = y - unweighted_control_path`, `tau = gap[T_treat:].mean() - lam @ gap[:T_treat]`).
#    Derive both slopes in $\ell$ on paper, then check
#    `np.isclose(did_at(3.0) - did_at(1.4), 1.6 * (f_post - f_pre.mean()))`,
#    `np.isclose(tau_lam_at(3.0) - tau_lam_at(1.4), 1.6 * (f_post - lam @ f_pre))` and
#    `abs(tau_lam_at(3.0) - tau_true) < 0.5 < 2.0 < abs(did_at(3.0) - tau_true)`.
# 3. *Stretch — when the placebo interval fails.* Run `mc3 = coverage(R_YOU, load0=3.0)` in a new
#    cell. Predict first: does the placebo s.e. change? Check `np.allclose(mc3["se_placebo"],
#    mc["se_placebo"])` (it is computed from the donors alone), then compare `mc3["tau"].std()`
#    with it and read the coverage. Which assumption of Algorithm 4 fails when the treated unit is
#    unlike every donor (p. 29)? Finally, size the piece the bootstrap misses at the baseline: the
#    treated unit's own noise contributes about `NOISE_SD * np.sqrt(1 / (T_periods - T_treat) + (lam**2).sum())`
#    to the sd of $\hat\tau$; compare it with `np.sqrt(sd_tau**2 - mc["se_boot"].mean()**2)`.
#
# ## Key takeaway
#
# - Use **`synthetic_did`** for regional, state or firm-level policies where parallel trends are
#   doubtful and a handful of units, or one, is treated. It keeps SC's unit matching and DiD's
#   invariance to level shifts, and its time weights remove the factor bias that the unit weights
#   leave behind.
# - With one treated unit, report the **placebo** standard error and say that it assumes the
#   treated unit's noise is like the donors'. `se_method="bootstrap"` needs several treated units.
# - **How comprehensive is this?** `sdid_multi_cohort` extends SDID to staggered adoption,
#   `puremacro.synthetic_control` implements the Abadie estimator with placebo gaps (notebook 24),
#   and notebook 10 covers staggered DiD with Callaway-Sant'Anna and Sun-Abraham.
