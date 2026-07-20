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
# # The tax multiplier, three ways
#
# **What happens to US GDP after a legislated tax increase of 1% of GDP?** The
# literature's three canonical answers disagree by a factor of three: about
# **−1** (Blanchard-Perotti 2002 QJE), about **−3** (Romer-Romer 2010 AER), and
# something in between once narrative information is used as an *instrument*
# (Mertens-Ravn 2013 AER, 2014 JME). Same country, same national accounts. This
# notebook runs all three identification philosophies on **one frozen quarterly
# dataset** — so every difference you see is identification, not data.

# %% [markdown]
# ## The three identifications in math
#
# All three start from the same reduced-form VAR in $x_t = (\tau_t, g_t, y_t)'$
# — log real federal tax revenue, log real federal spending, log real GDP:
# $$ x_t = A_1 x_{t-1} + \cdots + A_p x_{t-p} + u_t, \qquad u_t = B\,\varepsilon_t,\quad \Sigma_u = BB'. $$
#
# **(a) Blanchard-Perotti** identify the tax shock with an *institutional* fact:
# within a quarter, revenue moves with output only through the tax code, with
# elasticity $\theta = 2.08$ measured from tax-bracket and timing data, so
# $$ u^\tau_t = \theta\,u^y_t + \varepsilon^\tau_t, \qquad u^g_t = \varepsilon^g_t, \qquad u^y_t = b_\tau u^\tau_t + b_g u^g_t + \varepsilon^y_t, $$
# and $\varepsilon^\tau_t$ (the *cyclically-adjusted* tax residual) is a valid instrument for the $u^y$ equation.
#
# **(b) Romer-Romer** skip the VAR: read the legislative record, keep only tax
# changes motivated by deficits or long-run goals (not the cycle), and put that
# narrative series $z_t$ (in % of GDP) straight into a local projection
# $$ y_{t+h} - y_{t-1} = \alpha_h + m(h)\, z_t + \text{lags} + e_{t+h}, $$
# so $m(h)$ is the multiplier path (Jordà LP, lag-augmented per Montiel Olea-Plagborg-Møller 2021).
#
# **(c) Mertens-Ravn** use the same narrative series but only as an *external
# instrument* for the VAR's tax residual — relevance and exogeneity,
# $$ \mathbb{E}[z_t \varepsilon^\tau_t] = \phi \neq 0, \qquad \mathbb{E}[z_t \varepsilon^{-\tau}_t] = 0 \;\Rightarrow\; B_{\cdot 1} \propto \Sigma_u \Pi, \quad \Pi = \mathbb{E}[u_t z_t]/\mathbb{E}[z_t^2], $$
# which is robust to *measurement error* in narrative magnitudes — provided the first stage is strong.
#
# **Normalization.** We scale every identified shock so taxes rise by 1% of GDP
# on impact: $\Delta\tau_0 = (Y/T)\times 1\%$ in log points. Then the log-GDP
# response in percent *is* the dollar-for-dollar cumulative multiplier: the
# level change in GDP at horizon $h$ per initial tax dollar.

# %% [markdown]
# **Intuition.** The reduced-form correlation between taxes and output is
# hopelessly contaminated: recessions cut revenue automatically, and Congress
# legislates in response to the cycle. Each method breaks the circle with a
# different *kind* of knowledge. BP bring an out-of-sample number (the tax
# code's mechanical elasticity); RR bring archival reading (which bills were
# *not* about the cycle); MR bring an econometric bridge (narrative dates as an
# instrument, immune to sloppy magnitudes). None of the three estimates more
# data than the others — they *assume differently*. That is why their answers
# differ, and why the honest deliverable is the whole menu, not one number.

# %% [markdown]
# ## Setup — one frozen dataset
#
# Two snapshots ship with the package (regenerate with
# `tools/gen_notebook_data_tax14.py`; loaded via the package-data helper
# `puremacro.replication._data.load_csv`, the same convention as the
# `gali1999` / `kilian2009` replication snapshots):
#
# - `tax14_us_fiscal.csv` — from FRED's key-free fredgraph endpoint: `GDPC1`
#   (real GDP), `GDP` (nominal GDP), `GDPDEF` (deflator), `W006RC1Q027SBEA`
#   (federal current tax receipts), `FGEXPND` (federal current expenditures),
#   `CPIAUCSL` (CPI, the alternative deflator for the spec curve).
# - `tax14_narrative_tax_shocks.csv` — the narrative measures in % of GDP
#   (positive = tax increase), frozen from Valerie Ramey's public Handbook of
#   Macroeconomics tax archive: `rr_exog` (Romer-Romer exogenous changes),
#   `mtr_u` / `mtr_a` (Mertens-Ravn unanticipated / anticipated changes). The
#   GitHub mirror hard-coded in `puremacro.narrative.replication` is dead, so
#   the loaders below read this snapshot via `csv_path=`.
#
# Simplifications vs the papers, stated up front: total receipts instead of
# BP's net taxes (receipts minus transfers), no per-capita scaling, no
# deterministic trends, and a common 1950Q1-2006Q4 sample (the narrative
# record ends in the mid-2000s).

# %%
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.replication._data import load_csv
from puremacro.narrative.replication import load_romer_romer_2010, load_mertens_ravn_2013
from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf as var_irf
from puremacro.inference.wild_bootstrap import wild_bootstrap_var
from puremacro.inference.weak_iv import olea_pflueger_f
from puremacro.var.identify.proxy import proxy_svar
from puremacro.lp.la_lp import la_lp
from puremacro.inference.spec_curve import enumerate_specs, run_spec_curve

# --- frozen fiscal aggregates (FRED fredgraph snapshot) ----------------------
fiscal = load_csv("tax14_us_fiscal")
fiscal["date"] = pd.to_datetime(fiscal["date"])
fiscal = fiscal.set_index("date")

# --- narrative measures through the package loaders --------------------------
NARR_CSV = str(resources.files("puremacro.replication.data")
               .joinpath("tax14_narrative_tax_shocks.csv"))
rr_inst = load_romer_romer_2010(csv_path=NARR_CSV)               # column rr_exog
mr_u_inst = load_mertens_ravn_2013(csv_path=NARR_CSV, kind="unanticipated")
mr_a_inst = load_mertens_ravn_2013(csv_path=NARR_CSV, kind="anticipated")
print(f"narrative events: RR exogenous = {len(rr_inst.events)}, "
      f"MR unanticipated = {len(mr_u_inst.events)}, "
      f"MR anticipated = {len(mr_a_inst.events)}")
assert len(rr_inst.events) == 45 and len(mr_u_inst.events) == 31

def build_dataset(start="1950-01-01", end="2006-12-31", deflator="gdpdef"):
    """Common sample: logs x100 for the VAR, narrative series as % of GDP."""
    d = fiscal.loc[start:end].copy()
    d["tau"] = 100 * np.log(d["fedtax"] / d[deflator])   # log real federal taxes
    d["g"] = 100 * np.log(d["fedspend"] / d[deflator])   # log real federal spending
    d["y"] = 100 * np.log(d["gdpc1"])                    # log real GDP
    # Narrative instruments: zero in quarters with no legislated change.
    d["rr"] = rr_inst.quarterly.reindex(d.index).fillna(0.0)
    d["mtu"] = mr_u_inst.quarterly.reindex(d.index).fillna(0.0)
    d["mta"] = mr_a_inst.quarterly.reindex(d.index).fillna(0.0)
    return d

d = build_dataset()
tax_share = (d["fedtax"] / d["gdp"]).mean()
SCALE = 1.0 / tax_share       # % change in tax revenue per 1%-of-GDP tax shock
print(f"sample: {d.index[0].date()} .. {d.index[-1].date()}  (T = {len(d)})")
print(f"mean federal tax / GDP = {tax_share:.3f}  ->  a 1%-of-GDP tax increase "
      f"raises revenue by {SCALE:.1f}%")
assert len(d) == 228
assert 0.10 < tax_share < 0.14

P, H = 4, 16                  # VAR lags and IRF horizon (quarters)
hgrid = np.arange(H + 1)

# %% [markdown]
# ### The narrative record at a glance
#
# Both series are *event* series: zero except in quarters when a legislated
# change took effect. Sign is in % of GDP, positive = tax increase. Note what
# the narrative reading *excludes*: the 1968 surcharge — a textbook tax hike —
# is classified as countercyclical (endogenous) by Romer-Romer, so it is
# **absent** from `rr_exog`. That editorial judgment *is* the identification.

# %%
cols = _nbstyle.palette(2)
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, sharey=True)
for ax, col, lbl, c in [(axes[0], "rr", "Romer-Romer exogenous", cols[0]),
                        (axes[1], "mtu", "Mertens-Ravn unanticipated", cols[1])]:
    nz = d[col] != 0
    ax.vlines(d.index[nz], 0, d.loc[nz, col], color=c, linewidth=1.6)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_ylabel("% of GDP")
    ax.set_title(lbl, fontsize=10)
for ts, txt in [("1964-04-01", "'64 Kennedy-\nJohnson cut"),
                ("1982-01-01", "'81-'83 ERTA\nphase-ins"),
                ("2003-07-01", "'01/'03\nBush cuts")]:
    axes[0].annotate(txt, xy=(pd.Timestamp(ts), d.loc[ts, "rr"]),
                     xytext=(pd.Timestamp(ts), -2.6), fontsize=7, color="0.35",
                     ha="center", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
axes[1].set_xlabel("Quarter")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1. Blanchard-Perotti (2002): the institutional elasticity
#
# The whole identification is one number: $\theta = 2.08$, the within-quarter
# elasticity of federal revenue to output implied by tax-bracket structure and
# collection lags — measured from the tax code, not estimated from the VAR.
# Below, the impact-matrix algebra is written out in ~15 lines on top of
# `estimate_var`: subtract the automatic response ($\varepsilon^\tau = u^\tau -
# \theta u^y$), let spending ignore output within the quarter, and use both
# structural shocks as instruments for the output equation. Bands come from
# reusing `wild_bootstrap_var` with this impact function plugged in.

# %%
def make_bp_impact(theta):
    """Blanchard-Perotti (2002 QJE) impact matrix on [tau, g, y] residuals."""
    def bp_impact(A_list, Sigma, resid):
        u_tau, u_g, u_y = resid[:, 0], resid[:, 1], resid[:, 2]
        e_tau = u_tau - theta * u_y            # cyclically-adjusted tax shock
        e_g = u_g                              # spending can't react within quarter
        Z = np.column_stack([e_tau, e_g])      # valid instruments for the y equation
        X = np.column_stack([u_tau, u_g])
        b = np.linalg.solve(Z.T @ X, Z.T @ u_y)   # IV: u_y = b1*u_tau + b2*u_g + e_y
        e_y = u_y - X @ b
        A0 = np.array([[1.0,   0.0,  -theta],     # u_tau - theta*u_y          = e_tau
                       [0.0,   1.0,   0.0],       # u_g                        = e_g
                       [-b[0], -b[1], 1.0]])      # u_y - b1*u_tau - b2*u_g    = e_y
        return np.linalg.inv(A0) * np.array([e_tau.std(), e_g.std(), e_y.std()])
    return bp_impact

THETA_BP = 2.08
Y_var = d[["tau", "g", "y"]].to_numpy()
pt, lo, hi = wild_bootstrap_var(Y_var, p=P, horizon=H,
                                impact_fn=make_bp_impact(THETA_BP),
                                n_boot=200, ci=0.9, seed=0)
pt, lo, hi = [np.transpose(a, (2, 0, 1)) for a in (pt, lo, hi)]   # -> (H+1, n, n)
c_bp = SCALE / pt[0, 0, 0]                 # rescale: tax impact = 1% of GDP
m_bp, m_bp_lo, m_bp_hi = pt[:, 2, 0] * c_bp, lo[:, 2, 0] * c_bp, hi[:, 2, 0] * c_bp
bp_peak = m_bp[:13].min()
print(f"BP multiplier: impact {m_bp[0]:+.2f} | 2yr {m_bp[8]:+.2f} | "
      f"peak {bp_peak:+.2f} at h={int(m_bp[:13].argmin())}")
assert m_bp[0] > -1.0                          # small impact effect...
assert -2.2 < m_bp[8] < -0.6                   # ...builds toward ~ -1
assert -2.6 < bp_peak < -0.9                   # BP's published ballpark

# %% [markdown]
# **Read the output.** The BP multiplier starts near zero and builds slowly to
# about **−1.2 after two years** (peak ≈ −1.5) — Blanchard and Perotti's
# famous "close to one dollar for a dollar". The impact response is small by
# construction: after purging the automatic $\theta u^y$ component, what is
# left of the tax residual barely covaries with output within the quarter.
# Everything rests on $\theta$ being the *right* out-of-sample number — hold
# that thought for the fill-in at the end.

# %% [markdown]
# ## 2. Romer-Romer (2010): the narrative regression
#
# RR bypass the VAR: their exogenous series (already in % of GDP) enters a
# lag-augmented local projection directly. Note the LP left-hand side is the
# *change form* $y_{t+h}-y_{t-1}$, so the coefficient is the response of the
# GDP **level** at $t+h$ — with our 1%-of-GDP units, the multiplier itself.
# No revenue equation, no elasticity: the identifying assumption is that the
# archival reading really did isolate cycle-independent tax changes.

# %%
H_LP = 20      # LP horizons; lag augmentation p_aug = 4 + 20 (PMW 2021 default)
lp_rr = la_lp(d, y="y", x="rr", horizons=range(0, H_LP + 1), n_lags=4, alpha=0.10)
m_rr, m_rr_lo, m_rr_hi = (lp_rr["beta"].to_numpy(), lp_rr["lo"].to_numpy(),
                          lp_rr["hi"].to_numpy())
rr_peak = m_rr[:13].min()
rr_peak_h = int(m_rr[:13].argmin())
print(f"RR multiplier: impact {m_rr[0]:+.2f} | 2yr {m_rr[8]:+.2f} | "
      f"peak {rr_peak:+.2f} at h={rr_peak_h}")
assert -4.5 < rr_peak < -1.8                       # RR's published -2.5..-3 zone
assert 4 <= rr_peak_h <= 12
assert abs(rr_peak) > abs(bp_peak) + 0.5           # narrative >> SVAR, same data

# %% [markdown]
# **Read the output.** The same 1%-of-GDP tax increase now costs about **−3%
# of GDP after two years** — roughly *triple* the BP answer, on the identical
# dataset. This is Romer-Romer's headline (their Figure 4 bottoms out just
# past −3% at ten quarters). Nothing about the estimator explains the gap; the
# narrative series simply embodies a different claim about which tax changes
# are exogenous.

# %% [markdown]
# ## 3. Mertens-Ravn (2013): narrative meets the SVAR
#
# MR's move: don't put the narrative series in a regression — use it as an
# **external instrument** for the VAR's tax residual (`proxy_svar`). If the
# narrative *dates* are right, mismeasured *magnitudes* no longer bias the
# multiplier. The price is a first stage: the proxy must actually correlate
# with the tax innovation. We check that first, with the Olea-Pflueger
# effective F for each candidate instrument — including MR's key split into
# unanticipated vs anticipated changes (fiscal foresight: a pre-announced
# change is not a surprise when it takes effect).

# %%
est = estimate_var(Y_var, P)
u_tau = est.resid[:, 0]
f_stats = {}
for name, col in [("RR exogenous (all)", "rr"),
                  ("MR unanticipated", "mtu"),
                  ("MR anticipated", "mta")]:
    z = d[col].to_numpy()[-len(u_tau):]           # align to VAR residuals
    f_stats[name] = olea_pflueger_f(u_tau, z.reshape(-1, 1))
    print(f"first-stage effective F | {name:20s} = {f_stats[name]:6.2f}")
assert f_stats["MR anticipated"] < f_stats["MR unanticipated"] < f_stats["RR exogenous (all)"]
assert f_stats["MR unanticipated"] < 10           # weak on aggregate revenue data

prox = proxy_svar(Y_var, p=P, horizon=H,
                  instrument_series=d["mtu"].to_numpy(),
                  n_boot=200, ci=0.9, seed=0)
c_pr = SCALE / prox.irf_point[0, 0, 0]
m_prox = prox.irf_point[:, 2, 0] * c_pr
print(f"MR proxy-SVAR: effective F = {prox.first_stage_F:.2f} | "
      f"impact {m_prox[0]:+.2f} | 2yr {m_prox[8]:+.2f} | 3yr {m_prox[12]:+.2f}")
assert abs(m_prox[8]) < 1.0                       # weak proxy -> unstable point

# %% [markdown]
# **Read the output.** The honest headline here is the **F statistic, not the
# multiplier**. On aggregate federal receipts the MR proxy is *weak* (effective
# F ≈ 1.4, far below the Olea-Pflueger comfort zone; even the full RR series
# only reaches ≈ 5). With a weak first stage the unit normalization divides by
# a noisy near-zero revenue response, so the point path (≈ +0.6 on impact,
# drifting to ≈ −0.5) is not interpretable — exactly the fragility
# Jentsch-Lunsford (2019 AER) documented for MR's setup. MR's own strong
# results use *tax-specific* average tax rates (personal, corporate), not one
# aggregate revenue pile; the anticipated series' F ≈ 0 confirms their
# foresight logic beautifully.
#
# So where does narrative-as-instrument leave the multiplier? Mertens-Ravn's
# 2014 JME reconciliation extracts the answer differently: the narrative
# information implies the true output elasticity of revenue is **3.13**, not
# 2.08 — BP's number is too low because it misses the within-quarter response
# of collections. Impose $\theta = 3.13$ in the *same* BP machinery:

# %%
THETA_MR = 3.13     # Mertens-Ravn (2014 JME): narrative-implied tax-output elasticity
pt2, lo2, hi2 = wild_bootstrap_var(Y_var, p=P, horizon=H,
                                   impact_fn=make_bp_impact(THETA_MR),
                                   n_boot=200, ci=0.9, seed=0)
pt2, lo2, hi2 = [np.transpose(a, (2, 0, 1)) for a in (pt2, lo2, hi2)]
c_mr = SCALE / pt2[0, 0, 0]
m_mr, m_mr_lo, m_mr_hi = pt2[:, 2, 0] * c_mr, lo2[:, 2, 0] * c_mr, hi2[:, 2, 0] * c_mr
mr_peak = m_mr[:13].min()
print(f"MR (theta=3.13) multiplier: impact {m_mr[0]:+.2f} | 2yr {m_mr[8]:+.2f} | "
      f"peak {mr_peak:+.2f} at h={int(m_mr[:13].argmin())}")
# The reconciliation lands between BP and RR — the task's 'three ways' ordering:
assert m_rr[8] < m_mr[8] < m_bp[8] < 0
assert abs(bp_peak) < abs(mr_peak) < abs(rr_peak)

# %% [markdown]
# **Read the output.** With the narrative-implied elasticity, the identical
# VAR now delivers a two-year multiplier of about **−2.1** (peak ≈ −2.4) —
# squarely *between* BP's −1 and RR's −3, which is exactly Mertens-Ravn's
# reconciliation: the BP-vs-RR dispute is not SVAR-vs-LP, it is a dispute
# about one elasticity, and the narrative record votes for the higher value.

# %% [markdown]
# ### Hero figure — one question, three answers
#
# Cumulative output response (the GDP level change, in % of GDP — i.e. dollars
# of GDP per initial tax dollar) to a legislated tax increase of 1% of GDP,
# under the three identification schemes, with 90% bands. The weak-proxy path
# is drawn as a thin reference line, without a band, as a caution — not a
# result.

# %%
c3 = _nbstyle.palette(3)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.fill_between(hgrid, m_bp_lo, m_bp_hi, color=c3[0], alpha=0.14)
ax.plot(hgrid, m_bp, color=c3[0], linewidth=1.8,
        label=f"Blanchard-Perotti ($\\theta$=2.08): peak {bp_peak:+.1f}")
ax.fill_between(hgrid, m_mr_lo, m_mr_hi, color=c3[1], alpha=0.14)
ax.plot(hgrid, m_mr, color=c3[1], linewidth=1.8, linestyle="--",
        label=f"Mertens-Ravn reconciled ($\\theta$=3.13): peak {mr_peak:+.1f}")
ax.fill_between(hgrid, m_rr_lo[:H + 1], m_rr_hi[:H + 1], color=c3[2], alpha=0.14)
ax.plot(hgrid, m_rr[:H + 1], color=c3[2], linewidth=1.8, linestyle="-.",
        label=f"Romer-Romer narrative LP: peak {rr_peak:+.1f}")
ax.plot(hgrid, m_prox, color="0.55", linewidth=1.0, linestyle=":",
        label=f"MR proxy-SVAR (weak: F={prox.first_stage_F:.1f})")
ax.set_xlabel("Quarters after a tax increase of 1% of GDP")
ax.set_ylabel("GDP response (% of GDP) = dollar multiplier")
ax.set_title("The US tax multiplier under three identification schemes\n"
             "(one dataset: 1950Q1-2006Q4)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The specification curve — is it really identification?
#
# Maybe the gaps above are luck: a sample quirk, a deflator choice. The clean
# way to check is a **specification curve** (Simonsohn-Simmons-Nelson 2020):
# estimate the two-year multiplier for the full Cartesian grid of
# *identification × sample × deflator* and look at what actually moves the
# estimate. (The LP here adds `tau` and `g` as controls so the deflator enters
# the RR spec too.)

# %%
SAMPLES = {"1950-2006": ("1950-01-01", "2006-12-31"),
           "1954-2006": ("1954-01-01", "2006-12-31"),
           "1950-1979": ("1950-01-01", "1979-12-31")}
H8 = 8   # estimand: the two-year multiplier m(8)

def estimator(_, spec):
    ds = build_dataset(*SAMPLES[spec["sample"]], deflator=spec["deflator"])
    sc = 1.0 / (ds["fedtax"] / ds["gdp"]).mean()
    Yv = ds[["tau", "g", "y"]].to_numpy()
    ident = spec["identification"]
    if ident in ("BP 2.08", "MR 3.13"):
        th = 2.08 if ident == "BP 2.08" else 3.13
        p_, l_, h_ = wild_bootstrap_var(Yv, p=P, horizon=H8, n_boot=100,
                                        impact_fn=make_bp_impact(th), ci=0.9, seed=0)
        k = sc / p_[0, 0, 0]
        return {"sigma_hat": p_[2, 0, H8] * k,
                "se": abs(h_[2, 0, H8] - l_[2, 0, H8]) * abs(k) / (2 * 1.645)}
    if ident == "RR LP":
        lp = la_lp(ds, y="y", x="rr", horizons=[H8], n_lags=4, controls=["tau", "g"])
        return {"sigma_hat": float(lp["beta"].iloc[0]), "se": float(lp["se"].iloc[0])}
    pr = proxy_svar(Yv, p=P, horizon=H8, instrument_series=ds["mtu"].to_numpy(),
                    n_boot=100, ci=0.9, seed=0)
    k = sc / pr.irf_point[0, 0, 0]
    return {"sigma_hat": pr.irf_point[H8, 2, 0] * k,
            "se": abs(pr.irf_upper[H8, 2, 0] - pr.irf_lower[H8, 2, 0]) * abs(k) / (2 * 1.645),
            "first_stage_F": pr.first_stage_F}

grid = {"identification": ["BP 2.08", "MR 3.13", "MR proxy", "RR LP"],
        "sample": list(SAMPLES), "deflator": ["gdpdef", "cpi"]}
curve = run_spec_curve(data=None, specs=enumerate_specs(grid), estimator=estimator,
                       ci_level=0.90)
med = curve.groupby("identification")["sigma_hat"].median()
print(curve[["identification", "sample", "deflator", "sigma_hat", "se"]]
      .round(2).to_string(index=False))
print("\nmedian two-year multiplier by identification:")
print(med.round(2).to_string())
assert len(curve) == 24
assert med["RR LP"] < med["BP 2.08"] - 0.5        # identification gap >> ...
assert med["MR 3.13"] < med["BP 2.08"] - 0.25
assert abs(med["MR proxy"]) < 0.75                # weak proxy hugs zero
spread_ident = med.max() - med.min()
spread_defl = curve.groupby(["identification", "sample"])["sigma_hat"] \
                   .agg(lambda s: s.max() - s.min()).median()
print(f"\nspread across identifications (medians): {spread_ident:.2f} "
      f"| median spread across deflators, all else fixed: {spread_defl:.2f}")
assert spread_ident > 3 * spread_defl

# %%
order = curve.sort_values("sigma_hat").reset_index(drop=True)
marks = {"BP 2.08": "o", "MR 3.13": "s", "MR proxy": "^", "RR LP": "D"}
c4 = _nbstyle.palette(4)
colr = dict(zip(grid["identification"], c4))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1.4]})
x = np.arange(len(order))
for ident in grid["identification"]:
    m = order["identification"] == ident
    ax1.errorbar(x[m], order.loc[m, "sigma_hat"],
                 yerr=1.645 * order.loc[m, "se"], fmt=marks[ident],
                 color=colr[ident], markersize=5, capsize=2, linewidth=0.9,
                 label=ident)
ax1.axhline(0, color="0.6", linewidth=0.8, linestyle=":")
ax1.set_ylabel("Two-year multiplier m(8)")
ax1.set_title("Specification curve: 24 specs, ordered by estimate")
ax1.legend(fontsize=8, ncol=2)
rows = [("identification", grid["identification"]),
        ("sample", list(SAMPLES)), ("deflator", grid["deflator"])]
ytick, ylab = [], []
yy = 0
for dim, values in rows:
    for v in values:
        on = order[dim] == v
        ax2.scatter(x[on], np.full(on.sum(), yy), s=14, color="0.25", marker="|")
        ytick.append(yy); ylab.append(f"{v}")
        yy -= 1
    yy -= 0.6
ax2.set_yticks(ytick); ax2.set_yticklabels(ylab, fontsize=7)
ax2.set_xlabel("Specification (sorted)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **The punchline.** Read the bottom panel against the top: the sorted curve
# is *segmented by identification scheme*, not by sample or deflator. Swapping
# the deflator moves the two-year multiplier by ~0.2; swapping the
# identification moves it by ~2 (medians: BP ≈ −1.2, MR ≈ −2.1, RR ≈ −2.3,
# weak proxy ≈ 0). On one dataset, with one estimand, **identification — not
# estimation — drives the answer.** When someone quotes you "the" tax
# multiplier, the first question is not "what data?" but "what did they assume
# to make the shock exogenous?"

# %% [markdown]
# ## Your turn — the multiplier is a function of one assumption
#
# The entire BP-vs-RR dispute compresses into the within-quarter elasticity
# $\theta$. Change it below and watch the two-year multiplier travel the whole
# BP → MR → RR range: $\theta = 0$ is a pure Cholesky ordering (taxes first),
# 2.08 is Blanchard-Perotti's institutional value, 3.13 is Mertens-Ravn's
# narrative-implied value.

# %%
# ← Change this: the assumed within-quarter tax-output elasticity
#   (try 0.0, 1.0, 2.08, 3.13, 4.0).
THETA_TRY = 2.08
est_try = estimate_var(Y_var, P)
B_try = make_bp_impact(THETA_TRY)(est_try.A_list, est_try.Sigma, est_try.resid)
path_try = var_irf(est_try.A_list, B_try, H)[:, 2, 0] * (SCALE / (var_irf(est_try.A_list, B_try, 0)[0, 0, 0]))
print(f"theta = {THETA_TRY:.2f}  ->  two-year multiplier m(8) = {path_try[8]:+.2f}")
# Holds for the default and the whole suggested sweep: more automatic
# stabilizer purged -> a (weakly) more negative multiplier than theta=0.
theta0_m8 = var_irf(est_try.A_list, make_bp_impact(0.0)(est_try.A_list, est_try.Sigma, est_try.resid), H)[:, 2, 0]
theta0_m8 = theta0_m8[8] * (SCALE / make_bp_impact(0.0)(est_try.A_list, est_try.Sigma, est_try.resid)[0, 0])
assert path_try[8] <= theta0_m8 + 1e-6

# %% [markdown]
# **Prompts.** (1) *Basic*: set `THETA_TRY = 0.0` and compare with notebook 06's
# lesson — a Cholesky ordering with taxes first finds almost no multiplier.
# Why does treating the raw tax residual as the shock bias the answer toward
# zero? (Think about which way the automatic elasticity runs.) (2)
# *Intermediate*: re-run section 3's F table on the `1954-2006` sample
# (rebuild `d`) — does the weak-instrument verdict change? (3) *Stretch*: pass
# `d["mta"]` (anticipated changes) to `proxy_svar` and interpret the resulting
# path in light of its F ≈ 0 — why are pre-announced tax changes almost
# uninformative about tax *surprises*, and what would a "fiscal foresight" VAR
# need to fix this?
#
# **How comprehensive is this?** The pieces are all reusable:
# `puremacro.var.identify` adds sign restrictions, Blanchard-Quah, max-share
# and heteroskedasticity identification to the proxy-SVAR used here;
# `puremacro.lp` has state-dependent, IV, panel, smooth and quantile LPs
# beyond `la_lp`; `puremacro.narrative.replication` ships a dozen narrative
# datasets (Ramey defense news, Cloyne UK taxes, Guajardo-Leigh-Pescatori
# consolidations, ...) that plug into the same `NarrativeInstrument` API; and
# `puremacro.inference.spec_curve` powers the robustness grid for any
# estimator. The `examples/` gallery runs `svariv_mertens_ravn` end-to-end on
# synthetic data with a strong first stage, for contrast with the weak one
# found here.
