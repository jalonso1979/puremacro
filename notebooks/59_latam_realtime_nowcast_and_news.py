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
# # Latin America Real-Time Nowcasting & Bańbura-Modugno News Attribution: Ragged Edges, Revision Decomposition, and Density Calibration
#
# **How can a central-bank desk nowcast an activity index from a ragged-edge monthly panel, explain each update of the nowcast release by release with the Bańbura & Modugno (2014) news decomposition, replay the nowcast exactly as it would have looked on an earlier date, and grade density forecasts with probability integral transform (PIT) tests?**
#
# **The panel in this notebook is simulated. Read no fact about Mexico or Brazil out of it.** Every value is generated in the first code cell from the seed `np.random.default_rng(42)`: each country has its own random-walk factor, and each indicator is a noisy multiple of it. The provider names and series identifiers are labels borrowed from the catalogue of `puremacro.fetch.realtime` (which itself marks several identifiers as unverified), and the simulated values ignore those series' true frequencies and units: Brazil's `433`, for example, is a monthly percentage change, not an index. The publication calendar is stylised. The portable `.pmz` cartridge records `SIMULATED` in its provenance notes. Nothing is downloaded, so the notebook also runs in the browser.
#
# Central banks cannot wait for national accounts that arrive weeks after the month or quarter ends, so they track monthly indicators that arrive on different calendars. On any given day the most recent months of the panel are therefore incomplete: a "ragged edge". Dynamic factor models (Giannone, Reichlin & Small 2008; Doz, Giannone & Reichlin 2011) fill that edge with the Kalman smoother. When new data arrive, the nowcast moves, and the Bańbura & Modugno (2014) decomposition attributes the move to each release's surprise and to revisions of earlier data, with an exact accounting identity. Density forecasts are graded with PIT tests (Berkowitz 2001).

# %% [markdown]
# ## The method in math — Dynamic Factors, News Decomposition, and Density Calibration
#
# **1. Dynamic factor model in state-space form.** Let $X_t = [x_{1, t}, \dots, x_{n, t}]^\top$ be the $n$ monthly indicators, each standardised by its sample mean and standard deviation $s_j$. They share $r$ latent factors $F_t \in \mathbb{R}^r$ plus idiosyncratic errors $\xi_t$:
# $$ X_t = \Lambda F_t + \xi_t, \quad \xi_t \sim \text{i.i.d.} \, \mathcal{N}(0, R), \quad R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2). $$
# The factors follow a vector autoregression of order $p$:
# $$ F_t = A_1 F_{t-1} + \dots + A_p F_{t-p} + u_t, \quad u_t \sim \text{i.i.d.} \, \mathcal{N}(0, Q). $$
# The parameters $(\Lambda, A, Q, R)$ are estimated in two steps, principal components and then the Kalman filter and smoother, as in Doz, Giannone and Reichlin (2011). The nowcast of the target $y_{t^*}$ is $\hat{y}_{t^*|v} = \mathbb{E}[y_{t^*} \mid \Omega_v]$, the Kalman-smoothed value given the information set $\Omega_v$ of vintage $v$. Here the target is the `gdp` index for the last reference month, which neither vintage has published yet.
#
# **2. Bańbura & Modugno (2014) news attribution.** Let $\Omega_{v-1} \subset \Omega_v$ be the information sets of two successive vintages. The new information consists of releases for recent periods $j \in \mathcal{I}_{\text{new}}$ and revisions of earlier observations $k \in \mathcal{I}_{\text{rev}}$:
# $$ I_{j, v} \equiv x_{j, t_j} - \mathbb{E}[x_{j, t_j} \mid \Omega_{v-1}], \quad R_{k, v} \equiv x_{k, t_k}^{(v)} - x_{k, t_k}^{(v-1)}. $$
# With the model parameters held fixed at their estimates on $\Omega_v$, the change in the nowcast $\Delta \hat{y}_{t^*|v} \equiv \hat{y}_{t^*|v} - \hat{y}_{t^*|v-1}$ decomposes into
# $$ \Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot I_{j, v} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot R_{k, v}, $$
# with weights given by the Kalman gain and the state covariances:
# $$ \omega = \operatorname{Cov}\left(y_{t^*}, \begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right) \left[\operatorname{Var}\left(\begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right)\right]^{-1}. $$
# The decomposition is exact by construction, so its reported error,
# $$ \text{Decomposition Error} \equiv \left| \Delta \hat{y}_{t^*|v} - \left(\sum \text{Impact}_{\text{releases}} + \sum \text{Impact}_{\text{revisions}}\right) \right|, $$
# should be of the order of rounding error. That is an internal consistency check, not evidence that the nowcast is accurate.
#
# **3. A model-based fan chart.** With state variance $P_{T|T}$ in the last month $T$ of the panel, the $h$-month-ahead state variance and the predictive variance of indicator $j$ are
# $$ P_{T+h|T} = A P_{T+h-1|T} A^\top + Q, \qquad \operatorname{Var}(x_{j,T+h} \mid \Omega_v) = s_j^2 \left(\lambda_j^\top P_{T+h|T} \lambda_j + R_{jj}\right). $$
# The fan bands are Gaussian quantiles of these variances; `RealtimeNowcastResult.fan_chart` computes them from the fitted model, starting at the first unpublished month of the target. They ignore parameter uncertainty, so they are too narrow if the model is estimated on a short sample.
#
# **4. PIT tests of density forecasts (Berkowitz 2001).** For outcomes $\{y_t\}_{t=1}^T$ and predictive densities $\mathcal{N}(\mu_t, \sigma_t^2)$, the PIT is
# $$ p_t = \Phi\left(\frac{y_t - \mu_t}{\sigma_t}\right), \quad z_t = \Phi^{-1}(p_t). $$
# If the densities are correct, $p_t \sim \text{i.i.d.} \, \mathcal{U}(0, 1)$ and $z_t \sim \text{i.i.d.} \, \mathcal{N}(0, 1)$. Berkowitz fits
# $$ (z_t - \mu) = \rho (z_{t-1} - \mu) + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.} \, \mathcal{N}(0, \sigma_\varepsilon^2), $$
# and tests $H_0: \mu = 0, \sigma_\varepsilon^2 = 1, \rho = 0$ with
# $$ \text{LR} = -2 \left[ \ln L(0, 1, 0) - \ln L(\hat{\mu}, \hat{\sigma}_\varepsilon^2, \hat{\rho}) \right] \sim \chi^2(3). $$
# `pit_uniformity_test` also reports the Kolmogorov-Smirnov (KS) test of uniformity of $p_t$, which looks only at the marginal distribution of the PITs.

# %% [markdown]
# ## Intuition
#
# **Intuition.** Policy committees meet on a fixed schedule, whether or not the national accounts are out. Monthly indicators arrive earlier, but each covers only part of the economy and each has its own publication lag, so the latest months of the panel are incomplete in different ways for different series.
#
# A dynamic factor model assumes that a few common forces drive all the indicators. The Kalman smoother then estimates those forces from whatever has been published and fills in the missing values, including the target. When a new figure comes out, what moves the nowcast is not the figure itself but its *surprise*: the gap between the published value and what the model expected from the earlier information set. A strong release that the model already expected changes nothing; a weak release that was expected to be strong lowers the nowcast.
#
# The Bańbura-Modugno decomposition turns each update into an account: the change in the nowcast equals the sum of each release's surprise times its weight, plus the effect of revisions to earlier data. Staff can then say which release moved the nowcast and by how much.
#
# A nowcast is only useful if we know which information set produced it. Replaying the nowcast "as of" an earlier date must use only the data published by then; otherwise the replay looks better than the real-time nowcast ever was. Finally, a density forecast is honest only if its bands are neither too narrow nor too wide. PIT tests can detect such errors, but only with enough forecasts; the Your-turn cell measures how often they do.

# %%
# Preamble: numerical libraries, plotting style, and realtime nowcast modules
import sys
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.fetch.realtime import (
    VintagePanel,
    pack_realtime_cartridge,
    load_realtime_cartridge,
)
from puremacro.nowcast import (
    realtime_nowcast,
    pit_uniformity_test,
)

# Deterministic random seed: every value below is simulated
rng = np.random.default_rng(42)

# %%
# --- Experiment 1: a simulated two-country vintage panel with a publication calendar ---
# ALL VALUES BELOW ARE SIMULATED from seed 42. The provider names and series IDs are labels
# borrowed from puremacro.fetch.realtime's catalogue; the simulated values ignore those
# series' true frequencies and units.
dates = pd.date_range("2022-01-01", periods=36, freq="MS")   # reference months 2022-01 to 2024-12
v1 = pd.Timestamp("2025-01-15")                               # vintage v-1
v2 = pd.Timestamp("2025-02-15")                               # vintage v
# Stylised publication lags in months: a vintage dated in month m holds data up to month m - 1 - lag
pub_lag = {"gdp": 2, "activity": 1, "ip": 1, "cpi": 0, "policy_rate": 0}
revised_month, revision_size = 20, 0.45                       # vintage v revises 'activity' for 2023-09

mex_series = [
    ("gdp", "inegi", "735848", "index"),
    ("activity", "inegi", "736184", "index"),
    ("ip", "inegi", "736184_IP", "index"),
    ("cpi", "inegi", "628197", "index"),
    ("policy_rate", "banxico", "SF61745", "rate"),
]

bra_series = [
    ("gdp", "bcb", "22099", "index"),
    ("activity", "bcb", "24363", "index"),
    ("ip", "bcb", "21859", "index"),
    ("cpi", "bcb", "433", "index"),
    ("policy_rate", "bcb", "432", "rate"),
]

rows = []
for country, series_list in [("MEX", mex_series), ("BRA", bra_series)]:
    # Each country has its own random-walk common factor
    f_latent = np.cumsum(rng.normal(scale=0.25, size=len(dates)))
    for var, prov, sid, un in series_list:
        load = rng.uniform(0.7, 1.3)
        noise = rng.normal(scale=0.15, size=len(dates))
        y_sim = 100.0 + 1.5 * f_latent * load + noise if un == "index" else 8.0 - 0.1 * f_latent * load + noise

        for vint in (v1, v2):
            last_month = (vint.to_period("M") - 1 - pub_lag[var]).to_timestamp()
            for t_idx, d in enumerate(dates):
                if d > last_month:
                    continue  # not yet published in this vintage: the ragged edge
                val = float(y_sim[t_idx])
                if vint == v2 and var == "activity" and t_idx == revised_month:
                    val += revision_size  # the agency revises an old month
                rows.append({
                    "country": country, "variable": var, "date": d, "vintage": vint,
                    "value": val, "provider": prov, "series_id": sid, "units": un,
                })

df_panel = pd.DataFrame(rows)
panel_raw = VintagePanel(df_panel)

# Package into a self-verifying .pmz cartridge and reload it
with tempfile.TemporaryDirectory() as td:
    cart_path = Path(td) / "latam_realtime_nowcast.pmz"
    pack_realtime_cartridge(
        panel_raw,
        cart_path,
        source="Banxico, INEGI, BCB (identifiers only)",
        notes="SIMULATED panel on identifiers borrowed from the puremacro.fetch.realtime catalogue",
    )
    loaded_panel = load_realtime_cartridge(cart_path)

calendar = (df_panel[df_panel["country"] == "MEX"].groupby(["variable", "vintage"])["date"].max()
            .dt.strftime("%Y-%m").unstack("vintage"))
calendar.columns = [f"published by {c:%Y-%m-%d}" for c in calendar.columns]

print("Simulated vintage panel (identifiers borrowed, values simulated):")
print(f"  Observations      : {len(panel_raw):,}")
print(f"  Countries         : {panel_raw.countries}")
print(f"  Variables         : {panel_raw.variables}")
print(f"  Reference months  : {dates[0]:%Y-%m} to {dates[-1]:%Y-%m}")
print(f"  Vintages          : {v1:%Y-%m-%d} (v-1) and {v2:%Y-%m-%d} (v)")
print("\nLast reference month published, Mexico:")
print(calendar.to_string())

assert panel_raw.countries == ["BRA", "MEX"]
assert len(loaded_panel) == len(panel_raw)
assert (df_panel["date"] < df_panel["vintage"]).all(), "no vintage may hold data for months after its own date"
assert not ((df_panel["variable"] == "gdp") & (df_panel["date"] == dates[-1])).any(), "the target month is unpublished"

# %%
# --- Experiment 2: nowcasts, news attribution and a historical replay ---
target_month = dates[-1]
res_mex = realtime_nowcast(country="MEX", panel=loaded_panel, method="dfm", n_factors=1)
res_bra = realtime_nowcast(country="BRA", panel=loaded_panel, method="dfm", n_factors=1)

# Replay: what the same call returned on v-1, and the same call on a panel cut at v-1
res_mex_v1 = realtime_nowcast(country="MEX", panel=loaded_panel, method="dfm", n_factors=1, as_of=v1)
res_mex_cut = realtime_nowcast(country="MEX", panel=VintagePanel(df_panel[df_panel["vintage"] <= v1]),
                               method="dfm", n_factors=1)
nd_mex = res_mex.news_decomposition
nd_bra = res_bra.news_decomposition

print(res_mex.summary())
print("\nNews by release (Mexico):")
print(nd_mex.news_table.to_string(index=False, float_format="{:.4f}".format))
print("\nRevisions (Mexico):")
print(nd_mex.revision_table.to_string(index=False, float_format="{:.4f}".format))
print("\n" + "=" * 74)
print(f"Brazil nowcast for {target_month:%Y-%m}: {res_bra.nowcast:.4f} (s.d. {res_bra.forecast_sd:.4f}); "
      f"revision {nd_bra.revision:+.4f}, decomposition error {nd_bra.decomposition_error:.1e}")
print("=" * 74)
print(f"Replay as of {v1:%Y-%m-%d}: nowcast {res_mex_v1.nowcast:.4f} (s.d. {res_mex_v1.forecast_sd:.4f}), "
      f"news decomposition: {res_mex_v1.news_decomposition}")
print(f"Same call on the panel cut at {v1:%Y-%m-%d}: nowcast {res_mex_cut.nowcast:.4f}")
print(f"Previous nowcast inside the news decomposition (v-1 data, parameters estimated on v): {nd_mex.forecast_old:.4f}")
print(f"Effect of re-estimating the parameters: {nd_mex.forecast_old - res_mex_v1.nowcast:+.4f}")

# Internal checks: the identity, the target, and the information set of the replay
assert res_mex.target_period == target_month and res_mex.previous_vintage == v1
assert nd_mex.decomposition_error < 1e-10 and nd_bra.decomposition_error < 1e-10
assert abs(nd_mex.forecast_new - res_mex.nowcast) < 1e-10, "an unpublished target: the nowcast is the model's value"
assert res_mex_v1.news_decomposition is None, "no vintage precedes v-1"
assert abs(res_mex_v1.nowcast - res_mex_cut.nowcast) < 1e-12, "a replay may use only data published by as_of"
assert res_mex.forecast_sd < res_mex_v1.forecast_sd, "in this example the extra month of data narrows the band"

# %%
# --- Experiment 3a: PIT tests on a forecaster that is calibrated by construction ---
# These 60 forecasts are NOT the DFM's. The means and standard deviations are drawn at random
# and each outcome is drawn from the forecaster's own density, so the null of the tests is true.
# The experiment shows what the tests report in that case; it says nothing about the nowcasts above.
T_eval = 60
mu_eval = rng.normal(loc=2.0, scale=0.5, size=T_eval)
sd_eval = rng.uniform(0.4, 0.8, size=T_eval)
y_eval = mu_eval + sd_eval * rng.normal(size=T_eval)

pit_res = pit_uniformity_test(realised=y_eval, mu=mu_eval, sigma=sd_eval)
print(pit_res.summary())
assert 0.0 <= pit_res.lr_pvalue <= 1.0 and 0.0 <= pit_res.ks_pvalue <= 1.0

# --- Experiment 3b: a model-based fan chart for Mexico's gdp index ---
# res_mex.fan_chart builds the fan from the fitted DFM's state space (section 3 of the math):
# the unpublished months left in the panel, then 3 months beyond it; the history is published data.
fc_mex = res_mex.fan_chart(horizon=3, levels=(0.3, 0.6, 0.9), palette=_nbstyle.S2["color"])
fc_mean = fc_mex.forecast_mean
fc_sd = (fc_mex.intervals[0.9][1] - fc_mean) / norm.ppf(0.95)
gdp_hist = fc_mex.history

fan_table = pd.DataFrame({"mean": fc_mean, "s.d.": fc_sd,
                          "90% low": fc_mex.intervals[0.9][0], "90% high": fc_mex.intervals[0.9][1]})
fan_table.index = pd.DatetimeIndex(fan_table.index).strftime("%Y-%m")
print(f"\nMexico gdp index, one-factor DFM (last published month {gdp_hist.index[-1]:%Y-%m}, value {gdp_hist.iloc[-1]:.4f}):")
print(fan_table.round(4).to_string())

# Independent check of the library against the formula in section 3, from the same fitted matrices
fit = res_mex.model_result
j = list(fit.columns).index("gdp")
lam = np.zeros(fit.A.shape[0])
lam[: fit.n_factors] = fit.loadings[j]
P = np.asarray(fit.smoother_out["P_smooth"])[-1]   # state variance in the last month of the panel
hand_sd = []
for h in range(4):
    if h > 0:
        P = fit.A @ P @ fit.A.T + fit.Q             # one month further ahead
    hand_sd.append(float(fit.stds[j] * np.sqrt(lam @ P @ lam + fit.H[j, j])))

assert fc_mean.index[0] == target_month, "the fan starts at the first unpublished gdp month"
assert abs(fc_mean.iloc[0] - res_mex.nowcast) < 1e-10 and abs(fc_sd.iloc[0] - res_mex.forecast_sd) < 1e-10
assert np.allclose(fc_sd.to_numpy(), hand_sd, atol=1e-10), "library fan = formula in section 3"
assert np.allclose(fc_mean.to_numpy()[1:], fit.predict(steps=3)["gdp"].to_numpy(), atol=1e-10)
assert np.all(np.diff(fc_sd.to_numpy()) > 0), "the fan must widen with the horizon"
assert gdp_hist.equals(loaded_panel.as_of(v2).xs("MEX")["gdp"].dropna().iloc[-12:]), "history = published data"

# %%
# --- Dashboard: factors, news attribution, fan chart and PIT histogram ---
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.plot(res_mex.factors.index, res_mex.factors.iloc[:, 0], **_nbstyle.S1, label="Mexico, factor 1")
ax.plot(res_bra.factors.index, res_bra.factors.iloc[:, 0], **_nbstyle.S2, label="Brazil, factor 1")
ax.set_title("Estimated DFM factors, simulated panels (2022–2024)")
ax.set_xlabel("Reference month")
ax.set_ylabel("Factor (standardised units)")
ax.tick_params(axis="x", labelrotation=30)
ax.legend(loc="best", fontsize=8)

# News attribution: the library's waterfall, drawn relative to the previous nowcast
nd_mex.plot(ax=axes[0, 1], title=f"Mexico gdp nowcast for {target_month:%Y-%m}: update from v-1 to v")

# Fan chart: published gdp, then the DFM's own predictive bands (same as fc_mex above)
res_mex.plot_fan_chart(ax=axes[1, 0], horizon=3, palette=_nbstyle.S2["color"],
                       title="Mexico gdp index: one-factor DFM fan (simulated data)")
axes[1, 0].set_xlabel("Reference month")
axes[1, 0].tick_params(axis="x", labelrotation=30)

pit_res.plot(ax=axes[1, 1], title="PIT histogram: a forecaster calibrated by construction (not the DFM)")

fig.suptitle("Real-Time Nowcasting Dashboard (simulated Mexico and Brazil panels)")
fig.tight_layout()

# %% [markdown]
# ## Read the output
#
# **Read the output.** All data are simulated, so the numbers illustrate the method, not Mexico or Brazil.
#
# 1. **The ragged edge.** On 2025-01-15 the simulated `gdp` index is published up to 2024-10, `activity` and `ip` up to 2024-11, and `cpi` and the policy rate up to 2024-12. A month later each lagged series gains one month, and `activity` for 2023-09 is revised. The target, `gdp` for 2024-12, is unpublished in both vintages, so the nowcast is the model's estimate and not a published figure.
# 2. **The nowcast and its update.** On 2025-02-15 the one-factor DFM nowcasts the index at $100.6233$ with a standard deviation of $0.1108$ (90% interval $[100.4411, 100.8056]$). With the parameters held at their current estimates, the nowcast from the earlier vintage was $100.7341$, so it moved by $-0.1108$. The December `activity` release explains almost all of the move: it came in $0.2693$ below the model's expectation and, with a weight of $0.3633$, lowered the nowcast by $0.0978$. The `ip` release took off another $0.0111$, and the November `gdp` release only $0.0019$ (weight $0.0130$), because by then `activity` and `ip` already told the model about the last months. The $0.45$ revision to `activity` for 2023-09 gets a weight that rounds to $0.0000$: the fifteen months after it are observed, so it carries almost no information about December. The decomposition error of $1.05 \times 10^{-15}$ confirms the accounting identity, an internal check. The waterfall in the figure (`NewsDecompositionResult.plot`) draws the same impacts, summed by series, as changes from the previous nowcast.
# 3. **Replaying the past.** Asked `as_of` 2025-01-15, `realtime_nowcast` returns $100.7200$ with a standard deviation of $0.1471$ and no news decomposition, since no earlier vintage exists; the same call on a panel that simply ends on that date gives the same number, so the replay used only data published by then. That differs from the "previous nowcast" of the decomposition by $+0.0142$, the effect of re-estimating the parameters on the later vintage. The extra month of data narrowed the band from $0.1471$ to $0.1108$.
# 4. **Brazil.** Its nowcast is $99.3438$ (standard deviation $0.1757$) and moved by $-0.0173$ between the vintages, again with an exact decomposition ($6.0 \times 10^{-15}$). Brazil's factor is simulated independently of Mexico's, so the two factor paths in the first panel are unrelated by construction.
# 5. **The fan chart.** `res_mex.fan_chart` builds the fan from the DFM's own state-space matrices, and a separate computation of the formula in section 3 gives the same numbers: its standard deviation is $0.1108$ for the December nowcast and grows to $0.2689$, $0.3320$ and $0.3662$ for January to March 2025. The central path falls from $100.6233$ to $100.2257$ because the model's factor is a stationary autoregression, so its forecasts revert towards the sample mean, whereas the simulated factor is a random walk. The bands ignore parameter uncertainty and this misspecification, and the colours are styling only; nothing here is an official projection.
# 6. **The PIT tests.** The 60 forecasts of Experiment 3a are not the DFM's: they are correct by construction. The tests do not reject (Berkowitz $LR = 2.8021$, $p = 0.4232$; KS $= 0.1015$, $p = 0.5327$), as they should not in about 95% of such samples. The summary reports this as a failure to reject, not as proof of calibration, and it says nothing about the nowcasts above. The Your-turn cell measures how often the tests catch a miscalibrated forecaster.

# %% [markdown]
# ## Your turn
#
# Experiment 3a ran the PIT tests once, on a forecaster that is right by construction, so it could only show that the tests do not reject when they should not, and even that only once. How often do they catch a forecaster whose bands are too narrow or too wide? The cell below draws 500 samples of 60 forecasts, as in Experiment 3a, for a calibrated forecaster and for one that reports `sigma_scale` times the true standard deviation, and counts how often each test rejects at 5%.
#
# **Predict first:** against a wrong forecast variance, which test rejects more often, the Berkowitz LR test or KS? Think about what a too-narrow density does to the histogram of the PITs, and to their cumulative distribution. The checks confirm that both tests have about the right size, that the LR test detects your forecaster, and your prediction.

# %%
# Your turn: size and power of the PIT tests by Monte Carlo
sigma_scale = 0.6   # ← change this: forecast s.d. over the true s.d., 0.5 to 0.8 (too narrow) or 1.25 to 2.0 (too wide)
assert 0.5 <= sigma_scale <= 0.8 or 1.25 <= sigma_scale <= 2.0
R_mc, T_mc = 500, 60

def rejection_rates(scale, bias=0.0, phi=0.0, R=R_mc, T=T_mc):
    """Share of R samples in which the Berkowitz LR and the KS tests reject at 5%.

    The outcome is y = mu + bias * sd + sd * e, where e has unit variance and is an AR(1) with
    coefficient phi. The forecaster reports N(mu, (scale * sd)^2). Each replication has its own seed.
    """
    lr = ks = 0
    for s in range(R):
        g = np.random.default_rng(s)
        mu = g.normal(2.0, 0.5, T)
        sd = g.uniform(0.4, 0.8, T)
        u = g.normal(size=T)
        e = u.copy()
        for t in range(1, T):
            e[t] = phi * e[t - 1] + np.sqrt(1.0 - phi**2) * u[t]
        y = mu + bias * sd + sd * e
        test = pit_uniformity_test(realised=y, mu=mu, sigma=scale * sd)
        lr += test.lr_pvalue < 0.05
        ks += test.ks_pvalue < 0.05
    return lr / R, ks / R

size_lr, size_ks = rejection_rates(1.0)
power_lr, power_ks = rejection_rates(sigma_scale)
mc_se = np.sqrt(0.05 * 0.95 / R_mc)
print(f"Calibrated forecaster (size): LR {size_lr:.3f}, KS {size_ks:.3f} (nominal 0.05, Monte Carlo s.e. {mc_se:.3f})")
print(f"Forecast s.d. x {sigma_scale} (power): LR {power_lr:.3f}, KS {power_ks:.3f}")

assert abs(size_lr - 0.05) < 4 * mc_se and abs(size_ks - 0.05) < 4 * mc_se, "a test's size is far from 5%"
assert power_lr > size_lr + 4 * mc_se, "the LR test should detect this forecaster"
assert power_lr > power_ks, "your prediction: the LR test is more powerful against a wrong variance"

# %% [markdown]
# **Prompts.**
# 1. *Basic.* Move `sigma_scale` from $0.8$ to $0.5$ and from $1.25$ to $2.0$. How fast does the power of each test rise? Compare $0.8$ with its reciprocal $1.25$: which error is easier to detect with 60 forecasts, and why?
# 2. *Intermediate.* Now keep the variance right and shift the outcome: compare `rejection_rates(1.0, bias=0.25)` with `rejection_rates(1.0, bias=0.5)`. Which test wins against a biased forecaster, and which Berkowitz parameter ($\mu$, $\sigma_\varepsilon$, $\rho$) does the bias move? Self-check: `assert min(rejection_rates(1.0, bias=0.5)) > 0.85`.
# 3. *Stretch.* Keep every PIT marginally uniform but make the forecast errors serially correlated: `lr, ks = rejection_rates(1.0, phi=0.5)`. Which test notices, and why does KS still reject more often than 5% although every $p_t$ is $\mathcal{U}(0,1)$? Self-check: `assert lr > 0.8 and lr - ks > 0.3`. To grade the DFM itself you would replay it with `as_of` over many vintages and apply the same tests to its PITs.
#
# ## How comprehensive is this?
#
# - `puremacro.fetch.realtime`: vintage panels and connectors for Banxico, INEGI, BCB and BCCh (`VintagePanel`, `pack_realtime_cartridge`, `load_realtime_cartridge`); the connectors download data and are not called here.
# - `puremacro.nowcast.realtime_nowcast`: the orchestrator used above, with `as_of` and `previous_vintage` for historical replays and `method="mfvar"` for a mixed-frequency VAR; its result's `fan_chart` and `plot_fan_chart` draw the DFM's predictive fan after the published data (they refuse an `mfvar` result, which has no predictive variances).
# - `puremacro.nowcast.dfm` and `puremacro.nowcast.news`: the dynamic factor model (`DynamicFactorModel`) and the news decomposition (`banbura_modugno_news`, `NewsDecompositionResult` and its waterfall `plot`).
# - `puremacro.nowcast.evaluation`: `fan_chart`, `pit_uniformity_test`, `crps_gaussian` and `log_score_gaussian` for density forecasts.
# - Notebook 58 builds real-time vintage panels for Latin America; notebook 48 nowcasts with a held-out target; notebook 38 tests revisions for news and noise.
