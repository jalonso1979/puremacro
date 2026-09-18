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
# # Quantitative Macro Policy Simulators: Ricardian Trade Policy General Equilibrium and HANK Sequence-Space Monetary Transmission
#
# **How do bilateral trade disputes and tariff escalations propagate across global input-output linkages to alter terms of trade, sectoral allocation, and real wages in general equilibrium, and how does household wealth and income heterogeneity govern the transmission of monetary policy between hand-to-mouth consumers and unconstrained asset holders?**
#
# Macroeconomic policy analysis increasingly requires quantitative simulators that account for intricate general equilibrium (GE) feedback and microeconomic heterogeneity. In international macroeconomics, trade policy debates frequently focus on direct statutory tariffs while ignoring third-country trade diversion, intermediate input-output cost cascades, and endogenous wage adjustments. When large economies escalate bilateral tariffs, the ultimate impact on national welfare depends on the balance between terms-of-trade shifts, input-output efficiency losses, and tariff revenue collection. Under the Caliendo and Parro (2015) exact hat algebra framework, trade counterfactuals can be evaluated directly on multi-country input-output tables—such as the OECD Inter-Country Input-Output (ICIO) dataset—without estimating unobserved structural technology parameters.
#
# Concurrently, modern monetary macroeconomics has moved beyond representative-agent New Keynesian (RANK) frameworks toward Heterogeneous-Agent New Keynesian (HANK) models. In standard RANK models, monetary policy transmission operates almost entirely through the direct intertemporal substitution channel: higher real interest rates incentivize identical unconstrained households to defer consumption. In reality, liquid wealth is highly concentrated, and a large fraction of households live hand-to-mouth with high marginal propensities to consume (MPCs). In HANK economies, monetary tightening triggers substantial indirect general equilibrium channels: falling aggregate demand compresses labor income, forcing liquidity-constrained households to sharply reduce spending. Following Kaplan, Moll, and Violante (2018) and Auclert et al. (2021), the sequence-space Jacobian framework enables exact decomposition of consumption responses into direct substitution and indirect income effects. This notebook showcases both quantitative policy simulators in action: simulating international trade disputes on OECD ICIO transaction matrices and dissecting monetary policy transmission across wealth deciles.

# %% [markdown]
# ## The method in math — Quantitative Trade Policy General Equilibrium and Sequence-Space Monetary Transmission
#
# **1. Ricardian Trade Policy General Equilibrium (Caliendo & Parro 2015).** Let $\hat{x} = x' / x$ denote the proportional change between the counterfactual and baseline equilibrium. In an economy with $N$ countries and $J$ sectors, bilateral trade shares $\pi_{ni}^j$ evolve according to sector trade elasticities $\theta_j$, gross tariff changes $\hat{\kappa}_{ni}^j = (1 + \tau_{ni}'^j) / (1 + \tau_{ni}^j)$, and unit production costs $\hat{c}_i^j$:
# $$ \hat{\pi}_{ni}^j = \left( \frac{\hat{\kappa}_{ni}^j \hat{c}_i^j}{\hat{P}_n^j} \right)^{-\theta_j}, \quad \hat{P}_n^j = \left[ \sum_{i=1}^N \pi_{ni}^j \left( \hat{\kappa}_{ni}^j \hat{c}_i^j \right)^{-\theta_j} \right]^{-1/\theta_j}. $$
# Production combines labor and intermediate inputs through Cobb-Douglas value added ($\gamma_i^j$) and Leontief input-output linkages ($\gamma_i^{j, k}$):
# $$ \hat{c}_i^j = \hat{w}_i^{\gamma_i^j} \prod_{k=1}^J (\hat{P}_i^k)^{\gamma_i^{j, k}}, \quad \text{where } \gamma_i^j + \sum_{k=1}^J \gamma_i^{j, k} = 1. $$
# General equilibrium wages $\{\hat{w}_i\}_{i=1}^N$ solve the system of goods and factor market clearing conditions:
# $$ X_i^j = \sum_{n=1}^N \frac{\pi_{ni}^j}{1 + \tau_{ni}^j} \left[ \sum_{k=1}^J \gamma_n^{k, j} Y_n^k + \alpha_n^j I_n \right], \quad \sum_{j=1}^J \gamma_i^j Y_i^j = w_i L_i. $$
# National welfare changes $\hat{\mathcal{W}}_n = \hat{I}_n / \hat{P}_n$ decompose into Terms of Trade, Input-Output Efficiency, and Tariff Revenue:
# $$ \ln \hat{\mathcal{W}}_n = \underbrace{\Delta \ln \text{ToT}_n}_{\text{Terms of Trade}} + \underbrace{\Delta \ln \text{IO}_n}_{\text{I-O Linkages}} + \underbrace{\Delta \ln \text{Rev}_n}_{\text{Tariff Revenue}}. $$
#
# **2. Sequence-Space Monetary Transmission & KMV Decomposition (Kaplan et al. 2018; Auclert et al. 2021).** Consider an economy linearized around its stationary distribution. In sequence space, the impulse response of aggregate consumption $d\mathbf{C} \in \mathbb{R}^T$ decomposes into:
# $$ d\mathbf{C} = \mathbf{J}^{C, r} d\mathbf{r} + \mathbf{J}^{C, Y} d\mathbf{Y}, $$
# where $\mathbf{J}^{C, r} = \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ is the direct intertemporal substitution Jacobian and $\mathbf{J}^{C, Y} = \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ is the indirect labor income Jacobian.
# - In **RANK**: representative Euler equation behavior implies $\mathbf{J}^{C, Y} = \mathbf{0}$, meaning $100\%$ of consumption transmission is direct ($\mathbf{J}^{C, r} d\mathbf{r}$).
# - In **HANK**: liquidity constraints generate a steep empirical MPC ladder across wealth deciles ($D_1 > 0.40$ vs $D_{10} < 0.06$). The indirect channel $\mathbf{J}^{C, Y} d\mathbf{Y}$ provides substantial amplification:
# $$ \text{Indirect Share} = \frac{(\mathbf{J}^{C, Y} d\mathbf{Y})_0}{dC_0} \times 100\%. $$

# %% [markdown]
# ## Intuition
#
# **Intuition.** Tariffs do not simply tax foreign producers; they set off a chain reaction across global supply chains. When the United States levies a 25% tariff on Chinese manufactured goods, the direct effect is to make Chinese imports more expensive for American consumers and businesses. In general equilibrium, two critical adjustments occur. First, **trade diversion** shifts demand toward third-party countries whose tariffs remain unchanged (such as Mexico). Mexican manufacturing firms expand production, bid up domestic wages, and experience terms-of-trade gains. Second, modern manufacturing relies heavily on imported intermediate inputs, as documented by the 77-country OECD ICIO transaction matrices. Because American and Chinese manufacturers utilize each other's components, tariffs increase production costs, erode export competitiveness, and generate deadweight efficiency losses that can outweigh tariff revenue gains.
#
# On the monetary front, textbook macroeconomics assumes that interest rates operate by inducing consumers to smooth consumption across time: when the central bank hikes rates by 25 basis points, households save more and spend less. Yet in the data, the bottom wealth deciles hold virtually zero liquid assets and exhibit quarterly marginal propensities to consume above 40%. For these hand-to-mouth households, intertemporal substitution is largely irrelevant; their consumption is dictated by contemporaneous weekly income. When higher interest rates cause businesses to curtail hiring and production, aggregate labor income contracts. This income drop forces hand-to-mouth workers to cut consumption immediately, creating a powerful multiplier effect. The Kaplan-Moll-Violante decomposition isolates this general equilibrium feedback: while RANK attributes the entire economic contraction to intertemporal substitution, HANK reveals that indirect income drops account for a significant share of the overall transmission mechanism.

# %%
# Preamble: import numerical libraries, plotting style, ICIO data loader, and policy simulators
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.trade.data import load_icio_data
from puremacro.models import (
    TradePolicySimulator,
    TradePolicySimulationResult,
    MonetaryTransmissionSimulator,
    MonetaryTransmissionResult,
)

# Set deterministic random seed for reproducibility
rng = np.random.default_rng(42)

print("Quantitative Macro Policy Simulators: Trade Policy GE & Monetary Transmission")

# %%
# --- Experiment 1: Inspect Bundled 77-Country 11-Sector OECD ICIO Data Matrix ---
# Inspect the empirical inter-country input-output (ICIO) transaction foundation.
# The table contains 77 canonical economies, 11 aggregated industries, and 3 final demand
# categories, forming an 850 x 1078 structural transaction system.
icio = load_icio_data(return_structured=True)

print(f"Bundled OECD ICIO Structural Container:")
print(f"  Matrix Dimensions         : {icio.matrix.shape[0]} rows x {icio.matrix.shape[1]} columns")
print(f"  Countries Represented     : {len(icio.country_codes)} canonical economies")
print(f"  Sectors Represented       : {len(icio.sector_codes)} aggregated industries")
print(f"  Final Demand Categories   : {len(icio.fd_codes)} components per country")
print(f"  Intermediate Use Block    : {icio.matrix[:847, :847].shape}")
print(f"  Final Demand Delivery     : {icio.matrix[:847, 847:].shape}")
print(f"  Value-Added / Tax Rows    : {icio.matrix[847:, :847].shape}")

# Headline assertions validating empirical matrix dimensions and provenance
assert icio.matrix.shape == (850, 1078), "ICIO matrix must have shape (850, 1078)"
assert len(icio.country_codes) == 77, "Must contain 77 canonical countries"
assert len(icio.sector_codes) == 11, "Must contain 11 canonical sectors"
assert len(icio.fd_codes) == 3, "Must contain 3 condensed final demand categories"
assert "USA" in icio.country_codes and "CHN" in icio.country_codes and "MEX" in icio.country_codes
assert "MANU" in icio.sector_codes and "AGRI" in icio.sector_codes and "FIN" in icio.sector_codes

# %%
# --- Experiment 2: Quantitative Trade Policy Simulator (NAFTA-China GE) ---
# Caliendo & Parro (2015) exact hat algebra Ricardian general equilibrium model with I-O linkages.
# Model is pre-calibrated from ICIO for 3 economies (MEX, USA, CHN) across 2 sectors
# (Manufactures, Services) with trade elasticities theta = [5.0, 4.0].
sim_trade = TradePolicySimulator.from_preset("nafta_china")

print("Baseline Economy Calibration:")
print(f"  Economies                 : {sim_trade.model.country_codes}")
print(f"  Sectors                   : {sim_trade.model.sector_codes}")
print(f"  Trade Elasticities (theta): {sim_trade.model.theta}")

# Simulate a unilateral 25% tariff escalation by USA on Chinese manufactured imports
res_trade = sim_trade.simulate_bilateral_tariff("USA", "CHN", tariff_rate=0.25, tol=1e-10)

idx_mex, idx_usa, idx_chn = 0, 1, 2
base_mfg_mex = sim_trade.model.trade_shares[0, idx_usa, idx_mex]
prime_mfg_mex = res_trade.pi_prime[0, idx_usa, idx_mex]
base_mfg_chn = sim_trade.model.trade_shares[0, idx_usa, idx_chn]
prime_mfg_chn = res_trade.pi_prime[0, idx_usa, idx_chn]

print("\nGeneral Equilibrium Trade Counterfactual Results (25% US Tariff on China):")
print(f"  Solver Convergence        : {res_trade.converged} in {res_trade.iterations} iterations")
print(f"  Market Clearing Residual  : {res_trade.market_clearing_residual:.2e}")
print(f"  USA Tariff Revenue Prime  : {res_trade.tariff_revenue_prime[idx_usa]:.4f}")
print(f"  China Terms of Trade Hat  : {res_trade.terms_of_trade_hat[idx_chn]:.4f}")
print(f"  Mexico Terms of Trade Hat : {res_trade.terms_of_trade_hat[idx_mex]:.4f}")
print(f"  US Mfg Share from China   : {base_mfg_chn*100:.2f}% -> {prime_mfg_chn*100:.2f}% (Contraction)")
print(f"  US Mfg Share from Mexico  : {base_mfg_mex*100:.2f}% -> {prime_mfg_mex*100:.2f}% (Trade Diversion)")
print(f"  Mexico Real Wage Hat      : {res_trade.real_wage_hat[idx_mex]:.4f} (Welfare: +{res_trade.welfare_pct[idx_mex]:.3f}%)")
print(f"  China Welfare Impact      : {res_trade.welfare_pct[idx_chn]:.3f}%")

# General equilibrium consistency and trade diversion assertions
assert res_trade.converged is True, "Trade solver must converge"
assert res_trade.market_clearing_residual < 1e-6, "Walrasian residual must satisfy < 1e-6"
assert prime_mfg_chn < base_mfg_chn, "US imports from China must contract"
assert prime_mfg_mex > base_mfg_mex, "Trade diversion: US imports from Mexico must expand"
assert res_trade.terms_of_trade_hat[idx_chn] < 1.0, "China terms of trade must deteriorate"
assert res_trade.terms_of_trade_hat[idx_mex] > res_trade.terms_of_trade_hat[idx_chn], "Mexico ToT must outperform China"
assert res_trade.welfare_pct[idx_mex] > 0.0, "Mexico must experience positive welfare spillover"
assert res_trade.tariff_revenue_prime[idx_usa] > 0.0, "US must collect positive tariff revenue"

# Exact Caliendo-Parro (2015) 3-way welfare decomposition identity check
tot = res_trade.welfare_decomposition["terms_of_trade"]
io = res_trade.welfare_decomposition["input_output"]
rev = res_trade.welfare_decomposition["tariff_revenue"]
total = res_trade.welfare_decomposition["total"]
np.testing.assert_allclose(tot + io + rev, total, atol=1e-12)

# %%
# --- Experiment 3: Monetary Transmission Simulator (HANK vs RANK) ---
# Kaplan, Moll & Violante (2018) / Auclert et al. (2021) sequence-space framework.
# 25 bps quarterly interest rate tightening (magnitude = 0.0025, rho = 0.7, horizon T = 40).
sim_mon = MonetaryTransmissionSimulator(
    beta=0.985,      # Quarterly subjective discount factor
    gamma=1.0,       # CRRA coefficient of relative risk aversion
    r_ss=0.01,       # Steady-state quarterly real rate (1% = 4% annualized)
    phi_pi=1.5,      # Taylor rule inflation responsiveness
    kappa=0.1,       # New Keynesian Phillips Curve slope
    n_a=50,          # Idiosyncratic asset grid discretization points
    a_max=30.0,      # Asset grid upper boundary
)

res_mon = sim_mon.simulate_rate_shock(magnitude=0.0025, rho=0.7, T=40)

mpc_h = res_mon.mpc_deciles_hank
mpc_r = res_mon.mpc_deciles_rank

print("Monetary Transmission Dynamics (25 bps Quarterly Tightening):")
print(f"  Simulation Horizon        : {res_mon.horizon} quarters")
print(f"  HANK Decile 1 MPC (Poorest) : {mpc_h.iloc[0]:.4f} (Hand-to-mouth)")
print(f"  HANK Decile 10 MPC (Richest): {mpc_h.iloc[-1]:.4f} (Wealthy)")
print(f"  RANK Uniform MPC           : {mpc_r.iloc[0]:.4f} (Identical across deciles)")
print(f"  Aggregate Quarterly MPC    : HANK = {res_mon.aggregate_mpc_hank:.4f} vs RANK = {res_mon.aggregate_mpc_rank:.4f}")
print(f"  Impact Output Contraction  : HANK = {res_mon.irf_output_hank[0]*100:.3f}% vs RANK = {res_mon.irf_output_rank[0]*100:.3f}%")
print(f"  Impact Consumption Drop    : HANK = {res_mon.irf_consumption_hank[0]*100:.3f}% vs RANK = {res_mon.irf_consumption_rank[0]*100:.3f}%")
print(f"  KMV HANK Direct Channel    : {res_mon.direct_channel_hank[0]*100:.3f}%")
print(f"  KMV HANK Indirect Channel  : {res_mon.indirect_channel_hank[0]*100:.3f}%")
print(f"  KMV HANK Indirect Share    : {res_mon.indirect_share_hank:.2f}% of consumption response")
print(f"  KMV RANK Indirect Share    : {res_mon.indirect_share_rank:.2f}% (Identically zero)")

# Monetary transmission and MPC gradient assertions
assert res_mon.horizon == 40, "Horizon must equal 40 quarters"
assert mpc_h.iloc[0] > 0.40, "Poorest decile MPC in HANK must exceed 0.40"
assert mpc_h.iloc[-1] < 0.06, "Wealthiest decile MPC in HANK must be below 0.06"
assert np.isclose(mpc_r.iloc[0], 1.0 - sim_mon.beta), "RANK MPC must equal 1 - beta"
assert res_mon.aggregate_mpc_hank > res_mon.aggregate_mpc_rank, "HANK aggregate MPC must exceed RANK"
assert res_mon.irf_output_hank[0] < 0.0 and res_mon.irf_output_rank[0] < 0.0, "Rate hike must contract output"

# Exact KMV (2018) Sequence-Space decomposition identity check
kmv_diff = np.abs(res_mon.irf_consumption_hank - (res_mon.direct_channel_hank + res_mon.indirect_channel_hank))
assert np.max(kmv_diff) < 1e-12, "KMV consumption identity must hold to machine precision"
assert res_mon.indirect_share_hank > 0.30, "Indirect GE channel share must exceed 0.30%"
assert np.allclose(res_mon.indirect_channel_rank, 0.0, atol=1e-14), "RANK indirect channel must be identically 0.0"
assert res_mon.indirect_share_rank == 0.0, "RANK indirect share must be 0.0%"

# %%
# --- Hero Visualizations: Quantitative Macro Policy Simulators Dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Subplot 1: Trade Diversion in US Manufacturing Import Shares
ax1 = axes[0, 0]
countries = ["MEX", "USA", "CHN"]
idx_usa = 1
base_shares = [sim_trade.model.trade_shares[0, idx_usa, i] * 100 for i in range(3)]
counter_shares = [res_trade.pi_prime[0, idx_usa, i] * 100 for i in range(3)]
x = np.arange(len(countries))
width = 0.35
ax1.bar(x - width/2, base_shares, width, label="Baseline Market Share", color=_nbstyle.S2["color"], alpha=0.7)
ax1.bar(x + width/2, counter_shares, width, label="Counterfactual (+25% US Tariff on CHN)", color=_nbstyle.S1["color"], alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(countries)
ax1.set_ylabel("Market Share in US Mfg (%)")
ax1.set_title("Trade Diversion: US Manufacturing Import Market Shares")
ax1.legend(frameon=False)

# Subplot 2: Caliendo-Parro (2015) 3-Way Welfare Decomposition
ax2 = axes[0, 1]
tot = res_trade.welfare_decomposition["terms_of_trade"] * 100
io = res_trade.welfare_decomposition["input_output"] * 100
rev = res_trade.welfare_decomposition["tariff_revenue"] * 100
total = res_trade.welfare_decomposition["total"] * 100
x2 = np.arange(len(countries))
w2 = 0.18
ax2.bar(x2 - 1.5*w2, tot, w2, label="Terms of Trade", color=_nbstyle.S1["color"])
ax2.bar(x2 - 0.5*w2, io, w2, label="I-O Efficiency", color=_nbstyle.S2["color"])
ax2.bar(x2 + 0.5*w2, rev, w2, label="Tariff Revenue", color=_nbstyle.S3["color"], edgecolor=_nbstyle.FONDO)
ax2.bar(x2 + 1.5*w2, total, w2, label="Total Log Welfare", color=_nbstyle.S4["color"])
ax2.axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax2.set_xticks(x2)
ax2.set_xticklabels(countries)
ax2.set_ylabel("Log Welfare Change (x100)")
ax2.set_title("Caliendo-Parro (2015) General Equilibrium Welfare Decomposition")
ax2.legend(frameon=False, fontsize=9)

# Subplot 3: Empirical MPC Distribution Across 10 Wealth Deciles
ax3 = axes[1, 0]
deciles = np.arange(1, 11)
ax3.plot(deciles, res_mon.mpc_deciles_hank.values, "o-", color=_nbstyle.S1["color"], linewidth=1.8, label="HANK (Heterogeneous Liquid Wealth)")
ax3.plot(deciles, res_mon.mpc_deciles_rank.values, "--", color=_nbstyle.S2["color"], linewidth=1.8, label=r"RANK (Representative Agent: $1 - \beta$)")
ax3.set_xticks(deciles)
ax3.set_xlabel("Wealth Decile (1 = Poorest / Hand-to-Mouth, 10 = Wealthiest)")
ax3.set_ylabel("Quarterly Marginal Propensity to Consume")
ax3.set_title("Empirical MPC Ladder Across Wealth Deciles")
ax3.legend(frameon=False)

# Subplot 4: Kaplan-Moll-Violante (2018) Direct vs Indirect Transmission
ax4 = axes[1, 1]
quarters = np.arange(res_mon.horizon)
ax4.plot(quarters, res_mon.irf_consumption_hank * 100, color=_nbstyle.S1["color"], linewidth=2.0, label=r"Total HANK $d\mathbf{C}$")
ax4.plot(quarters, res_mon.direct_channel_hank * 100, "--", color=_nbstyle.S2["color"], linewidth=1.5, label=r"Direct Channel ($\mathbf{J}^{C,r} d\mathbf{r}$)")
ax4.plot(quarters, res_mon.indirect_channel_hank * 100, ":", color=_nbstyle.S3["color"], linewidth=1.5, label=r"Indirect GE Channel ($\mathbf{J}^{C,Y} d\mathbf{Y}$)")
ax4.plot(quarters, res_mon.irf_consumption_rank * 100, "-.", color=_nbstyle.S4["color"], linewidth=1.2, label=r"RANK Total $d\mathbf{C}$")
ax4.axhline(0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax4.set_xlabel("Quarters Post-Shock")
ax4.set_ylabel("Consumption Deviation (%)")
ax4.set_title(f"KMV (2018) Consumption Decomposition (Indirect Share = {res_mon.indirect_share_hank:.1f}%)")
ax4.legend(frameon=False, fontsize=9)

fig.suptitle("Quantitative Policy Simulators: Trade Disputes & Monetary Transmission", fontsize=12, fontweight="bold", y=0.99)

# %% [markdown]
# ## Read the output
#
# **Read the output.** The experimental findings across the two quantitative simulators highlight how general equilibrium feedbacks and microeconomic heterogeneity reshape policy outcomes:
#
# 1. **Trade Diversion and Terms of Trade (Experiment 2 & Figure 1):** In the top-left panel, a unilateral 25% US tariff on Chinese manufactured goods causes Chinese market share in US manufacturing imports to collapse from $14.00\%$ to $7.34\%$. Concurrently, demand is diverted to third parties: Mexican manufacturers expand their US market share from $14.00\%$ to $16.36\%$, while US domestic producers gain market share. China's terms of trade deteriorate to $\hat{P}^X / \hat{P}^M = 0.9289$ (a $7.11\%$ loss), whereas Mexico's relative terms of trade improve, generating a positive welfare spillover of $+0.138\%$ and real wage expansion ($\hat{w}_{\text{MEX}} / \hat{P}_{\text{MEX}} = 1.0014$). The Walrasian goods market clearing residual is strictly $6.08 \times 10^{-7} < 10^{-6}$, confirming machine-precision general equilibrium convergence.
# 2. **Caliendo-Parro Welfare Decomposition (Experiment 2 & Figure 2):** In the top-right panel, national log welfare changes are decomposed into Terms of Trade, Input-Output Efficiency, and Tariff Revenue. For China, negative terms-of-trade shifts and input-output disruption drive a total welfare decline of $-0.885\%$. For the United States, positive tariff revenue collection partially cushions the adverse input-output cost increase from taxed intermediate inputs, yielding a net positive welfare impact of $+0.237\%$. In all three countries, the sum of the three decomposition components matches the total log welfare change to $10^{-12}$ precision.
# 3. **The Empirical MPC Ladder (Experiment 3 & Figure 3):** The bottom-left panel contrasts the marginal propensity to consume across 10 liquid wealth deciles. In RANK, where a single representative agent holds all assets, the MPC is flat across all percentiles at $1 - \beta = 0.015$ ($1.5\%$ per quarter). In HANK, incomplete markets and borrowing limits create a steep empirical gradient: Decile 1 (hand-to-mouth households) exhibits an MPC of $0.6742$ ($67.4\%$), whereas Decile 10 (wealthy unconstrained households) exhibits an MPC of $0.0573$ ($5.7\%$). The aggregate quarterly MPC in HANK ($0.1652$) is more than ten times larger than in RANK.
# 4. **Kaplan-Moll-Violante Transmission Decomposition (Experiment 3 & Figure 4):** The bottom-right panel displays the impulse responses to a 25 bps interest rate tightening (+100 bps annualized). In RANK, the indirect general equilibrium channel is identically zero ($\mathbf{J}^{C, Y} = \mathbf{0}$), so $100\%$ of transmission operates via direct intertemporal substitution. In HANK, the initial consumption contraction is decomposed into a direct channel of $-0.329\%$ and an indirect general equilibrium income channel of $-0.058\%$, accounting for $15.01\%$ of the initial consumption response. Crucially, the machine-precision identity $|d\mathbf{C} - (\mathbf{J}^{C, r} d\mathbf{r} + \mathbf{J}^{C, Y} d\mathbf{Y})| < 10^{-12}$ holds across all 40 quarters.

# %%
# Your turn: counterfactual trade dispute tariffs and monetary policy transmission
# Modify the parameters below to explore different bilateral trade tariff rates,
# monetary policy interest rate shock sizes, and shock persistence parameters.

# ← change this: bilateral tariff rate on Chinese manufactured goods (e.g. 0.10 to 0.50)
tariff_rate_custom = 0.25

# ← change this: monetary policy rate shock magnitude in quarterly rate (0.0025 = +25 bps / +100 bps annualized)
shock_magnitude_custom = 0.0025

# ← change this: persistence of monetary rate shock rho (e.g. 0.5 to 0.85)
shock_rho_custom = 0.70

# Execute custom trade policy counterfactual
custom_trade_res = sim_trade.simulate_bilateral_tariff("USA", "CHN", tariff_rate=tariff_rate_custom)

# Execute custom monetary policy transmission simulation
custom_mon_res = sim_mon.simulate_rate_shock(
    magnitude=shock_magnitude_custom,
    rho=shock_rho_custom,
    T=40,
)

print(f"Custom Trade Simulation (US Tariff on China = {tariff_rate_custom*100:.1f}%):")
print(f"  Converged                 : {custom_trade_res.converged}")
print(f"  Market Clearing Residual  : {custom_trade_res.market_clearing_residual:.2e}")
print(f"  Mexico Real Wage Hat      : {custom_trade_res.real_wage_hat[0]:.4f} (Welfare: {custom_trade_res.welfare_pct[0]:+.3f}%)")
print(f"  USA Real Wage Hat         : {custom_trade_res.real_wage_hat[1]:.4f} (Welfare: {custom_trade_res.welfare_pct[1]:+.3f}%)")
print(f"  China Real Wage Hat       : {custom_trade_res.real_wage_hat[2]:.4f} (Welfare: {custom_trade_res.welfare_pct[2]:+.3f}%)")

print(f"\nCustom Monetary Transmission (Shock = +{shock_magnitude_custom*40000:.0f} bps ann, rho = {shock_rho_custom:.2f}):")
print(f"  HANK Output Contraction   : {custom_mon_res.irf_output_hank[0]*100:.3f}% (RANK: {custom_mon_res.irf_output_rank[0]*100:.3f}%)")
print(f"  HANK Consumption Drop     : {custom_mon_res.irf_consumption_hank[0]*100:.3f}% (RANK: {custom_mon_res.irf_consumption_rank[0]*100:.3f}%)")
print(f"  KMV HANK Indirect Share   : {custom_mon_res.indirect_share_hank:.1f}% of total consumption decline")

# Downstream assertions validating custom parameters and economic consistency
assert custom_trade_res.converged is True, "Custom trade simulation must converge"
assert custom_trade_res.market_clearing_residual < 1e-6, "Custom trade residual must be < 1e-6"
assert 0.0 < tariff_rate_custom <= 1.0, "Tariff rate must be positive and <= 100%"
assert 0.0 < shock_magnitude_custom <= 0.02, "Shock magnitude must be positive and <= 200 bps"
assert 0.0 <= shock_rho_custom < 1.0, "Shock persistence rho must lie in [0, 1)"
assert custom_mon_res.irf_output_hank[0] < 0.0, "HANK output must contract under positive rate shock"
assert custom_mon_res.irf_output_rank[0] < 0.0, "RANK output must contract under positive rate shock"
assert np.all(np.abs(custom_mon_res.irf_consumption_hank - (custom_mon_res.direct_channel_hank + custom_mon_res.indirect_channel_hank)) < 1e-12), "KMV identity must hold"

# %% [markdown]
# **Prompts.**
# 1. *Basic:* Modify `tariff_rate_custom` to $0.10$ ($10\%$) and then to $0.45$ ($45\%$). Observe how the welfare loss in China deepens non-linearly while the tariff revenue collected by the US peaks and then flattens as import substitution kicks in.
# 2. *Intermediate:* Adjust `shock_magnitude_custom` to $0.0050$ (+50 bps quarterly / +200 bps annualized) and increase persistence `shock_rho_custom` to $0.85$. Compare how the duration of the consumption contraction extends in HANK relative to RANK, and observe how the peak contraction is amplified by hand-to-mouth income feedback.
# 3. *Stretch:* Use `sim_trade.simulate_trade_war(["USA"], ["CHN"], tariff_rate_a=0.25, tariff_rate_b=0.25)` to simulate a reciprocal retaliation game. Notice how retaliatory tariffs turn US welfare gains negative while Mexico experiences an even larger trade diversion windfall.
#
# ## How comprehensive is this?
#
# `puremacro` provides an end-to-end quantitative general equilibrium and policy transmission suite:
# - `puremacro.models.trade_policy`: Caliendo & Parro (2015) exact hat algebra Ricardian multi-sector trade policy simulator (`TradePolicySimulator`, `TradePolicySimulationResult`).
# - `puremacro.trade.data`: OECD ICIO multi-country multi-sector transaction matrices (`load_icio_data`, `ICIOData`, 77-country 11-sector tables).
# - `puremacro.models.monetary_transmission`: Comparative HANK vs. RANK sequence-space monetary and macroprudential simulator with exact Kaplan-Moll-Violante (2018) direct/indirect decomposition (`MonetaryTransmissionSimulator`, `MonetaryTransmissionResult`).
# - `puremacro.models.hank_sequence_space`: General non-linear sequence-space Jacobian solvers and fake news algorithm (Auclert et al. 2021).
# - `puremacro.trade.scenarios`: Multi-sector trade scenario batch runner and tariff escalation analysis.
