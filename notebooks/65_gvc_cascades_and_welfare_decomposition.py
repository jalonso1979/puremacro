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
# # GVC cost propagation, provenance, and welfare validation limits
#
# **How far does a supplier's cost increase travel through a global value chain (GVC), and who bears a tariff on imported goods?** The second question also shows how to tell a solved trade counterfactual from an unsupported welfare claim: nominal GDP and tariff revenue are not welfare, and a welfare number is only as good as its closure and numeraire.
#
# All inputs are synthetic teaching data. The first half uses small generated tables laid out like the OECD Inter-Country Input-Output (ICIO), Eurostat FIGARO and EXIOBASE databases; they are not observations from those databases. The second half uses a hand-balanced two-country table.

# %% [markdown]
# ## The method in math
#
# A node is a country-sector pair. Let $Z_{ij}$ be node $j$'s purchases from node $i$ and $X_j$ node $j$'s gross output. The input coefficient $A_{ij}=Z_{ij}/X_j$ is the purchase from supplier $i$ per unit of buyer $j$'s output. With fixed coefficients, unit costs satisfy $p=A^Tp+v$, so a change $dv$ in primary cost per unit of output moves prices by
#
# $$dp=(I-A^T)^{-1}dv=dv+A^Tdv+(A^T)^2dv+\cdots$$
#
# The $k$-th term is the cost increase that has passed through $k$ layers of customers. The series converges when the spectral radius $\rho(A)<1$, which for a nonnegative $A$ is equivalent to the Hawkins–Simon productivity condition. This calculation holds factor prices and per-unit taxes fixed. (The library's production taxes are ad valorem, $p=D(A^Tp+v)$ with $D=\mathrm{diag}(1/(1-t))$, which gives $dp=(I-DA^T)^{-1}D\,dv$.) General equilibrium also moves factor prices, quantities, transfers and tax receipts.
#
# Welfare uses an expenditure function $e(P,U)$. The equivalent variation $EV=e(P_0,U_1)-e(P_0,U_0)$ values the utility change at baseline prices; the compensating variation $CV=e(P_1,U_1)-e(P_1,U_0)$ values it at counterfactual prices. Both are positive for gains. With one Leontief consumption basket, $U=C$ and $e(P,U)=PU$, so $EV=P_0(C_1-C_0)$ and $CV=P_1(C_1-C_0)$.

# %% [markdown]
# ## Intuition
#
# **Intuition.** A supplier's cost increase reaches its customers through their input purchases, then their customers' customers, and each round is smaller by roughly the factor $\rho(A)$. A tariff is a cost increase at a border. Who ends up paying depends on whether buyers can switch suppliers. With fixed-coefficient (Leontief) sourcing they cannot: the importing country keeps buying the taxed goods, its own costs rise, and the exporter's relative price can even go up. Equivalent variation turns the endpoint prices and quantities into a money measure only once the expenditure function, the fiscal closure, the foreign-saving closure and the numeraire are stated.

# %% [markdown]
# ## Worked code
#
# Generate the data explicitly and display their source before reporting results.

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.trade.data import load_exiobase, load_figaro, load_oecd_icio_granular
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade.regularize import compute_spectral_radius

# %% [markdown]
# ### Provenance
#
# Real-data loaders require a source file by default; `fallback_to_synthetic=True` is an explicit opt-in to generated tables. The three layouts below come from one generator. The provider name sets the country and sector labels, the final-use roster and the currency, not the coefficients, so at three countries and three sectors the three coefficient matrices are identical by construction; the cell asserts it. The table shows what each calibration records: FIGARO and EXIOBASE layouts are generated in million EUR and rescaled at an assumed exchange rate, the labor share of value added is a default assumption, and balancing changes the flows only at rounding level.

# %%
loaders = {"oecd": load_oecd_icio_granular, "figaro": load_figaro, "exiobase": load_exiobase}
# Synthetic provider layouts; each loader condenses the final uses before calibrating.
calibrations = {source: loader(custom_c=3, custom_s=3, seed=42, fallback_to_synthetic=True)
                for source, loader in loaders.items()}
provenance = pd.DataFrame([
    {"layout": name, "source": c.metadata["source"], "synthetic": c.metadata["is_synthetic"],
     "seed": c.metadata["seed"], "unit": c.metadata["unit"],
     "generated_in": c.metadata.get("original_unit") or c.metadata["unit"],
     "usd_per_eur": c.metadata.get("exchange_rate_usd_per_eur") or np.nan,
     "labor_share_of_va": round(c.metadata["labor_share_assumption"], 4),
     "balancing_below_1e-9": max(v["max_abs_change"] for v in c.metadata["adjustments"].values()) < 1e-9}
    for name, c in calibrations.items()
])
print(provenance.to_string(index=False))
assert provenance["synthetic"].all() and provenance["balancing_below_1e-9"].all()
A_by_layout = {name: c.a.reshape(c.n_countries * c.n_sectors, -1, order="F") for name, c in calibrations.items()}
layout_gap = max(np.abs(A - A_by_layout["oecd"]).max() for A in A_by_layout.values())
assert layout_gap < 1e-12  # the same network under three label sets
print("Coefficient matrices identical across layouts (max |dA| < 1e-12):", layout_gap < 1e-12)
min_share = min(c.theta.min() for c in calibrations.values())
print(f"Smallest final-use expenditure share in the provider layouts: {min_share:.4f}")

# %% [markdown]
# ### Fixed-price cost propagation
#
# Since the three layouts share one network, the cascade is computed once, on the OECD-layout labels. A 1% unit-cost increase hits the first node. The spectral upper bound certifies $\rho(A)<1$; rebuilding the cascade round by round (the series above) and comparing it with the direct linear solve is an internal consistency check of two numerical routes, not an independent oracle.

# %%
cal = calibrations["oecd"]
n = cal.n_countries * cal.n_sectors
A = cal.a.reshape(n, n, order="F")  # A[i, j]: input from node i per unit of node j's output
nodes = [f"{cal.country_codes[k // cal.n_sectors]}_{cal.sector_codes[k % cal.n_sectors]}" for k in range(n)]
rho, lower, upper = compute_spectral_radius(A)
assert upper < 1.0  # certifies rho(A) < 1, so the cascade converges
shock_node, shock_size = 0, 0.01
dv = np.zeros(n); dv[shock_node] = shock_size
dp = np.linalg.solve(np.eye(n) - A.T, dv)  # factor prices and per-unit taxes held fixed
partial, layer, gaps = np.zeros(n), dv.copy(), []
for _ in range(40):  # add one layer of customers per round
    partial, layer = partial + layer, A.T @ layer
    gaps.append(np.abs(partial - dp).max())
decay = gaps[20] / gaps[19]
assert gaps[-1] < 1e-14 and abs(decay - rho) < 0.01  # internal check: the series reproduces the solve
home = np.arange(n) // cal.n_sectors == shock_node // cal.n_sectors
indirect = dp - dv  # everything beyond the direct shock
cascade = pd.DataFrame({"home_country": home, "pass_through": dp / shock_size}, index=pd.Index(nodes, name="node"))
print(cascade.round(4).to_string())
print(f"Spectral radius bounds [{lower:.4f}, {upper:.4f}]; each round shrinks the gap by {decay:.4f}")
print(f"Own-node pass-through {dp[shock_node] / shock_size:.4f}; total over all nodes {dp.sum() / shock_size:.4f}")
split = {"own-node feedback": indirect[shock_node], "other home sectors": indirect[home].sum() - indirect[shock_node],
         "foreign nodes": indirect[~home].sum()}
print("Split of the indirect increase:", {k: f"{v / indirect.sum():.1%}" for k, v in split.items()})

# %% [markdown]
# ### A solved tariff experiment
#
# The provider layouts cannot carry the welfare experiment. After the loaders' foreign-balance closure, one country's investment share is negative (the smallest share printed above), and the consistent accounting behind Hicksian EV rejects negative expenditure shares. The experiment therefore uses a separate hand-balanced table: two countries, A and B, one good each, and final uses split into consumption (C) and investment (I), in one currency unit. Value added is split 2/3 labor and 1/3 capital, the loaders' default. Foreign saving is nonzero at baseline: A runs a trade surplus of 14 and B a deficit of 14 (printed below).
#
# **Experiment.** Country A levies a 10% ad valorem tariff on all imports from B, on intermediate inputs and on final goods; B levies none. The duty revenue is rebated lump-sum to A's households. Sourcing is Leontief (`sigma=0`, the solver default): no buyer can change the origin mix of its basket. With fixed factor endowments and one good per country, output quantities do not change.
#
# **Units, numeraire and closures.** Both states are solved with `accounting="consistent"`, the library's audited accounting mode: duties and trade are valued at producer prices, duty and tax revenue is rebated lump-sum, and baseline foreign balances are held fixed. A's producer price is the numeraire, equal to one in both states. Nominal GDP (factor income plus production taxes, final-use taxes and duties) and tariff revenue are therefore in units of A's good, and their changes depend on which country is listed first. Foreign saving is held fixed as a share of world factor income (`foreign_saving_units="world_income"`). The alternative closure, which fixes it in numeraire units, makes real outcomes depend on the country order when foreign saving is nonzero; the welfare cell shows this.

# %%
# Hand-balanced teaching table: one good per country, values in one currency unit.
Z = np.array([[10., 12.], [8., 14.]])                      # intermediate sales: rows seller A, B; columns buyer A, B
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])  # final sales to A-C, A-I, B-C, B-I
production_tax = np.array([4., 6.])                        # net production taxes paid by A, B
final_tax = np.array([2., 1., 3., 2.])                     # taxes on A-C, A-I, B-C, B-I purchases


def teaching_table(order=("A", "B")):
    """Rows: goods, net taxes, labor, capital; countries listed in `order` (the first is the numeraire)."""
    k = ["AB".index(code) for code in order]
    cols = [2 * i + j for i in k for j in (0, 1)]
    Zo, Fo, tax = Z[np.ix_(k, k)], F[np.ix_(k, cols)], production_tax[k]
    va = Zo.sum(1) + Fo.sum(1) - Zo.sum(0) - tax  # value added net of production taxes
    return np.vstack([np.hstack([Zo, Fo]), np.r_[tax, final_tax[cols]],
                      np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])


def tariff_experiment(order=("A", "B"), closure="world_income", sigma=0.0, rate=0.10):
    """Solve free trade and A's tariff on all imports from B; return calibration, both states, welfare."""
    calib = calibrate_trade_model(teaching_table(order), ns=1, nc=2, nfd=2,
                                  country_codes=list(order), sector_codes=["GOOD"])
    rates = np.zeros(2); rates[calib.country_codes.index("A")] = rate  # ad valorem RATES by importing country
    options = dict(accounting="consistent", tol=1e-10, sigma=sigma, foreign_saving_units=closure)
    base = solve_trade_equilibrium(calib, **options)
    counterfactual = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, base_result=base, **options)
    assert base.converged and counterfactual.converged, f"no equilibrium at sigma={sigma}"
    welfare = {code: compute_hicksian_welfare(calib, counterfactual, base_result=base, target_country=code)
               for code in ("A", "B")}
    return calib, base, counterfactual, welfare


print(pd.DataFrame(teaching_table(), index=["A", "B", "net taxes", "labor", "capital"],
                   columns=["A", "B", "A-C", "A-I", "B-C", "B-I"]).round(2).to_string())
calib, base, counterfactual, welfare = tariff_experiment()
print("Baseline foreign saving (A, B):", calib.invforT.ravel())
dq = (counterfactual.c_fd - base.c_fd)[0]  # rows: C, I; columns: A, B (basket quantities)
outcomes = pd.DataFrame({
    "nominal_GDP_%": 100 * (counterfactual.gdp / base.gdp - 1),
    "tariff_revenue": counterfactual.tariffs,
    "C_price_%": 100 * (counterfactual.Pfd_final[0, 0] / base.Pfd_final[0, 0] - 1),
    "I_price_%": 100 * (counterfactual.Pfd_final[0, 1] / base.Pfd_final[0, 1] - 1),
    "consumption_%": 100 * dq[0] / base.c_fd[0, 0],
    "investment_%": 100 * dq[1] / base.c_fd[0, 1],
}, index=pd.Index(calib.country_codes, name="country"))
print(outcomes.round(4).to_string())
p, w = counterfactual.p_sol.ravel(), counterfactual.w_sol.ravel()
p_ratio = p[1] / p[0]
print(f"B's producer price relative to A's: {p_ratio:.4f}; B's wage relative to A's: {w[1] / w[0]:.4f}")
print("Equilibrium residual below 1e-10:", counterfactual.max_residual < 1e-10)
# Output is fixed and every final-use basket is one unit of goods, so the world C gain is the world I loss.
assert np.allclose(counterfactual.y_sol, base.y_sol, atol=1e-9)
assert abs(dq[0].sum() + dq[1].sum()) < 1e-9 and dq[1, 0] < 0
print(f"World consumption {dq[0].sum():+.4f}, world investment {dq[1].sum():+.4f} (basket units)")

# %% [markdown]
# ### Hicksian consumption welfare
#
# Category 0 (consumption) enters utility; investment is excluded. EV is valued at baseline prices, where every producer price is one, so it is in the table's currency unit. `EV_%` is EV over baseline consumption spending and `CV_%` is CV over counterfactual consumption spending; both are ratios of utilities and carry no price level. The attribution splits EV into purchaser prices (what buyers pay, including taxes and duties), factor income and fiscal transfers (which count the duty rebate exactly once). It is an endpoint accounting split, not a causal terms-of-trade/efficiency decomposition, and it is valued partly at counterfactual prices, which carry the numeraire. The second table re-solves the experiment with B listed first, under both foreign-saving closures, to show which numbers survive relabelling.

# %%
welfare_table = pd.DataFrame([
    {"country": code, "EV": r.ev, "EV_%": r.ev_pct_consumption, "CV_%": 100 * r.cv / r.consumption_counterfactual,
     "prices": r.price_effect, "factor_income": r.factor_income_effect, "fiscal_transfers": r.fiscal_transfer_effect}
    for code, r in welfare.items()
]).set_index("country")
print(welfare_table.round(4).to_string())
runs = {(closure, order): tariff_experiment(order, closure=closure)
        for closure in ("numeraire", "world_income") for order in (("A", "B"), ("B", "A"))}
relabel = pd.DataFrame([
    {"closure": closure, "listed_first": order[0], "EV_A_%": w["A"].ev_pct_consumption,
     "EV_B_%": w["B"].ev_pct_consumption, "GDP_A_%": 100 * (cf.gdp[order.index("A")] / b.gdp[order.index("A")] - 1),
     "A_prices": w["A"].price_effect, "A_factor_income": w["A"].factor_income_effect,
     "A_fiscal": w["A"].fiscal_transfer_effect}
    for (closure, order), (_, b, cf, w) in runs.items()
])
print(relabel.round(4).to_string(index=False))
AB, BA = runs[("world_income", ("A", "B"))][3], runs[("world_income", ("B", "A"))][3]
for code in "AB":  # world-income closure: EV does not depend on which country is the numeraire
    assert abs(AB[code].ev - BA[code].ev) < 1e-8
assert abs(AB["A"].price_effect - BA["A"].price_effect) > 0.1  # the attribution does
assert runs[("numeraire", ("A", "B"))][3]["A"].ev * runs[("numeraire", ("B", "A"))][3]["A"].ev < 0  # sign flips
# Headline results of this Leontief benchmark (elastic sourcing reverses them; see the exercise).
assert p_ratio > 1 and welfare["B"].ev > welfare["A"].ev  # A's terms of trade worsen; B gains more
assert outcomes.loc["A", "nominal_GDP_%"] > 3 and abs(welfare["A"].ev_pct_consumption) < 0.1  # GDP is not welfare

# %% [markdown]
# ### Display the numerical results

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(13, 5))
shade = [_nbstyle.TINTA if k == shock_node else _nbstyle.BARRA_COLORES[1] if home[k] else _nbstyle.BARRA_COLORES[2]
         for k in range(n)]
hatch = ["" if k == shock_node else "//" if home[k] else ".." for k in range(n)]
bars = ax1.bar(nodes, indirect / shock_size, color=shade, edgecolor=_nbstyle.FONDO)
for bar, pattern in zip(bars, hatch):
    bar.set_hatch(pattern)
ax1.legend([bars[shock_node], bars[1], bars[-1]], ["shocked node (feedback)", "other home sectors", "foreign nodes"],
           loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.tick_params(axis="x", rotation=60)
ax1.set(title=f"Cost cascade of a 1% shock at {nodes[shock_node]} (ρ = {rho:.3f})",
        ylabel="Indirect cost increase per unit of direct shock")
measures = pd.DataFrame({"Nominal GDP (A-price units)": outcomes["nominal_GDP_%"],
                         "Consumption EV": welfare_table["EV_%"],
                         "Real investment": outcomes["investment_%"]})
x = np.arange(len(measures.index))
for j, column in enumerate(measures.columns):
    ax2.bar(x + (j - 1) * 0.26, measures[column], width=0.26, label=column, color=_nbstyle.BARRA_COLORES[j],
            hatch=_nbstyle.BARRA_HATCH[j], edgecolor=_nbstyle.FONDO)
_nbstyle.etiquetar_barras(ax2, fmt="{:+.3f}")
ax2.axhline(0, color=_nbstyle.SPINE, linewidth=.8)
ax2.margins(y=0.12)
ax2.set_xticks(x, [f"country {code}" for code in measures.index])
ax2.set(title="A levies 10% on all imports from B (Leontief sourcing)", ylabel="Change from baseline (%)")
ax2.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ## Read the output
#
# **The cascade.** The spectral radius is 0.3870 and the upper bound certifies it is below one, so the cost series converges; the round-by-round gaps shrink by about that factor (0.3876 per round), as the math predicts. A 1% cost increase at ARG_A01_02 raises that node's own unit cost by 1.2555% once the network feeds it back, and the pass-through summed over all nine nodes is 1.6143. In the left panel, the feedback to the shocked node is the largest single bar, but a third of the indirect increase (33.9%) lands on foreign nodes: a local cost shock is also an imported cost shock for trading partners. The three provider layouts give this same answer because they are the same network, not because three databases agree.
#
# **The tariff.** A collects 3.8102 in duties and its nominal GDP rises 3.6183%, yet its consumption EV is only 0.0147% of baseline consumption (0.0069 in currency units). GDP adds the duty receipts, which A's own buyers pay, so it is not a welfare measure. With Leontief sourcing, A's buyers cannot switch from B's goods to A's, so A cannot shift the duty onto B. Relative factor prices move toward B instead: B's wage rises to 1.0197 times A's and B's producer price to 1.0074 times A's. The tariff worsens the imposer's terms of trade, the opposite of the textbook optimal-tariff result, and B, the country facing the duty, gains 0.2959% of its consumption. With A as numeraire, the attribution tells the same story: A's fiscal transfer (+2.1409) roughly offsets the higher purchaser prices (−1.6641) and lower factor income (−0.4698).
#
# **Consumption versus investment.** Both consumption EVs are positive, yet this is not a world efficiency gain. Output is fixed, so the world consumption gain (+0.2137 baskets) is exactly the world investment loss (−0.2137). A's investment basket buys 15 of its 23 units from B, against 15 of 45 for its consumption basket, so the duty raises A's investment price by 7.0495% and its consumption price by 3.6031%; with fixed nominal spending shares, A's real investment falls 1.1913%. A consumption-only EV counts consumption bought at the expense of investment as a gain.
#
# **What survives relabelling.** Under the world-income closure EV and EV_% are identical whichever country is listed first. The price, factor-income and transfer terms are not: A's price effect is −1.6641 with A as numeraire and −1.3194 with B as numeraire, and only their sum is invariant. Nominal GDP is not invariant either (A's change is 3.6183% or 2.8615%), so the right panel's GDP bars depend on the numeraire and its EV bars do not. Under the numeraire-fixed foreign-saving closure even the sign of A's EV depends on the label: +0.0019% with A first, −0.1864% with B first.

# %% [markdown]
# ## Your turn
#
# Replace Leontief sourcing with CES sourcing of intermediate inputs. `sigma` is the elasticity of substitution between origins in intermediate purchases; final-use baskets stay fixed-coefficient. The cell solves A's 10% tariff at your `sigma` with each country listed first, under the world-income closure. Advertised range: 0 to 0.3 or 1 to 5. Between them this table has a singular point near 0.45: the terms-of-trade response to a small tariff changes sign there, the 10% solve fails to converge close to it, and converged EVs nearby are very large and not credible policy numbers.
#
# 1. **Basic.** Predict who gains at the default `sigma = 2.0` before you run it, then try 1 and 5. Which way does B's relative price move now, and why can A shift part of the duty onto B once its buyers can substitute away from B's goods?
# 2. **Intermediate.** Set `sigma = 0.2`, then `0.3`. B's relative price still rises. Why does A now lose, and more than at `sigma = 0`, even though it still collects the duty? Compare A's EV with its nominal GDP change from `cf_you.gdp` and `base_you.gdp`.
# 3. **Stretch.** In a new cell, loop over `sigma` in (0.4, 0.44, 0.46, 0.5, 0.6) with `tariff_experiment(sigma=s, rate=1e-3)` and record the small-tariff response $\ln(p_B/p_A)/0.001$. Then try `rate=0.10` inside `try`/`except`. Locate the sign change and explain why a converged but huge EV next to it is not evidence. Finally, call `tariff_experiment` with `closure="numeraire"` for both country orders at one `sigma`: which assert of the cell below would fail, and why?

# %%
sigma = 2.0  # ← change this: sourcing elasticity, 0 to 0.3 or 1 to 5
assert 0 <= sigma <= 0.3 or 1 <= sigma <= 5, "this table is near-singular for sigma between 0.3 and 1"
you = {order: tariff_experiment(order, sigma=sigma) for order in (("A", "B"), ("B", "A"))}
calib_you, base_you, cf_you, w_you = you[("A", "B")]
p_you = cf_you.p_sol.ravel()
p_ratio_you = p_you[1] / p_you[0]
print(f"sigma = {sigma}: B's relative price {p_ratio_you:.4f}; EV % of consumption: "
      f"A {w_you['A'].ev_pct_consumption:+.4f}, B {w_you['B'].ev_pct_consumption:+.4f}")
for code in "AB":  # relabelling invariance under the world-income closure
    assert abs(you[("A", "B")][3][code].ev - you[("B", "A")][3][code].ev) < 1e-8
assert (p_ratio_you > 1) == (sigma < 0.45)  # inelastic sourcing: A's duty raises B's relative price
assert (p_ratio_you > 1) == (w_you["B"].ev > 0)  # B gains exactly when its terms of trade improve

# %% [markdown]
# ## How comprehensive is this?
#
# `docs/trade_welfare.md` runs the same table and documents the welfare function, and `docs/trade_accounting.md` explains consistent accounting, the numeraire and the `foreign_saving_units` closures. Notebook 62 solves a consistent-accounting tariff equilibrium with CES intermediate sourcing, notebook 63 searches for optimal and Nash tariffs with a Hicksian EV objective (this notebook computes no Nash equilibrium), and notebook 64 brackets spectral radii and studies singular points of the solver. This notebook does not validate the ingestion of native multi-regional input-output (MRIO) archives, empirical cross-database welfare rankings or a causal terms-of-trade/efficiency decomposition; the historical TOT/Alloc (terms-of-trade/allocative-efficiency) labels remain unavailable. See `docs/STRUCTURAL_VALIDATION_STATUS.md` for current validation limits.
