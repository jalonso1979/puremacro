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
# # Tariff wars: optimal tariffs, retaliation and a numerical Nash check
#
# **What happens when two countries set tariffs against each other without cooperating: where do the tariffs end up, and who gains?**
#
# This hand-balanced two-country economy is synthetic. Country A and country B are illustrative; no actual national policy is being predicted. Every policy comparison uses the same consumption expenditure function, the same free-trade reference equilibrium and the same audited accounting.

# %% [markdown]
# ## The method in math
#
# Country $i$ values two final-use baskets $k\in\{0,2\}$; investment (category 1) is left out. Each basket is a fixed Leontief bundle of home and foreign goods with purchaser price $P_{ik}$, including tariffs and final-use taxes. The baskets are combined Cobb-Douglas with the calibrated expenditure shares $\theta_{ik}$:
#
# $$U_i=\prod_{k\in\{0,2\}}c_{ik}^{\omega_{ik}},\qquad e_i(P,U)=U\prod_{k\in\{0,2\}}\Big(\frac{P_{ik}}{\omega_{ik}}\Big)^{\omega_{ik}},\qquad \omega_{ik}=\frac{\theta_{ik}}{\theta_{i0}+\theta_{i2}}.$$
#
# A tariff profile $\tau=(\tau_A,\tau_B)$ pays the equivalent variation at free-trade prices $P_0$, reported as a share of baseline consumption spending $m_{i0}=e_i(P_0,U_{i0})$. The best response searches the declared interval $[0,\bar\tau]$:
#
# $$EV_i(\tau)=e_i\big(P_0,U_i(\tau)\big)-e_i(P_0,U_{i0}),\qquad BR_i(\tau_{-i})=\arg\max_{t\in[0,\bar\tau]}EV_i(t,\tau_{-i}).$$
#
# A Nash equilibrium on $[0,\bar\tau]$ satisfies $\tau_i=BR_i(\tau_{-i})$ for both players. The search updates the players in turn, $\tau_i\leftarrow(1-\lambda)\tau_i+\lambda\,BR_i(\tau_{-i})$ with relaxation $\lambda$, and accepts a candidate only if
#
# $$\max_i|BR_i(\tau_{-i})-\tau_i|\leq\epsilon_\tau,\qquad \max_i r_i/m_{i0}\leq\epsilon_r,\qquad r_i=\max_{t\in[0,\bar\tau]}EV_i(t,\tau_{-i})-EV_i(\tau).$$

# %% [markdown]
# ## Intuition
#
# **Intuition.** A tariff lowers the price a country pays to foreign exporters, so the partner pays part of it through worse terms of trade, and the revenue is rebated to domestic households. Against this terms-of-trade gain stands a substitution loss: buyers switch away from the taxed imports. The textbook optimal tariff balances the two and equals the inverse of the partner's export-supply elasticity: the less the partner can redirect its exports, the higher the tariff. In this model each country's fixed labor and capital produce a single good, and final-use baskets are Leontief across origins. Households therefore never substitute between home and foreign goods, and firms substitute intermediates only with elasticity sigma = 2. The substitution loss stays too small to offset the terms-of-trade gain, so each country's welfare rises monotonically in its own tariff; the code checks this up to 300%, with the partner at free trade or at 30%. Every "optimal tariff" and every Nash candidate below is therefore the ceiling we impose, not an interior optimum. Retaliation decides who gains. When both countries tariff, the terms-of-trade effects partly offset, and the outcome depends on the asymmetries of the table. A Prisoner's Dilemma, where mutual tariffs leave both worse off than free trade, is a property to check in the payoff matrix; retaliation alone does not imply it. The payoff counts consumption only, so resources that tariffs move out of investment into consumption show up as gains.

# %% [markdown]
# ## Worked code
#
# Build a balanced transaction table with one producing sector per country, two consumption baskets (categories 0 and 2), a separate investment category (1), domestic taxes and nonzero foreign balances. Labor and capital split each sector's residual value added 2/3 : 1/3 and close the production accounts; in the model, factor endowments are fixed and wages and rents are endogenous. Foreign saving is held at a fixed share of world factor income (`foreign_saving_units="world_income"`), so no country's price serves as the unit of account and results do not depend on which country is listed first.

# %%
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.trade import (
    calibrate_trade_model, solve_policy_equilibrium, build_strategic_tariffs,
    compute_hicksian_welfare, compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix, solve_multilateral_nash_tariffs,
)

# Columns: intermediate use in A/B, then final use C1/I/C2 in A and C1/I/C2 in B.
flows = np.array([[10., 12., 24., 8., 6., 4., 20., 16.],
                  [8., 14., 4.5, 15., 10.5, 35., 18., 15.]])
taxes = np.array([4., 6., 1.2, 1., .8, 1.8, 2., 1.2])
value_added = flows.sum(1) - flows[:, :2].sum(0) - taxes[:2]
data = np.vstack([flows, taxes, np.r_[2*value_added/3, np.zeros(6)],
                  np.r_[value_added/3, np.zeros(6)]])
calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
calib = replace(calib, metadata={**calib.metadata, "is_synthetic": True,
    "source": "hand-balanced teaching table", "unit": "illustrative value units"})
players = tuple(calib.country_codes)
print(f"Imports at base prices: A buys {flows[1, [0, 2, 3, 4]].sum():g} units from B, "
      f"B buys {flows[0, [1, 5, 6, 7]].sum():g} units from A; foreign balances {calib.invforT.ravel()}")
closure = dict(sigma=2., foreign_saving_units="world_income")
base = solve_policy_equilibrium(calib, tol=1e-9, **closure)
options = dict(metric="hicksian_ev", consumption_categories=(0, 2),
               base_equilibrium=base, ge_tol=1e-9, **closure)

def ev_pct(eq):
    """Consumption EV of A and B, % of each country's baseline consumption."""
    return np.array([compute_hicksian_welfare(calib, eq, base_result=base, target_country=p,
                     consumption_categories=(0, 2)).ev_pct_consumption for p in players])

def solve_profile(profile, seed):
    ta, tf = build_strategic_tariffs(calib, profile)
    return solve_policy_equilibrium(calib, ta, tf, x0=seed, tol=1e-9, **closure)

# %% [markdown]
# ### A bounded unilateral search
#
# Select `metric="hicksian_ev"` explicitly. Country A chooses a universal import tariff on $[0, 30\%]$ while B stays at free trade. The model uses CES intermediate sourcing with sigma 2, fixed Leontief final-use baskets and lump-sum rebates of all taxes and duties. Percentages are EV divided by baseline consumption spending. We then re-solve the economy far beyond the ceiling, up to a 300% tariff, with the partner at free trade or at 30%. The helpers `solve_profile` and `ev_pct` call the same public functions the searches use (`build_strategic_tariffs`, `solve_policy_equilibrium`, `compute_hicksian_welfare`).

# %%
tariff_ceiling = 0.30
opt = compute_unilateral_optimal_tariff(calib, country_idx="A", tariff_max=tariff_ceiling,
    num_grid=9, method="bounded", **options)
print(f"A's best tariff on [0, {tariff_ceiling:.0%}]: {opt.optimal_tariff_rate:.4f} "
      f"({opt.metadata['boundary']} bound), EV = {opt.welfare_gain_pct:+.2f}% of baseline consumption")
assert opt.metadata["boundary"] == "upper" and opt.optimal_tariff_rate == tariff_ceiling

# Own-tariff payoff far beyond the ceiling, partner at free trade or at the ceiling.
own_grid = np.r_[0., .1, .2, .3, .5, 1., 1.5, 2., 2.5, 3.]
curves = {}
for k, p in enumerate(players):
    for rival_rate in (0., tariff_ceiling):
        seed, values = base.x_sol, []
        for rate in own_grid:  # warm-start each solve from the previous tariff
            eq = solve_profile({p: float(rate), players[1-k]: rival_rate}, seed)
            seed = eq.x_sol
            values.append(ev_pct(eq)[k])
        curves[p, rival_rate] = np.array(values)
        print(f"{p}, partner at {rival_rate:.0%}: EV at own tariff 30% / 100% / 300% = "
              f"{values[3]:+.2f}% / {values[5]:+.2f}% / {values[9]:+.2f}%")
# Internal consistency: the helper route reproduces the search's payoff at the ceiling.
assert abs(curves["A", 0.][3] - opt.welfare_gain_pct) < 1e-6
assert all(np.all(np.diff(c) > 0) for c in curves.values())  # no interior optimum up to 300%

# %% [markdown]
# ### A fixed-action tariff game
#
# Each player chooses free trade (0%) or its own unilateral optimum on $[0, 30\%]$, which the library computes when no action is supplied; for both countries it is the 30% ceiling. Each cell prints (A's EV %, B's EV %), each relative to that country's baseline consumption. A Prisoner's Dilemma requires a strictly dominant tariff for both players and mutual tariffs that leave both worse off than free trade. We then rebuild the same game with B listed first; with foreign saving tied to world income, the payoffs must not change.

# %%
payoffs = compute_welfare_payoff_matrix(calib, player_a="A", player_b="B",
    tariff_max=tariff_ceiling, num_grid=9, **options)
M = payoffs.payoff_matrix  # M[i, j, k]: EV % of player k when A plays i and B plays j (1 = tariff)
actions = payoffs.metadata["action_tariffs"]
print("Positive actions:", {p: f"{t:.0%}" for p, t in actions.items()})
print(payoffs.summary())
dominant = {"A": bool(M[1, 0, 0] > M[0, 0, 0] and M[1, 1, 0] > M[0, 1, 0]),
            "B": bool(M[0, 1, 1] > M[0, 0, 1] and M[1, 1, 1] > M[1, 0, 1])}
print("Tariff strictly dominant:", dominant)
print(f"Mutual tariffs vs free trade: A {M[1, 1, 0]:+.2f}%, B {M[1, 1, 1]:+.2f}%")
print("Prisoner's Dilemma?", payoffs.is_prisoners_dilemma)
assert all(dominant.values()) and M[1, 1, 0] < 0 < M[1, 1, 1]
assert payoffs.is_prisoners_dilemma is False  # fails only through B, who gains from the war

# Relabel the table with B first: permute the producer rows and the country columns.
data_ba = data[:, [1, 0, 5, 6, 7, 2, 3, 4]]
data_ba[:2] = data_ba[[1, 0]]
calib_ba = calibrate_trade_model(data_ba, ns=1, nc=2, nfd=3, country_codes=["B", "A"])
base_ba = solve_policy_equilibrium(calib_ba, tol=1e-9, **closure)
payoffs_ba = compute_welfare_payoff_matrix(calib_ba, player_a="A", player_b="B",
    optimal_a=actions["A"], optimal_b=actions["B"], **{**options, "base_equilibrium": base_ba})
order_gap = np.abs(payoffs_ba.payoff_matrix - M).max()
print("Largest payoff change with B listed first is below 1e-8 percentage points:", order_gap < 1e-8)
assert order_gap < 1e-6

# %% [markdown]
# ### Numerical best-response diagnostics
#
# The continuous game lets each country choose any tariff in $[0, 30\%]$. With relaxation $\lambda=1$ each sweep moves every tariff all the way to its best response; a smaller $\lambda$ damps the update, which can help when best responses overshoot. Convergence requires both tests at the final profile: the best-response gap within `tol` and each player's regret within `regret_tol` of its baseline consumption. Every equilibrium solve is audited: all market-clearing and accounting equations are recomputed and must hold to `ge_tol`. If Newton's method fails, the solver tries SciPy's Powell hybrid root-finder and then pseudo-arclength (Keller) continuation, a path-following method shown in notebook 64. It never relaxes the tolerance, and exhausted recovery raises an error. The result can be inconclusive; report that outcome. After the search we value the candidate: welfare, terms of trade, world consumption EV and real investment.

# %%
nash = solve_multilateral_nash_tariffs(calib, player_countries=players,
    tariff_max=tariff_ceiling, best_response_grid_size=7, max_iter=20,
    relaxation=1., tol=1e-4, regret_tol=1e-6, **options)
last_ge = nash.equilibrium.metadata["policy_solver_attempts"][-1]
gaps = [h["undamped_update_gap"] for h in nash.iteration_history]
print("Candidate tariffs:", {p: round(t, 4) for p, t in nash.nash_tariffs.items()},
      "| best responses at bound:", nash.metadata["best_response_boundaries"])
print(f"Converged: {nash.converged} | best-response gap per sweep {gaps} | final gap "
      f"{nash.outer_error:.1e} (tol 1e-4) | relative regret {nash.metadata['relative_max_regret']:.1e} (tol 1e-6)")
print(f"Final GE solve: {last_ge['method']}, accepted {last_ge['accepted']}, "
      f"audited residual <= 1e-9: {last_ge['audited_max_residual'] <= 1e-9}")
print(nash.summary())
w, tot = nash.welfare_changes_pct, nash.terms_of_trade_changes_pct
investment = 100*(nash.equilibrium.c_fd[0, 1]/base.c_fd[0, 1] - 1)
at_base_prices = base.Pfd_final[0]*(nash.equilibrium.c_fd[0] - base.c_fd[0])  # (use, country)
print(f"World consumption EV {nash.world_welfare_change_pct:+.2f}% | real investment "
      f"A {investment[0]:+.2f}%, B {investment[1]:+.2f}%")
print(f"Change at baseline prices (value units): consumption {at_base_prices[[0, 2]].sum():+.2f}, "
      f"investment {at_base_prices[1].sum():+.2f}, all final demand {at_base_prices.sum():+.2f}")
assert nash.converged and set(nash.metadata["best_response_boundaries"].values()) == {"upper"}
# The best reply is the ceiling whatever the rival does: one sweep lands on it, the next confirms.
assert gaps == [tariff_ceiling, 0.]
assert w["A"] < 0 < w["B"] and tot["A"] < 0 < tot["B"]
assert nash.world_welfare_change_pct > 0 and np.all(investment < 0) and at_base_prices.sum() < 0

# %% [markdown]
# Two checks on the candidate. First, an internal consistency check (a second route through puremacro, not an independent oracle): re-solve the economy at 31 evenly spaced deviations for each player, with the rival at its candidate tariff, and confirm that no deviation beats the candidate by more than `regret_tol`. Second, a deliberately truncated search, one sweep with relaxation 0.0003, shows why a small damped step is not a convergence test.

# %%
for k, p in enumerate(players):
    seed, best = nash.equilibrium.x_sol, -np.inf
    for rate in np.linspace(0., tariff_ceiling, 31):
        eq = solve_profile({p: float(rate), players[1-k]: nash.nash_tariffs[players[1-k]]}, seed)
        seed, best = eq.x_sol, max(best, ev_pct(eq)[k])
    print(f"{p}: best of 31 deviations {best:+.4f}% vs candidate {w[p]:+.4f}%")
    assert (best - w[p])/100 <= 1e-6  # regret as a fraction of baseline consumption

short = solve_multilateral_nash_tariffs(calib, player_countries=players,
    tariff_max=tariff_ceiling, best_response_grid_size=7, max_iter=1,
    relaxation=3e-4, tol=1e-4, regret_tol=1e-6, **options)
step = short.iteration_history[-1]["step"]
print(f"Truncated search: step {step:.1e} (below tol 1e-4), gap {short.outer_error:.2f}, "
      f"relative regret {short.metadata['relative_max_regret']:.3f}, converged {short.converged}")
assert step < 1e-4 and not short.converged and short.metadata["relative_max_regret"] > 1e-2

# %%
# Both countries at the same tariff t: for t = 30% this is the Nash candidate.
mutual = np.linspace(0., .4, 9)
m0 = opt.metadata["baseline_consumption_by_country"]  # baseline consumption spending
investment0 = base.Pfd_final[0, 1] @ base.c_fd[0, 1]
seed, war = base.x_sol, []
for t in mutual:
    eq = solve_profile({"A": float(t), "B": float(t)}, seed)
    seed, ev = eq.x_sol, ev_pct(eq)
    investment_t = base.Pfd_final[0, 1] @ eq.c_fd[0, 1]  # at baseline prices
    war.append([*ev, ev @ m0/m0.sum(), 100*(investment_t/investment0 - 1)])
war = np.array(war)
# Columns: EV of A, EV of B, world consumption EV, world real investment (all %).
assert np.all(np.diff(war, axis=0)*np.array([-1, 1, 1, -1]) > 0)

fig, axes = _nbstyle.figura(1, 2, alto=True)
ink = _nbstyle.palette(2)
# Lightness and marker tell the countries apart; solid vs dotted, the partner's tariff.
for (p, rival_rate), values in curves.items():
    axes[0].plot(100*own_grid, values, ls="-" if rival_rate == 0 else (0, (1, 1.7)),
                 marker="o" if p == "A" else "s", ms=4, color=ink[players.index(p)],
                 label=f"{p}, partner at {rival_rate:.0%}")
axes[0].plot([100*tariff_ceiling]*2, [w["A"], w["B"]], "*", ms=14, color=_nbstyle.TINTA,
             markeredgecolor=_nbstyle.FONDO, label="Nash candidate")
axes[0].set(title="(a) Own-tariff payoff never turns down", xlabel="Own tariff (%)",
            ylabel="EV / baseline consumption (%)")
for j, (label, ls) in enumerate([("A: consumption EV", "-"), ("B: consumption EV", (0, (5, 2.2))),
                                 ("World: consumption EV", (0, (1, 1.7))),
                                 ("World: real investment", (0, (10, 3)))]):
    axes[1].plot(100*mutual, war[:, j], ls=ls, color=_nbstyle.palette(4)[j], label=label)
axes[1].set(title="(b) Both countries at the same tariff", xlabel="Tariff of both countries (%)",
            ylabel="Change from free trade (%)")
for ax, note in zip(axes, ("30% ceiling", "Nash candidate (30%)")):
    ax.axvline(100*tariff_ceiling, color=_nbstyle.NOTA, ls=":", lw=1.2, label=note)
    ax.axhline(0, color=_nbstyle.SPINE, lw=0.8)
    ax.legend(fontsize=9, ncol=2, handlelength=3.4, loc="upper center", bbox_to_anchor=(0.5, -0.17))

# %% [markdown]
# ## Read the output
#
# **Unilateral search.** A's best tariff on $[0, 30\%]$ is the ceiling itself (0.3000, upper bound), worth +7.07% of baseline consumption. The printed curves and panel (a) show why the ceiling binds: each country's EV keeps rising with its own tariff up to 300%, whether the partner stays at free trade or retaliates with 30% (A reaches +37.14% and +25.42%, B +46.79% and +44.29%). No finite optimal tariff exists in this calibration, so the reported "optimum" only restates the ceiling we chose.
#
# **Fixed-action game.** With 30% as each player's positive action, the tariff is strictly dominant for both players: whatever the partner does, a tariff raises one's own EV. Yet the game is not a Prisoner's Dilemma. Mutual tariffs leave A worse off than free trade (-3.17%) but B better off (+6.27%), so free trade does not Pareto-dominate the war; B's inequality is the one that fails. Listing B first leaves every payoff unchanged, as the world-income closure promises.
#
# **Nash candidate.** The search stops at (0.3, 0.3) with both best responses at the upper bound: the candidate is the ceiling pair, not an interior equilibrium. The gaps per sweep, [0.3, 0.0], say the same thing: the first sweep jumps to the ceiling from free trade, and the second finds the same best responses because they do not depend on the rival's tariff. Both tests pass with a final gap and relative regret of 0.0e+00, because the best responses at the candidate are the candidate itself, and the 31-point deviation scans find no deviation that beats it. The truncated search shows why the tests look at the best-response gap and regret rather than at the step: its step (9.0e-05) is below `tol`, yet the gap is 0.30 and a deviation would gain 0.117 of baseline consumption, so it is correctly reported as not converged.
#
# **Who wins the war.** B's EV is +6.27% and A's is -3.17%. B's terms of trade change by +8.54% and A's by -7.87%: symmetric tariffs do not cancel in an asymmetric economy (for example, B imports 52 units from A while A imports 38 from B). World consumption EV is +2.57%, but that is not a gain from the war: real investment changes by -9.66% in A and -4.31% in B. Valued at baseline prices, consumption changes by +3.29 units and investment by -4.04, so total final demand changes by -0.75 units. A consumption-only EV counts the resources moved out of investment as a gain. Panel (b) repeats the comparison for common tariffs from 0 to 40%: A's loss, B's gain, the world consumption "gain" and the fall in investment all grow with the tariff. The stars in panel (a) are the Nash payoffs, on the dotted "partner at 30%" curves at a 30% own tariff.

# %% [markdown]
# ## Your turn
#
# Treat the ceiling as a tariff binding negotiated in a trade agreement. Before running, predict: does the Nash candidate still sit at the ceiling, and is A better or worse off than in the 30% war? The cell asserts the answer for any ceiling in $[1\%, 40\%]$.

# %%
custom_ceiling = 0.10  # ← change this: a negotiated tariff binding in [0.01, 0.40]
assert 0.01 <= custom_ceiling <= 0.40
custom = solve_multilateral_nash_tariffs(calib, player_countries=players,
    tariff_max=custom_ceiling, best_response_grid_size=7, max_iter=20,
    relaxation=1., tol=1e-4, regret_tol=1e-6, **options)
wc = custom.welfare_changes_pct
print(f"Candidate at a {custom_ceiling:.0%} binding:", {p: round(t, 4) for p, t in custom.nash_tariffs.items()},
      "| bounds:", custom.metadata["best_response_boundaries"], "| converged:", custom.converged)
print(f"EV: A {wc['A']:+.2f}%, B {wc['B']:+.2f}%, world {custom.world_welfare_change_pct:+.2f}% "
      f"(30% war: A {w['A']:+.2f}%, B {w['B']:+.2f}%, world {nash.world_welfare_change_pct:+.2f}%)")
assert custom.converged and set(custom.metadata["best_response_boundaries"].values()) == {"upper"}
assert wc["A"] < 0 < wc["B"]  # no binding in the range makes both countries gain
if custom_ceiling < tariff_ceiling:  # a tighter binding helps A and costs B
    assert wc["A"] > w["A"] and wc["B"] < w["B"]
elif custom_ceiling > tariff_ceiling:
    assert wc["A"] < w["A"] and wc["B"] > w["B"]

# %% [markdown]
# **Graded prompts.**
#
# 1. *Basic.* Try `custom_ceiling = 0.05` and then `0.40`, predicting the direction first. Which country would push for a tighter binding in a negotiation? Does any binding in the range leave both countries better off than free trade? The cell's assertions check both answers.
# 2. *Intermediate: the closure matters.* Rebuild the fixed-action game in both country orders with `foreign_saving_units="numeraire"`, which fixes foreign saving in units of the first-listed country's price. Predict which of these depend on the order: the payoff magnitudes, their signs, the dominant action, the Prisoner's Dilemma verdict.
#
#    ```python
#    games = {}
#    for name, table, order in (("AB", data, ["A", "B"]), ("BA", data_ba, ["B", "A"])):
#        cal = calibrate_trade_model(table, ns=1, nc=2, nfd=3, country_codes=order)
#        b = solve_policy_equilibrium(cal, tol=1e-9, sigma=2., foreign_saving_units="numeraire")
#        games[name] = compute_welfare_payoff_matrix(cal, player_a="A", player_b="B",
#            optimal_a=actions["A"], optimal_b=actions["B"],
#            **{**options, "foreign_saving_units": "numeraire", "base_equilibrium": b})
#    G1, G2 = games["AB"].payoff_matrix, games["BA"].payoff_matrix
#    ```
#
#    Check: `assert np.abs(G1 - G2).max() > 0.05 and np.array_equal(np.sign(G1), np.sign(G2))` and `assert not games["AB"].is_prisoners_dilemma and not games["BA"].is_prisoners_dilemma`.
# 3. *Stretch: when is the optimal tariff interior?* Re-solve the baseline with `sigma=20.` and the same closure (`base20`), and set `opts20 = {**options, "sigma": 20., "base_equilibrium": base20}`. For A, run `compute_unilateral_optimal_tariff` with `policy_mode` equal to `"universal"`, `"final_only"` and `"intermediate_only"`, `tariff_max` 1.0 and 2.0, `num_grid=21` and `method="bounded"`, and tabulate the rate and `metadata["boundary"]`. Which instrument has an interior optimum, does it move when the ceiling doubles, and what do the Leontief final baskets have to do with it? Then run the Nash search with `policy_mode="intermediate_only"`, `tariff_max=1.0`, `best_response_grid_size=11`, `max_iter=30` and `relaxation=0.8`, and explain why A's Nash tariff differs from its unilateral optimum. Check: `"final_only"` and `"universal"` are `"upper"` at both ceilings, `"intermediate_only"` is `"interior"` at both with rates within 1e-3 of each other, and the Nash search converges with boundaries `{"A": "interior", "B": "upper"}` and an A tariff more than 0.05 above A's unilateral optimum.

# %% [markdown]
# ## How comprehensive is this?
#
# This is a reproducible teaching exercise on a synthetic table, not an empirical trade-war forecast. On this same table (sigma 2, categories 0 and 2), a separately coded scalar version of the model, [`validate_hicksian_policy.py`](https://github.com/jalonso1979/puremacro/blob/v4.3.0/tools/reference_validation/validate_hicksian_policy.py), evaluates all 1,681 profiles of a 41-by-41 grid over $[0, 40\%]^2$ and agrees with the runtime's unilateral optima, Nash candidate, deviations and fixed-action payoffs ([recorded results](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-hicksian-policy/REPORT.md)). That reference shares the model and used the numeraire closure, so it is an implementation check, not independent validation, and it does not cover the world-income closure used here. Grid and local searches can still miss narrow peaks. Causal terms-of-trade, allocative-efficiency and tariff-revenue decompositions, and existence or uniqueness proofs, are unavailable. The same machinery appears in notebook 62 (the trade CGE and `solve_trade_equilibrium`), notebook 64 (the continuation solvers behind the recovery chain) and notebook 65 (Hicksian EV and CV with price, factor-income and fiscal attribution); see also `docs/trade_policy.md`.
