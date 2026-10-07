> 🇬🇧 English · 🇪🇸 [Español](es/trade_ces_newton.md)

# Exact CES block Newton on consistent accounting

`puremacro.trade.ces_newton` solves the consistent-accounting trade equilibrium
of [consistent trade accounting](trade_accounting.md) with a three-tier nested
CES production technology, using the exact Jacobian instead of finite
differences. Each Newton step costs two dense `M x M` LU factorizations plus a
small macro Schur complement. On a synthetic 77 country by 11 sector table
(`M = 847`, 2001 unknowns) a 10% tariff with mild elasticities (0.1, 0.1, 0.5)
solves in under a second on a laptop (0.7 to 0.9 s measured, 4 Newton
iterations), and a high-elasticity case (origin elasticity 4) with a random 0
to 30% schedule takes a few seconds (2.6 to 3.7 s, 10 iterations). Every
returned state has passed an independent flow certificate; nothing else is
returned.

The module is ported from the IO research workspace (under
`headlinePaper/rebuild/`: `ces_newton.py`, `ces_continuation.py` and the
vendored `vendor/puremacro/trade/corrected/ces.py`, with the proof
`empirical_2019/CES_NEWTON_PROOF.md`). The quotations below come from that
workspace's `README.md` and `CHANGES.md`, and the elasticity audit from its
`results/elasticity_2019.json`. The IO engine solves a different model (one
composite factor, DEK closures); what carries over is the elimination, the
oracle tests and the homotopy rule.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model
from puremacro.trade.ces_newton import (NestedCESTechnology, solve_ces_block_newton,
                                        continue_tariff_homotopy, certify_ces_equilibrium)

# Two countries, one sector, two final uses (C, I): a hand-balanced table in the
# calibrate_trade_model layout (rows: deliveries, net taxes, labour, capital).
Z = np.array([[10., 12.], [8., 14.]])
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
production_tax = np.array([4., 6.]); final_tax = np.array([2., 1., 3., 2.])
output = Z.sum(1) + F.sum(1)
va = output - Z.sum(0) - production_tax
table = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                   np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])
calib = calibrate_trade_model(table, ns=1, nc=2, nfd=2, country_codes=["A", "B"])

tau = np.ones((2, 1, 2)); tau[1, 0, 0] = 1.2        # A taxes intermediate imports from B at 20%
tau_fd = np.ones((2, 2, 2)); tau_fd[1, :, 0] = 1.2  # and final imports from B at 20%
technology = NestedCESTechnology(sigma_va_materials=0.3, sigma_sectors=0.5,
                                 sigma_origins=1.5, rho_va=0.8)
result = solve_ces_block_newton(calib, tau, tau_fd, technology=technology)
print(result.summary())
print(result.equilibrium.w_sol.ravel(), result.certificate["current_account"])

path = continue_tariff_homotopy(calib, tau, tau_fd, technology=technology)
print(path.stages_frame())
audit = certify_ces_equilibrium(calib, path.equilibrium.x_sol, tau=tau, tau_fd=tau_fd,
                                technology=technology)
print(max(audit.values()))
```

`result.equilibrium` is an ordinary `TradeEquilibriumResult` with
`metadata["accounting"] == "consistent"` and
`metadata["effective_method"] == "ces_block_newton"`. Inside the repository the
frozen OECD 2019 three-region fixture used by the tests is loaded with
`tools.reference_validation.validate_oecd.load_fixture()` followed by the
private helper `puremacro.trade._oecd_icio.condense_final_demand` and
`puremacro.trade.data.package_mrio_to_calibration_result`. The bundled 77 by 11
table is rejected by consistent accounting (three merged investment cells are
negative), so native tables must be condensed to nonnegative C/I/X aggregates
first.

## What it computes

The equations are exactly those of `accounting="consistent"`: producer and
purchaser valuation, homogeneous factor costs, an output-revenue production tax,
tax-inclusive final expenditure shares, lump-sum rebates, foreign saving fixed
in numeraire units and the first producer price equal to one. Only the
intermediate technology changes. Cell `j` (country `c`, sector `s`) buys origin
cell `i` at `r_ij = p_i tau_ij` and combines inputs through

```
h_sj = CES_so({r_ij}_{i in s}; u0_ij)            origins within a material sector
m_j  = CES_ss({h_sj}_s; eta0_sj)                 material sectors
v_j  = CES_rho({w_c, r_c}; (1 - alpha_j, alpha_j))   labour and capital
c_j  = CES_sv({v_j, m_j}; (v0_j, A0_j) / (1 - t_j))  value added and materials
```

with `CES_sigma(x; s) = (sum_i s_i x_i^(1 - sigma))^(1 / (1 - sigma))`, the
geometric mean at `sigma = 1` and the value one on an empty nest. The
benchmark shares are `u0_ij = a0_ij / sum_{k in s} a0_kj`,
`eta0_sj = sum_{i in s} a0_ij / A0_j`, `A0_j = sum_i a0_ij` and
`v0_j = (l_j + k_j) / y_j`; the identity `A0_j + v0_j = 1 - t_j` holds cell by
cell in every `calibrate_trade_model` table. Shephard's lemma gives the input
requirements per unit of output

```
a_ij  = a0_ij (c_j / m_j)^sv (m_j / h_sj)^ss (h_sj / r_ij)^so
b_Lj  = b0_Lj (c_j / v_j)^sv (v_j / w_c)^rho,   b0_Lj = (1 - alpha_j) v0_j
b_Kj  = b0_Kj (c_j / v_j)^sv (v_j / r_c)^rho,   b0_Kj = alpha_j v0_j
(1 - t_j) c_j = sum_i a_ij r_ij + b_Lj w_c + b_Kj r_c
```

and the zero-profit row is `c_j - p_j`. Everything else (household income
`w L + r K + T`, expenditure `theta` times income less foreign saving on
investment, Leontief origin baskets `afd`, duties on producer values, the
government budget, the fixed-saving closure and the numeraire row that replaces
the first goods equation) is unchanged, so at **flat technology**
(`sigma_va_materials = 0`, `sigma_sectors = sigma_origins`, `rho_va = 1`) the
residual equals `_accounting.evaluate(sigma=sigma_origins)` row by row. The
flat CES of `_accounting` uses weights `a_ij / A_j` over all `M` origins; a
nested CES with equal elasticities on the sector and origin tiers collapses to
exactly that index. `NestedCESTechnology.is_flat` reports this case and
`NestedCESTechnology.from_flexible` maps a `FlexibleTechnologyConfig`
(`sigma_y`, `sigma_inter`, `rho_va`). The elasticities are stylized inputs: the
IO record states that "all CES elasticities are explicitly stylized"
(`CHANGES.md`).

## The exact Jacobian action

The unknowns are `x = [log p (M); log y (M); log r (N); log w (N); T (N); XN (N - 1)]`
and the rows are goods (row 0 is the numeraire `p_0 - 1`), prices, labour,
capital, balances `XN - B0` and budgets, in the order and level scaling of
`_accounting.evaluate`. With current cost shares
`alpha_ij = a_ij r_ij / ((1 - t_j) c_j)`, `beta_Lj = b_Lj w_c / ((1 - t_j) c_j)`
and `beta_Kj = b_Kj r_c / ((1 - t_j) c_j)` (Euler: they sum to one), nest shares
`u_ij`, `eta_sj`, `theta_Lj` built from them, and log directions `l` (prices),
`lw`, `lr`:

```
dlog c_j = sum_i alpha_ij l_i + beta_Lj lw_c + beta_Kj lr_c
dlog h_sj = sum_{i in s} u_ij l_i,   dlog m_j = sum_s eta_sj dlog h_sj,   dlog v_j = theta_Lj lw_c + theta_Kj lr_c
da_ij = a_ij [sv dlog c_j + (ss - sv) dlog m_j + (so - ss) dlog h_{s(i)j} - so l_i]
db_Lj = b_Lj [sv dlog c_j + (rho - sv) dlog v_j - rho lw_c]      (db_Kj likewise with lr_c)
d[y_j sum_i (tau_ij - 1) p_i a_ij] = duty_j dlog y_j + y_j sum_i (tau_ij - 1) p_i (da_ij + a_ij l_i)
```

The elimination follows the IO proof (`CES_NEWTON_PROOF.md`) with two changes
forced by puremacro's closure: there are two factors, and the numeraire
replaces the first goods equation while every national budget stays in the
system. `CESBlockJacobian.solve(r)` computes `J^{-1} r` by

1. reading the balance block directly (`dXN = r_bal`, an identity block);
2. solving the price block `(diag(p) - diag(c) alpha^T) l = diag(c)(beta_L C lw + beta_K C lr) - r_p`
   once for the forcing term and once per unit factor-price direction (`L_r`, `L_w`);
3. solving the goods block `(I - a) diag(y) dlog y = (da) y + afd . dq + rhs` where the
   omitted goods equation carries an unknown value `mu` in `rhs[0]`;
4. assembling the `(3N + 1) x (3N + 1)` Schur complement of the labour, capital,
   budget and numeraire rows in `(log r, log w, T, mu)` by pushing unit
   directions through `response` in blocks of `block_size` columns, LU-factoring
   it and back-substituting.

`apply(v)` evaluates `J v` from the same differentials; `as_preconditioner()`
and `as_operator()` wrap both as `scipy.sparse.linalg.LinearOperator`s.
`macro_condition` is the condition number of the row- and column-scaled Schur
block (`3N + 1 = 232` rows at `N = 77` rather than `4N - 1 = 307`: the `N - 1`
balance rows are an identity block and are eliminated first).
`CESBlockJacobian.n` is the number of cells `M`; `dim` (equal to `shape[0]`)
is the number of unknowns `2M + 4N - 1`.

Invertibility is a condition, not a theorem. The price block
`diag(p) - diag(c) alpha^T` is a Z-matrix whose row `j` dominates when
`p_j > c_j (1 - beta_Lj - beta_Kj)`, which holds near any root (where `p = c`)
as long as every cell pays some factor income; there it is an M-matrix and its
inverse is the convergent Neumann series. `I - a` is invertible when the current
input matrix has spectral radius below one (Hawkins-Simon). The Schur block is
factored with partial pivoting and its condition number is recorded. The IO
README puts it this way: "The proof is a conditional local identity, not a
convergence or global-equilibrium theorem."

## Acceptance and failure contract

`solve_ces_block_newton` is a damped Newton: direction `-J^{-1} R(x)`, first
step `min(1, 1 / max|dx_log|)`, Armijo decrease of the squared scaled residual
with factor `1 - 1e-4 step`, at most `max_backtracks` halvings, and an
admissibility filter (finite positive prices, outputs, factor prices and
incomes; positive final quantities on active categories and exactly zero on
structurally absent ones).

The loop has one acceptance test, the same one postprocessing and
`compute_hicksian_welfare` apply afterwards. It stops only when (i) every
imposed residual row divided by its benchmark scale (output, endowment or
income) is at most `tol` (default `2e-11`), and (ii) every level row is at most
`tol * max(scale)`, including the rows that no Newton equation imposes and that
hold only through Walras' law: the omitted goods equation and the realized
foreign balances. It also requires that no final expenditure is negative.
`equilibrium.metadata["tol"]` records that absolute bound. When (i) holds but
(ii) does not, the loop takes further polishing Newton steps; one step usually
suffices. When the flows are so large relative to the benchmark scales that
rounding alone exceeds the bound, the loop raises. For example, a 1e8 import
multiplier with fixed final origin baskets makes the rebated duties about 3e7
times benchmark income; the error then gives the level audit and the bound,
and a looser `tol` is the remedy.

A state is returned only if, in addition, `certify_ces_equilibrium` is at most
`certificate_tol` (default `1e-8`). The certificate rebuilds intermediate and
factor flows from the technology at `(p, r, w)` with explicit per-country
loops. Final quantities, final expenditure and recorded duty receipts come
from the demand assembly of the residual, and the certificate checks them
through the goods-value, omitted-equation, budget, current-account, duty and
GDP keys. It reports zero profit, cost adding-up, goods values, the omitted
goods equation, the current account, bilaterally reconstructed duties, factor
clearing, household and government budgets, the GDP identity and the scaled
root residual. Each key is tested against the violation it names: 1e-5
perturbations of each unknown, faults injected into the assembly, and a
technology that breaks the Euler identity by 1e-6.

Iteration limits, line-search failures, inadmissible iterates, singular macro
blocks and failed certificates raise `CESNewtonError` with the iteration
`history` and the last `macro_condition`. A line-search failure says which
test rejected the trial steps. If the Armijo test rejected them, the message
reports the condition number, because a nearly singular Jacobian is the usual
cause. If every trial left the admissible domain, the message names the binding
quantity, category and country at the smallest trial step. Each history record
counts both kinds of rejection (`rejected_armijo`, `rejected_inadmissible`) and
stores the level audit (`level_audit_before`). No unconverged state is ever
returned with `converged=True`.

`continue_tariff_homotopy` moves the intermediate and final schedules linearly
from a start schedule (free trade by default) to the target with an adaptive
fraction: first step `0.25`, growth `1.5` capped at `0.4` after a stage needing
at most `fast_iterations` Newton steps, halving after a failure, abort below
`min_step`. Every accepted stage passes the same residual, level-audit and
certificate tests as a direct solve and is packaged when it is accepted (a
stage whose packaging fails counts as a failed attempt). Failed attempts are
recorded in `failed_stages`. Every abort raises `CESNewtonError` carrying
`stages`, `failed_stages` and `last_result` (the last accepted stage, or
`None` when the start schedule itself is unresolved). `progress` receives
`{"fraction", "step"}` before each trial stage and the Newton history records
of every stage.

On a random 20 by 11 table with origin elasticity 4, a direct Newton from the
benchmark fails in the line search: the Armijo test rejects every trial and
the macro condition number reaches 8e10. The homotopy reaches the target in
three stages; that is the tool's purpose. Not every failure is a singularity.
Under other random schedules on the same table the direct Newton fails because
the admissibility filter binds: the iterate is pressed against zero investment
in one country, and the error says that every trial step left the admissible
domain. In the case examined (seed 0, flat elasticity 2), relaxing the filter
leads to a root whose investment expenditure in one country is -865, because
the fixed nominal foreign saving (7268) exceeds the investment share of income
(6402). That root is inadmissible under this closure, and a tariff homotopy
cannot change that.

Singular points are a property of the model, not of the port. On the frozen
OECD three-region fixture the determinant of the benchmark Jacobian changes
sign twice between origin elasticities 0.5 and 1. The smallest singular value
of the scaled Jacobian is 4.8e-6 at 0.99, against 5e-3 at 0 or 2. The tests
check this with central differences of `_accounting.evaluate`, independently of
this module. With a 10% US tariff at unit origin elasticity the direct Newton
fails, and so did puremacro's finite-difference Newton and `hybr` when this was
checked. The homotopy certifies the (1,1,1,1) endpoint, but the (0,1,1,1) path
stalls next to the benchmark. Both failures are raised with the
condition-number growth in `history`. The homotopy is not a recovery
guarantee: in the IO elasticity audit 12 of 21 native profiles were
accepted and 9 were unresolved, their first homotopy attempts stalling at
residuals between 1.7e-4 and 1.4e-3 (`results/elasticity_2019.json`; all
`unit_origin` and `middle` profiles, two `cobb_douglas` and one `top`). The IO
README states: "The discrete checks do not certify a continuous branch between
stages. The sign is not robust across the accepted elasticity profiles; failed
procedures are not nonexistence certificates."

## Welfare and results

`CESBlockNewtonResult` wraps the `TradeEquilibriumResult` (`equilibrium`), the
technology, `iterations`, `calls`, `seconds`, the scaled `residual_max`, the
`certificate`, `macro_condition`, the Newton `history` and the homotopy
`stages` and `failed_stages`; `to_dataframe()` renders the history and
`stages_frame()` and `certificate_frame()` the other two, with
`to_markdown()`, `to_latex()`, `to_typst()` and `summary()` as elsewhere in
puremacro.

At flat technology the result is a consistent-accounting equilibrium in every
respect and [`compute_hicksian_welfare`](trade_welfare.md) accepts it unchanged
(`metadata["hicksian_welfare_supported"]` is true and `metadata["sigma"]` is
`sigma_origins`). At nested technology `metadata["hicksian_welfare_supported"]`
is false. The flag is informational: `compute_hicksian_welfare` does not read
it. Instead its audit re-evaluates the state with the flat CES of
`_accounting.evaluate`, which rejects (with a `ValueError`) every nested state
whose root differs from the flat model's. A nested state that also solves the
flat model is accepted. An example is technology (0, .5, .5, .7) on the OECD
three-region fixture, where uniform capital shares make the capital/labour nest
inert (`w = r`). The EV it returns is still correct, because the technology
does not change consumption preferences. For the same reason, Hicksian
consumption welfare for nested results needs the household expenditure
functions evaluated at the result's purchaser prices (`Pfd_final`) rather than
a re-solve; see [household preferences](trade_household.md).

## Memory guard

Measured with `tracemalloc` on a synthetic 77 by 11 table (`M = 847`; one
`M x M` double array is 5.7 MB). The problem data retain about 3.3 such arrays,
the flows of the current state 5.9, and the operator itself 6.2 (two LU
factors, cost and nest shares, duty flows). Together that is about 15 arrays
(88 MB). A full Newton solve peaks at about 22 arrays (127 MB), because the line
search evaluates a trial state while the current one is alive. The peak scales
with `M^2`: about 2.1 GB at `M = 3465` (77 countries by 45 sectors) and 2.8 GB at
the default `max_cells = 4000`. `max_cells` refuses larger tables before
allocating, and the error message states both estimates. Nothing is allocated
at import, so the module is importable in the browser playground. There the
WebAssembly memory limit (a few GB at most) makes the default guard an upper
bound rather than a working size.

## What is and is not validated

`tests/test_trade_ces_newton.py` checks, on a hand-built nonnegative 3 by 2
table with an absent purchases-abroad category and bilateral counter-tariffs,
its active-X variant, the frozen OECD 2019 three-region fixture and a seeded
20 by 11 synthetic table (77 by 11 marked `slow`):

- the exact-inverse oracle at a random non-equilibrium point: the complex-step
  Jacobian of the module's residual satisfies `J solve(r) = r` and
  `apply(solve(r)) = r` to `3e-10` in scaled units for technologies
  (0,0,0,1), (.1,.3,1.5,1), (1,1,1,1), (.5,1,4,.7), (.1,.1,.5,.7) and three flat
  ones (observed `6e-14` on the small tables, `2e-12` at 20 by 11);
- central differences of `_accounting.evaluate` at flat technology invert the
  operator to `1e-6` and agree with the complex-step Jacobian;
- row-by-row residual parity with `_accounting.evaluate` at `sigma` in
  {0, .5, 1, 2} to `1e-12` relative, the Leontief limit against
  `solve_trade_equilibrium(accounting="consistent")` to `1e-10`, flat-CES
  equilibria and Hicksian EV against the finite-difference solvers, and the
  postprocessed flows against `_accounting.postprocess`;
- benchmark reproduction (`a = a0`, `b = b0`, residual below `1e-12`) for every
  technology, the Euler identity, Shephard derivatives by central differences,
  shares summing to one;
- certificate at most `1e-8` and the accounting ledger at most `1e-10` times the
  largest benchmark scale (output, endowment or income; about `1e-11` relative
  observed on the OECD fixture) on every accepted solution, every level row
  within `metadata["tol"]`, nominal homogeneity, currency-unit invariance of
  prices and welfare percentages;
- one acceptance test: a sweep of 66 solves on the hand-built tables, 4 of which
  need a polishing step, returns only states that pass the postprocessing audit,
  including a regression case that an earlier loop stopped at scaled residual
  `1.45e-11` with a net-export gap of `4.8e-9` against the bound `3.9e-9`;
- certificate sensitivity: each key trips (above `1e-7`) on the violation it
  names, and a technology violating the Euler identity by `1e-6` is refused;
- failure messages that name their cause (admissibility-bound direction, Armijo
  failure, rounding floor at prohibitive multipliers), homotopy aborts that
  carry their record, and the OECD near-singularity by central differences of
  `_accounting.evaluate`;
- the failure contract, homotopy stage certification, forced-failure recording,
  resumption from a partial schedule, block-size independence to `1e-12`, an
  independent dense `hybr` root of the same residual, a self-generated OECD
  golden, and the shared CES kernel against the IO `ces_index` in a subprocess
  when the research volume is mounted.

Not validated: parity with the IO engine's equilibria (one composite factor,
DEK closures, exogenous inventories: a different model), convergence for any
elasticity profile beyond the fixtures above, welfare at nested technology,
GPU execution, LES preferences, markups, capacity penalties and non lump-sum
fiscal closures (the flexible module keeps those). The residual and certificate
thresholds are floating-point checks, not interval proofs.

Related: [consistent trade accounting](trade_accounting.md),
[Hicksian consumption welfare](trade_welfare.md),
[Hicksian tariff policy and solver recovery](trade_policy.md),
[structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
