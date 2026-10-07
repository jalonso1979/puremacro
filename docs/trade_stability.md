> 🇬🇧 English · 🇪🇸 [Español](es/trade_stability.md)

# Reduced local stability of a trade equilibrium (experimental)

`reduced_stability` asks one narrow question about a converged trade equilibrium: if relative factor prices moved in response to their own excess demand while producer prices, gross outputs and national fiscal budgets cleared instantly, would a small displacement die out or grow? The answer is a classification (`stable`, `unstable`, `nonhyperbolic` or `degenerate`) of the eigenvalues of a reduced Jacobian, together with the checks that justify it. It is evidence about that one adjustment process. It is never a uniqueness result, and every `StabilityResult` carries the IO qualification text in `qualification`.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.stability import reduced_stability

# Two countries, one sector, no intermediates, no taxes, balanced trade.
# Rows: goods A, goods B, taxes, labour, capital; columns: 2 intermediate, 2 final uses per country.
y0, alpha = np.array([100., 150.]), np.array([.3, .4])
F = np.array([[60., 40.], [40., 110.]])          # origin x destination final deliveries
data = np.zeros((5, 6))
data[3, :2], data[4, :2] = (1-alpha)*y0, alpha*y0
for c in range(2):
    data[:2, 2+2*c:4+2*c] = F[:, c][:, None]*np.array([.7, .3])[None, :]
calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-12)

report = reduced_stability(calib, base)
print(report.summary())
print(report.to_dataframe())
print(report.maximum_real_eigenvalue)         # 0.2222... = 2/9, the closed-form unstable root
print(report.qualification)
```

The reduced Jacobian of this toy has a closed form (the test file derives it), and its unstable root is exactly `2/9`. The instability is economic, not numerical: with fixed physical baskets, demand for a country's good is price inelastic, so raising its factor prices raises its revenue and its excess factor demand.

## What is computed

Write the physical residuals of the model as `r(x)` in the puremacro unknown layout `x = [log p; log y; log r; log w; T; XN]` (`n = ns*nc` cells, `nc` countries). The unknowns are split into a slow block `z = (log r, log w)` and a fast block `u = (log p, log y, T)`; the foreign balances `XN` are held fixed. The rows are split into the factor rows `f` (labour and capital gaps) and the fast rows `g` (goods clearing, zero profit, fiscal budgets). The current-account rows are not imposed on the fast block: they certify the converged state and nothing else.

At every factor-price vector the fast block is cleared, `g(u(z), z) = 0`, and the excess factor-demand ratios are `e(z) = -f(u(z), z) / endowment` (positive means demand above endowment). Their Jacobian is the Schur complement

```
J_F = f_z - f_u g_u^{-1} g_z,        A = -J_F / endowment
```

with rows ordered (capital, labour) to match the columns (log r, log w); `factor_labels` gives the order `r[c0], ..., w[c0], ...`. `A` is built from a central finite-difference Jacobian of `puremacro.trade._accounting.evaluate` (consistent results) or `puremacro.trade.equilibrium.evaluate_equilibrium_residuals` (legacy results). Complex step is unavailable because the state unpacker casts to float. The steps on the level unknowns (`T`, and `XN` in the naive partition below) are scaled by national income, because those unknowns enter the model linearly in calibration units and an unscaled step would drown their effect in roundoff.

Foreign balances are re-denominated so that the map is homogeneous of degree zero in all factor prices, `A 1 = 0`. The default `balance_units="world_factor_income"` fixes them in units of `sum_k (w_k L_k + r_k K_k)`, which is the numeraire of the IO reduced-stability code. Under the consistent accounting the identity

```
p.goods + (1 - tax) y.(pp - p) + w.labour_gap + r.capital_gap + 1.budget = 0
```

holds at every `x`, so on the fast-cleared manifold the value-weighted excess factor demand `W.e` vanishes (`W = (r K, w L)`), and differentiating it gives `W A + W*e = 0`, which is `(r K, w L) A = 0` at an equilibrium. The IO code restores one dropped budget row from the same identity; puremacro evaluates every budget row directly and reports the identity as a check (`metadata['world_identity_error']`).

The relative law `d log(f_i / f_ref) / dt = e_i - e_ref` for every factor price other than the reference (default: the wage of country 0) gives the `(2 nc - 1)`-dimensional matrix `R = A[keep, keep] - A[ref, keep]`. Because `A 1 = 0`, every reference yields a similar `R`, so the spectrum does not depend on the reference. Optional country-factor `speeds` replace `A` by `diag(speeds) A` in the law, so the spectrum of `R` is then that of `diag(speeds) A` without its zero.

The optional dense check (`dense=True`, the default) recomputes `A` independently: it re-solves the fast block at perturbed factor prices by chord iteration (at most 30 iterations, stopping when goods rows are below `1e-12 max(1, max y)`, zero-profit rows below `1e-12 max(1, max p)` and budget rows below `1e-12 max(1, max income)`) and differences the resulting excess-demand ratios. This differs from the IO code, which eliminates the fast unknowns from a complex-step Jacobian of the full system: the re-solve checks the nonlinear fast-cleared map itself rather than repeating the same linear elimination.

Classification, with sign band `tol` and rank threshold `rank_tol` (both 1e-7 by default):

| outcome | condition |
|---|---|
| `degenerate` | `R` is numerically rank deficient: singular values at or below `rank_tol * max(1, largest)` (a structural null space, or a fold) |
| `unstable` | `max Re eig(R) > tol` |
| `stable` | `max Re eig(R) < -tol` |
| `nonhyperbolic` | otherwise |

The naive partition that keeps the current-account rows in the fast block together with `XN` (`current_account="fast"`) is structurally degenerate: its reduced matrix has rank `nc`, whatever the units of the table. The diagnostic reports this instead of a sign.

## Closures and the result fields

| argument | meaning |
|---|---|
| `balance_units` | `"world_factor_income"` (default, IO numeraire, reference-invariant spectrum), `"reference_factor_price"` (the assessment prototype's convention; homogeneous, but the spectrum moves with the reference whenever balances are nonzero, by 3.5e-2 on the synthetic fixtures), `"numeraire"` (the model's own closure, balances fixed in numeraire units; not homogeneous, reported with a warning). Not applicable with `current_account="fast"` |
| `reference`, `reference_factor` | country and factor (`"w"` or `"r"`) of the reference price |
| `current_account` | `"certificate"` (default) or `"fast"` (naive, degenerate) |
| `tau`, `tau_fd`, `tauf`, `tauf_fd` | tariff schedules; consistent results carry theirs in `metadata` and explicit values must agree, legacy results carry none and need them |
| `speeds` | optional positive adjustment speeds, length `2 nc` |
| `step`, `fd_order` | finite-difference step for log unknowns, at most 1e-2 (levels `T` and `XN` are scaled by income; they enter linearly), and stencil order 2 or 4 |
| `dense`, `dense_step`, `dense_tol` | independent check by re-solving the fast block at perturbed factor prices (`dense_step` at most 1e-2); raises `StabilityError` on disagreement |
| `tol`, `rank_tol` | sign band for the classification and relative singular-value threshold for the rank, set separately so that a wide sign band does not turn ordinary equilibria into degenerate ones |
| `homogeneity_tol`, `equilibrium_tol` | threshold for the homogeneity and Walras identities, and for accepting the state as an equilibrium of its own equations |

Where an identity is exact for the evaluated map, its failure is an error, not a caveat. Under the consistent accounting homogeneity is exact (with `balance_units` other than `"numeraire"`, or in the naive partition) and so are the Walras and world identities; a violation beyond `homogeneity_tol` (world identity: 1e-10) can only come from finite-difference error, for example a step that is too small or too large, and raises `StabilityError`.

`StabilityResult` exposes `classification`, `maximum_real_eigenvalue`, `eigenvalues` (sorted by real part), `reduced_jacobian` (`A`), `relative_matrix` (`R`), `rank`, `homogeneity_error` (`max|A 1| / max|A|`), `walras_error` (`max|W A + W*e| / max|W_i A_ij|`, relative), `fast_block_condition` (after row and column equilibration, because the raw block mixes log unknowns with transfer levels in calibration units), `equilibrium_residual` and `dense_check_error` (both absolute), the `adjustment`, `closure` and `qualification` texts, `reference`, `factor_labels`, and `metadata` (eigenvectors, singular values, absolute spectrum of `A`, the world-identity check, the partition and all tolerances; its arrays are read-only). `to_dataframe()` lists the eigenvalues; `mode_loadings(k)` gives the right eigenvector of mode `k` on the factor-price labels (zero at the reference); `to_markdown()`, `to_latex()` and `to_typst()` render the table.

## Legacy results

The diagnostic runs on legacy results, but the legacy accounting is not homogeneous of degree one: production taxes are levied on quantities and the default `replicate_matlab_precedence=True` cost formula is homogeneous of degree `1 + alpha`. The reduced map is then not homogeneous of degree zero, the relative reduction is conditional on the reference price, and `reduced_stability` issues a `RuntimeWarning`, sets `metadata['homogeneous'] = False` and appends the caveat to `closure`, so `summary()` shows it next to the verdict; `metadata['absolute_eigenvalues']` holds the spectrum of the full map. With zero production taxes, zero final-use taxes and `replicate_matlab_precedence=False`, legacy and consistent results give the same reduced Jacobian to 1.3e-10.

## What the tests establish and what they do not

Verified in `tests/test_trade_stability.py` (tolerances stated there; observed values in parentheses):

- closed-form reduced Jacobian of the two-country toy for all three balance denominations and both references, atol 1e-8 (2.3e-10 with the default second-order step 1e-6; 1.8e-13 with `fd_order=4`, `step=1e-3`);
- Schur complement versus an independent `fsolve` re-solve of the fast block on the two-country fixture, 1e-8;
- the built-in chord re-solve check on every fixture, 1e-8 (1.1e-9 to 2.8e-9);
- homogeneity and the value-weighted Walras identity, 1e-8 (below 2e-9), also at an off-equilibrium point of the fast-cleared manifold where `W A` alone is not zero;
- reference invariance of the spectrum under world factor income units (1.8e-10) and its documented dependence under reference factor price units;
- adjustment speeds: 8 random draws in [1/4, 4] on the three-country fixture, where the relative spectrum equals the spectrum of `diag(speeds) A` without its zero (1e-8; observed at most 2.2e-10) and the classification stays `unstable`;
- mode loadings are right eigenvectors of the law modulo the constant vector (1e-8; observed 5e-16);
- degenerate naive partition (rank `nc`) on both synthetic fixtures, on the three-country table with flows multiplied by 1e6 and on the OECD 3x3 fixture in million USD; `A` itself is unchanged by that rescaling (1e-8);
- the sign band and the rank threshold act separately;
- eigenvalue continuity across 1e-3 and 1e-2 tariff changes;
- Euler tatonnement on the nonlinear fast-cleared map: the excess-demand norm falls along the stable eigenvector and grows along the unstable one;
- legacy versus consistent agreement when the accountings coincide (1.3e-10) and the MATLAB-precedence discrepancy (0.19);
- the OECD 3x3 fixture, the Hicksian two-country CES model and a slow-marked bundled 77x11 legacy run (about 50 seconds for the finite-difference Jacobian).

Observations recorded by those tests, all closure dependent: the two synthetic puremacro fixtures classify `unstable` under fixed-coefficient sourcing (max Re +0.245 and +0.239); the maximum real eigenvalue falls monotonically with the intermediate substitution elasticity and turns negative at `sigma = 1` (2c2s: +0.245, +0.114, -0.016, -0.278 at sigma 0, 0.5, 1, 2); the OECD 3x3 fixture does the same under the consistent accounting (+0.145, +0.050, -0.022 at sigma 0, 0.5, 1); it is `stable` under the legacy accounting with a 10 percent USA tariff (-0.115, homogeneity error 0.6), so the same table yields different verdicts under different accountings.

Not established, quoting the IO documentation: "A local stability result is conditional on the specified relative-wage adjustment process, not a uniqueness theorem" (README) and "Numerical verification does not establish empirical identification, global uniqueness, or stability under every possible adjustment process" (VALIDATION.md). The classification is never a uniqueness certificate, says nothing about basins of attraction, and does not estimate how factor prices actually adjust. Only the lump-sum fiscal closure without capacity margins or revenue recycling is supported. The finite-difference Jacobian costs `2 (2 n + 3 nc)` residual evaluations, which is fine at the bundled 77x11 size and impractical at 45-sector native sizes without an analytic Jacobian.

## Related

- [Consistent trade accounting](trade_accounting.md) for the residual layout the diagnostic differentiates.
- [Hicksian consumption welfare](trade_welfare.md) and [Hicksian tariff policy](trade_policy.md) for the equilibria this diagnostic is applied to.
- [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md) for the validation row.
