# Fertility DSGE Blanchard-Kahn failure — diagnosis

**Stream:** fertility-dx · **Date:** 2026-07-19 · **Scope:** analysis only, no package changes.
**Environment:** conda numpy 2.4.6 / scipy 1.18.0 (OpenBLAS, arm64 mac), python 3.13.

## Verdict (one paragraph)

The model is **not indeterminate** and the QZ/Jacobian numerics are **not fragile**.
`solve_fertility()` fails Blanchard-Kahn because of a **mis-specified calibration
stage**: `solve_bgp()` runs Levenberg-Marquardt on a mutually inconsistent
13-equation system whose least-squares minimiser is **non-isolated** (the 13×13
BGP Jacobian at the LM endpoint has **two zero singular values**; residual norm
0.075). The LM endpoint therefore depends on floating-point arithmetic — which is
the entire OpenBLAS / Accelerate / manylinux story — and, on this machine, lands at
economically inadmissible parameters (`tau_n = -1.35`, `tau_b = -0.24`,
`barn = -13.77`, leisure `l_o = 2.71 > 1`). That point is then used directly as the
linearisation point even though the dynamic-model residuals there have norm
**1.48** (a steady state would give ~0): the port drops Dynare's `steady;` step.
At the intended calibration (the Bayesian `.mod` + `parameters.mat` values) with an
exactly solved steady state, the ported equations reproduce Dynare's `check`
eigenvalues to the printed 4 digits and BK **passes with margin 0.08 at every one
of 569 parameter points tested**. Fix the calibration stage (remedy R1 below) and
all solve-dependent tests pass deterministically on every LAPACK build.

## Established facts, reproduced here

`solve_fertility()` raises with `n_stable = 11` (expected 12). Companion moduli:
`[0 x7, 0.5, 0.595, 0.7807, 0.9, 1.0685, 1.1576, 1.9673, inf x10]` — identical to
the task brief.

## 1. Jacobian noise — exonerated

Same linearisation point, three differentiation schemes
(`output/02_roots_vs_h.csv`, `figures/02_roots_vs_h.pdf`):

| scheme | h | n_stable | max abs(mod - complex-step) |
|---|---|---|---|
| central | 1e-4 | 11 | 2.0e-07 |
| central | 1e-5 | 11 | 2.0e-09 |
| central (package default) | 1e-6 | 11 | 3.6e-10 |
| central | 1e-7 | 11 | 5.5e-09 |
| central | 1e-8 | 11 | 1.3e-08 |
| Richardson | 1e-4..1e-6 | 11 | <= 4.1e-10 |
| complex-step (exact) | — | 11 | 0 |

The eigenvalue structure is **h-invariant to >= 7 significant digits** across four
orders of magnitude of h; the classification never changes. Jacobian noise is not
the culprit, and analytic Jacobians would change nothing about the failure.

**Complex-step support:** the package `model_residuals` does **not** support
complex input — it allocates a float64 output array and numpy **silently discards
the imaginary part** (`ComplexWarning`), so a complex-step Jacobian through it
would be identically zero. The diagnosis used a verified local copy
(`scripts/common.py::model_residuals_cs`, max deviation from the package on real
inputs: exactly 0.0). If complex-step is ever wanted in the package, the one-line
enabler is allocating `R` with `np.result_type(z_lead, z, z_lag, eps, float)`.

## 2. Pencil structure — exactly as theory predicts, and clean

At the port's point (`output/01_run.log`, `01_pencil_eigs.csv`,
`01_singular_values.csv`):

- `svd(A) = [6.11, 2.505, 0, ..., 0]` → **rank(A) = 2** (consumption + fertility
  Euler rows only). `svd(C)` has 5 nonzero values → rank(C) = 5 (lag columns
  a, mun, ph, k, n).
- Predicted structure, confirmed by determinant interpolation (FFT contour, 64
  points): `deg det(A mu^2 + B mu + C) = n + rank(A) = 14`; `n - rank(C) = 7`
  roots at 0; `2n - 14 = 10` infinite companion eigenvalues.
- Finite spectrum = 7 zeros ∪ {rhon = 0.5, rhoa = 0.7807, rhop = 0.9} ∪ the
  **endogenous quartet {0.595, 1.0685, 1.1576, 1.9673}**. QZ and the
  QZ-independent determinant method agree to <= 1.4e-6 on the quartet.
- **|beta| clusters are separated by 16 orders of magnitude** (finite cluster min
  |beta| = 0.59, infinite cluster max |beta| = 3.8e-17, package threshold 1e-12,
  `figures/01_beta_clusters.pdf`). There is **no borderline (alpha, beta) pair**:
  nothing in `ordqz` is close to flipping.
- 200 random relative-1e-13 perturbations of (M1, M0): `n_stable = 11` in
  **200/200** trials. Companion-pencil eigenvalue sensitivities are modest
  (<= ~150). The QZ classification on this pencil is rock-solid.

**Where the cross-build nondeterminism actually lives:** perturbing the BGP
starting point `x0` by as little as **1e-10 relative noise** sends
`solve_bgp`'s LM to different endpoints with `tau_n` anywhere in [-10.2, +0.36]
and flips the verdict: over 59 draws, `n_stable ∈ {11 (x47), 12 (x11), 13 (x1)}`,
with the boundary root wandering over [0.932, 1.172]
(`output/02_bgp_basin.csv`, `figures/02_bgp_basin.pdf`). This reproduces, on one
machine, exactly the observed cross-platform behaviour — the Accelerate/py3.12
"pass" is a *different LM endpoint that happens to be weakly determinate*, not a
spurious QZ root, and CI-py3.11's differing 9th modulus (0.7192 vs 0.595) is a
third LM endpoint. Root cause of the nondeterminism: at the LM endpoint the BGP
Jacobian has singular values `[6.97, ..., 0.040, 2.3e-16, 3.8e-24]` — a **2-D
flat manifold of minimisers** of an inconsistent least-squares problem.

## 3. Which root is missing, and what is it economically?

Quadratic-eigenvector loadings (`output/03_loadings.csv`):

| root | loads on | identity |
|---|---|---|
| 0.5 | mun only | fertility-preference AR(1) (rhon) |
| 0.7807 | a (k secondary) | productivity AR(1) (rhoa = 0.94^4) |
| 0.9 | ph only | mortality AR(1) (rhop) |
| 0.595 | l_o, n, u, y, k mixed | fertility-block stable root |
| **1.0685** | **k dominant (0.49 n)** | **capital-accumulation root — the missing 12th stable root** |
| 1.1576 | k dominant | capital/consumption-Euler unstable root |
| 1.9673 | n dominant, b | fertility-Euler unstable root |

At the healthy Bayesian calibration the quartet is {0.684, **0.920**, 1.192,
1.549}: two stable (fertility-block 0.684, capital-block 0.920) as saddle-path
determinacy requires (2 endogenous states k, n × 2 Euler equations). At the
port's broken point the **stable capital root has been pushed across the unit
circle to 1.0685** — too few stable roots, i.e. explosive/no-solution, not
indeterminacy. Adjustment costs barely touch it (psik ±10%: delta ≈ 5e-5;
psin ±10%: delta ≈ 1e-7), so no adjustment-cost recalibration can rescue the
broken point — confirmed by the rescue sweeps (§4).

**Docstring/spec expectations vs reality.** The original `.mod` header says
"adjusted timing for children accumulation to satisfy the rank condition", and
both original Dynare runs verify the rank condition with 8 unstable eigenvalues
for 8 forward-looking variables (their 12-var listing has 5 predetermined:
a, mun, ph, k, n — same as the port). The port's module docstring correctly
documents rank(A) = 2 and the companion approach; what it mis-states is that the
"BK check ... still verifies" anything meaningful at a non-equilibrium
linearisation point built from inadmissible parameters.

**Equation-by-equation validation (the equations are NOT mis-ported).** With the
original Dynare parameter values and an exactly solved steady state
(`scripts/common.py::exact_steady_state`, residuals <= 3e-17), the ported
residual functions reproduce the logged Dynare `check` spectra
(`output/03_dynare_replication.csv`):

| model | Dynare finite moduli | this diagnosis | max diff |
|---|---|---|---|
| `fertility_adj_costs.mod` | 0.1, 0.5844, 0.8889, 0.9, 0.96, 1.254, 1.856 | same | 2.9e-4 (Dynare print precision) |
| `..._bayesian_estimation.mod` + `parameters.mat` | 0.5, 0.6838, 0.7807, 0.9, 0.9197, 1.192, 1.549 | same | 4.3e-4 |

**Ablation** from the BK-passing Bayesian calibration toward the port's
conventions (`output/03_ablation.csv`): `barp: log(1/1.03) → 0`,
`omega: 2 → 2.5`, `g: 0.0175 → 0.017`, all three at once, and even *skipping the
exact SS and linearising at the raw `parameters.mat` initval point* — **every
variant keeps n_stable = 12 with margin >= 0.069**. None of the constant-value
discrepancies matters. Conversely, no steady state in a sensible basin exists at
the port's LM parameters at all (negative time costs). The failure is caused by
the LM endpoint itself, full stop.

## 4. Parameter region (`output/04_*.csv`, `figures/04_sweep_1d.pdf`, `figures/04_heatmap.pdf`)

Around the healthy calibration, exact SS re-solved at every point, robustness =
(QZ n_stable = 12) AND (min abs(|mu| - 1) > 1e-3) AND (classification identical
for central h = 1e-6 and complex-step):

- 1-D sweeps (16 points each): beta ∈ [0.85, 0.995], psik/psin ∈ [0.25, 10],
  omega ∈ [1.15, 3.5], delta_k ∈ [0.06, 0.30], rhoa/rhon/rhop ∈ [0.05, 0.99] —
  **n_stable = 12 at 128/128 points**. Influence ranking by boundary-root range:
  beta (0.098) > psik (0.077) > omega (0.058) >> rhon (0.008) > delta_k (0.004) >
  psin (1e-4) > rhoa ≈ rhop (0). The shock persistences (the 0.5 / 0.9 values)
  only relabel their own AR roots; they cannot affect determinacy because the
  shock block is exogenous (block-triangular).
- 2-D grid over the two most influential (beta × psik, 21×21):
  **441/441 points ROBUSTLY determinate**. The heat map (`04_heatmap.pdf/.png`)
  is uniformly green with margin ≈ 0.07-0.13; the determinacy boundary is
  nowhere near any plausible calibration.
- Rescue sweeps at the port's frozen LM point (what
  `solve_fertility(params={...})` can reach today): beta, omega, psik, psin over
  wide ranges — **n_stable = 12 at 0/80 points**. The broken point cannot be
  dialled back to determinacy.

## 5. Verdict and remedies

**(i) genuinely indeterminate?** No — at the intended calibration the model is a
textbook saddle: robustly determinate over the entire plausible region.
**(ii) determinate but numerically fragile?** Not in the Jacobian/QZ sense —
h-invariant spectra, 16-orders-of-magnitude cluster separation. The *fragile*
stage is `solve_bgp`'s LM on an inconsistent system with a 2-D flat minimiser
manifold. **(iii) mis-specified in one equation?** The 12 dynamic equations are
verified correct against Dynare. The mis-specification is in the **calibration
stage of `solve_fertility`**: treating the LM least-squares point of the
over-determined BGP system as (a) a parameter calibration and (b) a steady
state, without admissibility checks and without ever solving the model's own
steady state (Dynare's `steady;` step, silently dropped in the port).

### Remedies, prioritized

**R1 (do this): pin the calibrated parameters and solve the exact steady state
before linearising.**
1. Freeze the seven calibrated parameters to the original `parameters.mat`
   values (already validated above): `barn = log(1.587983)`,
   `mu_l = 0.634818`, `beta = 0.936122`, `tau_n = 0.213943`,
   `tau_b = 0.469184`, `p_n = 0.094621`, `delta_k = 0.144`, together with the
   Bayesian-variant constants `barp = log(1/1.03)`, `omega = 2.0`,
   `g = 0.0175`.
2. Solve `model_residuals(z, z, z, 0) = 0` exactly for z (a, mun, ph are pinned
   by the AR means; the remaining 9 equations in 9 unknowns solve by
   `scipy.optimize.root(method="hybr")` from the `.mod` initval in
   milliseconds to 1e-17; accept on residual, not on hybr's progress flag).
   Reference SS: c 0.266225, k 1.033825, i 0.114395, u 1.137443, n 1.066294,
   b 0.067290, l_o 0.474782, l_w 0.265520, y 0.481514.
3. Linearise there (central differences at h = 1e-6 are already fine).
   Verified end-to-end in this diagnosis with the package's own
   `_solve_matrix_quadratic`: BK passes, `max|eig(P)| = 0.9197`,
   y-response to `ea` = +0.644 impact / +0.273 at h = 4, n-response to `ep`
   negative at h = 0..4 — i.e. **all solve-dependent tests
   (5 in `test_solve_fertility.py`, 3 in `test_fertility_irf_fevd.py`, the demo
   smoke test, and the remaining solve-dependent CI failure) pass with margins
   ~0.08, five orders of magnitude above any observed build-to-build noise**.
   The loose BGP tests (`test_fertility_bgp.py`: residual norm < 10, targets
   within ±0.1) remain green if `solve_bgp` is kept for reference or replaced
   by the pinned dict.

**R2 (if the least-squares calibration must stay): make it well-posed and
admissible.** Use `least_squares(method="trf")` with bounds
(`tau_n, tau_b, p_n, delta_k, c, b, l_w, u, k, n > 0`, `0 < beta < 1`,
`0 < l_o < 1` via a reformulated residual), a fixed deterministic multistart,
and a diagnostic error when the solution violates admissibility or when
`J'J` at the endpoint is rank-deficient (both conditions are the smoking guns
found here). Then **still** re-solve the exact steady state (R1 step 2) —
a least-squares compromise point is never a steady state, and linearising off
the manifold moved the capital root by +0.15.

**R3 (hygiene, optional): deflate the 10 infinite roots analytically.** Not
needed for correctness (the clusters are 16 orders apart), but it makes the
count `n_stable = 12` structural rather than threshold-based. Since
`A = E F` with `E = [e_8, e_9]` (Euler rows) and `F = A[7:9, :]` (2×12), define
`s_t = F z_{t+1}` and stack `x_t = [z_t; s_t]` (14-dim). The pencil

```
lambda [ B   E ]   [ C  0 ]
       [ -F  0 ] + [ 0  I ]  = 0
```

has, by the Schur complement on the identity block,
`det(lambda [B E; -F 0] + [C 0; 0 I]) = det(lambda^2 A + lambda B + C)` — a
**14×14 pencil whose 14 eigenvalues are exactly the finite spectrum** (7
structural zeros + 3 shock AR roots + the endogenous quartet), with no infinite
eigenvalues at all. BK then reads: count stable among 14, expect 12, with the 10
infinite roots removed by construction rather than by an |beta| > 1e-12 cut.

### Why the same code "passes" on Accelerate / py3.12

Not a spurious 12th stable root in QZ (no borderline (alpha, beta) pair exists;
the pencil classification survives 1e-13 perturbations 200/200). Different
LAPACK builds make scipy's LM take a different path across the 2-D flat
minimiser manifold of the inconsistent BGP system; some endpoints happen to sit
on the determinate side (11/59 in the x0-perturbation experiment), some are even
indeterminate (n_stable = 13 observed once). Any green CI obtained this way is
luck, not correctness — the "passing" Accelerate solution is itself built on an
economically inadmissible calibration and a non-equilibrium linearisation point.

## Reproduction

```
cd puremacro
python docs/research/fertility_bk_diagnosis/scripts/01_repro_and_pencil.py
python docs/research/fertility_bk_diagnosis/scripts/02_jacobian_noise.py
python docs/research/fertility_bk_diagnosis/scripts/03_root_identity.py
python docs/research/fertility_bk_diagnosis/scripts/04_param_grid.py
```

Each script writes CSVs to `output/` and figures to `figures/`; full console
logs of the runs behind this report are in `output/0?_run.log`.

External evidence consulted (read-only): `My Drive/Fertility/code/paper1_housing/
matlab/fertility_adj_costs.mod` and `bgp_fertility_calibration.m`,
`My Drive/Fertility/.worktrees/multi-shock-plan3/fertility_adj_costs_bayesian_
estimation.mod`, `My Drive/Fertility/output/data/parameters.mat`, and the
archived Dynare logs (`_archive/fertility_adj_costs*.log`) containing the
`STEADY-STATE RESULTS` and `EIGENVALUES` blocks quoted above.
