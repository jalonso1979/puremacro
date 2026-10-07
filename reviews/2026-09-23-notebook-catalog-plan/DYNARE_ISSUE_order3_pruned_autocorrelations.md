# Draft issue for the Dynare tracker

Target: <https://git.dynare.org/Dynare/dynare/-/issues>. Attach
`dynare_order3_autocovariance_fix.diff` (in this folder). It patches
`matlab/+pruned_SS/pruned_state_space_system.m`. Everything below the line is the
proposed issue text.

Pre-posting checks (re-verified 2026-10-02; not posted):

- **The diff applies.** `git apply --check` succeeds against the installed Dynare 8
  snapshot `8-2026-05-26-1803` and Dynare 7.0. It fails against 6.4 and the
  `7-2025-11-07-1744` snapshot: the `end` line at 1106 that the diff removes carries
  trailing whitespace there. `patch -l -p1 --dry-run` applies on both.
- **Dynare `master` was not checked.** git.dynare.org served an HTML page instead of
  the raw file to a script. Before posting, open the file on `master` and confirm that
  the line numbers and the hunks still match.
- **The tables match.** Every number in the tables below equals
  `tests/fixtures/dynare_order3_pruned_moments.json`, which holds live Dynare 8
  output. The closed form was recomputed independently from the formulas in the
  text. The "patched" rows equal puremacro's exact moments
  (`tests/test_dsge_order3_moments.py`, 29 passed).
- **Not re-run:** the patched Dynare function itself. That needs MATLAB; the
  2026-09-23 session ran it.
- **Derivative caveat.** The derivative caveat in "Proposed fix" was added after
  checking `+identification/get_jacobians.m`. It calls the function with
  `compute_derivs=1`.

---

## Title

Order-3 pruned autocovariances (`pruned_state_space_system`) omit the correlation of `kron(kron(xf(-1),u),u)` with past innovations

## Summary

At `order=3` with `pruning`, the theoretical autocorrelations that
`pruned_SS.pruned_state_space_system` returns (`Var_yi`, `Corr_yi`, hence
`oo_.autocorr`) are wrong at every lag of 1 or more. The means (`E_y`) and the
contemporaneous covariance (`Var_y`) are right. In a model with a closed-form
solution the reported lag-1 autocorrelation is 0.7085; the true value is
0.7634. Dynare's own long simulation of the same pruned system also disagrees
with its table. Centring one innovation on its conditional mean fixes it; the
patch is attached, and the patched function reproduces the closed form to
machine precision.

## Affected

- Versions: Dynare 8 snapshot 2026-05-26 (`8-2026-05-26-1803`, MATLAB R2026a),
  which we ran. The same code (lines 805-815 and 1030-1111, identical up to
  trailing whitespace) is in 6.4 and 7.0; we did not run those. The diff applies
  with `git apply` to the 8 snapshot and to 7.0, and with `patch -l -p1` to 6.4.
- Consumers of `Var_yi`/`Corr_yi` (and `dVar_yi`/`dCorr_yi`) at order 3:
  - `stoch_simul(order=3, pruning)`: the printed autocorrelation table and `oo_.autocorr`
    (`moments/disp_th_moments_pruned_state_space.m`);
  - identification at order 3 (`+identification/get_jacobians.m`, `numerical_objective.m`);
  - posterior theoretical moments with pruning (`estimation/dsge_simulated_theoretical_covariance.m`);
  - `method_of_moments` GMM at order 3 when autocovariances are matched (`+mom/objective_function.m`).
- Not affected: order 1 and order 2, and at order 3 the means and variances.

## Minimal example with a closed form

```
var y x; varexo e; parameters beta rho; beta=0.95; rho=0.8;
model; x = rho*x(-1) + e; y = beta*y(+1) + x^3; end;
initval; x=0; y=0; end;
shocks; var e; stderr 0.1; end;
stoch_simul(order=3, pruning, irf=0, ar=5);
```

`y` is the price of a claim to the payoff `x^3`. Its exact solution is
`y = A x^3 + B x` with `A = 1/(1 - beta rho^3)` and
`B = 3 beta A rho s^2/(1 - beta rho)`. Because `x` is linear, the third-order
pruned solution reproduces it exactly, so the pruned moments are the true
moments. With `x` Gaussian, `v = s^2/(1 - rho^2)` and `r = rho^k`, Isserlis'
theorem gives

```
Var(y)              = 15 A^2 v^3 + 6 A B v^2 + B^2 v
Cov(y_t, y_{t-k})   = A^2 v^3 (9 r + 6 r^3) + 6 A B v^2 r + B^2 v r
```

| Autocorrelation of `y`, lag | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Closed form | 0.7634 | 0.5920 | 0.4640 | 0.3663 | 0.2905 |
| Dynare (`oo_.autocorr`) | 0.7085 | 0.5199 | 0.3919 | 0.3013 | 0.2347 |
| Dynare with the attached patch | 0.7634 | 0.5920 | 0.4640 | 0.3663 | 0.2905 |

The variance agrees: 0.0038365 from the formula, 0.0038 printed.

## Dynare's own simulation disagrees with its table

```
var y z x; varexo e v;
parameters rho beta; rho=.7; beta=.95;
model; x=rho*x(-1)+e; z=.5*z(-1)+v;
y=beta*y(+1)+(x+2*z)^3+.2*x*z; end;
initval; x=0;z=0;y=0;end;
shocks;var e;stderr .1;var v;stderr .2;corr e,v=.2;end;
```

| Autocorrelation of `y`, lag | 1 | 2 | 3 |
|---|---|---|---|
| `stoch_simul(order=3, pruning)`, theoretical | 0.3299 | 0.1553 | 0.0863 |
| `stoch_simul(order=3, pruning, periods=4000000, drop=1000)`, `set_dynare_seed(20260923)` | 0.4750 | 0.2550 | 0.1450 |
| Theoretical with the attached patch | 0.4751 | 0.2546 | 0.1441 |

The simulated variance matches the theoretical one (0.7881 against 0.7883).

## Cause

At order 3 the function writes the pruned system as

```
z_t = c + A z_{t-1} + B inov_t,        y_t = ys + d + C z_{t-1} + D inov_t,
```

with `inov6_t = kron(kron(xf_{t-1}, u_t), u_t)` among the innovations. Its
conditional mean is `kron(xf_{t-1}, vec(Sigma_e)) ≠ 0`. The code accounts for the
resulting correlation between `inov_t` and `z_{t-1}` through `E_inovzlag1` and
`E_inovzlagi`. `inov6` is also correlated with *earlier innovations*:
for `s > t-i`, `E[inov6_s inov_{t-i}'] = kron(E[xf_{s-1} inov_{t-i}'], vec(Sigma_e))`,
which is nonzero because `xf_{s-1}` loads on `u_{t-i}`. The exact autocovariance is

```
Cov(y_t, y_{t-i}) = C Cov(z_{t-1}, z_{t-1-i}) C' + C E[z_{t-1} inov_{t-i}'] D'
                  + D E[inov_t z_{t-1-i}'] C' + D E[inov_t inov_{t-i}'] D'.
```

In lines 1089-1106, `C*Ai*tmp` computes the second term as
`C A^(i-1) E[z_{t-i} inov_{t-i}'] D'`. That propagation assumes the innovations
dated `t-i+1, ..., t-1` are uncorrelated with `inov_{t-i}`. The last term,
`D E[inov_t inov_{t-i}'] D'`, is left out. Both omitted pieces come from `inov6`.
At lag 0 the formula at lines 1039-1042 is complete, which is why `Var_y` is
right. Removing exactly these two pieces from an exact implementation
reproduces Dynare's `Corr_yi` to 5e-14 on six test models (the two above and four
others).

## Proposed fix

Centre `inov6` on its conditional mean: use
`kron(xf_{t-1}, kron(u_t,u_t) - vec(Sigma_e))` as the innovation and move
`kron(xf_{t-1}, vec(Sigma_e)) = kron(eye(x_nbr), E_uu(:)) * xf_{t-1}` into the
`xf` columns of `A` and `C`. Every innovation then has zero conditional mean,
so the innovations are uncorrelated with `z_{t-1}` and with each other at
different dates. `E_inovzlag1` becomes zero, and the order-2 autocovariance formula
`Var_yi = C A^(i-1) (A Var_z C' + B Varinov D')` becomes exact at order 3. The
process is unchanged, so `E_y`, `Var_z` and `Var_y` are unchanged. The only
other change is the `inov6` block of `Varinov`, which becomes
`kron(E_xfxf, E[kron(u,u) kron(u,u)'] - vec(Sigma_e) vec(Sigma_e)')`.

The attached patch changes lines 810-814 and 1089-1106. We checked it by calling the
patched function (with `compute_derivs=0`) on the `oo_.dr` of the six models
after `stoch_simul(order=3, pruning)`:

- `E_y` identical and `Var_y` within 1.1e-16 of the unpatched function;
- `Corr_yi` equal to the closed form above and to an independent implementation
  within 5e-14.

The `compute_derivs` branch builds `dA`, `dB`, `dC`, `dD`, `dVarinov` and
`dE_inovzlag1` for the uncentred system and needs the same change:

- for each parameter `jp`, add
  `dB(:,id_inov6_xf_u_u,jp)*kron(eye(x_nbr),E_uu(:)) + B(:,id_inov6_xf_u_u)*kron(eye(x_nbr),vec(dE_uu(:,:,jp)))`
  to the `xf` columns of `dA(:,:,jp)`, and the same with `dD`/`D` to the `xf` columns of `dC(:,:,jp)`;
- replace the `inov6` block of `dVarinov` with the derivative of the centred block;
- set `dE_inovzlag1` to zero.

We did not patch or test that branch. **The attached diff should not be merged
without it.** With `compute_derivs=1`, which order-3 identification uses
(`+identification/get_jacobians.m`), the patched `A`, `C` and `E_inovzlag1` would be
combined with the uncentred `dA`, `dC` and `dE_inovzlag1`. That would make `dVar_z`
and `dVar_y` inconsistent, and those two are correct today. The diff is meant to
demonstrate the fix for `compute_derivs=0`. Once the derivative branch is changed,
`E_inovzlagi`, `dE_inovzlagi` and the order-3 branches of the `Var_y` and `Var_yi`
computations (and their derivatives) can be removed.

## Context

We found this while validating an independent implementation of the pruned
third-order moments (puremacro, Python), which centres the innovation as above.
Its means and variances match Dynare's to 2e-13 on six models, and its
autocorrelations match the closed form, Dynare's own long simulations and the
patched function.
