> 🇬🇧 English · 🇪🇸 [Español](es/sovereign_extensions_validation.md)

# Sovereign extensions: independently checked equations

PureMacro 6.1.0 adds independent small-grid checks for long-term
debt, restructuring and the collateral planner. These complement the completed
[short-term Arellano study](arellano_replication.md); they do not replicate
published long-term or Mendoza empirical moments.

## Long-term debt and restructuring

`tools/reference_validation/sovereign_extensions.py` implements the value and
bond-price fixed point separately using NumPy/SciPy. At 17 debt by 5 income
states, maturity rates 0.2, 0.5 and 1 pass: maximum value error is below `1.8e-9`,
price error below `1.9e-14`, and modal debt policies match exactly. Both the
20% haircut and a 50% duration increase agree with independently evaluated
post-restructuring default probabilities within `7e-15`.
These comparisons disable imposed monotonicity and Howard acceleration;
a projected price derivative alone is not a structural dilution test.

For `delta < 0.999`, the existing solver smooths debt choices with Boltzmann
weights at `tau = min(0.005, max(0.0001, sigma_eps/2))`. The repayment value is
their weighted objective **without an entropy or taste-shock bonus**. This is
a quantal-policy approximation, not the deterministic optimal policy or the
inclusive value of a random-utility debt-choice model. It is distinct from
the separately modeled default/repayment taste shock. The independently
measured gap to the deterministic maximization is about 0.00528 and 0.00425
value units in the two long-term test cases and is recorded rather than hidden.

The returned `policy_b` remains the modal debt choice. Simulation, ergodic
spread diagnostics, recovery paths and survival-claim pricing now use the full
`metadata['debt_choice_probabilities']`. Short-term solves and simulations remain
bit-identical to the pinned 6.0.1 algorithm; long-term simulated moments change.

A haircut alone reuses the equilibrium and changes the inherited debt state.
A maturity change re-solves the same calibration with the new maturity rate.
Unconverged inputs/counterfactuals are rejected, period zero uses the selected
distress income, and debt-grid rounding is reported. These are imposed
counterfactuals, without bargaining or endogenous recovery negotiations.

For risk-neutral zero-recovery survival claims,
$q_h=S_h/(1+r^*)^h$. The legacy `risk_premium_component` column is the difference
between the discrete annual spread and the continuously compounded survival
loss. It is a compounding adjustment, not an identified risk or dilution premium.
Independent constant-hazard and full-choice transition recursions check pricing.

## Collateral planner and measured diagnostics

This is an illustrative adjustment-cost planner with
$b'\leq\kappa q k'$. It does not implement Mendoza's complete competitive model
with working-capital financing and imported inputs. An exhaustive reference
without Howard acceleration matches all debt, capital and consumption policies
exactly on the 7 × 6 × 3 grid at collateral shares 0.3 and 0.9. The first case
has binding states. Value errors are below `2.2e-9`, and final Bellman residuals
are below `1e-10`. Infeasible states are rejected instead of receiving artificial
positive consumption.

- `collateral_binding_fraction` is the fraction of grid states where collateral
  excludes the next higher debt node at the chosen capital. It is not a crisis rate.
- `crisis_probability` is measured after burn-in from 2,500 simulated periods:
  collateral binds, consumption falls and the current account increases.
  `simulate()` returns those observations and the relevant capacity/slack.
- Capacity is averaged as `kappa*q*k_next` over the path. It is not the product
  of separately averaged prices and capital. The buffer compares that capacity
  with average next-period debt.
- Amplification is the ratio of average mid-to-low-productivity consumption
  drops in the constrained and unconstrained solutions, evaluated at the same
  simulated debt/capital states and weighted by their visitation frequencies.
  A zero/increasing benchmark response or unconverged solution gives `NaN` and
  a status. No maximum-state selection, 1.35 fallback or imposed lower bound
  remains. The ratio may be below one; it is not a causal crisis elasticity.

These definitions make diagnostics reproducible. Economic grid convergence,
published moments and a full competitive collateral model remain separate work.

## Reproduce

Frozen parameters, source hashes, thresholds and errors are in
`tests/fixtures/sovereign_extensions/independent_validation.json`.

```bash
python -m tools.reference_validation.check_sovereign_extensions --output /tmp/sovereign-extensions.json
python -m pytest tests/test_sovereign/test_extension_validation.py -q
```

Notebook 73 is an exploratory interface tour with updated measurements. Its
coarse-grid output is not the independent reference or a policy calibration.
