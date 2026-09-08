> 🇬🇧 English · 🇪🇸 [Español](es/replication.md)

# Replication Gallery

> Distinct from `puremacro.validation` (which guards against algorithmic drift relative to reference software packages), `puremacro.replication` is designed to scientifically reproduce published empirical and structural headline findings from peer-reviewed academic literature.

Every replication case is declared as a self-contained, frozen `ReplicationCase` running deterministically in the browser under the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero network access and executing in sub-second time.

## Running the Replication Suite

```python
from puremacro.replication import run_all, scorecard

# Run all offline replication cases and display the scorecard
df = scorecard()
print(df[["id", "family", "paper", "target_kind", "tol", "passed", "margin"]])

# Assert 100% reproduction of published findings
assert df["passed"].all()
```

---

## Replication Families

### 1. Bayesian DSGE Estimation (`dsge_estimation`)

Replicates headline empirical results from the seminal Smets & Wouters (2007) medium-scale DSGE model of the US economy (AER 97(3):586–606):

- **`dsge_estimation.sw07_log_posterior_at_mode`**
  - **Target**: Exact negative log-posterior value of `-1673.72` at the posterior mode on 1966Q1–2004Q4 US data (155 quarters, 7 observables).
  - **Methodology**: Evaluates the full Kalman log-likelihood initialized from the unconditional Lyapunov stationary covariance ($P_0 = T P_0 T' + R Q R'$) with 36 informative priors.
  - **Tolerance**: `Tol.TIGHT` ($\text{rtol} \le 0.02$).

- **`dsge_estimation.sw07_laplace_marginal_data_density`**
  - **Target**: Laplace asymptotic marginal data density of `-1686.09`.
  - **Methodology**: Evaluates $\log p(Y) \approx \log p(Y, \theta^*) + \frac{d}{2}\log(2\pi) + \frac{1}{2}\log|\Sigma^*|$ using the inverse Hessian at the mode.
  - **Tolerance**: `Tol.TIGHT`.

- **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`**
  - **Target**: Geweke (1999) modified harmonic mean MDD mean of `-2524.36` with spread across truncation levels $\le 2.20$ log points.
  - **Methodology**: Evaluates parameter draws reweighted by truncated Gaussian kernels across truncation levels $p \in [0.1, 0.3, 0.5, 0.7, 0.9]$, testing invariance to truncation.
  - **Tolerance**: `Tol.TIGHT`.

- **`dsge_estimation.sw07_structural_parameters_mode`**
  - **Target**: Table 1 published mode parameters: habit persistence $\lambda = 0.71$, intertemporal elasticity $\sigma_c = 1.38$, investment adjustment cost $S'' = 5.74$, Calvo price stickiness $\xi_p = 0.66$, wage stickiness $\xi_w = 0.70$, fixed costs $\Phi = 1.60$, Taylor rule inflation $\phi_\pi = 2.04$, interest rate smoothing $\rho = 0.81$, and trend growth $\bar{\gamma} = 0.43$.
  - **Tolerance**: `Tol.COARSE` ($\text{rtol} \le 0.25$, reflecting numerical mode-finding neighborhood stability).

---

### 2. Pure-NumPy Econometric Regressions (`regression`)

Replicates foundational micro- and macro-econometric findings using puremacro's zero-dependency, pure-NumPy regression subsystem (`puremacro.regress`):

- **`regression.card1995_iv_vs_ols`**
  - **Paper**: Card, D. (1995), *"Using Geographic Variation in College Proximity to Estimate the Return to Schooling"*.
  - **Findings**: Replicates the classical return to education: OLS coefficient $\beta_{\text{educ}} = 0.0740$ vs Two-Stage Least Squares (2SLS) IV estimate $\beta_{\text{educ}} = 0.1323$ instrumented by county-level 4-year college proximity (`nearc4`).
  - **Tolerance**: `Tol.TIGHT`.

- **`regression.long_ervin2000_hc_hierarchy`**
  - **Paper**: Long, J. S. and Ervin, L. H. (2000), *"Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model"*, The American Statistician 54(3):217–224.
  - **Findings**: Verifies the strict monotonic standard error hierarchy under heteroskedasticity with high leverage points:
    $$\mathrm{SE}_{\text{OLS}} < \mathrm{SE}_{\text{HC0}} < \mathrm{SE}_{\text{HC1}} < \mathrm{SE}_{\text{HC2}} < \mathrm{SE}_{\text{HC3}}$$
  - **Target Kind**: `TargetKind.SIGN` (positive differences at each tier).

- **`regression.mroz1987_logit_participation`**
  - **Paper**: Mroz, T. A. (1987), *"The Sensitivity of an Empirical Model of Married Women's Hours of Work"*, Econometrica 55(4):765–799.
  - **Findings**: Binary Logit model of female labor force participation (`inlf`): log-likelihood of `-401.77`, intercept `0.4255`, young children effect (`kidslt6`) `-1.4434`, and non-wife income (`nwifeinc`) `-0.0213`.
  - **Tolerance**: `Tol.TIGHT`.

- **`regression.romer_romer_tax_multiplier_ols`**
  - **Paper**: Romer, C. D. and Romer, D. H. (2010), *"The Macroeconomic Effects of Tax Changes"*, AER 100(3):763–801.
  - **Findings**: Cumulative 8-quarter GDP response to narrative tax changes using distributed lag OLS with Newey-West HAC covariance: multiplier of `-2.9004` (published benchmark $\approx -3.0$).
  - **Tolerance**: `Tol.MEDIUM` ($\text{rtol} \le 0.10$).

---

### 3. Heterogeneous-Agent & Structural Macro (`ha_dsge_causal`)

- **`ha_dsge_causal.aiyagari_precautionary_wedge`**: Replicates the Aiyagari (1994) precautionary savings wedge driving the equilibrium interest rate below the subjective discount rate ($r^* < \rho$).
- **`ha_dsge_causal.aiyagari_r_decreasing_in_sigma`**: Validates comparative statics: equilibrium $r^*$ decreases as idiosyncratic earnings risk $\sigma$ increases.
- **`ha_dsge_causal.aiyagari_r_decreasing_in_rho`**: Validates comparative statics: equilibrium $r^*$ decreases as earnings persistence $\rho$ increases.
- **`ha_dsge_causal.huggett_precautionary_riskfree_rate`**: Replicates Huggett (1993) pure credit economy risk-free rate depression due to borrowing constraints.
- **`ha_dsge_causal.sw07_seven_shocks_seven_observables`**: Smets-Wouters (2007) covariance stationary solution reproducing empirical business cycle variances.

---

### 4. Macroeconomic Stylized Facts (`stylized_facts`)

- **`stylized_facts.okun_law_fred`**: Replicates the negative cyclical comovement between output growth and changes in the unemployment rate (Okun's Law, correlation $\le -0.60$).
