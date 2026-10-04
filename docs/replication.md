> 🇬🇧 English · 🇪🇸 [Español](es/replication.md)

# Replication Gallery

> Distinct from `puremacro.validation` (which guards against algorithmic drift relative to reference software packages), `puremacro.replication` is designed to scientifically reproduce published empirical and structural headline findings from peer-reviewed academic literature. Where a case pins a number computed by puremacro instead of a published one, its `citation` says so ("puremacro regression value, not published").

Every replication case is declared as a self-contained, frozen `ReplicationCase` running deterministically in the browser under the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero network access and executing in sub-second time.

## Running the Replication Suite

```python
from puremacro.replication import run_all, scorecard

# Run all offline replication cases and display the scorecard
df = scorecard()
print(df[["id", "family", "paper", "target_kind", "tol", "passed", "margin"]])

# Every offline case passes (published targets, or labelled regression values)
assert df["passed"].all()
```

---

## Replication Families

### 1. Bayesian DSGE Estimation (`dsge_estimation`)

Smets & Wouters (2007), *"Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach"*, AER 97(3):586–606. Tables and pages below are from the ECB Working Paper 722 version (February 2007, <https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf>).

- **Model**: `puremacro.dsge.smets_wouters`. Its impulse responses to all seven shocks coincide with the bundled Pfeifer `.mod` (`puremacro/dsge/_references/sw07_pfeifer.mod`) solved by puremacro at the same parameters (max gap below 1e-12). The price and wage markup disturbances are ARMA(1,1) with a contemporaneous innovation, $\varepsilon^p_t = \rho_p \varepsilon^p_{t-1} + \eta^p_t - \mu_p \eta^p_{t-1}$ and the same for wages (WP 722 printed pp. 14–15).
- **Data**: `puremacro/dsge/_sw07_data.csv`, 1966Q1–2004Q4 (156 quarters), rebuilt from FRED by `tools/build_sw07_data.py` with the definitions of the SW07 data appendix (printed p. 47). GDP, consumption and investment are per capita; nominal consumption and fixed investment are deflated by the GDP deflator. The real wage is NFB hourly compensation over the GDP deflator. Hours are NFB average hours × civilian employment ÷ population 16+. All of these are in 100 × log. Inflation and the federal funds rate are quarterly percent. These are today's FRED vintages, not SW's 2006 files.
- **Fixture**: `puremacro/replication/data/sw07_parity_seed0_200draws.npz`, shipped as package data so the cases also run from an installed wheel, and rebuilt with `python tools/build_sw07_data.py fixture`. It holds the optimised posterior mode, the inverse Hessian there, and 200 thinned random-walk Metropolis draws. It also records the SHA-256 of the CSV it was built on, and a mismatch fails the cases.

- **`dsge_estimation.sw07_structural_parameters_mode`** (published target)
  - **Target**: the posterior **Mode** column of Table 1a (PDF p. 35) and Table 1b (PDF p. 36). The Mean column is not the target: for example it has $\varphi = 5.74$ and $\sigma_c = 1.38$, against the modes 5.48 and 1.39.
  - **What is compared**: the posterior mode of puremacro's model on the bundled data. It was found by L-BFGS-B from two starting points and polished with Newton steps, and it is a stationary point: no coordinate step raises the log posterior (tested).

| Parameter | SW07 symbol | Table | SW07 mode | puremacro mode | Difference |
|---|---|---|---:|---:|---:|
| `csadjcost` | $\varphi$ | 1a | 5.48 | 5.694 | +3.9% |
| `csigma` | $\sigma_c$ | 1a | 1.39 | 1.405 | +1.1% |
| `chabb` | $h$ (habit) | 1a | 0.71 | 0.700 | -1.4% |
| `csigl` | $\sigma_L$ | 1a | 1.92 | 2.035 | +6.0% |
| `cprobp` | $\xi_p$ | 1a | 0.65 | 0.690 | +6.2% |
| `cfc` | $\Phi$ | 1a | 1.61 | 1.617 | +0.4% |
| `crr` | $\rho$ | 1a | 0.81 | 0.809 | -0.1% |
| `crdy` | $r_{\Delta y}$ | 1a | 0.22 | 0.226 | +2.6% |
| `ctrend` | $\bar\gamma$ | 1a | 0.43 | 0.416 | -3.2% |
| `crhopinf` | $\rho_p$ | 1b | 0.90 | 0.879 | -2.3% |
| `cmap` | $\mu_p$ | 1b | 0.74 | 0.703 | -5.0% |
| `crhow` | $\rho_w$ | 1b | 0.97 | 0.972 | +0.2% |
| `cmaw` | $\mu_w$ | 1b | 0.88 | 0.886 | +0.6% |

  - **Tolerance**: `Tol.COARSE` ($\text{rtol} \le 0.25$), because the data vintage differs from SW's. Among the other published modes, 21 of the remaining 22 (including the seven shock standard deviations) are also within 25%; the exception is price indexation $\iota_p$ (`cindp`), 0.32 against 0.22. The hours intercept $\bar l$ (`constelab`) is not compared, because the bundled hours series is demeaned.

- **`dsge_estimation.sw07_log_posterior_at_mode`**, **`dsge_estimation.sw07_laplace_marginal_data_density`**, **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`** (puremacro regression values, **not published numbers**)
  - **Targets**:
    - log posterior (log likelihood + log prior) at the mode: `-822.04`, recomputed live;
    - Laplace log marginal data density: `-902.83`, from the live log posterior and the stored inverse Hessian;
    - Geweke (1999) modified harmonic mean on the 200 stored draws: `-908.03`, with spread `2.66` across truncation levels 0.1–0.9.

    All three use `Tol.TIGHT`.
  - **Why they are not published targets**: SW07 report no log posterior at the mode, and their marginal likelihood (Table 2, PDF p. 37: −905.8) is computed over 1966:1–2004:4 "using the period 1956:1–1965:4 as a training sample" with the Laplace approximation. These cases guard puremacro's own model, data, priors and marginal-likelihood estimators against unintended change.
  - **The Table 2 computation itself (4.6.0)**: `sw07_laplace_mdd` (`puremacro.dsge.sw07_marginal`) implements the training sample as Dynare's `presample` with a diffuse initialisation, on the authors' own series (`load_sw07_data("authors")`, the AER replication data). On 3 October 2026 it gave a Laplace log data density of **−932.3** (log posterior at the mode −849.6) for the paper's configuration (1956Q1 start, 40-quarter presample), **−921.4** (−840.0) for the configuration of the authors' replication `.mod` (1965Q1 start, 4-quarter presample, `lik_init=2`), **−926.2** (−845.1) for 1966–2004 with the unconditional initialisation, and **−902.8** (−822.0) on the bundled FRED rebuild, which reproduces the regression case above. At the authors' own posterior mode (`usmodel_mode.mat`) puremacro's log posterior is −840.81 against **−841.46 printed by Dynare 8** on the same files and options (Laplace −922.4 against −923.1 with Dynare's Hessian; with the 1956Q1 start and 40-quarter presample, −888.27 against −887.13). The replication case `dsge_estimation.sw07_log_posterior_at_authors_mode_vs_dynare` pins that agreement. The printed −905.8 is therefore reproduced neither by puremacro nor by Dynare 8 on the public replication files, with any of the sample and initialisation choices above; the gap is in the data or the computation behind the published table, not in puremacro's model.
  - Through 4.3.0 this page presented −1673.72, −1686.09 and −2524.36 as SW07 results. They were puremacro outputs of a model whose markup shocks lacked their MA terms and entered one quarter late, on data whose hours lacked the factor 100.

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
  - **Scope**: A modified horizon-eight local projection with Newey-West HAC covariance, revised outcome data and a 1950–2006 sample. Its approximately `-2.9004` response is compared coarsely with the paper's headline magnitude near `-3`; it is not the original distributed-lag estimator or a published horizon-eight LP coefficient.
  - **Tolerance**: `Tol.MEDIUM` ($\text{rtol} \le 0.10$), for that coarse comparison only. The [original-data replication](empirical_research.md) separately reproduces the authors' baseline specification and compares it with independent software output.

---

### 3. Heterogeneous-Agent & Structural Macro (`ha_dsge_causal`)

- **`ha_dsge_causal.aiyagari_precautionary_wedge`**: Replicates the Aiyagari (1994) precautionary savings wedge driving the equilibrium interest rate below the subjective discount rate ($r^* < \rho$).
- **`ha_dsge_causal.aiyagari_r_decreasing_in_sigma`**: Validates comparative statics: equilibrium $r^*$ decreases as idiosyncratic earnings risk $\sigma$ increases.
- **`ha_dsge_causal.aiyagari_r_decreasing_in_rho`**: Validates comparative statics: equilibrium $r^*$ decreases as earnings persistence $\rho$ increases.
- **`ha_dsge_causal.huggett_precautionary_riskfree_rate`**: Replicates Huggett (1993) pure credit economy risk-free rate depression due to borrowing constraints.
- **`ha_dsge_causal.sw07_seven_shocks_seven_observables`**: Smets-Wouters (2007) specification check: the model has the published seven structural shocks matched to seven observables (a count, not a comparison of moments).

---

### 4. Macroeconomic Stylized Facts (`stylized_facts`)

- **`stylized_facts.okun_law_fred`**: Replicates the negative cyclical comovement between output growth and changes in the unemployment rate (Okun's Law, correlation $\le -0.60$).
