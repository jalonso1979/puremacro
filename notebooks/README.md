# puremacro showcase notebooks

Static, publication-quality notebooks illustrating `puremacro` across the major
heterogeneous-agent paradigms (`puremacro.vfi`), the empirical-econometrics
tools (SVAR / LP / GARCH / GaR / DiD), and text-as-data uncertainty indices
(`puremacro.narrative`). **Edit the `.py` (jupytext percent) source,
not the `.ipynb`** — the `.ipynb` is a build artifact, regenerated with outputs.

| Notebook | Shows |
|---|---|
| `01_wealth_inequality` | Aiyagari + Huggett + permanent-type β-heterogeneity; Lorenz/Gini |
| `02_aggregate_shocks` | Krusell–Smith approximate aggregation; transition path; rep-agent growth |
| `03_life_cycle_and_demographics` | Finite-horizon life-cycle; cohort wealth by age; mortality weighting |
| `04_firm_dynamics` | Hopenhayn entry/exit, selection, comparative statics |
| `05_portfolios_and_preferences` | Two-asset portfolios; Epstein–Zin; EGM vs VFI |
| `06_svar_identification` | SVAR: Cholesky vs sign-restriction identification of a planted monetary shock |
| `07_local_projections` | Jordà LP-HAC + state-dependent (recession vs expansion) IRFs |
| `08_garch_volatility` | GARCH(1,1) MLE + Engle DCC time-varying correlation (pure-numpy) |
| `09_growth_at_risk` | Quantile-AR Growth-at-Risk fan + skew-t conditional density (ABG) |
| `10_staggered_did` | Staggered DiD: Callaway-Sant'Anna + Sun-Abraham event study |
| `11_narrative_uncertainty` | Build an EPU/MPU text-uncertainty index from a corpus (pure-numpy, no API key/LLM) |
| `12_validation_gallery` | Validation scorecard + coverage figure + puremacro-vs-reference overlays (statsmodels/scipy goldens) |
| `13_build_your_own_index` | Build four uncertainty indices from one toolkit — text→EPU, macro panel→JLN-style, financial→FCI, cross-section→comovement premium |
| `14_tax_multiplier_three_ways` | The US tax multiplier three ways on one frozen dataset — Blanchard-Perotti SVAR (−1), Romer-Romer narrative LP (−3), Mertens-Ravn narrative-as-instrument (between, with the effective first-stage F) — plus a spec curve showing identification, not estimation, drives the answer |
| `15_lp_did` | DiD meets local projections — why naive TWFE event studies break under staggered heterogeneous adoption, and LP-DiD (Dube-Girardi-Jordà-Taylor) as the fix; side-by-side with Callaway-Sant'Anna and Sun-Abraham agreeing on one panel |
| `16_regime_girf` | State-dependent transmission done right — Koop-Pesaran-Potter generalized IRFs for TVAR/MS-VAR with endogenous regime switching; the frozen-regime IRF overstates stress-state losses by ~33% |
| `17_identification_spec_curve` | Identification specification curve: how the identifying assumption, not the estimator, drives the answer |
| `18_beveridge_curve` | Beveridge curve: vacancies, unemployment and matching efficiency shifts |
| `19_model_confidence_set` | Hansen-Lunde-Nason model confidence set for competing forecasts |
| `20_unit_roots_with_power` | Elliott-Rothenberg-Stock DF-GLS unit root test with superior local power |
| `21_dynare_vfi_dsl` | Declarative Dynare-like Automated VFI specification with Howard acceleration & panel inequality simulation |
| `22_continuous_time_hjb` | Achdou et al. (2022) Continuous-Time HJB finite difference upwind scheme for consumption-saving models |
| `23_aiyagari_endogenous_labor` | Aiyagari GE incomplete markets model with endogenous intra-temporal labor choice |
| `24_synthetic_control` | Abadie et al. (2010) Synthetic Control Method for causal policy evaluation with donor placebos |
| `25_frequency_connectedness` | Baruník & Křehlík (2018) frequency-domain variance decomposition spillover networks |
| `26_cycles_and_bandpass` | Baxter-King, Christiano-Fitzgerald & Beveridge-Nelson vs Hamilton & HP cycle filtering |
| `27_garch_midas_macro_risk` | Engle-Ghysels-Sohn (2013) GARCH-MIDAS two-component mixed-frequency macro volatility |
| `28_weak_iv_anderson_rubin` | Weak-IV robust Anderson-Rubin (1949) quadratic HAC confidence sets for LP-IV |
| `29_synthetic_did` | Arkhangelsky et al. (2021) Synthetic DiD combining SCM unit weights and DiD time weights |
| `30_narrative_bursts_and_transcripts` | Kleinberg (2002) burst detection & high-frequency communication density |
| `31_sequence_space_hank` | Auclert et al. (2021) Sequence-Space Jacobian method for HANK models |
| `32_climate_macro_dice` | Nordhaus DICE integrated assessment model & optimal carbon taxation |
| `33_gdp_nowcasting_news` | Giannone-Reichlin-Small (2008) DFM GDP nowcasting & news decomposition |
| `34_penalized_macro_forecasting` | Elastic Net & Adaptive Lasso penalized macroeconomic forecasting |
| `35_empirical_benchmark_replications` | Benchmark replication scorecard across SVAR, LP, and DiD literatures |
| `36_climate_sovereign_debt_risk` | Physical/transition climate risk transmission to sovereign debt sustainability |
| `37_central_bank_narrative_sentiment` | Central bank communication tone extraction & high-frequency monetary LP |
| `38_real_time_vintages_and_revisions` | Real-time QNA vintages across 45+ countries, revision triangles $(T \times V)$, & Mankiw-Shapiro (1986) news vs. noise test |
| `39_multilingual_narrative_harvesting` | Multi-source narrative harvesting (50+ connectors), 8-language macro scoring, realization lags, & structured policy classification |
| `40_quarterly_national_accounts` | Three approaches to GDP in one panel: one price reference year (`qna_rebase`), the expenditure/output/income identities scored inside their own flows (`qna_identity`), growth decomposed with previous-period nominal weights (`qna_contributions`) |
| `41_dynare_frontier_showcase` | Smets-Wouters (2007) from Pfeifer's `.mod`: solve, FEVD, historical shock decomposition, OccBin ZLB, perfect-foresight Ramsey transition, Bayesian MCMC |
| `00_whats_new_in_puremacro_2_0` | Tour of the 2.0 unified API (`lags`/`horizon`/`ci`, result objects, exporters) |

The deepened showcases (`01`, `06`, `11`, `14`, `15`, `16`) follow the structure in
[`_TEMPLATE.md`](./_TEMPLATE.md): motivating question → the method in math → intuition →
worked code → read the output → a fill-in *Your turn* → "how comprehensive is this?".
`13_build_your_own_index` is a multi-kernel lab variant (four worked recipes, each with a
fill-in).

## Rebuild

```bash
pip install -e ".[notebooks]"
python tools/build_notebooks.py                 # build all
python tools/build_notebooks.py 01_wealth_inequality   # one
python tools/build_notebooks.py --check         # execute all, fail on error
```

Notebooks are numpy-only (Pyodide-safe), deterministic (fixed seeds), and carry
inline asserts on headline numbers so a numerical regression fails the build.
`14` runs on two frozen real-data snapshots shipped as package data
(`puremacro/replication/data/tax14_*.csv`; regenerate with
`tools/gen_notebook_data_tax14.py` — FRED fredgraph + Ramey's public HOM tax
archive), so it too executes offline and deterministically.

## Running on iPad Juno / Pyodide

To run notebooks on an iPad using **Juno** or in-browser Pyodide kernels (Juno.sh / JupyterLite):

1. **Installation via micropip (Pyodide / Juno.sh)**:
   Because `pyarrow` has no Pyodide wheel, install with `deps=False`:
   ```python
   import micropip
   await micropip.install("puremacro", deps=False)
   ```
2. **Compute Budgets & Memory Limits**:
   iPadOS terminates processes that exceed device memory (~1.5–3 GB). Use `puremacro.runtime` to fit costly operations to tablet ceilings:
   ```python
   from puremacro import runtime
   # Auto-scale bootstrap draws or posterior sampling to tablet size:
   svar = runtime.budgeted(cholesky_svar)
   ```
3. **Bundled Offline Data**:
   All benchmark datasets load offline via `puremacro.datasets` (`load_gali1999()`, `load_narrative_tax_shocks()`, etc.) without requiring network sockets or parquet engines.

