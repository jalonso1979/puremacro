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
