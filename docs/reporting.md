> 🇬🇧 English · 🇪🇸 [Español](es/reporting.md)

# Publication & Manuscript Reporting

Macroeconomic research requires clear, reproducible dissemination. Translating empirical estimates, standard errors, and confidence bands from Python into papers has historically required manual formatting, brittle scripts, or heavy external dependencies.

`puremacro 2.0` integrates a zero-dependency reporting pipeline in `puremacro.reports` and equips all major estimator results with built-in export methods for **LaTeX**, **Typst**, and **Markdown**.

---

## The Three Output Formats

Every export method in puremacro supports three target ecosystems:

1. **LaTeX (`.to_latex()`)**: Renders clean, standard `\begin{tabular}...\end{tabular}` environments formatted for Overleaf and academic journal templates (AER, JPE, QJE).
2. **Typst (`.to_typst()`)**: Generates `#table(...)` blocks for [Typst](https://typst.app/), the modern, fast typesetting system increasingly adopted in economics and quantitative sciences.
3. **Markdown (`.to_markdown()`)**: Generates GitHub-Flavored Markdown tables for Quarto documents (`.qmd`), Jupyter notebooks, and research notes.

No external libraries (such as `tabulate` or `stargazer`) are needed. All formatting is performed by lightweight pure-Python string formatters compliant with the Pyodide runtime contract.

---

## 1. Exporting Estimator Results

### Local Projections (`LPResult`)

All local projection estimators (`lp_hac`, `lp_iv`, `lp_state_dep`, `lp_state_dep_iv`, `panel_lp`) return an `LPResult` object:

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# A small synthetic quarterly dataset (replace with your own frame)
rng = np.random.default_rng(0)
T = 200
policy_rate = rng.standard_normal(T)
gdp = np.cumsum(-0.15 * policy_rate + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "policy_rate": policy_rate})

res = lp_hac(df, y="gdp", x="policy_rate", horizon=8, lags=2, ci=0.90)


# Export to LaTeX
latex_str = res.to_latex()

# Export to Typst
typst_str = res.to_typst()

# Export to Markdown
markdown_str = res.to_markdown()
```

Typst output of `res.to_typst()` (first horizons shown):
```typst
#table(
  columns: 6,
  [* h *],
  [* beta *],
  [* se *],
  [* t *],
  [* lo *],
  [* hi *],
  [0],
  [-0.168952],
  [0.022707],
  [-7.44069],
  [-0.206301],
  [-0.131603],
  [1],
  [-0.182836],
  [0.030485],
  [-5.997644],
  [-0.232979],
  [-0.132693],
  ...
)
```

---

### Structural VARs (`_IRFPlotMixin`)

All SVAR result objects (`CholeskySVARResult`, `ProxySVARResult`, `SignRestrictionResult`, `MaxShareResult`, etc.) support multi-format export:

```python
from puremacro.var.identify import cholesky_svar

# Three-variable VAR(1) simulated for the example
A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
Y = np.zeros((300, 3))
for t in range(1, 300):
    Y[t] = A @ Y[t - 1] + rng.standard_normal(3)

res = cholesky_svar(Y, p=2, horizon=12, ci=0.90)


# Export full multidimensional IRF grid as a tidy DataFrame
df_tidy = res.to_frame()

# Export path of variable 0 responding to shock 0
latex_irf = res.to_latex(target_idx=0, shock_idx=0)
typst_irf = res.to_typst(target_idx=0, shock_idx=0)
markdown_irf = res.to_markdown(target_idx=0, shock_idx=0)
```

---

### Factor-Augmented VAR (`FAVARResult`)

```python
from puremacro.var import favar

# Informational panel (T x N) and a policy series, simulated for the example
factor = np.cumsum(rng.standard_normal(300)) * 0.1
panel_df = pd.DataFrame(
    np.outer(factor, rng.uniform(0.5, 1.5, 12)) + 0.3 * rng.standard_normal((300, 12)),
    columns=[f"x{i}" for i in range(12)],
)
policy_series = pd.Series(0.5 * factor + 0.2 * rng.standard_normal(300), name="policy_rate")

favar_res = favar(panel_df, policy_series, n_factors=3, horizon=12, n_boot=50)


# Export full cross-sectional IRF panel table
print(favar_res.to_latex())
print(favar_res.to_typst())
```

---

### Difference-in-Differences Results

All staggered DiD result dataclasses (`CallawaySantannaResult`, `SunAbrahamResult`, `BorusyakJaravelSpiessResult`, `SyntheticDiDResult`) feature `.to_markdown()`, `.to_latex()`, and `.to_typst()`:

```python
from puremacro.did import callaway_santanna

# Staggered-adoption panel: 40 units, 12 years, cohorts treated in 2006 and 2009
rows = []
for i in range(40):
    g = {0: 2006, 1: 2009, 2: np.nan}[i % 3]
    for year in range(2000, 2012):
        effect = 2.0 if (not np.isnan(g) and year >= g) else 0.0
        rows.append({"id": i, "year": year, "g": g, "y": 0.1 * (year - 2000) + effect + rng.standard_normal()})
did_df = pd.DataFrame(rows)

res_cs = callaway_santanna(did_df, unit="id", time="year", outcome="y", treat_time="g", n_boot=50)


# Export event-study dynamic impacts
latex_did = res_cs.to_latex()
typst_did = res_cs.to_typst()
```

---

## 2. Standalone Regression Tables (`coef_table`)

`puremacro.reports.coef_table` formats arbitrary coefficient estimates and standard errors into academic regression tables with significance stars:

- `***` $p < 0.01$
- `**` $p < 0.05$
- `*` $p < 0.10$

```python
import numpy as np
from puremacro.reports import coef_table

beta = np.array([0.452, -1.238, 0.081])
se = np.array([0.085, 0.312, 0.074])
names = ["Interest Rate", "Government Spending", "Tax Cut"]

# 1. LaTeX table with significance stars and parenthetical SEs
print(coef_table(beta, se, names=names, stars=True, fmt="latex"))

# 2. Typst table
print(coef_table(beta, se, names=names, stars=True, fmt="typst"))

# 3. Markdown table
print(coef_table(beta, se, names=names, stars=True, fmt="markdown"))
```

### Generated LaTeX output (what `coef_table(..., fmt="latex")` prints):
```latex
\begin{tabular}{lrrrrrr}
variable & coef & se & t & p & lo\_95\% & hi\_95\% \\
\hline
Interest Rate & 0.452*** & 0.085 & 5.318 & 0.000 & 0.285 & 0.619 \\
Government Spending & -1.238*** & 0.312 & -3.968 & 0.000 & -1.850 & -0.626 \\
Tax Cut & 0.081 & 0.074 & 1.095 & 0.274 & -0.064 & 0.226 \\
\end{tabular}
```

---

## 3. Workflow Integration

### Quarto (`.qmd`)
In Quarto documents, use `output: asis` to insert LaTeX or Markdown tables directly into compiled PDF or HTML outputs:

```text
```{python}
#| output: asis
from puremacro.lp import lp_hac

res = lp_hac(df, y="gdp", x="shock", horizon=8)
print(res.to_markdown())
```
```

### Typst (`.typ`)
Pipe output into your Typst source:

```python
import os
os.makedirs("tables", exist_ok=True)   # your manuscript's tables folder
with open("tables/multipliers.typ", "w") as f:
    f.write(res.to_typst())
```

In your `main.typ` document:
```typst
#include "tables/multipliers.typ"
```
