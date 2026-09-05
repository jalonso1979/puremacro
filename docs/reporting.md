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
from puremacro.lp import lp_hac

res = lp_hac(df, y="gdp", x="policy_rate", horizon=8, lags=2, ci=0.90)

# Export to LaTeX
latex_str = res.to_latex()

# Export to Typst
typst_str = res.to_typst()

# Export to Markdown
markdown_str = res.to_markdown()
```

Sample Typst output:
```typst
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  table.header([h], [beta], [se], [t], [lo], [hi]),
  [0], [-0.1423], [0.0381], [-3.73], [-0.2050], [-0.0796],
  [1], [-0.2811], [0.0512], [-5.49], [-0.3653], [-0.1969],
  [2], [-0.3540], [0.0624], [-5.67], [-0.4566], [-0.2514],
)
```

---

### Structural VARs (`_IRFPlotMixin`)

All SVAR result objects (`CholeskyResult`, `ProxySVARResult`, `SignRestrictionResult`, `MaxShareResult`, etc.) support multi-format export:

```python
from puremacro.var.identify import cholesky_svar

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

favar_res = favar(panel_df, policy_series, n_factors=3, horizon=12)

# Export full cross-sectional IRF panel table
print(favar_res.to_latex())
print(favar_res.to_typst())
```

---

### Difference-in-Differences Results

All staggered DiD result dataclasses (`CallawaySantannaResult`, `SunAbrahamResult`, `BorusyakJaravelSpiessResult`, `SyntheticDiDResult`) feature `.to_markdown()`, `.to_latex()`, and `.to_typst()`:

```python
from puremacro.did import callaway_santanna

res_cs = callaway_santanna(df, unit="id", time="year", outcome="y", treat_time="g")

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

### Generated LaTeX Output:
```latex
\begin{tabular}{lrr}
Variable & Coef & Std.Err \\
\hline
Interest Rate & 0.4520*** & (0.0850) \\
Government Spending & -1.2380*** & (0.3120) \\
Tax Cut & 0.0810 & (0.0740) \\
\end{tabular}
```

---

## 3. Workflow Integration

### Quarto (`.qmd`)
In Quarto documents, use `output: asis` to insert LaTeX or Markdown tables directly into compiled PDF or HTML outputs:

```python
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
with open("tables/multipliers.typ", "w") as f:
    f.write(res.to_typst())
```

In your `main.typ` document:
```typst
#include "tables/multipliers.typ"
```
