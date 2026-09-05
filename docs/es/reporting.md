> 🇬🇧 [English](../reporting.md) · 🇪🇸 Español

# Publicación e informes para manuscritos

La investigación macroeconómica exige una comunicación clara y reproducible. Tradicionalmente, trasladar estimaciones empíricas, errores estándar y bandas de confianza desde scripts de Python a artículos académicos requería formateo manual, scripts frágiles o dependencias externas pesadas.

`puremacro 2.0` integra una pipeline de generación de informes sin dependencias externas en `puremacro.reports` y dota a todos los objetos de resultado principales de métodos de exportación nativos para **LaTeX**, **Typst** y **Markdown**.

---

## Los tres ecosistemas de salida

Cada método de exportación en puremacro admite tres formatos principales:

1. **LaTeX (`.to_latex()`)**: Genera entornos limpios `\begin{tabular}...\end{tabular}` listos para compilar en Overleaf y plantillas de revistas internacionales (AER, JPE, QJE, REStud).
2. **Typst (`.to_typst()`)**: Produce bloques `#table(...)` para [Typst](https://typst.app/), el sistema moderno y veloz de composición tipográfica que está ganando amplia adopción en economía y ciencias cuantitativas.
3. **Markdown (`.to_markdown()`)**: Genera tablas en GitHub-Flavored Markdown compatibles con documentos Quarto (`.qmd`), cuadernos de Jupyter y notas de investigación.

No se requieren librerías de terceros (como `tabulate` o `stargazer`). Todo el formateo se realiza mediante funciones ligeras en Python puro compatibles con el contrato de ejecución de Pyodide.

---

## 1. Exportación de resultados de estimadores

### Proyecciones locales (`LPResult`)

Todos los estimadores de proyecciones locales (`lp_hac`, `lp_iv`, `lp_state_dep`, `lp_state_dep_iv`, `panel_lp`) devuelven un objeto `LPResult`:

```python
from puremacro.lp import lp_hac

res = lp_hac(df, y="gdp", x="policy_rate", horizon=8, lags=2, ci=0.90)

# Exportar a LaTeX
latex_str = res.to_latex()

# Exportar a Typst
typst_str = res.to_typst()

# Exportar a Markdown
markdown_str = res.to_markdown()
```

Salida típica para Typst:
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

### VAR Estructural (`_IRFPlotMixin`)

Todos los resultados de SVAR (`CholeskyResult`, `ProxySVARResult`, `SignRestrictionResult`, `MaxShareResult`, etc.) admiten exportación multiformato:

```python
from puremacro.var.identify import cholesky_svar

res = cholesky_svar(Y, p=2, horizon=12, ci=0.90)

# Exportar malla completa de FRI como DataFrame ordenado
df_tidy = res.to_frame()

# Exportar la respuesta de la variable 0 al choque 0
latex_irf = res.to_latex(target_idx=0, shock_idx=0)
typst_irf = res.to_typst(target_idx=0, shock_idx=0)
markdown_irf = res.to_markdown(target_idx=0, shock_idx=0)
```

---

### VAR aumentado con factores (`FAVARResult`)

```python
from puremacro.var import favar

favar_res = favar(panel_df, policy_series, n_factors=3, horizon=12)

# Exportar tabla completa de respuestas de corte transversal
print(favar_res.to_latex())
print(favar_res.to_typst())
```

---

### Resultados de Diferencias en Diferencias

Todas las dataclasses de resultados de DiD escalonado (`CallawaySantannaResult`, `SunAbrahamResult`, `BorusyakJaravelSpiessResult`, `SyntheticDiDResult`) incorporan `.to_markdown()`, `.to_latex()` y `.to_typst()`:

```python
from puremacro.did import callaway_santanna

res_cs = callaway_santanna(df, unit="id", time="year", outcome="y", treat_time="g")

# Exportar dinámica del estudio de eventos
latex_did = res_cs.to_latex()
typst_did = res_cs.to_typst()
```

---

## 2. Tablas de regresión independientes (`coef_table`)

`puremacro.reports.coef_table` estructura coeficientes y errores estándar arbitrarios en tablas de regresión con estrellas de significancia estadística:

- `***` $p < 0.01$
- `**` $p < 0.05$
- `*` $p < 0.10$

```python
import numpy as np
from puremacro.reports import coef_table

beta = np.array([0.452, -1.238, 0.081])
se = np.array([0.085, 0.312, 0.074])
names = ["Tasa de interés", "Gasto público", "Recorte de impuestos"]

# 1. Tabla LaTeX con errores entre paréntesis y estrellas
print(coef_table(beta, se, names=names, stars=True, fmt="latex"))

# 2. Tabla Typst
print(coef_table(beta, se, names=names, stars=True, fmt="typst"))

# 3. Tabla Markdown
print(coef_table(beta, se, names=names, stars=True, fmt="markdown"))
```

### Ejemplo de salida en LaTeX:
```latex
\begin{tabular}{lrr}
Variable & Coef & Std.Err \\
\hline
Tasa de interés & 0.4520*** & (0.0850) \\
Gasto público & -1.2380*** & (0.3120) \\
Recorte de impuestos & 0.0810 & (0.0740) \\
\end{tabular}
```

---

## 3. Integración en flujos de trabajo de publicación

### Quarto (`.qmd`)
En documentos Quarto, utilice la opción `#| output: asis` para insertar tablas directamente en documentos PDF o HTML:

```python
```{python}
#| output: asis
from puremacro.lp import lp_hac

res = lp_hac(df, y="gdp", x="shock", horizon=8)
print(res.to_markdown())
```
```

### Typst (`.typ`)
Guarde la tabla generada e inclúyala en su archivo principal de Typst:

```python
with open("tablas/multiplicadores.typ", "w") as f:
    f.write(res.to_typst())
```

En su documento `main.typ`:
```typst
#include "tablas/multiplicadores.typ"
```
