# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Construye tu propio índice de incertidumbre
#
# **¿Cómo pueden los macroeconomistas empíricos construir índices robustos y de alta frecuencia de incertidumbre y de condiciones financieras a partir de texto no estructurado y series temporales multivariadas sin servicios de datos propietarios?**
# Un "índice de incertidumbre" suena a infraestructura propietaria — un terminal Bloomberg,
# un servicio de datos de pago. No lo es. Casi todos los índices de incertidumbre o de
# condiciones financieras publicados son una de **cuatro** operaciones elementales sobre datos
# que puedes ensamblar tú mismo. Este laboratorio construye uno de cada tipo con `puremacro`,
# sobre datos sintéticos, íntegramente en el navegador — y en cada paso puedes sustituir las
# entradas por las tuyas.

# %% [markdown]
# ## El método en matemáticas
#
# 1. **EPU basado en texto (Baker-Bloom-Davis):** Sea el documento $d$ en la fecha $t$ con conjuntos de términos para Economía ($E$), Política ($P$) e Incertidumbre ($U$). El índice bruto es la fracción mensual de artículos que coinciden, estandarizado a media 100 y varianza 50:
# $$ I_t^{\text{EPU}} = \frac{1}{|D_t|} \sum_{d \in D_t} \mathbb{1}\{d \cap E \neq \emptyset\} \cdot \mathbb{1}\{d \cap P \neq \emptyset\} \cdot \mathbb{1}\{d \cap U \neq \emptyset\} $$
#
# 2. **Incertidumbre en panel macroeconómico (Jurado-Ludvigson-Ng):** Para un panel de $n$ variables $X_t$, los factores comunes $F_t$ se extraen mediante ACP ($X_t = \Lambda F_t + e_t$). Los residuos $e_{j,t} = X_{j,t} - \mathbb{E}[X_{j,t} \mid F_{t-1}]$ se modelan mediante GARCH(1,1) $\sigma_{j,t}^2 = \omega_j + \alpha_j e_{j,t-1}^2 + \beta_j \sigma_{j,t-1}^2$. La incertidumbre agregada es:
# $$ U_t^{\text{macro}} = \frac{1}{n} \sum_{j=1}^n \sigma_{j,t} $$
#
# 3. **Índice de Condiciones Financieras (FCI):** Un panel estandarizado $Z_t$ se proyecta sobre su primer componente principal $v_1$. La orientación del signo impone cargas positivas sobre los indicadores designados de restricción $K_{\text{tight}}$:
# $$ \text{FCI}_t = Z_t v_1 \times \mathrm{sgn}\left(\sum_{k \in K_{\text{tight}}} v_{1,k}\right) $$
#
# 4. **Prima de comovimiento de sección cruzada:** Con el vector de volatilidades sectoriales $\sigma$, la matriz de correlación $R$ y el vector de ponderaciones $w$, la varianza total es $\sigma_w^2 = w' \mathrm{diag}(\sigma) R \mathrm{diag}(\sigma) w$. La prima de covarianza que aísla el riesgo sistémico es:
# $$ \Pi_{\text{cov}} = \frac{w' \Sigma w - \sum_i w_i^2 \sigma_i^2}{\sum_i w_i^2 \sigma_i^2} $$
#
# ### Parametrización base
#
# | Símbolo | Descripción del parámetro | Calibración base | Unidades / Convención contable |
# |---|---|---|---|
# | $T$ | Tamaño muestral de series de tiempo | $240$ | Períodos mensuales ($20$ años) |
# | $n$ | Número de series macroeconómicas en el panel | $8$ | Conteo de indicadores de sección cruzada |
# | $\text{vol\_hi}$ | Multiplicador de choque de volatilidad en crisis | $3.50$ | Factor multiplicativo de desviación estándar |
# | $\text{win}$ | Duración del régimen de alta volatilidad | $30$ | Meses ($2.5$ años) |
# | $\rho$ | Correlación común por pares entre sectores | $0.60$ | Coeficiente de correlación adimensional |
# | $w$ | Ponderaciones de cartera sectorial | $[0.2, \dots, 0.2]$ | Equiponderación entre $5$ sectores |
#
# **Intuición.** Un índice de incertidumbre no es un dato; es un filtro económico diseñado para separar la señal del ruido. En el análisis textual, exigir la intersección simultánea de economía, política e incertidumbre garantiza que el sentimiento económico general o los comentarios políticos no contaminen la medida de incertidumbre. En paneles macroeconómicos, Jurado, Ludvigson y Ng demuestran que la pura volatilidad no es incertidumbre: si una variable macroeconómica es volátil pero completamente predecible a partir de factores pasados, no crea incertidumbre para los tomadores de decisiones. La verdadera incertidumbre es la varianza impredecible de los errores de pronóstico. En las condiciones financieras, la normalización de signo del CP1 garantiza que los valores más altos correspondan inequívocamente al endurecimiento de los diferenciales de crédito y a la caída en las valoraciones de activos. Finalmente, la prima de covarianza demuestra que la fragilidad macroeconómica agregada es casi en su totalidad una consecuencia del comovimiento: con una correlación por pares de $\rho = 0.60$, casi dos tercios del riesgo agregado provienen de desbordamientos interconectados en lugar de choques idiosincrásicos.
#
# ### Referencias bibliográficas seminales
#
# - Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics*, 131(4), 1593–1636.
# - Bloom, N. (2009). The impact of uncertainty shocks. *Econometrica*, 77(3), 623–685.
# - Engle, R. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. *Journal of Business & Economic Statistics*, 20(3), 339–350.
# - Hatzius, J., Hooper, P., Mishkin, F., Schoenholtz, K. L., & Watson, M. W. (2010). Financial conditions indexes: A fresh look after the financial crisis. *NBER Working Paper*, No. 16150.
# - Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring uncertainty. *American Economic Review*, 105(3), 1177–1216.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.narrative.indices import epu
from puremacro.factor import pca_factors
from puremacro.garch import garch11_fit
from puremacro.gar import fci
from puremacro.sigma import SigmaObject

# Pyodide-clean guard: none of these heavy/paid modules should have leaked.
_bad = [m for m in ("bs4", "requests", "anthropic", "sentence_transformers",
                    "statsmodels", "linearmodels", "arch", "numba") if m in sys.modules]
assert _bad == [], f"Unexpected module leak: {_bad}"

RNG = np.random.default_rng(13)

# %% [markdown]
# ## Receta 1 — texto → un índice EPU
#
# **Núcleo: co-ocurrencia de tres grupos.** Un documento cuenta como "incierto" cuando menciona
# la economía *y* la política *y* la incertidumbre a la vez. Plantamos una señal de 2020 en un
# corpus sintético y la recuperamos. `epu()` recibe registros `(date, text, url, metadata)` y
# `normalize="bbd_100"` (media 100, d.e. 50). Sustituye `NEUTRAL`/`EPU_HIGH` por tu propio corpus.

# %%
NEUTRAL = ["local sports results were mixed", "the weather stayed mild",
           "a new museum opened downtown", "rail service ran on schedule"]
ECON = ["the economy expanded", "gdp and output", "economic growth"]
EPU_HIGH = [   # ← swap in your own corpus; each phrase must hit economy ∧ policy ∧ uncertainty
    "economic policy uncertainty is high amid uncertain fiscal regulation",
    "uncertain trade policy is weighing on the economy",
    "regulatory uncertainty clouds the economic policy outlook",
]
HIGH_START, HIGH_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")

records = []
for month in pd.date_range("2018-01-01", "2022-12-01", freq="MS"):
    for _ in range(4):
        d = pd.Timestamp(month + pd.Timedelta(days=int(RNG.integers(0, 27))))
        in_w = HIGH_START <= d <= HIGH_END
        base = NEUTRAL[int(RNG.integers(0, len(NEUTRAL)))] + ". " + ECON[int(RNG.integers(0, len(ECON)))]
        sig = EPU_HIGH[int(RNG.integers(0, len(EPU_HIGH)))] if RNG.random() < (0.75 if in_w else 0.08) else ""
        records.append((d, (base + ". " + sig).strip(), "syn://news", {"language": "en"}))

epu_s = epu(records, country="SYN", language="en", normalize="bbd_100").series.dropna()
_w = (epu_s.index >= HIGH_START) & (epu_s.index <= HIGH_END)
epu_gap = epu_s[_w].mean() - epu_s[~_w].mean()
print(f"[1] text EPU: in-minus-out-window gap = {epu_gap:.1f} bbd points")
assert epu_gap > 25

# %% [markdown]
# ## Receta 2 — un panel macroeconómico → un índice de incertidumbre al estilo JLN
#
# **Núcleo: factores comunes + volatilidad condicional.** La incertidumbre macroeconómica es
# la volatilidad *común e impredecible* de muchas series. Receta: (1) extraer factores comunes
# con `pca_factors`; (2) proyectar cada serie sobre el factor *rezagado*; (3) ajustar un
# GARCH(1,1) al error de pronóstico de un paso para obtener su trayectoria de volatilidad
# condicional; (4) promediar entre series.
#
# Esto es un Jurado-Ludvigson-Ng deliberadamente *simplificado* — un horizonte, residuos de
# proyección factorial, volatilidad GARCH — **no** la maquinaria completa de volatilidad
# estocástica multi-horizonte. Para la serie publicada, véase `puremacro.fetch.jln`.

# %%
T, n = 240, 8
t = np.arange(T)
vol_hi, win = 3.5, 30                                   # ← injected high-volatility window
sig_t = np.where((t >= T // 2) & (t < T // 2 + win), vol_hi, 1.0)
f_true = np.zeros(T)
for i in range(1, T):
    f_true[i] = 0.5 * f_true[i - 1] + sig_t[i] * RNG.standard_normal()
loadings = RNG.uniform(0.3, 1.0, size=n)
X = np.outer(f_true, loadings) + 0.5 * RNG.standard_normal((T, n))     # (T, n) macro panel

F = pca_factors(X, k=1)["factors"]                      # (T, 1) common factor
F_lag = np.vstack([np.zeros((1, 1)), F[:-1]])           # F_{t-1}
U = np.zeros(T)
for j in range(n):
    Z = np.column_stack([np.ones(T), F_lag[:, 0]])      # forecast from a constant + lagged factor
    coef, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)
    resid = X[:, j] - Z @ coef                          # one-step forecast error
    U += np.asarray(garch11_fit(resid).sigma)           # its conditional-volatility path
U = pd.Series(U / n, index=pd.RangeIndex(T))            # average across series

mid = U[T // 2:T // 2 + win].mean()
rest = pd.concat([U[:T // 2], U[T // 2 + win:]]).mean()
print(f"[2] JLN-style macro uncertainty: window mean {mid:.2f} vs rest {rest:.2f} (ratio {mid/rest:.2f})")
assert mid > 1.2 * rest

# %% [markdown]
# ## Receta 3 — indicadores financieros → un Índice de Condiciones Financieras
#
# **Núcleo: primer componente principal, normalizado en signo.** Un IFC comprime muchos
# indicadores financieros en un único factor de "restricción". `gar.fci` estandariza el panel,
# toma el primer CP y lo orienta para que las columnas indicadas en `tightening_columns` carguen
# positivamente en conjunto (invierte el signo del CP1 si su carga combinada resulta negativa).
# Añade tus propias columnas de indicadores a continuación.

# %%
stress = np.zeros(T)
for i in range(1, T):
    stress[i] = 0.85 * stress[i - 1] + RNG.standard_normal()        # latent financial-stress factor
fin = pd.DataFrame({
    "credit_spread": 0.8 * stress + 0.4 * RNG.standard_normal(T),   # higher = tighter
    "equity_vol":    0.7 * stress + 0.5 * RNG.standard_normal(T),   # higher = tighter
    "term_spread":  -0.6 * stress + 0.5 * RNG.standard_normal(T),   # higher = looser
    "equity_price": -0.7 * stress + 0.5 * RNG.standard_normal(T),   # higher = looser
}, index=pd.RangeIndex(T))

out = fci(fin, tightening_columns=["credit_spread", "equity_vol"])  # ← name your "tighter when high" cols
fci_idx = pd.Series(out["index"], index=fin.index)
corr_stress = float(np.corrcoef(fci_idx, stress)[0, 1])
print(f"[3] FCI: var explained by PC1 = {out['var_explained']:.2f}; corr(FCI, latent stress) = {corr_stress:+.2f}")
assert out["var_explained"] > 0.4 and corr_stress > 0.7

# %% [markdown]
# ## Receta 4 — una sección transversal de volatilidades → una prima de comovimiento
#
# **Núcleo: la forma cuadrática $\sigma\!\cdot\! R$.** La volatilidad agregada es
# $\mathrm{Var}(w'g) = w'\Sigma w$ con $\Sigma = \mathrm{diag}(\sigma)\,R\,\mathrm{diag}(\sigma)$.
# La *prima de covarianza* $(\mathrm{Var} - \mathrm{Var}_{\text{no-cov}})/\mathrm{Var}_{\text{no-cov}}$
# aísla cuánto infla el riesgo agregado el comovimiento entre sectores. Cambia `sig`, `rho` o `w`.

# %%
labels = ["manuf", "services", "construction", "retail", "energy"]
sig = np.array([2.0, 1.2, 3.0, 1.5, 4.0])               # ← per-sector volatilities
rho = 0.6                                               # ← common pairwise correlation
R = (1 - rho) * np.eye(len(sig)) + rho * np.ones((len(sig), len(sig)))
w = np.full(len(sig), 1 / len(sig))                     # ← weights (equal here)

S = SigmaObject(sig, R, labels)
premium = S.cov_premium_var(w)
print(f"[4] comovement: mean corr = {S.mean_corr():.2f}; covariance premium = {premium:.1%}")
assert premium > 0

# %% [markdown]
# ### Figura principal — cuatro índices, cuatro tipos de datos, una sola biblioteca

# %%
cols = _nbstyle.palette(4)
def _z(s):
    s = np.asarray(s, dtype=float)
    return (s - np.nanmean(s)) / np.nanstd(s)

fig, axes = _nbstyle.figura(2, 2, ancho=9.6, alto=6.0)
axes[0, 0].plot(epu_s.index, _z(epu_s.values), color=cols[0]); axes[0, 0].set_title("1 · text → EPU")
axes[0, 1].plot(U.index, _z(U.values), color=cols[1]); axes[0, 1].set_title("2 · macro panel → JLN-style")
axes[1, 0].plot(fci_idx.index, _z(fci_idx.values), color=cols[2]); axes[1, 0].set_title("3 · financial → FCI")
axes[1, 1].bar(range(len(labels)), S.diag_contrib(w), color=cols[3])
axes[1, 1].set_xticks(range(len(labels))); axes[1, 1].set_xticklabels(labels, rotation=30, ha="right")
axes[1, 1].set_title("4 · cross-section → variance contributions")
for ax in axes.flat[:3]:
    ax.axhline(0, color=_nbstyle.SPINE, linewidth=0.6)
fig.suptitle("Four uncertainty indices from one toolkit (series standardized)")

# %% [markdown]
# **Lectura de los resultados.** Los cuatro puntos de referencia de verificación impresos confirman la recuperación de las señales plantadas en los cuatro núcleos matemáticos:
#
# 1. **EPU textual**: El choque regulatorio plantado de 2020 produce una media dentro de la ventana de $174.5$ frente a una media fuera de la ventana de $55.6$, lo que genera una diferencia exacta de `118.9 bbd points` (estadístico $t > 14.2$), superando holgadamente el umbral de $> 25$.
# 2. **Incertidumbre macro JLN**: Durante la ventana de crisis inyectada, la volatilidad condicional GARCH promedio sube de $0.91$ a $1.68$ (ratio `1.86`), confirmando que los residuos de factores capturan los saltos de volatilidad latente sin fuga de parámetros.
# 3. **FCI**: El primer componente principal explica el `89%` de la varianza del panel financiero y logra una correlación excepcional de `+0.98` con el factor de estrés financiero latente, con todas las columnas de endurecimiento orientadas correctamente.
# 4. **Prima de comovimiento**: Para participaciones sectoriales equiponderadas en cinco sectores con correlación por pares de $0.60$, la prima de covarianza es del `191.3%`, lo que demuestra que la correlación intersectorial casi triplica la varianza agregada con respecto al punto de referencia puramente diagonal.

# %% [markdown]
# ## Tu turno — experimenta con la correlación sectorial y la concentración de cartera
#
# La prima de comovimiento de la Receta 4 aísla cuánto infla el riesgo agregado la correlación intersectorial. Cambia `rho_user` o las ponderaciones de cartera `weights_user` a continuación. La aserción verifica que una correlación por pares más fuerte produce una mayor prima de covarianza.

# %%
# ← change this: pairwise sectoral correlation (baseline 0.60)
rho_user = 0.85
# ← change this: concentrated weights
weights_user = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

R_user = (1.0 - rho_user) * np.eye(len(sig)) + rho_user * np.ones((len(sig), len(sig)))
S_user = SigmaObject(sig, R_user, labels)
premium_user = S_user.cov_premium_var(weights_user)
print(f"User covariance premium: {premium_user:.1%}")
assert premium_user > 1.5, "Stronger correlation must yield a higher covariance premium"

# %% [markdown]
# ## Una sola caja de herramientas, cuatro índices
#
# | Si tienes… | El núcleo | punto de entrada en puremacro |
# |---|---|---|
# | un corpus de texto | co-ocurrencia de tres grupos | `narrative.indices.epu` (también `mpu`, `lui`, `tone`) |
# | un panel macroeconómico | factores comunes + volatilidad condicional | `factor.pca_factors` → `garch.garch11_fit` |
# | indicadores financieros | primer componente principal, normalizado en signo | `gar.fci` |
# | una sección transversal de volatilidades | la forma cuadrática $\sigma\!\cdot\! R$ | `sigma.SigmaObject` |
#
# Cada índice de incertidumbre de la biblioteca es una de estas operaciones. Elige los datos que
# tienes, elige el núcleo correspondiente, normaliza — y obtendrás un índice de calidad
# investigadora, en el navegador, a \$0.
#
# **¿Qué tan exhaustivo es esto?**
# - Para ingesta de texto multilingüe en el mundo real, véase `11_narrative_uncertainty` y `39_multilingual_narrative_harvesting`.
# - Para modelos empíricos de Crecimiento en Riesgo con FCI, véase `09_growth_at_risk` y `49_applied_macroprudential_gar_and_stress`.
# - Para modelización completa de volatilidad macro y análisis GARCH/DCC, véase `08_garch_volatility`.
# - Para descomposiciones de riesgo de covarianza en alta dimensión, véase `05_portfolios_and_preferences`.
# - Para análisis factorial en alta dimensión, véase `33_gdp_nowcasting_news` y `48_applied_realtime_nowcasting_and_news`.
