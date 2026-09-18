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
# # Nowcasting del PIB en Frecuencia Mixta — Seguimiento de la Economía en Tiempo Real
#
# **¿Cómo pueden los bancos centrales y las autoridades fiscales estimar el crecimiento del PIB del trimestre en curso antes de las publicaciones estadísticas oficiales, y cómo atribuir rigurosamente las revisiones a las sorpresas en los datos?**
#
# Las cifras oficiales del Producto Interno Bruto (PIB) se compilan con frecuencia trimestral y típicamente se difunden con un rezago significativo de uno a tres meses posteriores al cierre del trimestre de referencia. En contraste, las autoridades económicas y los participantes del mercado financiero requieren diagnósticos inmediatos sobre el pulso de la actividad para calibrar la tasa de política monetaria o desplegar medidas fiscales. Los indicadores macroeconómicos de mayor frecuencia (producción industrial, ventas minoristas, índices PMI de gerentes de compras y nóminas no agrícolas) se publican semanal o mensualmente, pero llegan con desfases asíncronos y calendarios escalonados. Esto genera un panel desbalanceado con datos faltantes al final de la muestra, conocido en la literatura econométrica como el **borde irregular (ragged edge)**.
#
# Domenico Giannone, Lucrezia Reichlin y David Small (2008, *Journal of Monetary Economics*) introdujeron el marco del **Modelo de Factores Dinámicos (DFM)** en espacio de estados para abordar este desafío de frecuencia mixta. Al postular que un amplio panel de indicadores mensuales responde a un número reducido de factores latentes comunes no observados $F_t$, el DFM cumple dos objetivos esenciales:
# 1. **Extracción de Señales con Datos Faltantes**: El filtro de Kalman y el suavizador de Rauch-Tung-Striebel procesan patrones de observación variables en el tiempo, proyectando los factores comunes a través del borde irregular.
# 2. **Descomposición Formal de Noticias**: Marta Bańbura y Michele Modugno (2014, *Journal of Applied Econometrics*) formalizaron cómo descomponer la revisión del nowcast del PIB $\Delta \hat{y}_{t|v} = \hat{y}_{t|v} - \hat{y}_{t|v-1}$ en la suma ponderada de las *noticias* (la sorpresa inesperada de cada serie publicada respecto al pronóstico del modelo).
#
# En este tutorial interactivo simulamos un panel macroeconómico mensual asíncrono con bordes irregulares, estimamos el nowcast del PIB mediante factores dinámicos y calculamos la matriz de atribución de noticias con `puremacro.nowcast.nowcast_gdp`.

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

from puremacro.nowcast import nowcast_gdp

# %% [markdown]
# ## 1. Simulación de Panel Macroeconómico Asíncrono
#
# La representación en espacio de estados del Modelo de Factores Dinámicos de frecuencia mixta comprende:
#
# 1. **Ecuación de Medida**:
#    Sea $X_t = (x_{1,t}, \dots, x_{N,t})^\top$ el vector estandarizado de $N$ variables mensuales. Cada serie se descompone en fluctuaciones del factor común y ruido idiosincrásico:
#    $$ X_t = \Lambda F_t + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \Sigma_\xi) $$
#    donde $\Lambda \in \mathbb{R}^{N \times r}$ es la matriz de cargas factoriales, $F_t \in \mathbb{R}^r$ es el vector de $r \ll N$ factores comunes y $\Sigma_\xi = \text{diag}(\sigma_{\xi, 1}^2, \dots, \sigma_{\xi, N}^2)$ garantiza ortogonalidad transversal.
# 2. **Ecuación de Transición**:
#    Los factores latentes siguen un proceso autorregresivo vectorial estacionario:
#    $$ F_t = A_1 F_{t-1} + \dots + A_p F_{t-p} + u_t, \quad u_t \sim \mathcal{N}(0, Q) $$
# 3. **Ecuación Puente Trimestral**:
#    El crecimiento trimestral del PIB $y_q^Q$ se vincula a los factores comunes mensuales mediante un filtro de agregación $\bar{F}_q = \frac{1}{3} \sum_{m \in q} F_{q, m}$:
#    $$ y_q^Q = \mu + \beta^\top \bar{F}_q + \varepsilon_q, \quad \varepsilon_q \sim \mathcal{N}(0, \sigma_\varepsilon^2) $$
#
# Cuando una publicación se retrasa, la observación $x_{i,t}$ no está disponible. La ecuación de medida se proyecta sobre el subespacio observado mediante una matriz de selección variable en el tiempo $M_t$: $X_t^{obs} = M_t X_t = M_t \Lambda F_t + M_t \xi_t$.
#
# A continuación generamos $T=72$ meses para $N=10$ indicadores representativos, introduciendo valores faltantes en la última fecha para emular el rezago de publicación de datos duros.

# %%
rng = np.random.default_rng(42)
n_months = 72
dates_m = pd.date_range("2018-01-01", periods=n_months, freq="MS")

F_true = np.zeros((n_months, 2))
for t in range(1, n_months):
    F_true[t, 0] = 0.85 * F_true[t-1, 0] + rng.normal(scale=0.8)
    F_true[t, 1] = 0.60 * F_true[t-1, 1] + rng.normal(scale=0.5)

var_names = [
    "Producción Industrial", "Empleo Formal", "Ventas Minoristas",
    "Inicios de Vivienda", "Utilización de Capacidad", "PMI Manufactura",
    "Inflación Subyacente", "Ingreso Personal Real", "Pedidos de Exportación", "Confianza del Consumidor"
]
N = len(var_names)
X = np.zeros((n_months, N))
for i in range(N):
    load = rng.uniform(0.4, 1.6, size=2)
    X[:, i] = F_true @ load + rng.normal(scale=0.4, size=n_months)

df_X = pd.DataFrame(X, index=dates_m, columns=var_names)
# Rezagos escalonados de publicación: series duras con retraso de 1 mes
df_X.iloc[-1, [3, 4, 8, 9]] = np.nan

# PIB trimestral histórico
dates_q = pd.date_range("2018-01-01", periods=n_months // 3, freq="QS")
F_q = df_X.resample("QE").mean().to_numpy().mean(axis=1)[:len(dates_q)]
gdp = 2.2 + 1.4 * F_q + rng.normal(scale=0.3, size=len(dates_q))
s_gdp = pd.Series(gdp, index=dates_q.to_period("Q").astype(str), name="PIB")

# %% [markdown]
# ## 2. Estimación del Nowcast DFM y Descomposición de Noticias
#
# La estimación se realiza mediante el algoritmo de Esperanza-Maximización (EM) de Shumway y Stoffer (1982), que itera entre:
# 1. **Paso E**: Ejecutar el suavizador de Kalman condicional en los parámetros actuales $(\hat{\Lambda}, \hat{A}, \hat{\Sigma}_\xi, \hat{Q})$ para estimar la trayectoria esperada de los factores $\mathbb{E}[F_t \mid X_{1:T}^{obs}]$.
# 2. **Paso M**: Actualizar las cargas factoriales y las matrices autorregresivas mediante mínimos cuadrados generalizados sobre los momentos suavizados.
#
# Al converger el modelo, la ecuación puente traduce la secuencia de factores en una estimación puntual del PIB del trimestre en curso, ofreciendo un nowcast objetivo antes de la divulgación oficial.

# %%
res = nowcast_gdp(df_X, s_gdp, n_factors=2)
print(res.summary())

# %% [markdown]
# ## 3. Factores Mensuales Latentes y Cargas Factoriales
#
# Los factores latentes descomponen el ciclo económico en impulsos macroeconómicos diferenciados:
# - **Factor 1 (Actividad Real)**: Sintetiza el comovimiento de los datos duros de oferta (producción industrial, empleo y utilización de capacidad).
# - **Factor 2 (Demanda y Expectativas)**: Recoge la confianza del consumidor, el gasto minorista y las presiones de precios.
#
# La matriz de cargas factoriales $\Lambda$ (Panel Derecho) ilustra la ponderación de cada indicador sobre los factores subyacentes. Las series con mayores coeficientes sobre el Factor 1 actúan como los termómetros principales del ciclo real.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

ax1.plot(df_X.index, res.factors["Factor_1"], **_nbstyle.S1, label="Factor 1 (Actividad Real)")
ax1.plot(df_X.index, res.factors["Factor_2"], **_nbstyle.S2, label="Factor 2 (Demanda/Sentimiento)")
ax1.set_title("Factores Comunes Mensuales Suavizados", fontsize=11, fontweight="bold")
ax1.set_xlabel("Fecha", color=_nbstyle.TEXTO)
ax1.set_ylabel("Nivel del Factor", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

im = ax2.imshow(res.loadings.to_numpy(), cmap=_nbstyle.CMAP_SEQ, aspect="auto")
ax2.set_yticks(range(N))
ax2.set_yticklabels(var_names, fontsize=8)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["Factor 1", "Factor 2"])
ax2.set_title("Cargas Factoriales Estimadas Λ", fontsize=11, fontweight="bold")
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Ponderación", color=_nbstyle.TEXTO)
cbar.ax.tick_params(colors=_nbstyle.NOTA)

# %% [markdown]
# ## 4. Impacto de las Noticias en la Revisión del Nowcast
#
# La pieza central del monitoreo en tiempo real es la **Atribución de Noticias de Bańbura y Modugno (2014)**. Cuando se publica una nueva vendimia de datos $v$ con observaciones recientes $X_{new, v}$, la modificación del nowcast del PIB se descompone exactamente como:
#
# $$ \hat{y}_{t \mid v} - \hat{y}_{t \mid v-1} = \sum_{j \in \text{publicadas}} \omega_{j, v} \cdot \left[ x_{j, v} - \mathbb{E}(x_{j, v} \mid \Omega_{v-1}) \right] $$
#
# donde:
# - $\text{Sorpresa}_{j, v} = x_{j, v} - \mathbb{E}(x_{j, v} \mid \Omega_{v-1})$ representa la innovación inesperada del indicador $j$ frente a la expectativa previa del modelo.
# - $\omega_{j, v} = \beta^\top \frac{\partial \mathbb{E}[F_t \mid \Omega_v]}{\partial x_{j, v}}$ es la ponderación econométrica determinada por la ganancia de Kalman y la relación señal-ruido del indicador.
#
# En los paneles inferiores:
# - **Panel Izquierdo (Sorpresas en los Datos)**: Muestra las sorpresas estandarizadas de cada publicación. Una barra positiva indica un desempeño superior al previsto.
# - **Panel Derecho (Impacto en la Revisión)**: Ilustra la contribución neta de cada sorpresa a la variación del PIB en puntos porcentuales. Las series de alta volatilidad idiosincrásica reciben pesos menores, evitando revisiones espurias.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

if not res.news_decomposition.empty:
    surp = res.news_decomposition["surprise"].to_numpy()
    colors1 = [_nbstyle.TINTA if s >= 0 else _nbstyle.NOTA for s in surp]
    bars1 = ax1.barh(res.news_decomposition["series"], surp, color=colors1, edgecolor=_nbstyle.SPINE)
    ax1.set_title("Sorpresa en la Publicación (Observado - Esperado)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Sorpresa (unidades σ)", color=_nbstyle.TEXTO)
    ax1.axvline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
    ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

    contrib = res.news_decomposition["contribution"].to_numpy()
    colors2 = [_nbstyle.TINTA if c >= 0 else _nbstyle.NOTA for c in contrib]
    bars2 = ax2.barh(res.news_decomposition["series"], contrib, color=colors2, edgecolor=_nbstyle.SPINE)
    ax2.set_title("Impacto en la Revisión del Nowcast del PIB", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Contribución (puntos porcentuales)", color=_nbstyle.TEXTO)
    ax2.axvline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
    ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
