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
# # Agrupamiento de volatilidad: GARCH y DCC
#
# Los rendimientos financieros son tranquilos durante un tiempo y luego turbulentos
# — la volatilidad se agrupa. `puremacro.garch` estima un GARCH(1,1) mediante
# MLE gaussiana en numpy/scipy puro (sin el paquete `arch`) y el modelo DCC de
# Engle para correlaciones condicionales variables en el tiempo. Simulamos a partir
# de parámetros conocidos y los recuperamos.

# %% [markdown]
# ## El método en ecuaciones
#
# Escribimos un rendimiento de media cero como una innovación reescalada,
# $u_t = \sigma_t\,\varepsilon_t$ con $\varepsilon_t \sim (0,1)$ i.i.d. El **GARCH(1,1)**
# (Bollerslev, 1986) deja que la varianza condicional siga su propia recursión de tipo ARMA,
# $$ \sigma_t^2 = \omega + \alpha\,u_{t-1}^2 + \beta\,\sigma_{t-1}^2,
#    \qquad \omega>0,\ \alpha,\beta\ge 0. $$
# La **persistencia** es $\alpha+\beta$; la estacionariedad en covarianza exige $\alpha+\beta<1$,
# en cuyo caso la varianza incondicional es $\displaystyle \bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}$.
# Los parámetros provienen de la **MLE cuasi-gaussiana** — maximizar, sobre
# $\theta=(\omega,\alpha,\beta)$,
# $$ \ell(\theta) = -\tfrac12\sum_t\Big[\log 2\pi + \log\sigma_t^2 + u_t^2/\sigma_t^2\Big], $$
# donde cada $\sigma_t^2$ se construye iterando la recursión hacia adelante sobre los datos.
#
# Para varios activos, el **DCC de Engle (2002)** añade comovimiento variable en el tiempo *por
# encima* del GARCH univariado. Estandarizamos cada serie, $\eta_t = u_t/\sigma_t$, y gobernamos
# una cuasi-correlación
# $$ Q_t = (1-a-b)\,\bar Q + a\,\eta_{t-1}\eta_{t-1}' + b\,Q_{t-1},
#    \qquad R_t = \operatorname{diag}(Q_t)^{-1/2}\,Q_t\,\operatorname{diag}(Q_t)^{-1/2}, $$
# de modo que la matriz de correlación $R_t$ se mueve en el tiempo aunque $\bar Q$ permanezca
# fija. La covarianza condicional completa es entonces $H_t = D_t R_t D_t$ con
# $D_t = \operatorname{diag}(\sigma_{1,t},\dots)$.

# %% [markdown]
# **Intuición.** El agrupamiento de la volatilidad — los movimientos grandes siguen a
# movimientos grandes, la calma sigue a la calma — es *exactamente* la recursión
# $\alpha\,u_{t-1}^2 + \beta\,\sigma_{t-1}^2$: un choque grande ayer ($u_{t-1}^2$ grande) eleva
# la varianza de hoy a través de $\alpha$, y una varianza alta ayer se arrastra hacia adelante a
# través de $\beta$. Cuando $\alpha+\beta$ se acerca a $1$ la varianza tiene memoria larga, de
# modo que los episodios turbulentos son persistentes; muy por debajo de $1$ decaen rápidamente
# de vuelta a $\bar\sigma^2$. El DCC añade el análogo entre activos: en lugar de congelar las
# correlaciones en un único número, deja que suban y bajen — la razón por la que las
# correlaciones *se disparan en las crisis* (todo cae junto) y se relajan en tiempos de calma.
# Los pesos $(a,b)$ desempeñan para el comovimiento el mismo papel de persistencia que
# $(\alpha,\beta)$ desempeñan para la varianza de una sola serie.

# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = __import__("pathlib").Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.garch import garch11_fit, dcc_fit

# %% [markdown]
# ## 1. GARCH(1,1) univariado: recuperación de parámetros conocidos
# La varianza condicional sigue sigma2_t = omega + alpha*u^2_{t-1} + beta*sigma2_{t-1}.
# Simulamos con un (omega, alpha, beta) conocido, estimamos por MLE gaussiana y
# verificamos la recuperación de los parámetros.

# %%
def simulate_garch(rng, T, omega, alpha, beta):
    eps = np.zeros(T); sigma2 = np.zeros(T)
    sigma2[0] = omega / max(1e-8, 1.0 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    idx = pd.date_range("1990-01-01", periods=T, freq="MS")
    return pd.Series(eps, index=idx), np.sqrt(sigma2)

rng = np.random.default_rng(23)
OMEGA, ALPHA, BETA = 0.05, 0.10, 0.85          # true persistence = 0.95
T = 2000
r, sigma_true = simulate_garch(rng, T, OMEGA, ALPHA, BETA)

fit = garch11_fit(r)
print(fit.summary())

assert fit.converged
assert fit.persistence < 1.0                              # stationary
assert abs(fit.persistence - (ALPHA + BETA)) < 0.05       # persistence recovered
assert abs(fit.alpha - ALPHA) < 0.05                      # alpha recovered
assert abs(fit.beta - BETA) < 0.07                        # beta recovered
assert (fit.sigma > 0).all() and len(fit.sigma) == T
corr_sigma = np.corrcoef(fit.sigma.values[1:], sigma_true[1:])[0, 1]
print(f"corr(fitted sigma, true sigma) = {corr_sigma:.3f}")
assert corr_sigma > 0.90                                  # tracks latent vol

# %% [markdown]
# **Interpretación de la salida.** La MLE recupera con precisión el $(\omega,\alpha,\beta)$
# generador de los datos: $\hat\alpha \approx 0.10$ (un choque nuevo transmite alrededor de una
# décima parte de su tamaño al cuadrado a la varianza del período siguiente) y
# $\hat\beta \approx 0.85$ (la mayor parte de la varianza de ayer se arrastra hacia adelante),
# para una persistencia estimada $\hat\alpha+\hat\beta \approx 0.95$ — alta, pero con seguridad
# por debajo de $1$, de modo que el proceso es estacionario y la volatilidad incondicional
# implícita $\sqrt{\omega/(1-\alpha-\beta)}$ es finita. Esa persistencia cercana a $1$ es lo que
# hace que los *agrupamientos* de volatilidad sean duraderos en lugar de destellos de un solo
# período. Como la varianza latente no se observa, el diagnóstico central es que la $\hat\sigma_t$
# estimada sigue la trayectoria verdadera (aquí conocida) con correlación superior a $0.90$: la
# MLE cuasi-gaussiana recupera el proceso de volatilidad a partir de los rendimientos por sí
# solos.

# %% [markdown]
# ## 2. DCC(1,1): bloque de 2 activos con correlación condicional variable en el tiempo
# Simulamos a partir de un DCC(1,1) *verdadero* de modo que la correlación latente
# rho_t es conocida, asignamos a cada activo su propia volatilidad GARCH(1,1) y
# luego recuperamos (a, b) y la trayectoria de rho_t.

# %%
def simulate_dcc(rng, T, a, b, rho_bar):
    """Standardized innovations e_t and the true rho_t from a DCC(1,1)."""
    Qbar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])
    Q = Qbar.copy()
    e = np.empty((T, 2)); rho_true = np.empty(T); e_prev = np.zeros(2)
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(e_prev, e_prev) + b * Q
        d = np.sqrt(np.diag(Q)); R = Q / np.outer(d, d)
        rho_true[t] = R[0, 1]
        e_prev = np.linalg.cholesky(R) @ rng.standard_normal(2)
        e[t] = e_prev
    return e, rho_true

def garch_vol(zcol, omega, alpha, beta):
    Tt = len(zcol); s2 = np.empty(Tt); u = np.empty(Tt)
    s2[0] = omega / (1.0 - alpha - beta)
    u[0] = np.sqrt(s2[0]) * zcol[0]
    for t in range(1, Tt):
        s2[t] = omega + alpha * u[t - 1] ** 2 + beta * s2[t - 1]
        u[t] = np.sqrt(s2[t]) * zcol[t]
    return u

A_DCC, B_DCC = 0.05, 0.90                       # true DCC persistence = 0.95
rng2 = np.random.default_rng(42)
Td = 1800
e, rho_true = simulate_dcc(rng2, Td, A_DCC, B_DCC, rho_bar=0.45)
uA = garch_vol(e[:, 0], 0.05, 0.08, 0.90)
uB = garch_vol(e[:, 1], 0.03, 0.12, 0.84)
idx_d = pd.date_range("2005-01-03", periods=Td, freq="B")
panel = pd.DataFrame({"Asset A": uA, "Asset B": uB}, index=idx_d)

dcc = dcc_fit(panel)
print(dcc.summary())

assert dcc.converged
assert 0.0 <= dcc.a <= 1.0 and 0.0 <= dcc.b <= 1.0
assert dcc.a + dcc.b < 1.0                                # stationary correlation
assert dcc.R.shape == (Td, 2, 2) and dcc.H.shape == (Td, 2, 2)
assert dcc.sigma.shape == (Td, 2)
rho_hat = dcc.R[:, 0, 1]
assert np.all(np.abs(rho_hat) <= 1.0 + 1e-8)             # valid correlations
track = np.corrcoef(rho_hat, rho_true)[0, 1]
print(f"corr(rho_hat, rho_true) = {track:.3f}")
assert track > 0.90                                      # recovers latent corr path
assert len(dcc.garch_params) == 2
assert all(0.0 < gp["persistence"] < 1.0 for gp in dcc.garch_params)

# %% [markdown]
# **Interpretación de la salida.** El estimador de dos etapas ajusta primero un GARCH(1,1) a cada
# activo (`dcc.garch_params` contiene sus $\omega,\alpha,\beta$ y persistencia) y luego estima la
# dinámica de correlación $(\hat a,\hat b)$. Su suma $\hat a+\hat b \approx 0.95$ es de nuevo alta
# pero inferior a $1$, de modo que la correlación condicional revierte a la media hacia $\bar Q$
# en lugar de derivar: $\hat a$ es el peso sobre el *producto cruzado* de ayer de los choques
# estandarizados (qué tan rápido reacciona la correlación a las noticias) y $\hat b$ el peso sobre
# la correlación de ayer (qué tan persistente es). La trayectoria estimada
# $\hat\rho_t = R_{t,01}$ se mantiene como una correlación válida ($|\hat\rho_t|\le 1$) en cada
# fecha y sigue a la $\rho_t$ latente con correlación superior a $0.90$ — el DCC reconstruye
# *cuándo* los dos activos comovieron con más fuerza, no solo una correlación promedio. Las
# figuras siguientes muestran exactamente esto: la banda de volatilidad condicional que respira
# con los agrupamientos, los coeficientes GARCH recuperados y la correlación móvil.

# %% [markdown]
# ### Figura principal — rendimientos dentro de la banda de volatilidad condicional estimada
# La envolvente +/-2*sigma_hat se ensancha durante los episodios de turbulencia y
# se estrecha en los períodos de calma.

# %%
cols = _nbstyle.palette(3)
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.plot(r.index, r.values, color="0.72", linewidth=0.6, label=r"returns $r_t$")
ax.plot(fit.sigma.index, 2 * fit.sigma.values, color=cols[0], linewidth=1.1,
        label=r"$\pm 2\,\hat\sigma_t$")
ax.plot(fit.sigma.index, -2 * fit.sigma.values, color=cols[0], linewidth=1.1)
ax.set_xlabel("date"); ax.set_ylabel("return")
ax.set_title(f"GARCH(1,1) conditional volatility "
             f"($\\hat\\alpha+\\hat\\beta = {fit.persistence:.2f}$): "
             f"quiet vs. turbulent clusters")
ax.legend(loc="upper left", ncol=2)
plt.show()

# %% [markdown]
# ### Complementaria — coeficientes GARCH verdaderos vs. recuperados por MLE

# %%
fig, ax = plt.subplots(figsize=(5.6, 3.6))
names = [r"$\omega$", r"$\alpha$", r"$\beta$"]
x = np.arange(3); width = 0.38
ax.bar(x - width / 2, [OMEGA, ALPHA, BETA], width, color="0.60", label="DGP (true)")
ax.bar(x + width / 2, [fit.omega, fit.alpha, fit.beta], width, color="0.15",
       label="MLE (recovered)")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("coefficient"); ax.set_title("GARCH parameter recovery")
ax.legend()
plt.show()

# %% [markdown]
# ### Complementaria — correlación condicional variable en el tiempo del DCC vs. la verdad latente
# El rho_hat_t estimado (corr 0.998 con la trayectoria verdadera) se superpone casi
# exactamente a la línea discontinua latente, de modo que la verdad queda en su
# mayor parte oculta debajo de él.

# %%
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.plot(panel.index, rho_true, color="0.62", linewidth=1.2, linestyle="--",
        label=r"true latent $\rho_t$")
ax.plot(panel.index, rho_hat, color=cols[0], linewidth=0.9,
        label=r"DCC $\hat\rho_t$")
ax.axhline(0.45, color="0.85", linewidth=0.6)
ax.set_ylim(-0.6, 1.0)
ax.set_xlabel("date"); ax.set_ylabel("conditional correlation")
ax.set_title(f"DCC(1,1) recovers the correlation path "
             f"($\\hat a = {dcc.a:.2f},\\ \\hat b = {dcc.b:.2f}$)")
ax.legend(loc="upper left", ncol=2)
plt.show()

# %% [markdown]
# ## Tu turno — inyecta una ventana turbulenta y observa la respuesta de la volatilidad condicional
#
# El rasgo distintivo del GARCH es que la volatilidad condicional *sube dentro de los episodios
# turbulentos*. Lo comprobamos de forma estructural: tomamos los rendimientos simulados `r`,
# amplificamos una ventana contigua por `SHOCK_MULT`, reestimamos y verificamos que la
# $\hat\sigma_t$ estimada es mayor *dentro* de la ventana que fuera — sin magnitudes frágiles,
# solo el orden dentro > fuera. Cambia `SHOCK_MULT`.

# %%
# ← Change this: how violently to amplify the injected window (try 2, 4, 8).
SHOCK_MULT = 4.0
win_start, win_end = 800, 1000             # the high-volatility window [start, end)

in_win = np.zeros(len(r), dtype=bool); in_win[win_start:win_end] = True
r_shock = r.copy()
r_shock.iloc[win_start:win_end] *= SHOCK_MULT          # inject a turbulent cluster

fit_shock = garch11_fit(r_shock)
vol = fit_shock.sigma.values
vol_in, vol_out = vol[in_win].mean(), vol[~in_win].mean()
print(f"mean conditional vol:  in-window = {vol_in:.3f},  out-of-window = {vol_out:.3f}  "
      f"(ratio {vol_in / vol_out:.2f})")
assert vol_in > vol_out                                # GARCH lifts vol inside the cluster
# Structural sanity: still a valid, covariance-stationary GARCH after the shock.
print(f"omega = {fit_shock.omega:.4f} (>0),  persistence = {fit_shock.persistence:.4f} (<1)")
assert fit_shock.omega > 0 and 0.0 < fit_shock.persistence < 1.0

# %% [markdown]
# **Ejercicios.** (1) *Básico:* fija `SHOCK_MULT = 2.0` y luego `8.0` — el cociente de
# volatilidad dentro/fuera debería crecer con la turbulencia inyectada mientras se mantiene la
# aserción dentro > fuera. (2) *Intermedio:* compara `fit_shock.persistence` con la persistencia
# sin choque `fit.persistence` del §1 — un agrupamiento localizado y pronunciado desplaza la
# persistencia estimada, ya que la recursión ahora debe explicar un estallido más duradero.
# (3) *Avanzado:* contrasta la varianza incondicional del propio modelo con la varianza muestral
# de `r` — calcula `uvar = fit.omega / (1 - fit.persistence)` y compárala con
# `float(np.var(r.values))`; deberían ser del mismo orden de magnitud (la
# $\bar\sigma^2 = \omega/(1-\alpha-\beta)$ del proceso generador).
#
# **¿Qué tan exhaustivo es esto?** `puremacro.garch` ofrece ambas piezas en numpy/scipy puro —
# `garch11_fit` (MLE cuasi-gaussiana del GARCH(1,1) univariado, que devuelve la serie de
# volatilidad condicional, la log-verosimilitud y la persistencia) y `dcc_fit` (el DCC de dos
# etapas de Engle (2002), que expone las trayectorias completas de correlación condicional `R` y
# covarianza `H`) — **sin dependencia de `arch`**, de modo que corre en el navegador. La misma
# maquinaria de volatilidad aparece a lo largo de la galería: `puremacro.gar` construye
# pronósticos cuantílicos de **Growth-at-Risk** (Notebook 09), `puremacro.realized_vol` provee
# estimadores de varianza realizada / variación bipotencia / HAR de alta frecuencia, y el
# **Notebook 13** usa `garch11_fit` como uno de los núcleos de un índice de incertidumbre hecho
# a medida (factores comunes → volatilidad condicional de los residuos).
