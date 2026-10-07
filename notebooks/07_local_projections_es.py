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
# # Proyecciones locales: efectos causales dinámicos
#
# **¿Cómo se propaga un choque económico a lo largo de la macroeconomía en los años siguientes, y se intensifica su transmisión durante recesiones en comparación con expansiones económicas?**
# Las proyecciones locales de Jordà (2005) responden a esto con una regresión por
# horizonte —robustas ante una especificación dinámica errónea y fáciles de volver
# *dependientes del estado del ciclo*. Plantamos un choque sintético cuyo efecto es
# más intenso en recesiones y recuperamos tanto las respuestas lineales como las
# específicas por régimen con `puremacro.lp`.
#

# %% [markdown]
# ## El método en matemáticas
#
# Para trazar el efecto causal dinámico de un choque $s_t$ sobre una variable $y$, Jordà
# (2005) corre **una regresión por horizonte** $h$, proyectando el cambio acumulado
# de la variable desde $t-1$ sobre el choque de hoy:
# $$ y_{t+h} - y_{t-1} = \alpha_h + \beta_h\, s_t + \gamma_h' x_t + e_{t+h}, \qquad h = 0, 1, \dots, H. $$
# La función de impulso-respuesta *es* la sucesión de coeficientes de pendiente
# $\{\beta_h\}_{h=0}^{H}$, leída horizonte a horizonte —sin un VAR que invertir e iterar
# hacia adelante. Los controles $x_t$ (aquí rezagos de $y$ y de $s$) absorben la dinámica
# predecible para que $\beta_h$ aísle el efecto del choque.
#
# **Inferencia.** Apilar horizontes implica que el residuo $e_{t+h}$ se solapa con su propio
# futuro: $e_{t+h}$ y $e_{t+1+h}$ comparten $h$ períodos de innovaciones, por lo que están
# correlacionados serialmente *por construcción*. Las pendientes de MCO siguen siendo
# consistentes, pero sus errores estándar de manual son incorrectos. Empleamos por ello
# errores estándar **HAC (Newey–West)** con un núcleo de Bartlett y rezago de truncamiento
# $h+1$, y reportamos bandas $\beta_h \pm z_{1-\alpha/2}\, \mathrm{se}(\beta_h)$. Ese rezago
# de truncamiento es una regla práctica, no una recomendación publicada: el residuo a $h$
# pasos está correlacionado serialmente hasta el orden $h$, así que el núcleo debe llegar
# al menos hasta ahí.
#
# **Dependencia del estado.** Como cada horizonte es su propia regresión, la no linealidad
# encaja sin costo. Interactúe el choque con un indicador de estado $I_t$ (p. ej. $1$ en
# recesiones) para obtener respuestas específicas por régimen en una sola ecuación:
# $$ y_{t+h} = I_t\big(\alpha_h^{R} + \beta_h^{R} s_t\big) + (1-I_t)\big(\alpha_h^{E} + \beta_h^{E} s_t\big) + \gamma_h' x_t + e_{t+h}, $$
# que entrega una FIR de recesión $\{\beta_h^{R}\}$ y una FIR de expansión $\{\beta_h^{E}\}$,
# cada una con su propia banda HAC.
#
# ### Parametrización base
#
# | Símbolo | Rol económico / econométrico | Calibración base | Unidades físicas |
# |---|---|---|---|
# | $s_t$ | Choque de política estructural exógeno | i.i.d. sintético | Desviaciones estándar ($\sigma$) |
# | $y_t$ | Variable de resultado macroeconómica | AR(1) persistente | Puntos de índice |
# | $I_t$ | Indicador binario de recesión | Umbral en ciclo latente ($< 35\%$) | Variable binaria $\{0, 1\}$ |
# | $h$ | Horizonte del impulso de proyección | $0, 1, \dots, 16$ | Meses |
# | $P$ | Rezagos de variables de control | $2$ | Rezagos (meses) |
# | $\alpha$ | Nivel de significancia del intervalo de confianza | $0.10$ (confianza del $90\%$) | Masa de probabilidad en colas |
# | $M$ | Ancho de banda de truncamiento HAC Newey-West | $h + 1$ | Rezagos (meses) |
# | $\beta_h^R, \beta_h^E$ | Coeficientes de respuesta específicos por régimen | Verdadero $\beta^R = -0.90, \beta^E = -0.25$ | Respuesta por unidad de choque |
#
# **Intuición.** Las proyecciones locales estiman cada horizonte *por separado*, de modo que
# —a diferencia del SVAR del Cuaderno 6, que impone una única ley de movimiento paramétrica y
# la itera hacia adelante— una dinámica de corto plazo mal especificada no contamina la
# respuesta de horizonte largo, y una partición por estado es apenas un término de
# interacción en lugar de un segundo modelo completo. El precio es la *eficiencia*: descartar
# las restricciones entre horizontes vuelve las estimaciones más ruidosas, sobre todo a lo
# lejos, donde las ventanas solapadas reducen la muestra efectiva y las bandas HAC se abren
# en abanico. Las PL compran robustez y flexibilidad con varianza; el SVAR compra precisión
# con estructura.
#
# ### Referencias bibliográficas seminales
#
# - Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review*, 95(1), 161–182.
# - Montiel Olea, J. L., & Plagborg-Møller, M. (2021). Local projection inference is simpler and more robust than you think. *Econometrica*, 89(4), 1789–1823.
# - Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
# - Plagborg-Møller, M., & Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. *Econometrica*, 89(2), 955–980.
# - Ramey, V. A., & Zubairy, S. (2018). Government spending multipliers in good times and in bad: Evidence from US historical data. *Journal of Political Economy*, 126(2), 850–901.

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

from puremacro.lp.jorda import lp_hac
from puremacro.lp.state_dep import lp_state_dep

# Pyodide contract: after importing the functions we use, no forbidden module
# should have been pulled in. Both LP paths go through the pure-numpy ols_hac.
_bad = [m for m in ("statsmodels", "linearmodels", "arch", "bs4", "requests", "numba")
        if m in sys.modules]
assert _bad == [], f"forbidden modules imported: {_bad}"

# %% [markdown]
# ## 1. Un choque sintético con propagación dependiente del estado
# Un choque i.i.d. unitario incide sobre una variable cuyo impacto es *mayor en
# recesiones*. El indicador dicotómico de recesión se construye a partir de un
# factor latente persistente del ciclo económico.

# %%
rng = np.random.default_rng(20240529)
T = 360

x = rng.standard_normal(T)                       # identified structural shock
g = np.zeros(T)                                  # persistent business-cycle factor
for t in range(1, T):
    g[t] = 0.85 * g[t - 1] + rng.standard_normal() * 0.5
recession = (g < np.quantile(g, 0.35)).astype(float)   # binary state dummy

beta_exp, beta_rec = -0.25, -0.90                # impact in expansion vs recession
y = np.zeros(T)
for t in range(1, T):
    impact = beta_rec if recession[t] else beta_exp
    y[t] = (0.55 * y[t - 1]
            + impact * x[t - 1]
            + 0.30 * impact * x[t]
            + rng.standard_normal() * 0.35)

df = pd.DataFrame(
    {"y": y, "shock": x, "recession": recession},
    index=pd.date_range("1985-01-01", periods=T, freq="MS"),
)
print(f"recession sample share = {df['recession'].mean():.2f}")

H = 16
horizons = range(0, H + 1)

# %% [markdown]
# ## 2. Proyección local lineal con corrección HAC
# `lp_hac` regresa `y_{t+h} - y_{t-1}` sobre el choque más sus rezagos, con
# errores estándar HAC de Newey-West (ancho de banda h+1) y bandas de confianza
# al 90%.

# %%
lin = lp_hac(df, y="y", x="shock", horizons=horizons, n_lags=2, alpha=0.10)
print(lin.head().round(3).to_string(index=False))

# The planted average response: the shock is independent of the state, so the linear LP
# estimates the state-average of the DGP's impulse response, with b_bar the mean impact.
b_bar = df["recession"].mean() * beta_rec + (1 - df["recession"].mean()) * beta_exp
psi_bar = np.r_[0.3 * b_bar, (1 + 0.55 * 0.3) * b_bar * 0.55 ** np.arange(H)]
covered = (lin["lo"].values <= psi_bar) & (psi_bar <= lin["hi"].values)
print(f"planted average IRF: h=0 {psi_bar[0]:+.3f}, h=1 {psi_bar[1]:+.3f}; "
      f"90% bands containing it: {int(covered.sum())} of {H + 1} horizons")
print(f"se: {lin['se'].iloc[0]:.3f} at h=0, {lin['se'].iloc[1:13].min():.3f}-"
      f"{lin['se'].iloc[1:13].max():.3f} at h=1-12, {lin['se'].iloc[13:].min():.3f}-"
      f"{lin['se'].iloc[13:].max():.3f} at h=13-16")

# Bands present, finite, and correctly ordered.
assert set(lin.columns) >= {"h", "beta", "se", "t", "lo", "hi"}
assert np.all(np.isfinite(lin[["beta", "se", "lo", "hi"]].values))
assert np.all(lin["se"].values > 0)
assert np.all(lin["lo"].values <= lin["beta"].values + 1e-9)
assert np.all(lin["beta"].values <= lin["hi"].values + 1e-9)
z95 = 1.6448536269514722                          # z_{0.95} for the 90% band
assert np.allclose(lin["hi"] - lin["lo"], 2 * z95 * lin["se"], atol=1e-6)

# %% [markdown]
# **Lectura de los resultados.** Cada fila es la regresión de un horizonte: `beta` es $\beta_h$, la
# respuesta de $y$ a un choque unitario $h$ meses adelante, y `[lo, hi]` es su banda HAC al
# 90%. La estimación es −0.126 en el impacto y alcanza su mínimo, −0.569, en $h=1$, porque
# el efecto principal del proceso generador entra por $x_{t-1}$; después decae hacia cero
# aproximadamente a la tasa AR de 0.55. El choque es independiente del estado, así que la PL
# lineal apunta a la respuesta promedio entre estados: los valores plantados son −0.143 en
# el impacto y −0.556 en $h=1$, y las bandas al 90% contienen la senda plantada en 16 de los
# 17 horizontes. Eso es lo que deben hacer unas bandas puntuales al 90% en una muestra; el
# segundo ejercicio de abajo lo comprueba a lo largo de muchas. Los errores estándar
# muestran el costo de eficiencia de estimar cada horizonte por separado: 0.026 en el
# impacto, 0.034–0.047 en $h=1$–12 y 0.049–0.055 en $h=13$–16, a medida que crece el rezago
# de truncamiento HAC y se reduce la muestra de ventanas solapadas. El último `assert` es la
# *definición* de la banda, ancho $=2 z_{0.95}\,\mathrm{se}$: el nivel de confianza cambia
# solo ese multiplicador, nunca la estimación puntual.

# %% [markdown]
# ## 3. Funciones de impulso-respuesta dependientes del estado: recesión vs. expansión
# `lp_state_dep` con `transition="threshold"` descompone el choque en una
# interacción con el estado alto (recesión) y el estado bajo (expansión), y
# devuelve coeficientes y bandas HAC separadas para cada régimen.

# %%
sd = lp_state_dep(df, y="y", x="shock", state="recession",
                  horizons=horizons, n_lags=2,
                  transition="threshold", alpha=0.10)
print(sd[["h", "beta_H", "beta_L"]].head().round(3).to_string(index=False))

assert {"beta_H", "se_H", "lo_H", "hi_H",
        "beta_L", "se_L", "lo_L", "hi_L"}.issubset(sd.columns)
assert np.all(np.isfinite(sd[["beta_H", "beta_L", "se_H", "se_L"]].values))
assert np.all(sd["lo_H"].values <= sd["hi_H"].values)
assert np.all(sd["lo_L"].values <= sd["hi_L"].values)

peak_H = sd["beta_H"].min()        # recession regime (state dummy = 1)
peak_L = sd["beta_L"].min()        # expansion regime
print(f"peak recession beta_H = {peak_H:.3f}   peak expansion beta_L = {peak_L:.3f}   "
      f"ratio {peak_H / peak_L:.1f}")
assert peak_H < peak_L - 0.10, (peak_H, peak_L)        # recession bites harder
gap = sd["hi_H"].values < sd["lo_L"].values            # H-band entirely below L-band
print(f"horizons with non-overlapping regime bands: {int(gap.sum())}")
assert gap.any(), "expected at least one horizon where the regimes separate"

# %% [markdown]
# **Lectura de los resultados.** Ahora hay *dos* funciones de impulso-respuesta a partir de una sola
# regresión. `beta_H` es la respuesta en recesión (estado alto, $I_t=1$) y `beta_L` la
# respuesta en expansión. Ambas son negativas, pero el mínimo en recesión es mucho más
# profundo: −0.832 frente a −0.398, una razón de 2.1, cuando el proceso generador plantó
# coeficientes de impacto $\beta^{R}=-0.90$ y $\beta^{E}=-0.25$. El objeto
# económicamente interesante es la *brecha*: en los 2 horizontes contados arriba, la banda al
# 90% de recesión queda *enteramente por debajo* de la banda de expansión (`hi_H < lo_L`), de
# modo que la diferencia no es solo una estimación puntual —sobrevive a la inferencia HAC.
# Este es el rédito de las proyecciones locales: una FIR dependiente del estado, con bandas de
# incertidumbre específicas por régimen, a partir de una regresión interactuada en lugar de
# dos modelos separados. (Las bandas son *por régimen*; una prueba formal de
# $\beta_h^{R}=\beta_h^{E}$ usaría su covarianza conjunta —el no solapamiento es una señal
# suficiente y conservadora.)

# %% [markdown]
# ### Figura principal — proyección local lineal con corrección HAC y bandas al 90%

# %%
cols = _nbstyle.palette(3)
fig, ax = _nbstyle.figura()
ax.axhline(0.0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax.fill_between(lin["h"], lin["lo"], lin["hi"], color=_nbstyle.BANDA, label="90% HAC band")
ax.plot(lin["h"], lin["beta"], color=cols[0], marker="o", markersize=3,
        label=r"$\beta_h$ (linear LP)")
ax.set_xlabel("Horizon h (months)")
ax.set_ylabel("Response of y to a unit shock")
ax.set_title("Local-projection IRF (Jordà, HAC bands)")
ax.legend(loc="lower right")

# %% [markdown]
# ### Figura complementaria — funciones de impulso-respuesta por régimen
# La respuesta en recesión (línea continua) es marcadamente más profunda que la
# respuesta en expansión (línea discontinua); las regiones sombreadas corresponden
# a las bandas HAC al 90% de cada régimen.

# %%
fig, ax = _nbstyle.figura()
ax.axhline(0.0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax.fill_between(sd["h"], sd["lo_H"], sd["hi_H"], color=cols[1], alpha=0.25)
ax.fill_between(sd["h"], sd["lo_L"], sd["hi_L"], color=cols[2], alpha=0.25)
ax.plot(sd["h"], sd["beta_H"], color=cols[1], marker="o", markersize=3,
        label="Recession")
ax.plot(sd["h"], sd["beta_L"], color=cols[2], marker="s", markersize=3,
        linestyle="--", label="Expansion")
ax.set_xlabel("Horizon h (months)")
ax.set_ylabel("Response of y to a unit shock")
ax.set_title("State-dependent IRFs: shocks bite harder in recessions")
ax.legend(loc="lower right")

# %% [markdown]
# ### Figura complementaria — las series sintéticas y el estado del ciclo

# %%
fig, (axL, axR) = _nbstyle.figura(1, 2, ancho=9.5, alto=3.4)
t_idx = np.arange(T)
rec = df["recession"].values
axL.plot(t_idx, df["y"].values, color=_nbstyle.TINTA, linewidth=0.8)
axL.fill_between(t_idx, df["y"].min(), df["y"].max(), where=rec > 0,
                 color=_nbstyle.BANDA, step="mid", label="recession")
axL.set_xlabel("t"); axL.set_ylabel("y")
axL.set_title("Outcome with recession shading"); axL.legend(loc="upper right")
axR.bar(["expansion", "recession"], [(1 - rec).mean(), rec.mean()],
        color=[cols[2], cols[1]])
axR.set_ylabel("sample share"); axR.set_title("State frequencies")

# %% [markdown]
# ## Tu turno — califica las proyecciones locales contra una verdad plantada
#
# Cuatro ejercicios, de básico a avanzado. Cada celda corre tal como está. Cambia el control
# marcado con `# ←` dentro de su rango anunciado, predice el resultado antes de correrlo y
# deja que los `assert` califiquen a los estimadores contra la verdad plantada en la
# simulación.
#
# ### 1. Básico — la PL suave anida a la de Jordà: $\lambda = 0$ apaga la penalización
#
# Las proyecciones locales suaves de Barnichon–Brownlees (2019) escriben
# $\beta_h = \sum_k b_k(h)\,\theta_k$ sobre una base de B-splines cúbicos, ajustan todos los
# horizontes a la vez y penalizan $\lambda\,\theta' P_d\,\theta$, donde $P_d$ es una
# penalización de rugosidad por diferencias de orden $d$ sobre los coeficientes del spline.
# Con `n_knots = H - 3` la base tiene $H+1$ funciones, una por horizonte. Predice tres cosas
# primero: (a) a qué debe ser igual el ajuste con $\lambda = 0$; (b) sus grados de libertad
# efectivos cuando $\lambda \to \infty$ (¿cuál es el espacio nulo de una penalización por
# diferencias de orden $d$?); (c) si el ajuste elegido por GCV queda entre los dos límites,
# tanto en grados de libertad como en rugosidad.

# %%
from puremacro.lp.smooth import lp_smooth

# ← change this: order d of the roughness penalty (advertised range: 1, 2 or 3)
your_order = 2
full = dict(y="y", x="shock", horizons=horizons, n_lags=2, n_knots=H - 3)
s_zero = lp_smooth(df, **full, lam=0.0)
s_inf = lp_smooth(df, **full, lam=1e12, penalty_order=your_order)
s_gcv = lp_smooth(df, **full, selection="gcv")
rough = lambda b: float(np.sum(np.diff(np.asarray(b), 2) ** 2))   # sum of squared 2nd differences
print(f"lambda=0: max |beta_smooth - beta_Jorda| = "
      f"{np.max(np.abs(s_zero['beta'].values - lin['beta'].values)):.1e}")
print(f"effective df: lambda=0 -> {s_zero.df_lambda:.2f} | GCV lambda={s_gcv.optimal_lambda:.3g} -> "
      f"{s_gcv.df_lambda:.2f} | lambda=inf, order {your_order} -> {s_inf.df_lambda:.3f}")
print(f"roughness: Jorda {rough(lin['beta']):.4f} | GCV-smooth {rough(s_gcv['beta']):.4f}")

assert np.allclose(s_zero["beta"].values, lin["beta"].values, atol=1e-10)   # lambda = 0 is Jorda
assert abs(s_inf.df_lambda - your_order) < 1e-3     # the penalty's null space has dimension d
assert 2.0 < s_gcv.df_lambda < H + 1                # GCV shrinks, but not all the way
assert rough(s_gcv["beta"]) < rough(lin["beta"])    # and the smoothed IRF is less rough

# %% [markdown]
# **Ejercicio.** Fija `your_order = 1` y predice los grados de libertad límite antes de
# correrlo. Suavizar es contraer hacia el espacio nulo de la penalización: una FIR plana para
# $d = 1$, y para $d = 2$ una curva cuyos coeficientes de spline están sobre una recta. ¿Por
# qué ese límite con $d = 2$ se parece mucho, pero no exactamente, a una recta en $h$? (Pista:
# los nodos de frontera están fijados.)
#
# ### 2. Intermedio — ¿cubren las bandas al 90% el 90% de las veces?
#
# Una banda al 90% es una promesa sobre muestras repetidas: a lo largo de muchas muestras
# debería contener el $\beta_h$ verdadero el 90% de las veces. Pon a prueba la promesa donde
# conoces la verdad. El proceso generador reutiliza el bucle de la sección 1 con un único
# impacto $b = -0.5$ y sin partición por recesión,
# $y_t = \phi\, y_{t-1} + b\, x_{t-1} + 0.3\, b\, x_t + 0.35\, e_t$ con $T = 360$, en
# $R = 200$ réplicas (semilla $1000 + r$; se sortea $x$ y luego $e$). Deriva a mano la FIR
# verdadera primero: $\psi_0 = 0.3 b$, $\psi_1 = \phi\psi_0 + b$, $\psi_h = \phi\,\psi_{h-1}$.
# (La variable dependiente de `lp_hac` es $y_{t+h} - y_{t-1}$, pero su pendiente sobre $x_t$ es
# la misma porque $y_{t-1}$ ya es un control.) Predice: ¿será la cobertura exactamente 0.90 en
# cada horizonte?
#
# La celda también corre `la_lp`, la PL aumentada con rezagos de Montiel Olea y
# Plagborg-Møller (2021), sobre las mismas simulaciones: dos rezagos más un rezago de aumento,
# con errores estándar de Eicker-Huber-White y sin corrección HAC.

# %%
import math
from puremacro.lp.la_lp import la_lp

# ← change this: persistence phi of y (advertised range 0.0 to 0.7)
phi_mc = 0.55
R, T_mc, b_mc = 200, 360, -0.5
psi = np.empty(H + 1)
psi[0] = 0.3 * b_mc
psi[1] = phi_mc * psi[0] + b_mc
for h in range(2, H + 1):
    psi[h] = phi_mc * psi[h - 1]

betas, ses = np.empty((R, H + 1)), np.empty((R, H + 1))
hit, hit_la = np.empty((R, H + 1), bool), np.empty((R, H + 1), bool)
width, width_la = np.empty((R, H + 1)), np.empty((R, H + 1))
for r in range(R):
    g_r = np.random.default_rng(1000 + r)
    x_r = g_r.standard_normal(T_mc)
    e_r = g_r.standard_normal(T_mc)
    y_r = np.zeros(T_mc)
    for t in range(1, T_mc):
        y_r[t] = phi_mc * y_r[t - 1] + b_mc * x_r[t - 1] + 0.3 * b_mc * x_r[t] + 0.35 * e_r[t]
    d_r = pd.DataFrame({"y": y_r, "shock": x_r})
    res = lp_hac(d_r, y="y", x="shock", horizons=horizons, n_lags=2, alpha=0.10)
    la = la_lp(d_r, y="y", x="shock", horizons=horizons, n_lags=2, alpha=0.10)
    betas[r], ses[r] = res["beta"].values, res["se"].values
    hit[r] = (res["lo"].values <= psi) & (psi <= res["hi"].values)
    hit_la[r] = (la["lo"].values <= psi) & (psi <= la["hi"].values)
    width[r], width_la[r] = (res["hi"] - res["lo"]).values, (la["hi"] - la["lo"]).values

cov, cov_la = hit.mean(0), hit_la.mean(0)
ratio = ses.mean(0) / betas.std(0)                  # average SE over the true sampling s.d.
pred = math.erf(z95 * ratio.mean() / math.sqrt(2))  # coverage of a normal band with that SE ratio
bias_sd = (betas.mean(0) - psi) / betas.std(0)
print("lp_hac coverage by h:", np.array2string(cov, precision=3))
print(f"lp_hac: mean coverage {cov.mean():.3f} (Monte Carlo s.e. of one horizon "
      f"{math.sqrt(0.09 / R):.3f}); se/sd {ratio.min():.2f}-{ratio.max():.2f}, mean "
      f"{ratio.mean():.3f}; erf prediction {pred:.3f} (gap {cov.mean() - pred:+.4f}); "
      f"max |bias|/sd {np.abs(bias_sd).max():.2f}")
print(f"la_lp:  mean coverage {cov_la.mean():.3f} (range {cov_la.min():.3f}-{cov_la.max():.3f}) "
      f"with p_aug = {la.attrs['p_aug']}; band width la_lp / lp_hac = "
      f"{(width_la.mean(0) / width.mean(0)).mean():.3f}")

assert 0.80 < cov.mean() < 0.95 and 0.80 < cov_la.mean() < 0.95
assert cov[0] >= 0.85 and cov[1] >= 0.85            # also checks the psi_0 and psi_1 derivation
assert abs(cov.mean() - pred) < 0.015               # the SE ratio explains the shortfall

# %% [markdown]
# **Ejercicios.** (a) ¿Es el faltante respecto a 0.90 mayor que el ruido de Monte Carlo? La
# línea `erf` es la cobertura de una banda normal cuyo error estándar está desviado en la
# razón promedio impresa: úsala para decir si las bandas cubren de menos porque son demasiado
# estrechas o porque las estimaciones están sesgadas. (b) Compara la línea de `la_lp`. ¿Son
# más anchas sus bandas, y cubren mejor? El choque es i.i.d. y dos rezagos ya capturan la
# dinámica, así que ¿qué compra aquí el rezago extra y qué cuesta? Luego vuelve a correr
# `la_lp` con `extra_lags=H`, el valor por defecto de la biblioteca antes de la corrección
# (puremacro 4.3.0 y anteriores: $2 + 16 = 18$ rezagos en cada horizonte), y predice primero
# si más rezagos ayudan. (c) *Avanzado*: fija `phi_mc = 0.95`, fuera del rango anunciado, y
# predice qué `assert` falla. Calcula `bias_sd[8:].mean()` con 0.95 y con 0.55: ¿crece el
# sesgo de horizonte largo, y puede la razón de errores estándar por sí sola seguir
# explicando la cobertura?
#
# ### 3. Intermedio — una asimetría de signo, y el promedio que reporta en su lugar una PL lineal
#
# Planta una asimetría de signo, en el espíritu de la idea de "empujar una cuerda" según la
# cual endurecer muerde más que relajar:
# $y_t = 0.5\, y_{t-1} + a_+ \max(x_{t-1}, 0) + a_- \min(x_{t-1}, 0) + e_t$ con $a_+ = 1.0$.
# `lp_asymmetric` divide $x_t$ en su parte positiva y su parte negativa; la `lp_hac` lineal
# corre sobre los mismos datos. Deriva primero las respuestas verdaderas: 0 en $h = 0$ y
# $a\, 0.5^{h-1}$ para $h \ge 1$. Luego predice dónde cae la PL lineal. No es ninguna de las
# dos respuestas sino un promedio ponderado, $w\, a_+ + (1 - w)\, a_-$ con
# $w = \mathrm{Cov}(x, x^+)/\mathrm{Var}(x)$ y $x^+ = \max(x, 0)$.

# %%
from puremacro.lp.asymmetric import lp_asymmetric

# ← change this: shock distribution, "normal" or "skewed" (right-skewed with mean zero)
shock_dist = "normal"
# ← change this: planted response to negative shocks, 0.3 (asymmetric) or 1.0 (= a_pos)
a_neg = 0.3
a_pos = 1.0
rng_a = np.random.default_rng(7)
Ta = 500
xa = rng_a.standard_normal(Ta) if shock_dist == "normal" else rng_a.exponential(1.0, Ta) - 1.0
ea = 0.5 * rng_a.standard_normal(Ta)
ya = np.zeros(Ta)
for t in range(1, Ta):
    ya[t] = 0.5 * ya[t - 1] + a_pos * max(xa[t - 1], 0.0) + a_neg * min(xa[t - 1], 0.0) + ea[t]
dfa = pd.DataFrame({"y": ya, "shock": xa})
asy = lp_asymmetric(dfa, y="y", x="shock", horizons=range(0, 9), n_lags=2, alpha=0.10)
lin_a = lp_hac(dfa, y="y", x="shock", horizons=range(0, 9), n_lags=2, alpha=0.10)
r1 = asy[asy["h"] == 1].iloc[0]                    # true h=1 responses: a_pos and a_neg
b1, se1 = float(lin_a["beta"].iloc[1]), float(lin_a["se"].iloc[1])
w_pos = np.cov(xa, np.maximum(xa, 0))[0, 1] / np.var(xa, ddof=1)
pred_lin = w_pos * a_pos + (1 - w_pos) * a_neg
separated = bool(r1["hi_neg"] < r1["lo_pos"])
print(f"h=1: beta_pos = {r1['beta_pos']:.3f} (planted {a_pos})  beta_neg = {r1['beta_neg']:.3f} "
      f"(planted {a_neg});  90% bands separated: {separated}")
print(f"h=1: linear LP beta = {b1:.3f}  vs  w*a_pos + (1-w)*a_neg = {pred_lin:.3f} (w = {w_pos:.2f})")

assert abs(r1["beta_pos"] - a_pos) < 3 * r1["se_pos"] and abs(r1["beta_neg"] - a_neg) < 3 * r1["se_neg"]
assert abs(b1 - pred_lin) < 3 * se1                 # the linear LP is the w-weighted average
assert separated == (a_pos - a_neg > 0.5)           # the bands separate when the planted gap is large

# %% [markdown]
# **Ejercicios.** (a) Fija `a_neg = 1.0`, de modo que no haya asimetría que encontrar, y
# predice qué imprime `separated`. (b) Cambia a `shock_dist = "skewed"`. Predice primero cómo
# se mueve $w$ y hacia dónde va con él la FIR lineal. El "efecto promedio" que reporta una PL
# lineal depende de la distribución de los choques, no solo de la economía.
#
# ### 4. Avanzado — ¿vale la pena suavizar? Un chequeo de Monte Carlo de sesgo y varianza sobre una joroba plantada
#
# Una sola muestra no puede decir cuánta varianza elimina el suavizado, así que corre un Monte
# Carlo. La joroba plantada es $y_t = 0.4\, y_{t-1} + \sum_{k=0}^{9} w_k\, x_{t-k} + e_t$ con
# $w = (0, .2, .5, .8, 1, 1, .8, .5, .2, 0)$, $x_t$ i.i.d. $N(0, 1)$,
# $e_t \sim N(0, \sigma_e^2)$ y $T = 200$. La FIR verdadera sigue
# $\psi_h = 0.4\, \psi_{h-1} + w_h$, con su máximo en $h = 5$. Para las semillas 0 a 59,
# estima `lp_hac` y `lp_smooth(..., n_knots=4, selection="gcv")` en $h = 0, \dots, 12$ y
# compara la varianza sumada, el ECM sumado contra $\psi$, el sesgo en el máximo y la
# correlación de los errores de Jordà en $h = 5$ y $h = 6$.

# %%
# ← change this: noise s.d. sigma_e (advertised range 1.5 to 4.0)
noise_sd = 2.0
w_hump = np.array([0, .2, .5, .8, 1, 1, .8, .5, .2, 0])
Hm, T_sm = 12, 200
psi_h = np.zeros(Hm + 1)
for h in range(Hm + 1):
    psi_h[h] = (0.4 * psi_h[h - 1] if h else 0.0) + (w_hump[h] if h < len(w_hump) else 0.0)
raw, smo = [], []
for s in range(60):
    g_s = np.random.default_rng(s)
    x_s = g_s.standard_normal(T_sm + 20)
    e_s = noise_sd * g_s.standard_normal(T_sm + 20)
    y_s = np.zeros(T_sm + 20)
    for t in range(10, T_sm + 20):                  # 20 burn-in periods are dropped below
        y_s[t] = 0.4 * y_s[t - 1] + w_hump @ x_s[t - np.arange(10)] + e_s[t]
    d_s = pd.DataFrame({"y": y_s[20:], "shock": x_s[20:]})
    raw.append(lp_hac(d_s, y="y", x="shock", horizons=range(Hm + 1), n_lags=2)["beta"].values)
    smo.append(lp_smooth(d_s, y="y", x="shock", horizons=range(Hm + 1), n_lags=2,
                         n_knots=4, selection="gcv")["beta"].values)
raw, smo = np.array(raw), np.array(smo)
var_ratio = smo.var(0).sum() / raw.var(0).sum()
mse_raw, mse_smo = ((raw - psi_h) ** 2).mean(0).sum(), ((smo - psi_h) ** 2).mean(0).sum()
bias5_raw, bias5_smo = raw[:, 5].mean() - psi_h[5], smo[:, 5].mean() - psi_h[5]
corr56 = np.corrcoef(raw[:, 5] - psi_h[5], raw[:, 6] - psi_h[6])[0, 1]
print(f"noise_sd={noise_sd}: var(smooth)/var(Jorda) = {var_ratio:.3f}; "
      f"MSE Jorda {mse_raw:.3f} vs smooth {mse_smo:.3f}")
print(f"peak (h=5, true {psi_h[5]:.3f}) bias: Jorda {bias5_raw:+.3f}, smooth {bias5_smo:+.3f}; "
      f"corr of Jorda errors at h=5,6: {corr56:.2f}")

assert var_ratio < 0.95                             # smoothing removes variance...
assert bias5_smo < bias5_raw                        # ...shaves the peak (bias)...
assert mse_smo < mse_raw                            # ...and wins on MSE at this noise level

# %% [markdown]
# **Ejercicio.** Fija `noise_sd = 0.5`, por debajo del rango anunciado; es el nivel de ruido
# de `puremacro/examples/lp_smooth_demo.py`. Predice qué `assert` fallan antes de correrlo y
# luego usa la correlación de errores impresa para explicar por qué. Cuando los errores de la
# PL en horizontes adyacentes están casi perfectamente correlacionados, el ruido de estimación
# es él mismo suave en $h$: una penalización de rugosidad no puede quitarlo y solo recorta la
# joroba. Suavizar paga cuando los errores horizonte a horizonte están menos correlacionados.
#
# **¿Qué tan exhaustivo es esto?** `puremacro.lp` es un conjunto de herramientas completo de
# proyecciones locales más allá de estos estimadores: PL aumentada con rezagos, cuyas bandas
# de Eicker-Huber-White siguen siendo válidas con datos muy persistentes y en un rango amplio
# de horizontes sin corrección HAC (`la_lp`, Montiel Olea–Plagborg-Møller 2021; ejercicio 2),
# PL-VI / identificación por proxy (`lp_iv`, Stock–Watson 2018), PL suave (`lp_smooth`,
# Barnichon–Brownlees 2019; ejercicios 1 y 4), asimetría por signo (`lp_asymmetric`;
# ejercicio 3), dependencia del estado con transición suave (`lp_state_dep` con
# `transition="logistic"`, su valor por defecto), PL por cuantiles (`lp_quantile`), y
# variantes de **panel** con efectos fijos en dos vías y errores estándar de Driscoll–Kraay,
# de grupo medio y CCE (`panel_lp`, `panel_lp_dk`, `panel_lp_iv`, `mean_group_panel_lp`,
# `cce_panel_lp`). La alternativa estructural —imponer todo el sistema dinámico en lugar de
# una regresión por horizonte— es el SVAR del **Cuaderno 6** (`puremacro.var`). Las
# demostraciones de principio a fin viven en `puremacro/examples/` (`la_lp_pmw_demo`, cuyo
# nombre es histórico, `lp_smooth_demo`, `lp_asymmetric_tenreyro`, `lp_panel_dk`,
# `narrative_panel_lp`). Cada estimador aquí circula por la ruta `ols_hac` de numpy puro o
# por mínimos cuadrados simples, por lo que el cuaderno permanece compatible con Pyodide (sin
# statsmodels).
