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

# %% [markdown] slideshow={"slide_type": "slide"}
# # Módulo 3 — Riesgo, incertidumbre y volatilidad
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Distinguir **riesgo** (distribución conocida) de **incertidumbre** (la distribución
#    misma se mueve), y por qué a la macro le importa la **cola izquierda**.
# 2. Estimar **Growth-at-Risk (GaR)**: proyectar los *cuantiles* del crecimiento del PIB
#    sobre las condiciones financieras y leer la densidad condicional (Adrian–Boyarchenko–Giannone, 2019).
# 3. Ajustar un **GARCH(1,1)** a las innovaciones del crecimiento y ver con datos la
#    **Gran Moderación** y el pico de volatilidad de **COVID**.
# 4. Discretizar un **AR(1)** con el método de **Tauchen** para tratar la volatilidad como
#    un *estado* — el primer paso hacia "la volatilidad como choque" — y **medir** el sesgo
#    de esa discretización frente a **Rouwenhorst**.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`): sin conexión y sin costo.

# %% slideshow={"slide_type": "skip"}
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
try:  # bajo Jupyter/ipykernel: conserva el backend inline (captura figuras)
    get_ipython()
except NameError:
    matplotlib.use("Agg")  # script plano / CLI: backend no interactivo
import matplotlib.pyplot as plt
_cwd = pathlib.Path.cwd()
_nb = _cwd if (_cwd / "_nbstyle.py").exists() else _cwd.parent
sys.path.insert(0, str(_nb)); sys.path.insert(0, str(_nb / "course"))
import _nbstyle; _nbstyle.apply_style()
from _tutor import tutor
DATA = (_nb / "course" / "data")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. Riesgo, incertidumbre, volatilidad
#
# La macro moderna usa tres palabras que conviene separar:
#
# - **Riesgo:** la distribución de los choques es *conocida y estable*. Sabemos que el
#   crecimiento del próximo trimestre tiene, digamos, media 2% y desviación estándar 3%.
# - **Incertidumbre:** *la distribución misma se mueve*. En una crisis no solo cae la media:
#   la varianza sube y la **cola izquierda** engorda. La segunda derivada importa.
# - **Volatilidad:** la magnitud *medible* de esas oscilaciones — la desviación estándar
#   condicional $\sigma_t$, que en los datos **no** es constante.
#
# Tres preguntas empíricas, tres herramientas:
# 1. ¿Cómo cambia *toda la distribución* del crecimiento con las condiciones financieras? → **GaR**.
# 2. ¿Cómo evoluciona $\sigma_t$ en el tiempo? → **GARCH**.
# 3. ¿Cómo metemos $\sigma_t$ como *estado* en un modelo? → **Tauchen**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Growth-at-Risk (GaR)
#
# Adrian, Boyarchenko y Giannone (2019, *"Vulnerable Growth"*) observan que las condiciones
# financieras mueven **la cola izquierda** del crecimiento futuro mucho más que su centro:
# cuando el crédito se endurece, el escenario *malo* empeora, pero el escenario *bueno*
# casi no cambia. La media engaña; hay que mirar los cuantiles.
#
# **Receta.** Para cada cuantil $\tau$ estimamos una regresión cuantílica del crecimiento
# sobre sus rezagos y un índice de condiciones financieras (aquí el **NFCI** de la Fed de
# Chicago, donde valores altos = condiciones *apretadas*):
#
# $$Q_\tau\big(g_{t+1}\big)=\alpha_\tau+\sum_{\ell=1}^{4}\beta_{\tau,\ell}\,g_{t-\ell+1}+\gamma_\tau\,\text{NFCI}_t.$$
#
# (Ese es literalmente el diseño que arma `gar.qar` con `p=4`, `horizons=(1,)`: `beta0` es el
# intercepto, `beta1…beta4` son $g_t,\dots,g_{t-3}$ —el $g_t$ contemporáneo cuenta como el
# primero de los cuatro rezagos— y `beta5` es el NFCI fechado en $t$.)
#
# Luego ajustamos una densidad **skew-t** a los cuantiles predichos (Azzalini) para leer la
# forma completa. En `puremacro`: `gar.qar`, `gar.fit_skewt_to_quantiles`, `gar.skewt_pdf`.
#
# > **Ficha de medición (regla del curso).** Serie: PIB real de EE. UU., tasa trimestral
# > anualizada, FRED `A191RL1Q225SBEA` (BEA); condiciones financieras: `NFCI` semanal de la
# > Fed de Chicago promediada a trimestre. Muestra de estimación: **1971Q1–2019Q4** (el
# > arranque lo fija el NFCI; el corte lo fijamos nosotros). Filtro: ninguno, la serie ya
# > viene en tasas. Base de precios: volúmenes encadenados de la BEA. Orden: alinear
# > (`dropna`) → recortar → estimar. Edición: CSV congelados en `course/data/`.
# >
# > **Por qué cortamos en 2019Q4.** Con la muestra completa, los dos atípicos de COVID
# > (−28.0% en 2020Q2 y +34.9% en 2020Q3) dominan las regresiones de los cuantiles extremos
# > y **le cambian el signo** a $\gamma_\tau$ en la cola derecha: el NFCI alto de 2020Q2 va
# > seguido del mayor rebote de la serie, así que la QAR "aprende" que apretar las
# > condiciones financieras *mejora* el escenario bueno ($\gamma_{0.95}=+0.76$). Ese
# > artefacto destruye justamente el resultado de ABG que queremos mostrar. La muestra de
# > *Vulnerable Growth* termina en 2015Q4, muy antes del problema.

# %% slideshow={"slide_type": "fragment"}
from puremacro.gar import qar, fit_skewt_to_quantiles, skewt_pdf

# PIB real de EE. UU.: tasa de crecimiento trimestral anualizada (%), FRED A191RL1Q225SBEA.
g = (pd.read_csv(DATA / "A191RL1Q225SBEA.csv", parse_dates=["observation_date"])
       .rename(columns={"A191RL1Q225SBEA": "gdp"})
       .set_index("observation_date"))
g.index = g.index.to_period("Q")

# NFCI (Chicago Fed National Financial Conditions Index): semanal -> promedio trimestral.
nf = pd.read_csv(DATA / "NFCI.csv", parse_dates=["observation_date"]).rename(columns={"NFCI": "nfci"})
nf["q"] = nf["observation_date"].dt.to_period("Q")
nfci_q = nf.groupby("q")["nfci"].mean()

FIN_GAR = "2019Q4"      # ver la ficha de medición: COVID contamina las colas de la QAR
gar_df = pd.DataFrame({"gdp": g["gdp"], "nfci": nfci_q}).dropna().loc[:FIN_GAR]
y = gar_df["gdp"].to_numpy()
z = gar_df["nfci"].to_numpy()
print(f"muestra GaR: {gar_df.index.min()}–{gar_df.index.max()}  (n={len(gar_df)})")

taus = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
qfit = qar(y, quantiles=taus, horizons=(1,), p=4, controls=z, n_boot=200, seed=0)

# El resultado de ABG en una sola columna: gamma_tau = efecto del NFCI sobre el cuantil tau.
gamma = qfit.set_index("tau")["beta5"]
print("\nefecto del NFCI por cuantil  (gamma_tau, pp de crecimiento por punto de NFCI):")
print(gamma.round(2).to_string())
assert gamma.loc[0.05] < -2.0        # cola izquierda: el NFCI la hunde
assert abs(gamma.loc[0.95]) < 0.5    # cola derecha: el NFCI casi no la mueve

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### De cuantiles a densidad condicional
# `qar` devuelve, por cuantil, el intercepto y los coeficientes de los 4 rezagos y del NFCI
# (`beta0…beta5`). Evaluamos esos coeficientes en dos *estados*: el trimestre de crisis
# **2008Q4** (NFCI muy alto, crecimiento reciente negativo) y un trimestre tranquilo de la
# Gran Moderación, **2005Q2**. Cada estado nos da un mapa $\{\tau \to Q_\tau\}$ que
# alimentamos al ajuste skew-t.

# %% slideshow={"slide_type": "fragment"}
idx = list(gar_df.index)

def _predictor(period):
    """x_t = [1, g_t, g_{t-1}, g_{t-2}, g_{t-3}, NFCI_t] en el orden de qar."""
    i = idx.index(pd.Period(period, "Q"))
    return np.array([1.0, y[i], y[i - 1], y[i - 2], y[i - 3], z[i]])

def _quantiles_at(period):
    x = _predictor(period)
    out = {}
    for _, r in qfit.iterrows():
        beta = np.array([r.beta0, r.beta1, r.beta2, r.beta3, r.beta4, r.beta5])
        out[float(r.tau)] = float(x @ beta)
    # Reordenamiento (Chernozhukov, Fernández-Val y Galichon 2010): las regresiones
    # cuantílicas se estiman por separado y pueden cruzarse; ordenar los cuantiles
    # predichos es la corrección estándar y no empeora la aproximación.
    ts = sorted(out)
    return dict(zip(ts, np.sort([out[t] for t in ts])))

q_crisis = _quantiles_at("2008Q4")   # crisis financiera global
q_calmo  = _quantiles_at("2005Q2")   # Gran Moderación

fit_crisis = fit_skewt_to_quantiles(q_crisis, maxiter=2000)
fit_calmo  = fit_skewt_to_quantiles(q_calmo,  maxiter=2000)

# Growth-at-Risk = cuantil condicional al 5% (peor escenario "razonable").
gar_crisis = fit_crisis.downside_quantile(0.05)
gar_calmo  = fit_calmo.downside_quantile(0.05)
es_crisis  = fit_crisis.expected_shortfall(0.05)   # E[g | g <= Q_5%]
print(f"GaR 5%  crisis 2008Q4 = {gar_crisis:6.2f}%   |   calmo 2005Q2 = {gar_calmo:6.2f}%")
print(f"Expected shortfall 5% crisis = {es_crisis:6.2f}%")
print(f"mediana condicional  crisis = {fit_crisis.ppf(0.5):6.2f}%   |   calmo = {fit_calmo.ppf(0.5):6.2f}%")
print(f"cuantil 95%          crisis = {fit_crisis.ppf(0.95):6.2f}%   |   calmo = {fit_calmo.ppf(0.95):6.2f}%")
print(f"escala sigma  crisis = {fit_crisis.sigma:5.2f} vs calmo = {fit_calmo.sigma:5.2f}   |   "
      f"asimetría alpha  crisis = {fit_crisis.alpha:5.2f} vs calmo = {fit_calmo.alpha:5.2f}")

assert gar_crisis < gar_calmo               # la cola izquierda se desploma en la crisis
assert fit_crisis.sigma > fit_calmo.sigma   # la densidad de crisis es más ancha
assert fit_crisis.alpha < 0 < fit_calmo.alpha   # y se vuelca hacia la izquierda

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
xx = np.linspace(-16, 16, 400)
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.plot(xx, skewt_pdf(xx, fit_calmo.mu, fit_calmo.sigma, fit_calmo.alpha, fit_calmo.nu),
        color=cols[0], lw=1.6, label="Densidad condicional · 2005Q2 (calmo)")
ax.plot(xx, skewt_pdf(xx, fit_crisis.mu, fit_crisis.sigma, fit_crisis.alpha, fit_crisis.nu),
        color="0.45", lw=1.6, ls=(0, (4, 2)), label="Densidad condicional · 2008Q4 (crisis)")
tail = xx[xx <= gar_crisis]
ax.fill_between(tail, 0, skewt_pdf(tail, fit_crisis.mu, fit_crisis.sigma, fit_crisis.alpha, fit_crisis.nu),
                color="0.45", alpha=0.25, label="cola 5% (crisis)")
ax.axvline(gar_crisis, color="0.30", lw=0.9, ls=":")
ax.axvline(0, color="0.85", lw=0.6)
ax.set_xlabel("crecimiento del PIB a 1 trimestre (% anualizado)")
ax.set_ylabel("densidad")
ax.set_title("Growth-at-Risk: la cola izquierda engorda cuando las condiciones se aprietan")
ax.legend(loc="upper left", fontsize=9)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lo que hay que leer
# La densidad de crisis (línea discontinua) no es solo la densidad calma *corrida a la
# izquierda*: es **más ancha** ($\sigma=7.79$ frente a $2.56$) y **se vuelca hacia la
# izquierda** ($\alpha=-5.68$ frente a $+1.55$ en calma). El GaR al 5% pasa de $+0.35\%$ en
# calma a $-12.03\%$ en la crisis, y el *expected shortfall* al 5% llega a $-14.95\%$.
#
# El mensaje de *Vulnerable Growth* se lee mejor en la columna $\gamma_\tau$ que imprimimos
# arriba: el NFCI mueve el cuantil 5% en $-2.34$ pp por punto de índice y el cuantil 95% en
# apenas $+0.20$ pp. **Las condiciones financieras deciden qué tan malo puede ser el
# escenario malo y casi no tocan el bueno.** Comparando los dos estados completos (rezagos
# incluidos) la asimetría es la misma: al pasar de 2005Q2 a 2008Q4 el cuantil 95% baja
# ~3.2 pp mientras el 5% se desploma ~12.4 pp, cuatro veces más.
#
# > Nota técnica: el ajuste skew-t hace *matching* de cuantiles por Nelder–Mead; su bandera
# > `success` puede quedar en `False` sin que la densidad deje de ser útil (los cuantiles
# > objetivo se reproducen bien). Los grados de libertad $\nu$ se acotan por abajo a
# > $\ge 2.5$ para que la varianza sea finita, pero **no** por arriba: cuando el
# > *matching* se resuelve bien con colas normales, $\nu$ se dispara y la skew-t degenera
# > en una *skew-normal*. Por eso aquí la asimetría se lee en $\alpha$, no en $\nu$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. GARCH(1,1): la volatilidad tiene memoria
#
# El GaR mira la distribución en un instante. El **GARCH** mira cómo evoluciona la
# desviación estándar condicional $\sigma_t$ en el tiempo. Con innovaciones $\varepsilon_t$
# (aquí, los residuos de un AR(1) del crecimiento),
#
# $$\varepsilon_t=\sigma_t z_t,\qquad z_t\sim N(0,1),\qquad
#   \sigma_t^2=\omega+\alpha\,\varepsilon_{t-1}^2+\beta\,\sigma_{t-1}^2.$$
#
# La **persistencia** $\alpha+\beta$ mide cuánto dura un episodio de alta volatilidad; cerca
# de 1 significa que los choques de volatilidad casi no se olvidan. En `puremacro`:
# `garch.garch11_fit` (MLE gaussiano).
#
# > **Ficha de medición.** Misma serie de PIB, pero aquí **sí** usamos la muestra completa
# > (1947Q3–2026Q1): el pico de COVID es parte de lo que queremos ver, no un estorbo.
# > Innovaciones: residuos MCO de un AR(1) sobre el crecimiento; media impuesta a cero en el
# > GARCH porque ya son residuos. Sin filtro, volúmenes encadenados de la BEA, CSV congelado.

# %% slideshow={"slide_type": "fragment"}
from puremacro.garch import garch11_fit

# Innovaciones = residuos de un AR(1) del crecimiento del PIB (muestra completa 1947–).
gv = g["gdp"].to_numpy()
Y1, X1 = gv[1:], np.column_stack([np.ones(len(gv) - 1), gv[:-1]])
b_ar = np.linalg.lstsq(X1, Y1, rcond=None)[0]
inno = pd.Series(Y1 - X1 @ b_ar, index=g.index[1:])
print(f"AR(1) del crecimiento ({inno.index.min()}–{inno.index.max()}):  "
      f"const={b_ar[0]:.2f}, rho={b_ar[1]:.2f}")

res = garch11_fit(inno, mean="zero")
print(res.summary())

sig = res.sigma                                     # sigma_t condicional (pd.Series)
vol_early = sig.loc[:"1983Q4"].mean()               # antes de la Gran Moderación
vol_gm    = sig.loc["1984Q1":"2007Q4"].mean()       # Gran Moderación (misma ventana que el mazo)
print(f"sigma media 1947Q3–1983Q4 = {vol_early:5.2f}   |   "
      f"Gran Moderación 1984Q1–2007Q4 = {vol_gm:5.2f}   "
      f"(razón {vol_gm / vol_early:.2f})")
print(f"pico de sigma: {sig.max():5.2f} en {sig.idxmax()}  "
      f"(el rebote del PIB fue {g['gdp'].max():.1f}% en {g['gdp'].idxmax()}, "
      f"tras {g['gdp'].min():.1f}% en {g['gdp'].idxmin()})")
assert vol_gm < vol_early                            # la volatilidad cae ~a la mitad

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.8))
tdx = sig.index.to_timestamp()
ax.plot(tdx, sig.to_numpy(), color="0.20", lw=1.1)
ax.axvspan(pd.Timestamp("1984-01-01"), pd.Timestamp("2007-12-31"),
           color="0.90", alpha=0.6, label="Gran Moderación")
# El pico está en sig.idxmax() (2020Q4), no en el trimestre del desplome: sigma_t reacciona
# a eps_{t-1}^2, así que llega un trimestre DESPUÉS del rebote de +34.9% de 2020Q3.
t_pico = sig.idxmax().to_timestamp()
ax.axvline(t_pico, color="0.45", lw=0.9, ls=(0, (4, 2)))
ax.annotate(f"COVID ({sig.idxmax()})", xy=(t_pico, sig.max()),
            xytext=(pd.Timestamp("2001-01-01"), sig.max() * 0.9), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
ax.set_xlabel("año"); ax.set_ylabel(r"$\sigma_t$ condicional (pp anualizados)")
ax.set_title("Volatilidad condicional del crecimiento: GARCH(1,1)")
ax.legend(loc="upper left", fontsize=9)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lo que hay que leer
# Tres regímenes saltan a la vista: (i) los años 70–principios de los 80, volátiles
# ($\bar\sigma_t = 4.69$ en 1947Q3–1983Q4); (ii) la **Gran Moderación** (1984–2007), con
# $\bar\sigma_t = 2.54$, es decir **el 54% del nivel previo**, más o menos la mitad;
# (iii) el **pico de COVID**, $\sigma_t = 25.4$ en **2020Q4**, un atípico enorme
# (el PIB anualizado cayó −28.0% en 2020Q2 y rebotó +34.9% en 2020Q3).
#
# Fíjate en la fecha del pico: llega en 2020**Q4**, un trimestre *después* del rebote. No es
# un error, es la mecánica del GARCH — $\sigma_t^2$ depende de $\varepsilon_{t-1}^2$, así que
# la volatilidad condicional siempre va un paso por detrás del choque. Ese atípico empuja la
# persistencia a $\alpha+\beta=0.999$ (casi IGARCH) y deja `converged=False`: la
# verosimilitud se aplana cuando un solo dato domina. Es un buen recordatorio de que los
# eventos extremos deforman las estimaciones de volatilidad — la misma razón por la que
# recortamos la muestra de la QAR en 2019Q4.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Tauchen: la volatilidad como *estado*
#
# Para *usar* la volatilidad dentro de un modelo (por ejemplo, un modelo de decisiones con
# incertidumbre variable) necesitamos convertir un proceso continuo AR(1)
#
# $$x_{t+1}=\rho\,x_t+\varepsilon_{t+1},\qquad \varepsilon\sim N(0,\sigma^2)$$
#
# en una **cadena de Markov** finita: una grilla de estados y una matriz de transición
# $P$. El método de **Tauchen (1986)** coloca una grilla equiespaciada de $\pm m$ desviaciones
# estándar incondicionales y asigna probabilidades integrando la normal entre puntos medios.
# Si $x_t$ es el log de la volatilidad, esto es justo una volatilidad estocástica discretizada.

# %% slideshow={"slide_type": "fragment"}
from scipy.stats import norm

def tauchen(rho, sigma, N=7, m=3.0):
    """Discretiza x' = rho*x + N(0, sigma^2) en N estados (Tauchen, 1986)."""
    sig_x = sigma / np.sqrt(1.0 - rho ** 2)          # desviación estándar incondicional
    xs = np.linspace(-m * sig_x, m * sig_x, N)        # grilla simétrica
    d = xs[1] - xs[0]                                  # paso de la grilla
    P = np.zeros((N, N))
    for i in range(N):
        mu = rho * xs[i]
        P[i, 0]  = norm.cdf((xs[0] + d / 2 - mu) / sigma)
        P[i, -1] = 1.0 - norm.cdf((xs[-1] - d / 2 - mu) / sigma)
        for j in range(1, N - 1):
            P[i, j] = (norm.cdf((xs[j] + d / 2 - mu) / sigma)
                       - norm.cdf((xs[j] - d / 2 - mu) / sigma))
    return xs, P

RHO, SIG = 0.95, 0.30                                  # AR(1) persistente para log-volatilidad
xs, P = tauchen(rho=RHO, sigma=SIG, N=7)
assert np.allclose(P.sum(axis=1), 1.0)                 # cada fila es una distribución

# Nuestra versión de 20 líneas coincide con la del paquete (firma: n primero).
from puremacro.vfi.discretize import tauchen as pm_tauchen, rouwenhorst, markov_stationary
xs_pm, P_pm = pm_tauchen(7, RHO, SIG, 3.0)
assert np.allclose(xs, xs_pm) and np.allclose(P, P_pm)

# distribución estacionaria (autovector izquierdo con autovalor 1)
pi = markov_stationary(P)

print("estados (log-vol):", np.round(xs, 2))
print("multiplicador de volatilidad exp(x):", np.round(np.exp(xs), 2))
print("distribución estacionaria:", np.round(pi, 3))
print("prob. de seguir en el estado más volátil P[-1,-1] =", round(P[-1, -1], 3))

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### ¿Cuánto miente la cadena? Tauchen frente a Rouwenhorst
# Discretizar **no es gratis**: la cadena finita tiene sus propios momentos, y no tienen por
# qué ser los del AR(1) que dice aproximar. Los medimos con la estacionaria $\pi$:
#
# $$\hat\sigma_x^2=\sum_i \pi_i x_i^2,\qquad
#   \hat\rho=\frac{\sum_i \pi_i\,x_i\,(P x)_i}{\hat\sigma_x^2}$$
#
# y los comparamos con los verdaderos $\sigma_x=\sigma/\sqrt{1-\rho^2}$ y $\rho$.
# **Rouwenhorst (1995)** construye la matriz por recursión, sin malla que truncar ni $m$ que
# elegir, y —Kopecky y Suen (2010)— reproduce *exactamente* $\rho$ y la varianza
# incondicional para cualquier $S$. Está en `puremacro.vfi.rouwenhorst`.

# %% slideshow={"slide_type": "fragment"}
def momentos_cadena(x, Pm):
    """(sd, rho) implícitos de una cadena de Markov bajo su distribución estacionaria."""
    p = markov_stationary(Pm)
    m = p @ x
    v = p @ (x - m) ** 2
    cov = float(sum(p[i] * (x[i] - m) * (Pm[i] @ x - m) for i in range(len(x))))
    return np.sqrt(v), cov / v

print(f"{'rho':>6} {'sigma_x real':>13} | {'Tauchen sd':>11} {'sesgo':>7} {'rho impl.':>10}"
      f" | {'Rouw. sd':>9} {'sesgo':>7} {'rho impl.':>10}")
for rho_i in (0.95, 0.99):
    sd_real = SIG / np.sqrt(1 - rho_i ** 2)
    sd_t, rho_t = momentos_cadena(*tauchen(rho=rho_i, sigma=SIG, N=7))
    sd_r, rho_r = momentos_cadena(*rouwenhorst(7, rho_i, SIG))
    print(f"{rho_i:>6.2f} {sd_real:>13.4f} | {sd_t:>11.4f} {sd_t/sd_real-1:>+6.1%} {rho_t:>10.4f}"
          f" | {sd_r:>9.4f} {sd_r/sd_real-1:>+6.1%} {rho_r:>10.4f}")

# Rouwenhorst clava ambos momentos; Tauchen no, y empeora con la persistencia.
assert np.allclose(momentos_cadena(*rouwenhorst(7, 0.99, SIG)),
                   (SIG / np.sqrt(1 - 0.99 ** 2), 0.99), rtol=1e-8)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Preguntas para pensar
# 1. **¿La volatilidad es en sí misma un choque?** En el GARCH, $\sigma_t$ *reacciona* a
#    choques pasados ($\varepsilon_{t-1}^2$). En Bloom (2009) y la literatura de "uncertainty
#    shocks", un salto en $\sigma_t$ es una *causa* independiente que frena inversión y
#    contratación. ¿Puedes distinguir empíricamente ambos relatos con una sola serie de $\sigma_t$?
# 2. La Gran Moderación bajó $\sigma_t$ pero no evitó 2008. ¿Menor volatilidad *medida* puede
#    esconder mayor riesgo de cola (más masa en la izquierda del GaR)? ¿"Calma" = seguridad?
# 3. En la tabla, al subir $\rho$ de 0.95 a 0.99 el sesgo de Tauchen sobre $\sigma_x$ pasa de
#    $+24\%$ a $+32\%$ y su $\hat\rho$ implícita se va a 0.9999, mientras Rouwenhorst clava
#    ambos momentos. ¿Por qué se degrada Tauchen justo donde más lo necesitamos? Relaciónalo
#    con la persistencia $\alpha+\beta\approx 1$ que estimó el GARCH: si la volatilidad
#    macro es casi IGARCH, ¿qué discretizador usarías?
# 4. Reestima el GaR con la muestra **completa** (`FIN_GAR = "2026Q1"`). El `assert` sobre
#    $\gamma_{0.95}$ falla. ¿Qué le hicieron dos trimestres de 2020 a la cola derecha, y qué
#    dice eso sobre publicar cuantiles extremos sin declarar la muestra?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Con una sola serie de $\sigma_t$ no se separan: el GARCH es *backward-looking* por
#    construcción. Hace falta identificación extra (un instrumento de incertidumbre, un
#    evento exógeno, o restricciones de signo en un VAR) para leer $\sigma_t$ como causa.
# 2. Sí: la varianza puede caer mientras la *asimetría* de cola empeora. El GaR (cuantil 5%)
#    puede deteriorarse aunque la desviación estándar baje — exactamente el punto de ABG.
# 3. $\rho$ mayor ⇒ estados más persistentes (diagonal de $P$ más pesada) y una estacionaria
#    más dispersa (la grilla se ensancha porque $\sigma_x=\sigma/\sqrt{1-\rho^2}$ crece).
#    Tauchen se degrada porque su malla es *equiespaciada y truncada* en $\pm m\sigma_x$:
#    cuando $\rho\to 1$ la masa se acumula en los extremos, la grilla no alcanza y las
#    probabilidades de salto se aplastan contra la diagonal. Con volatilidad casi IGARCH,
#    Rouwenhorst (o Farmer–Toda, también en `puremacro.vfi`) es la elección correcta.
# 4. Los dos atípicos de 2020 son un NFCI altísimo seguido del mayor rebote de la serie, así
#    que la regresión de $\tau=0.95$ le asigna al NFCI un coeficiente *positivo*
#    ($\gamma_{0.95}=+0.76$) y la cola derecha se "abre" con las condiciones apretadas —
#    lo contrario de ABG. Moraleja: sin ficha de medición (muestra incluida), un cuantil
#    extremo no significa nada.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 6. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "Explica en una frase por qué en *Vulnerable Growth* las condiciones financieras mueven
#   la cola izquierda del crecimiento pero casi no la cola derecha."
# - "¿En qué se diferencia un choque de volatilidad GARCH de un *uncertainty shock* a la Bloom (2009)?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase: ¿por qué las condiciones financieras afectan más la cola "
            "izquierda que la derecha del crecimiento futuro (Vulnerable Growth)?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Medimos el riesgo macro de tres formas complementarias: **GaR** para ver la
# *forma* de la distribución condicional (y su cola izquierda) con `gar.qar` +
# `gar.fit_skewt_to_quantiles`; **GARCH(1,1)** para la *trayectoria* de $\sigma_t$ con
# `garch.garch11_fit` (Gran Moderación y pico COVID); y **Tauchen** para convertir un AR(1)
# de volatilidad en un *estado* discreto utilizable en un modelo — midiendo su sesgo contra
# **Rouwenhorst** (`vfi.tauchen`, `vfi.rouwenhorst`, `vfi.markov_stationary`). Cada bloque
# viene con su **ficha de medición**: sin muestra declarada, un cuantil extremo o un segundo
# momento no significan nada. La pregunta abierta —
# ¿es la volatilidad un síntoma o una causa? — abre el siguiente módulo sobre choques de
# incertidumbre. Todo en `puremacro` (Python puro), sobre tu instalación local.
