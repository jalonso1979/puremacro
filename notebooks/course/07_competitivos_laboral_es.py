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
# # Módulo 7 — Márgenes laborales: margen extensivo, intensivo y elasticidad agregada del trabajo
#
# **Curso complementario · puremacro · mazo Slides07 — mecanismos y economía abierta (semanas 13–14)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Descomponer** la varianza de las horas totales en su margen **extensivo**
#    (empleo) e **intensivo** (horas por trabajador): $\mathrm{var}(\Delta\log H)=
#    \mathrm{var}(\Delta\log E)+\mathrm{var}(\Delta\log h)+2\,\mathrm{cov}$, y saber
#    **cuál** de las dos descomposiciones al uso corresponde al «80–90 % extensivo» del mazo.
# 2. Distinguir la **elasticidad micro** de la oferta de trabajo (finita, la que mide la
#    microeconometría) de la **elasticidad macro** que el modelo necesita, y ver por qué la
#    agregación con trabajo **indivisible** y seguro completo la vuelve **infinita**
#    (Hansen 1985; Rogerson 1988).
# 3. Construir la **q de Tobin agregada** con datos de balance del sector corporativo
#    no financiero, estimar la respuesta de la **inversión** ante $\Delta\log q$ con
#    **proyecciones locales** y errores estándar **HAC**, y **medir cuán poco explica**:
#    el fracaso empírico clásico de la teoría q.
# 4. Leer el **precio relativo de la inversión** como el hilo que une q, inversión y trabajo
#    en un mundo **competitivo** (sin cuñas — aunque sí con **costos de ajuste**).
#
# Estos son mecanismos *competitivos*: precios que igualan márgenes, **no cuñas**. Ojo con el
# matiz: «competitivo» no quiere decir «sin fricciones». La q de Tobin sólo se separa de 1
# porque **ajustar el capital cuesta** ($\phi>0$); con $\phi=0$ el mazo muestra que
# $q_t\equiv 1$ y no habría nada que estimar. Todo corre en Python puro sobre tu
# **instalación local** de `puremacro` (`pip install puremacro`), con los datos congelados
# del *bundle*: sin conexión y sin costo.

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
# ## 1. Dos márgenes del trabajo
#
# Las **horas totales** trabajadas en la economía se abren en dos factores:
#
# $$H_t = E_t \cdot h_t,\qquad \log H_t = \log E_t + \log h_t,$$
#
# donde $E_t$ es el **número de personas ocupadas** (margen **extensivo**) y $h_t$ son las
# **horas por trabajador** (margen **intensivo**). En crecimientos, la identidad es exacta:
#
# $$\Delta\log H_t = \Delta\log E_t + \Delta\log h_t.$$
#
# Por lo tanto la varianza de las horas se descompone sin residuo:
#
# $$\mathrm{var}(\Delta\log H)=\underbrace{\mathrm{var}(\Delta\log E)}_{\text{extensivo}}
# +\underbrace{\mathrm{var}(\Delta\log h)}_{\text{intensivo}}
# +\underbrace{2\,\mathrm{cov}(\Delta\log E,\Delta\log h)}_{\text{comovimiento}}.$$
#
# **Dos descomposiciones, dos números distintos.** La de arriba deja un tercer término
# —el comovimiento— que no es de ningún margen, así que las «cuotas» no suman 100 % entre
# extensivo e intensivo. La literatura (y la lámina del mazo «¿qué margen manda en cada
# país?») usa la variante **de covarianzas**, que reparte el comovimiento por mitades
# implícitas y sí suma uno:
#
# $$1=\underbrace{\frac{\mathrm{cov}(\Delta\log E,\Delta\log H)}{\mathrm{var}(\Delta\log H)}}_{\text{cuota extensiva}}
# +\underbrace{\frac{\mathrm{cov}(\Delta\log h,\Delta\log H)}{\mathrm{var}(\Delta\log H)}}_{\text{cuota intensiva}}.$$
#
# Cuando el mazo dice «en EUA y México el extensivo explica $\sim80$–$90\%$ de la varianza»,
# habla de **esta segunda**. Compararla con la cruda `var(E)/var(H)` es comparar peras con
# manzanas: abajo imprimimos las dos.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Por qué importa para el modelo
# En el modelo neoclásico competitivo el hogar iguala la **desutilidad marginal del trabajo**
# con el **salario real** $= $ producto marginal del trabajo. Si ese margen fuera *divisible*
# y suave, casi todo el ajuste cíclico ocurriría en las **horas por trabajador**. Los datos
# dicen lo contrario: domina el **empleo**. Ese hecho es la motivación de la **oferta laboral
# indivisible** (Hansen 1985; Rogerson 1988) y de los modelos con margen extensivo.
#
# > **Nota de datos (leer antes de creerse los números).** El *bundle* no trae series
# > separadas de empleo y horas por trabajador para EUA (no hay `PAYEMS`, `AWHI` ni
# > `HOANBS`; la ruta con datos reales del mazo usa `fetch_qna_labor` y FRED, y exige red).
# > Así que aquí **simulamos** las dos series con un factor cíclico común y semilla fija.
# > La calibración está **elegida para reproducir** el hecho estilizado del mazo —cuota
# > extensiva de covarianzas $\approx 80\%$ en EUA y México— de modo que ese $80\%$ es un
# > **insumo, no un hallazgo**: lo que el bloque *verifica* es la aritmética (que la
# > identidad de varianzas cierra sin residuo y que las dos descomposiciones difieren),
# > no el hecho empírico. El hecho empírico está en la lámina del mazo, con datos OCDE/BLS.
# > Recordatorio institucional del mazo: en Alemania, España y Francia manda el margen
# > **intensivo** (*Kurzarbeit* y esquemas afines), así que la calibración de abajo no es
# > universal — es la de EUA y México.

# %% slideshow={"slide_type": "fragment"}
rng = np.random.default_rng(20260721)          # SEMILLA declarada: márgenes SIMULADOS
T = 280                                         # 70 años trimestrales
z = np.zeros(T)                                 # factor cíclico común (AR(1))
for i in range(1, T):
    z[i] = 0.70 * z[i - 1] + rng.standard_normal()
z = z / z.std()

# crecimientos trimestrales en % (100·Δlog). Extensivo más volátil que intensivo.
# Coeficientes calibrados para que la CUOTA DE COVARIANZAS extensiva salga ≈ 80 % (mazo).
d_ext = 0.60 * z + 0.34 * rng.standard_normal(T)   # Δlog E : empleo (margen extensivo)
d_int = 0.12 * z + 0.22 * rng.standard_normal(T)   # Δlog h : horas por trabajador (intensivo)
d_tot = d_ext + d_int                              # Δlog H : horas totales (identidad exacta)

v_ext = np.var(d_ext, ddof=1)
v_int = np.var(d_int, ddof=1)
cov   = np.cov(d_ext, d_int, ddof=1)[0, 1]
v_tot = np.var(d_tot, ddof=1)

# cuotas "de covarianzas": suman exactamente 1 porque cov(E,H)+cov(h,H) = var(H)
s_ext = (v_ext + cov) / v_tot
s_int = (v_int + cov) / v_tot

print(f"sd(Δlog E) extensivo   = {d_ext.std(ddof=1):.3f}   (desviación estándar, %)")
print(f"sd(Δlog h) intensivo   = {d_int.std(ddof=1):.3f}")
print(f"corr(ΔE, Δh)           = {np.corrcoef(d_ext, d_int)[0,1]:.2f}")
print(f"var(Δlog H)            = {v_tot:.4f}")
print(f"var(E)+var(h)+2cov     = {v_ext + v_int + 2*cov:.4f}   (identidad)")
print("-" * 62)
print("descomposición CRUDA (tres términos; las dos cuotas NO suman 100 %)")
print(f"  extensivo     var(E)/var(H) = {100*v_ext/v_tot:5.1f} %")
print(f"  intensivo     var(h)/var(H) = {100*v_int/v_tot:5.1f} %")
print(f"  comovimiento  2cov /var(H)  = {100*2*cov/v_tot:5.1f} %")
print("descomposición DE COVARIANZAS (la del mazo; suma 100 %)")
print(f"  extensivo   cov(E,H)/var(H) = {100*s_ext:5.1f} %   ← comparable con el 80–90 % del mazo")
print(f"  intensivo   cov(h,H)/var(H) = {100*s_int:5.1f} %")

assert np.isclose(v_tot, v_ext + v_int + 2 * cov)   # la identidad no tiene residuo
assert np.isclose(s_ext + s_int, 1.0)               # las cuotas de covarianzas sí suman 1
assert v_ext > v_int                                # el extensivo domina
assert cov > 0                                      # los márgenes comueven
assert 0.75 <= s_ext <= 0.90                        # calibración en el rango del mazo (EUA/México)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lectura
# El margen **extensivo** (empleo) explica la mayor parte de la varianza de las horas; el
# **intensivo** (horas por trabajador) aporta poco por sí solo, y el término de
# **comovimiento** es positivo (en las expansiones entran personas *y* suben las horas de
# quienes ya trabajan). Un modelo con trabajo perfectamente divisible pone casi todo el peso
# en el intensivo: contradice el dato **de EUA y México**.
#
# Fíjate en la brecha entre las dos descomposiciones: la cruda deja ~⅕ de la varianza en un
# término de comovimiento que no es de nadie, y por eso su cuota extensiva sale ~10 pp por
# debajo de la de covarianzas. Si comparas tu número con el $80$–$90\%$ del mazo sin decir
# cuál calculaste, no estás comparando lo mismo. (Regla del curso: ningún segundo momento
# se publica sin decir cómo se midió.)

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(4)
labels = ["extensivo\nvar(ΔE)", "intensivo\nvar(Δh)", "comovimiento\n2·cov", "total\nvar(ΔH)"]
vals = [v_ext, v_int, 2 * cov, v_tot]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.7),
                              gridspec_kw={"width_ratios": [2.1, 1]})
bars = ax.bar(labels, vals, color=[cols[0], cols[2], cols[3], "0.80"],
              edgecolor="0.15", width=0.66)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{100*v/v_tot:.0f}%",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("varianza de Δlog (puntos²)")
ax.set_title("Descomposición CRUDA (tres términos)", fontsize=10)
ax.axhline(0, color="0.85", lw=0.6)

# la descomposición del mazo: cuotas de covarianzas, que sí suman 100 %
ax2.bar(["extensivo", "intensivo"], [100 * s_ext, 100 * s_int],
        color=[cols[0], cols[2]], edgecolor="0.15", width=0.55)
for xi, si in zip([0, 1], [s_ext, s_int]):
    ax2.text(xi, 100 * si + 1.5, f"{100*si:.0f}%", ha="center", va="bottom", fontsize=10)
ax2.axhline(80, color="0.35", ls="--", lw=0.9)
ax2.text(1.42, 80, "80 % (mazo)", ha="right", va="bottom", fontsize=8, color="0.35")
ax2.set_ylim(0, 100)
ax2.set_ylabel("% de var(Δlog H)")
ax2.set_title("Cuotas DE COVARIANZAS (mazo)", fontsize=10)

fig.suptitle("Márgenes del trabajo: la cuota depende de qué descomposición uses", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. La elasticidad agregada la fabrica la agregación, no la microeconometría
#
# El RBC necesita una oferta de trabajo muy elástica; la microeconometría mide elasticidades
# **pequeñas**. Hansen (1985) y Rogerson (1988) resuelven la tensión **sin tocar la micro**.
#
# Con un continuo de individuos $i\in[0,1]$ y preferencias
# $u(c_{it},h_{it})=\log c_{it}+\mu\frac{(1-h_{it})^{1-\nu}}{1-\nu}$:
#
# * **Trabajo divisible.** Cada quien elige $h$ en el margen. Manteniendo constante la
#   utilidad marginal del consumo, la elasticidad de **Frisch** es
#   $\;\varepsilon^{\text{micro}}=\dfrac{1-\bar h}{\nu\,\bar h}$: finita, y tanto menor cuanto
#   mayor es la curvatura $\nu$ del ocio.
# * **Trabajo indivisible + seguro completo.** Se trabaja $\bar h$ o nada, y una lotería con
#   mercados completos asegura el riesgo de desempleo. La utilidad esperada del hogar
#   colectivo colapsa a $\;\log c_t - B\,\theta_t\;$ con
#   $B=\frac{\mu}{1-\nu}\left[1-(1-\bar h)^{1-\nu}\right]$, donde $\theta_t$ es la **fracción
#   empleada**. La desutilidad es **lineal** en $\theta_t$: la oferta agregada de trabajo es
#   **horizontal** y su elasticidad, **infinita** — para cualquier $\nu$, es decir, para
#   cualquier elasticidad micro.
#
# El puente con la sección 1: el margen que se mueve en el modelo indivisible es $\theta_t$,
# o sea el **extensivo**, exactamente el que domina en el dato de EUA y México.

# %% slideshow={"slide_type": "fragment"}
hbar, mu = 1 / 3, 2.0                 # horas del que trabaja y peso del ocio


def B_indivisible(nu, hbar=hbar, mu=mu):
    """Desutilidad lineal del empleo, B, del hogar con loterías (Hansen–Rogerson)."""
    if np.isclose(nu, 1.0):           # límite log: μ(1-h)^{1-ν}/(1-ν) → μ·log(1-h)
        return -mu * np.log(1 - hbar)
    return mu / (1 - nu) * (1 - (1 - hbar) ** (1 - nu))


print(f"calibración: h̄ = {hbar:.3f} (≈ 1/3 del tiempo disponible), μ = {mu:.1f}")
print(f"{'ν':>5} {'ε Frisch micro (divisible)':>28} {'B (indivisible)':>18} {'ε macro':>10}")
for nu in (0.5, 1.0, 2.0, 4.0):
    eps_micro = (1 - hbar) / (nu * hbar)
    print(f"{nu:5.1f} {eps_micro:28.2f} {B_indivisible(nu):18.3f} {'∞':>10}")

# la elasticidad macro NO depende de ν: la desutilidad es lineal en θ para todo ν
assert all(B_indivisible(nu) > 0 for nu in (0.5, 1.0, 2.0, 4.0))
assert (1 - hbar) / (4.0 * hbar) < (1 - hbar) / (0.5 * hbar)   # más curvatura, menos micro

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lectura
# La columna «ε macro» no cambia con $\nu$ **por construcción del argumento**, no por un
# resultado numérico: al ser $-B\theta_t$ lineal, la condición de primer orden en $\theta_t$
# fija el salario, no la cantidad. Ése es todo el truco de Hansen–Rogerson, y es un
# resultado de **agregación con mercados completos**, no de preferencias exóticas.
#
# Advertencia honesta: la elasticidad infinita es una *idealización*. Con desutilidad
# **convexa** del empleo y de las horas (Cho–Cooley, el modelo de dos márgenes del mazo) se
# recupera una solución interior y la elasticidad agregada vuelve a ser finita, aunque mucho
# mayor que la micro. Y como muestra la lámina de IRF del mazo, ni el trabajo indivisible
# corrige el problema de fondo del RBC: tras el choque tecnológico **las horas siguen
# subiendo** en impacto, contra lo que estima Galí (1999).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. La q de Tobin agregada y la inversión
#
# La **q de Tobin** es el valor de mercado del capital instalado dividido entre su costo de
# reposición. Con costos de ajuste convexos ($\phi>0$ en el mazo), la primera condición de la
# inversión da $q_t=\lambda_{it}/\lambda_{ct}$ y la **inversión crece con q**. Insistimos en
# el matiz: si $\phi=0$ entonces $q_t\equiv 1$ y no hay nada que estimar; la fricción de
# ajuste es **necesaria** para que exista la variable. Lo que no hace falta es una **cuña**.
#
# Aproximamos la **q media** del sector corporativo no financiero con dos series de balance
# (Financial Accounts / FRED, ya en el bundle):
#
# $$q_t = \frac{\text{valor de mercado del capital accionario (NCBEILQ027S)}}
# {\text{patrimonio neto a valor de mercado (TNWMVBSNNCB)}}.$$
#
# **Control de unidades (el error clásico):** las dos series están en **millones** de
# dólares, así que $q$ sale **adimensional y del orden de la unidad**. Si te sale $0.001$
# mezclaste millones con miles de millones — comprueba la ficha de FRED, no el nombre.
#
# **Control de muestra:** en las Financial Accounts los niveles de 1945–1951 se publican
# **sólo anuales** (cuarto trimestre). Si arrancas ahí, `diff()` calcula cambios *anuales*
# disfrazados de trimestrales y `lp_hac` —que desplaza **filas**, no fechas— trata cinco
# saltos anuales como si fueran trimestres. Recortamos a **1952Q1 en adelante**, que es donde
# la serie es genuinamente trimestral, y lo verificamos con un `assert`. (Por eso la mediana
# de $q$ que verás aquí, $\approx0.95$, difiere del $0.93$ de la lámina del mazo, que no
# recorta el tramo anual: mismo rango $[0.29,\,1.94]$, distinta muestra. Es exactamente el
# tipo de diferencia que la **ficha de medición** del curso obliga a declarar.)
#
# Luego medimos cómo responde la **inversión real** (GPDIC1) a un impulso en $\Delta\log q$
# con **proyecciones locales** y bandas **HAC** (`lp_hac`), con `n_lags=4` como la figura
# del mazo.

# %% slideshow={"slide_type": "fragment"}
from puremacro.lp import lp_hac   # verificada con inspect: (df, y, x, horizons, n_lags, controls, alpha)

def _load(fname, col):
    """Lee un CSV congelado de FRED y devuelve [fecha, serie]."""
    return pd.read_csv(DATA / fname, parse_dates=["observation_date"])[
        ["observation_date", col]]

eq  = _load("NCBEILQ027S.csv", "NCBEILQ027S")   # equities (valor de mercado), MILLONES USD
nw  = _load("TNWMVBSNNCB.csv", "TNWMVBSNNCB")   # patrimonio neto (mercado),   MILLONES USD
inv = _load("GPDIC1.csv",      "GPDIC1")        # inversión privada real (miles de mill. 2017)

df = (eq.merge(nw, on="observation_date").merge(inv, on="observation_date")
        .dropna())
df = df[df["observation_date"] >= "1952-01-01"].reset_index(drop=True)  # tramo trimestral
df["q"]      = df["NCBEILQ027S"] / df["TNWMVBSNNCB"]      # q de Tobin agregada
df["dlogq"]  = 100 * np.log(df["q"]).diff()              # Δlog q  (%)
df["loginv"] = 100 * np.log(df["GPDIC1"])                # log inversión (×100)
df = df.dropna().reset_index(drop=True)

# la muestra tiene que ser trimestral CONTIGUA: sin esto, lp_hac mezcla frecuencias
_gap = df["observation_date"].diff().dt.days.dropna()
assert _gap.between(89, 93).all(), "hay saltos no trimestrales en la muestra"

print(f"muestra: {df.observation_date.min():%Y-%m} a {df.observation_date.max():%Y-%m}  "
      f"({len(df)} trimestres contiguos)")
print(f"q: media {df['q'].mean():.2f}, mediana {df['q'].median():.2f}, "
      f"rango [{df['q'].min():.2f}, {df['q'].max():.2f}]  (adimensional: control de unidades OK)")

# proyección local: (loginv_{t+h} - loginv_{t-1}) = a_h + b_h · Δlog q_t + rezagos
irf = lp_hac(df, y="loginv", x="dlogq", horizons=range(0, 17), n_lags=4, alpha=0.10)
print(irf.round(3).head(9).to_string(index=False))

hmax = int(irf.loc[irf["beta"].idxmax(), "h"])
print(f"\nrespuesta máxima en h = {hmax} trimestres: "
      f"{irf['beta'].max():.2f}% de inversión por 1% de Δlog q")

# ¿CUÁNTO explica q? La teoría dice: estadístico SUFICIENTE. El dato dice otra cosa.
d_inv = 100 * np.log(df["GPDIC1"]).diff()
_c = pd.DataFrame({"x": df["dlogq"], "y": d_inv}).dropna()
r2_0 = np.corrcoef(_c["x"], _c["y"])[0, 1] ** 2                     # contemporáneo
_h = pd.DataFrame({"x": df["dlogq"],
                   "y": df["loginv"].shift(-hmax) - df["loginv"].shift(1)}).dropna()
r2_h = np.corrcoef(_h["x"], _h["y"])[0, 1] ** 2                     # al horizonte del pico
print(f"R² bivariado, crecimiento contemporáneo de la inversión sobre Δlog q : {r2_0:.3f}")
print(f"R² bivariado, respuesta acumulada a h = {hmax}                        : {r2_h:.3f}")
print(f"sd(Δlog q) = {df['dlogq'].std():.1f}%  →  un choque de 1 sd mueve la inversión "
      f"{irf['beta'].max()*df['dlogq'].std():.1f}% en el pico")

assert (irf["beta"] > 0).sum() >= 12      # el SIGNO que predice la teoría q sí aparece
assert irf.loc[irf["h"] == hmax, "lo"].item() > 0   # el pico es significativo (banda 90%)
assert r2_0 < 0.10                        # ...pero q NO es un estadístico suficiente

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lectura de la respuesta: el signo sí, la suficiencia no
# **Lo que sale bien.** El perfil tiene el signo que predice la teoría: un aumento de
# $\Delta\log q$ eleva la inversión de forma acumulativa, el efecto crece durante poco más
# de un año y luego decae, y el pico es significativo al 90 % con bandas HAC. Cuando el
# mercado valora el capital instalado por encima de su costo de reposición, conviene
# instalar más capital.
#
# **Lo que sale mal, y es el punto de la lámina del mazo.** La teoría q no dice sólo «signo
# positivo»: bajo Hayashi (1982), $q$ es un **estadístico suficiente** para la inversión —
# nada más debería importar. El $R^2$ que acabas de imprimir lo desmiente: en el
# contemporáneo $q$ explica alrededor del **1 %** del crecimiento de la inversión, y ni
# siquiera al horizonte del pico llega a la décima parte. **La relación es débil**: es el
# fracaso empírico clásico de la teoría q, no su confirmación. Un coeficiente
# estadísticamente significativo con $R^2\approx 0.01$ es exactamente eso.
#
# El mazo ofrece tres diagnósticos, y ninguno es «el mercado se equivoca»:
# 1. **Intangibles fuera del denominador.** El costo de reposición del balance omite el
#    capital intangible; la *total q* de Peters–Taylor (2017) recupera buena parte del poder
#    predictivo. Gutiérrez y Philippon (2017) ligan la brecha inversión–q a concentración e
#    intangibles.
# 2. **No convexidades e inacción.** Si los costos de ajuste no son convexos, la relación
#    lineal q–inversión se rompe: la microevidencia muestra un pico de inacción en cero y
#    episodios de inversión $>20\%$, no ajuste suave.
# 3. **Agregación.** Con el *capex* de unos pocos *hyperscalers* y valoraciones bursátiles
#    concentradas (2023–25), ¿qué información lleva una q **agregada**?
#
# Dos advertencias de medición, además. El numerador y el denominador de $q$ son del sector
# **corporativo no financiero**, mientras que `GPDIC1` es la inversión privada real
# **total** (incluye residencial e inventarios): hay desalineación de universo, y una versión
# más limpia usaría `PNFI/GDP` como en el ejercicio del mazo. Y `lp_hac` **ya acumula**
# (regresa $y_{t+h}-y_{t-1}$), por eso pasamos el **nivel** $100\log(\texttt{GPDIC1})$ y no su
# diferencia: si pasas la diferencia, estimas la respuesta del *crecimiento*, que es otra cosa.

# %% slideshow={"slide_type": "slide"}
# --- El hilo: precio relativo de la INVERSIÓN (PIRIC), no del equipo (ése es PERIC) ---
pir = pd.read_csv(DATA / "PIRIC.csv", parse_dates=["observation_date"]).dropna()
yrs = pir["observation_date"].dt.year + (pir["observation_date"].dt.month - 1) / 12
p0, p1 = pir["PIRIC"].iloc[0], pir["PIRIC"].iloc[-1]
años = (pir["observation_date"].iloc[-1] - pir["observation_date"].iloc[0]).days / 365.25
tasa = 100 * np.log(p1 / p0) / años
print(f"PIRIC (inversión TOTAL / consumo): {p0:.2f} → {p1:.2f}   "
      f"factor {p0/p1:.1f}× más barato; {tasa:.2f}% anual, {años:.0f} años "
      f"({pir.observation_date.min():%Y}–{pir.observation_date.max():%Y})")
print("comparación del mazo — PERIC (EQUIPO / consumo, Fisher 2006): factor ≈41 desde 1947,")
print("  ≈4.8% anual. El equipo se abarata MUCHO más rápido que la inversión total,")
print("  que incluye estructuras y residencial. PERIC no viene en el bundle offline.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))

# (a) respuesta de la inversión a Δlog q, con banda HAC 90%
a1.axhline(0, color="0.85", lw=0.6)
a1.fill_between(irf["h"], irf["lo"], irf["hi"], color="0.85", label="banda HAC 90%")
a1.plot(irf["h"], irf["beta"], color=cols[0], lw=1.6, marker="o", ms=3,
        label="respuesta acumulada")
a1.set_xlabel("horizonte (trimestres)")
a1.set_ylabel("% de inversión por 1% Δlog q")
a1.set_title(f"q de Tobin → inversión (LP-HAC)\nsigno correcto, pero R²≈{r2_0:.2f}", fontsize=10)
a1.legend(loc="upper right")

# (b) precio relativo de la inversión: caída secular (escala log)
a2.plot(yrs, pir["PIRIC"], color="0.25", lw=1.5)
a2.set_yscale("log")
a2.set_xlabel("año")
a2.set_ylabel("precio relativo (log)")
a2.set_title(f"Precio rel. de la INVERSIÓN (PIRIC)\n{p0/p1:.1f}× desde 1947 ({tasa:.1f}%/año)",
             fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El hilo competitivo — y una precisión de nomenclatura que el mazo subraya
# El **precio relativo de la inversión** cae de forma secular: cada dólar de inversión compra
# cada vez más capital efectivo. Es progreso técnico *incorporado* en el capital, à la
# Greenwood–Hercowitz–Krusell (1997). Ese abaratamiento es una fuerza de demanda de inversión
# que **eleva la q deseada** y, al cambiar la relación capital/trabajo, reacomoda también los
# **márgenes laborales**.
#
# > **`PIRIC` ≠ `PERIC`.** `PIRIC` es el precio de la **inversión total** relativo al consumo
# > (la serie del bundle, la que acabas de graficar). `PERIC` es el del **equipo**, y es la
# > que usan Greenwood–Hercowitz–Krusell (1997), Fisher (2006) y Justiniano–Primiceri–
# > Tambalotti (2010) para los choques IST. No son intercambiables: el equipo cae un factor
# > $\approx41$ desde 1947 ($\approx 4.8\%$ anual), la inversión total apenas un factor $5$
# > ($\approx 2.2\%$ anual), porque el agregado arrastra estructuras y vivienda, que casi no
# > se abaratan. Si necesitas el hecho de GHK/Fisher, la serie es `PERIC`, y exige red: no
# > está en el bundle offline. La identificación IST con `PERIC` está en la lección 10.
#
# Tres piezas — márgenes del trabajo, q de Tobin, precio relativo de la inversión — encajan
# en un solo relato **competitivo**: precios que igualan márgenes, **sin cuñas**. Con
# fricciones sí: sin costos de ajuste del capital no hay q que estimar.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Preguntas para pensar
# 1. **¿Por qué domina el margen extensivo?** El empleo (entrar y salir del trabajo) es más
#    volátil que las horas por trabajador. ¿Qué lo explica —contratos, semana laboral estándar,
#    fatiga, costos fijos de emplear— y qué le exige eso a un modelo con oferta laboral suave?
#    ¿Y por qué en Alemania, España y Francia manda el **intensivo**?
# 2. **Elasticidad micro vs macro.** Con $\bar h=1/3$ y $\nu=4$ la elasticidad de Frisch micro
#    es $0.5$, compatible con la microeconometría. ¿Por qué la elasticidad **agregada** no
#    hereda ese $0.5$ bajo trabajo indivisible? ¿Qué supuesto —la lotería, el seguro
#    completo, o la indivisibilidad— hace todo el trabajo, y cuál de los tres te parece más
#    frágil?
# 3. **q media vs q marginal.** Medimos la q *media* (balance del sector). La teoría habla de la
#    q *marginal*. ¿Bajo qué supuesto (Hayashi 1982) coinciden, y qué hace ese supuesto
#    cuando el capital intangible no está en el balance?
# 4. **¿Por qué el $R^2$ es tan bajo?** Ordena los tres diagnósticos del mazo (intangibles,
#    no convexidades, agregación) por cuánto crees que explican la brecha, y di qué dato
#    pediría cada uno para distinguirse de los otros dos.
# 5. **¿Endogeneidad?** $\Delta\log q$ y la inversión pueden responder a un mismo choque. ¿La
#    proyección local con controles rezagados basta para leer $\beta_h$ como respuesta causal,
#    o qué instrumento buscarías?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Las horas por trabajador están ancladas por la semana laboral estándar, contratos y
#    rendimientos decrecientes del esfuerzo diario; el grueso del ajuste cíclico son **personas
#    entrando/saliendo del empleo**. Un modelo con trabajo divisible y suave predice lo
#    contrario, así que hace falta oferta indivisible (Hansen–Rogerson, loterías) o
#    búsqueda-emparejamiento para reproducir la volatilidad del empleo. En Alemania, España y
#    Francia el margen intensivo domina por instituciones —*Kurzarbeit* y esquemas de trabajo
#    a tiempo reducido— que subsidian el ajuste de horas en vez del despido: qué modelo elegir
#    depende del país.
# 2. Con seguro completo, la desutilidad esperada del hogar es **lineal** en la fracción
#    empleada $\theta_t$: $\log c_t - B\theta_t$. La curvatura $\nu$ sobrevive dentro de la
#    constante $B$ —afecta el *nivel* del salario de reserva— pero desaparece de la
#    *pendiente*, y la elasticidad es la pendiente. Lo que hace el trabajo pesado es la
#    combinación **indivisibilidad + mercados completos**; con desutilidad convexa del empleo
#    (Cho–Cooley) la elasticidad vuelve a ser finita.
# 3. Con tecnología de ajuste **lineal-homogénea** y competencia (Hayashi 1982), q marginal =
#    q media; entonces la q media observable del balance sería un estadístico suficiente. Con
#    intangibles fuera del denominador la q media medida deja de ser la q media teórica, y la
#    equivalencia se rompe por medición antes que por teoría (Peters–Taylor 2017).
# 4. Los intangibles se distinguirían con una q que los capitalice (I+D y capital
#    organizacional, *total q*); las no convexidades, con microdatos de plantas —inacción y
#    picos— y con no linealidad en la LP; la agregación, comparando la q agregada con q
#    sectoriales o excluyendo las mayores capitalizaciones.
# 5. La LP con rezagos controla la dinámica predecible pero no un choque contemporáneo común;
#    para causalidad se buscaría un instrumento (p. ej. innovaciones de q ortogonales a
#    fundamentales, o variación en el precio relativo del equipo à la Fisher 2006) — tema del
#    módulo de identificación.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 5. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué el margen extensivo del trabajo es más volátil que el intensivo, y qué modelo lo captura?"
# - "Explica en una frase el teorema de Hayashi (1982): ¿cuándo q marginal = q media?"
# - "¿Por qué la q agregada casi no predice la inversión, si la teoría dice que es un
#   estadístico suficiente?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase: ¿por qué la inversión crece con la q de Tobin en el modelo competitivo?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Descompusimos la varianza de las horas de dos maneras y vimos que el margen
# **extensivo** (empleo) domina al **intensivo** (horas por trabajador) en EUA y México
# —el hecho que motiva la oferta laboral indivisible—, con la advertencia de que la cuota
# cambia ~10 pp según qué descomposición uses. Vimos por qué la **elasticidad agregada** del
# trabajo la fabrica la **agregación** (Hansen–Rogerson: cualquier elasticidad micro es
# compatible con una macro infinita) y no la microeconometría. Construimos la **q de Tobin
# agregada** con datos de balance y, con **proyecciones locales HAC** (`lp_hac`), obtuvimos
# el **signo** que predice la teoría q — pero un $R^2$ que la desmiente como estadístico
# suficiente: **la relación es débil**, el fracaso empírico clásico. El **precio relativo de
# la inversión** (`PIRIC`; el del equipo es `PERIC`), en caída secular, es el hilo que une q,
# inversión y trabajo en un mundo **competitivo** — sin cuñas, aunque con costos de ajuste.
#
# **Siguiente módulo:** la lección **10 — utilización variable, costos de ajuste del capital
# y q de Tobin** (`10_b1_capital_ajuste_es`), que sigue en este mismo bloque de las semanas
# 13–14 del mazo Slides07 y toma en serio los dos ingredientes que aquí dimos por sentados:
# que los servicios de capital son $u_t K_t$ y no $K_t$, y que instalar capital cuesta.
# Después, la lección **10b** lleva estos mecanismos a la economía pequeña y abierta.
#
# (La contabilidad del ciclo económico con cuñas —la BCA— ya la vimos antes, en la lección
# 04b de la semana 8; esta lección es el paso *posterior*, no el previo.)
