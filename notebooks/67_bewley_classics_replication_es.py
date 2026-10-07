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
# # Clásicos de Bewley: una replicación de Huggett (1993) y Aiyagari (1994)
#
# **¿Cuánto empuja el riesgo de ingreso no asegurable la tasa de interés por debajo de la tasa de preferencia temporal, y reproducen los solucionadores actuales las dos tablas que respondieron esa pregunta?**
#
# Huggett (1993), "The risk-free rate in heterogeneous-agent incomplete-insurance economies", *Journal of Economic Dynamics and Control* 17(5-6), 953-969, y Aiyagari (1994), "Uninsured idiosyncratic risk and aggregate saving", *Quarterly Journal of Economics* 109(3), 659-684, calcularon equilibrios estacionarios de economías en las que los hogares se autoaseguran con un solo activo. Sus tablas son aquí los objetivos, transcritas de las páginas impresas; las de Aiyagari coinciden además con las tablas de su documento de trabajo (Federal Reserve Bank of Minneapolis Working Paper 502, revisado en diciembre de 1993, p. 35), que documenta su método con más detalle. No se usan datos.
#
# | Ficha de replicación | |
# |---|---|
# | Resultados replicados | Tabla I de Aiyagari (la cadena de Markov) y Tabla II (24 tasas de interés de equilibrio); Tablas 1-2 de Huggett (8 precios de bonos de equilibrio) |
# | Oráculos (independientes de puremacro) | (1) Las tablas publicadas. La Tabla II y las tablas de Huggett se comprueban primero con las identidades $s=\delta\alpha/(r+\delta)$ y $r=q^{-6}-1$; la Tabla I no tiene una identidad así y se comprueba reproduciéndola. (2) Las replicaciones en que se apoya Kirkby (2023), "Quantitative Macroeconomics: Lessons Learned from Fourteen Replications", *Computational Economics* 61, 875-896: otro software (el VFI Toolkit) y otro método (iteración de la función de valor discretizada), guardadas como números fijos. Cubren los ocho precios de Huggett y las seis tasas de Aiyagari con $\rho=0.9$ en una cadena de 27 estados |
# | Nuestro solucionador | `solve_egm` (políticas continuas), `continuous_stationary_distribution` (loterías de Young) y un buscador de raíces con intervalo |
# | Comprobaciones internas (no son oráculos) | Refinamiento de mallas; la función empaquetada `solve_aiyagari_continuous` (el mismo método, con un solucionador del hogar programado por separado); `VFIProblem` (iteración de la función de valor discreta, otro método) en la peor celda de Huggett |
# | Aceptación | Tabla I: igualdad con la precisión impresa. Tablas II, 1 y 2: todo ordenamiento publicado debe cumplirse, y la brecha de cada celda se clasifica frente a nuestro error numérico. Kirkby: coincidencia dentro de la resolución de sus mallas (un paso de precio más el redondeo para Huggett, dos pasos de tasa para Aiyagari) |
# | Veredicto | Lo imprime el cuadro de verificación al final, a partir de los valores calculados |

# %% [markdown]
# ## El método en matemáticas
#
# Un hogar con estado de ingreso $z$ y activos $a$ resuelve
# $$\max E_0\sum_t\beta^t\frac{c_t^{1-\mu}}{1-\mu}\quad\text{s.a.}\quad c_t+a_{t+1}=(1+r)a_t+w\,z_t,\qquad a_{t+1}\ge\underline a,$$
# con $z$ una cadena de Markov finita y $\psi$ la distribución estacionaria de los hogares sobre $(a,z)$. **Aiyagari**: $z$ es la dotación de trabajo $l$, $\underline a=0$, y una empresa Cobb-Douglas paga $r=\alpha K^{\alpha-1}L^{1-\alpha}-\delta$ y el salario $w$, donde $L=E[l]$ es el trabajo efectivo. El capital se vacía cuando $\int a\,d\psi=K(r)$, la demanda de la empresa, y la tasa de ahorro es $s=\delta\alpha/(r+\delta)$. El ingreso sigue
# $$\log l_t=\rho\log l_{t-1}+\sigma(1-\rho^2)^{1/2}\epsilon_t,$$
# de modo que $\sigma$ es la desviación estándar incondicional del logaritmo del ingreso, que se discretiza con una cadena de siete estados. **Huggett**: una economía de intercambio con dotación $e\in\{0.1,1\}$ y un bono en oferta neta nula, $c+q\,a'=a+e$ y $a'\ge\underline a$ en unidades del bono; $q$ vacía $\int a\,d\psi=0$. Con $x=q\,a$ esto es la ecuación de arriba escrita en $x$, con $wz=e$, $1+r=1/q$ y un límite que depende del precio, $x'\ge q\,\underline a$; así lo resuelve el código. Huggett escribe la aversión al riesgo como $\sigma$; aquí es $\mu$ en ambos artículos. Con seguro completo, $r=1/\beta-1$.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Un hogar que no puede asegurar su riesgo de ingreso ahorra para los tiempos difíciles. Cuando todos los hogares lo hacen, la oferta de ahorro aumenta y la tasa de interés debe caer por debajo de la tasa de preferencia temporal para vaciar el mercado de activos. El efecto crece con el tamaño y la persistencia del riesgo y con la aversión al riesgo, y se reduce cuando los hogares pueden endeudarse más. Ambos artículos sostienen ese argumento con números, así que los números mismos son el objetivo de la replicación. Se calcularon a principios de los noventa con simulación o mallas gruesas, y dependen de un proceso de ingreso discretizado cuya propia precisión forma parte del resultado.

# %% [markdown]
# ## Código resuelto
#
# Los valores publicados se guardan tal como se imprimieron, con los valores replicados de Kirkby al lado. Cada problema del hogar se resuelve con el método de la malla endógena, que invierte la ecuación de Euler sobre una malla de activos del período siguiente y por eso no necesita buscar una raíz en cada nodo (el cuaderno 52 lo deriva). La distribución estacionaria usa las loterías de Young sobre un histograma fino: la masa de cada política se reparte entre los dos nodos vecinos del histograma de modo que se conserve su media (`docs/es/vfi_continuous_equilibrium.md`, sección 1.2). El precio de equilibrio sale del método de Brent.

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.vfi import (VFIProblem, continuous_stationary_distribution, markov_stationary,
                           rouwenhorst, solve_aiyagari_continuous, solve_egm, tauchen)

# Aiyagari (1994), Table I: (sigma, rho) -> coefficient of variation and serial correlation of the chain.
AIYAGARI_TABLE_I = {(0.2, 0.0): (0.21, 0.0), (0.2, 0.3): (0.21, 0.3), (0.2, 0.6): (0.21, 0.59),
                    (0.2, 0.9): (0.24, 0.9), (0.4, 0.0): (0.43, 0.0), (0.4, 0.3): (0.43, 0.28),
                    (0.4, 0.6): (0.44, 0.58), (0.4, 0.9): (0.49, 0.89)}
# Aiyagari (1994), Table II: (sigma, rho, mu) -> (net return to capital %, aggregate saving rate %).
AIYAGARI_TABLE_II = {
    (0.2, 0.0, 1): (4.1666, 23.67), (0.2, 0.0, 3): (4.1456, 23.71), (0.2, 0.0, 5): (4.0858, 23.83),
    (0.2, 0.3, 1): (4.1365, 23.73), (0.2, 0.3, 3): (4.0432, 23.91), (0.2, 0.3, 5): (3.9054, 24.19),
    (0.2, 0.6, 1): (4.0912, 23.82), (0.2, 0.6, 3): (3.8767, 24.25), (0.2, 0.6, 5): (3.5857, 24.86),
    (0.2, 0.9, 1): (3.9305, 24.14), (0.2, 0.9, 3): (3.2903, 25.51), (0.2, 0.9, 5): (2.5260, 27.36),
    (0.4, 0.0, 1): (4.0649, 23.87), (0.4, 0.0, 3): (3.7816, 24.44), (0.4, 0.0, 5): (3.4177, 25.22),
    (0.4, 0.3, 1): (3.9554, 24.09), (0.4, 0.3, 3): (3.4188, 25.22), (0.4, 0.3, 5): (2.8032, 26.66),
    (0.4, 0.6, 1): (3.7567, 24.50), (0.4, 0.6, 3): (2.7835, 26.71), (0.4, 0.6, 5): (1.8070, 29.37),
    (0.4, 0.9, 1): (3.3054, 25.47), (0.4, 0.9, 3): (1.2894, 31.00), (0.4, 0.9, 5): (-0.3456, 37.63)}
# Huggett (1993), Tables 1-2: (mu = relative risk aversion, Huggett's sigma; credit limit) -> (annual rate %,
# its printed step, price q per model period).
HUGGETT_TABLES = {(1.5, -2): (-7.1, 0.1, 1.0124), (1.5, -4): (2.3, 0.1, 0.9962), (1.5, -6): (3.4, 0.1, 0.9944),
                  (1.5, -8): (4.0, 0.1, 0.9935), (3.0, -2): (-23.0, 1.0, 1.0448), (3.0, -4): (-2.6, 0.1, 1.0045),
                  (3.0, -6): (1.8, 0.1, 0.9970), (3.0, -8): (3.7, 0.1, 0.9940)}

# Second, independent oracle: Kirkby's VFI Toolkit replications (discretized value function iteration), numbers
# only, from github.com/vfitoolkit/vfitoolkit-matlab-replication. Huggett1993/Huggett1993.pdf, Tables 1-2 (commit
# 790fcdd27b, sha256 2f1e23b9e5eb8141514cb9611479e3cdb64f549ac188eb650011ace8d2c7ed0d): prices on 1551 grid points
# in [0.9, 1.1], assets on 1024 points up to a = 4. Aiyagari1994/Aiyagari1994.pdf, Table 3 (commit d09dd9aec6,
# sha256 ea207e53005dc601ff6a7b3aebb9281beddad9614949b9da3207c63eb0c2bb75): a 27-state Tauchen chain three
# standard deviations wide, positive rates on 301 grid points in [0, 1/beta - 1].
KIRKBY_HUGGETT_Q = {(1.5, -2): 1.0128, (1.5, -4): 0.9981, (1.5, -6): 0.9950, (1.5, -8): 0.9938,
                    (3.0, -2): 1.0459, (3.0, -4): 1.0075, (3.0, -6): 0.9987, (3.0, -8): 0.9955}
KIRKBY_AIYAGARI_R = {(0.2, 1): 4.0139, (0.2, 3): 3.5833, (0.2, 5): 3.0417,     # (sigma, mu) at rho = 0.9, %
                     (0.4, 1): 3.5833, (0.4, 3): 2.0972, (0.4, 5): 0.6806}
KIRKBY_Q_STEP, KIRKBY_A_TOP = 0.2 / 1550, 4.0

ALPHA, DELTA, BETA = 0.36, 0.08, 0.96            # Aiyagari: one period is a year
BETA_H = 0.99322                                  # Huggett: six periods a year
E_H = np.array([0.1, 1.0])                        # endowments (low, high)
P_H = np.array([[0.5, 0.5], [0.075, 0.925]])      # rows: from low, from high
RTP = 100.0 * (1.0 / BETA - 1.0)                  # Aiyagari's rate of time preference, %
KIRKBY_R_STEP = RTP / 300                         # step of Kirkby's rate grid, pp

checks = []  # one row per replicated result


def record(result, measured, bound, verdict, oracle):
    checks.append({"result": result, "oracle": oracle, "measured": float(measured), "bound": bound, "verdict": verdict})


def log_moments(log_grid, P):
    """Standard deviation and first-order autocorrelation of log income under the chain."""
    pi = markov_stationary(P)
    dev = log_grid - pi @ log_grid
    var = pi @ dev**2
    return np.sqrt(var), (pi * dev) @ P @ dev / var


def level_moments(log_grid, P):
    """Coefficient of variation and first-order autocorrelation of exp(log_grid) under the chain."""
    pi = markov_stationary(P)
    level = np.exp(log_grid)
    dev = level - pi @ level
    var = pi @ dev**2
    return np.sqrt(var) / (pi @ level), (pi * dev) @ P @ dev / var


def aiyagari_market(log_grid, P, mu, r, *, n_a=250, n_hist=1200, a_max=200.0):
    """Capital supply minus the firm's demand at rate r, the demand, and the mass in the top tenth of the grid."""
    z = np.exp(log_grid)
    labor = markov_stationary(P) @ z
    grid = a_max * np.linspace(0.0, 1.0, n_a) ** 2        # nodes concentrated near the constraint
    hist = a_max * np.linspace(0.0, 1.0, n_hist) ** 2
    k_per_l = ((r + DELTA) / ALPHA) ** (1.0 / (ALPHA - 1.0))
    wage = (1.0 - ALPHA) * k_per_l**ALPHA
    pol = solve_egm(grid, z, wage * z, P, beta=BETA, r=r, gamma=float(mu), tol=1e-10, max_iter=100_000)
    a_next = np.column_stack([np.interp(hist, grid, pol.aprime[:, j]) for j in range(z.size)])
    dist = continuous_stationary_distribution(a_next, hist, shock_transition=P, shock_grid=z)
    demand = labor * k_per_l
    return dist.mean() - demand, demand, dist.pdf[hist > 0.9 * a_max].sum()


def aiyagari_rate(log_grid, P, mu, *, xtol=1e-9, **grids):
    """Net return to capital that clears the capital market, with zero borrowing."""
    excess = lambda r: aiyagari_market(log_grid, P, mu, r, **grids)[0]
    split = 0.041      # solves near 1/beta - 1 are slow, so search below 4.1% whenever the root lies there
    lo, hi = (-0.06, split) if excess(split) > 0.0 else (split, 1.0 / BETA - 1.0 - 1e-5)
    return brentq(excess, lo, hi, xtol=xtol)


def huggett_balance(q, mu, limit, *, n_a=250, n_hist=1200, x_max=24.0):
    """Mean credit balance in bond units at price q: excess demand for credit balances, zero in equilibrium.

    With x = q a (goods spent on credit), c + q a' = a + e becomes c + x' = (1 + r) x + e,
    1 + r = 1/q, which is solve_egm's problem with the borrowing limit x' >= q * limit."""
    lo = q * limit
    grid = lo + (x_max - lo) * np.linspace(0.0, 1.0, n_a) ** 2
    hist = lo + (x_max - lo) * np.linspace(0.0, 1.0, n_hist) ** 2
    pol = solve_egm(grid, E_H, E_H, P_H, beta=BETA_H, r=1.0 / q - 1.0, gamma=mu, tol=1e-9, max_iter=400_000)
    x_next = np.column_stack([np.interp(hist, grid, pol.aprime[:, j]) for j in range(2)])
    return continuous_stationary_distribution(x_next, hist, shock_transition=P_H, shock_grid=E_H).mean() / q


def huggett_price(mu, limit, **grids):
    """Bond price q that sets mean credit balances to zero."""
    # q > beta keeps the distribution stationary (the 1e-4 keeps that endpoint's solve fast), and
    # q > 1 + e_low/limit keeps consumption positive at the limit; annual rates up to 4.10% are searched.
    q_low = max(BETA_H, 1.0 + E_H[0] / limit) + 1e-4
    return brentq(lambda q: huggett_balance(q, mu, limit, **grids), q_low, 1.15, xtol=1e-10)


def annual_rate(q):
    return 100.0 * (q**-6 - 1.0)


def saving_rate(r_pct):
    return 100.0 * DELTA * ALPHA / (r_pct / 100.0 + DELTA)


def rounding_ratio(f, x, x_half, y, y_half):
    """|f(x) - y| relative to the rounding of both printed numbers (f monotone); above 1 flags a misprint."""
    lo, hi = sorted((f(x - x_half), f(x + x_half)))
    return abs((lo + hi) / 2.0 - y) / ((hi - lo) / 2.0 + y_half)


# The printed tables check themselves: Aiyagari's saving rate is delta*alpha/(r + delta) (working paper 502,
# footnote 42) and Huggett's annual rate is q**-6 - 1, each within the rounding of both printed numbers.
ratio_saving = max(rounding_ratio(saving_rate, r, 5e-5, s, 5e-3) for r, s in AIYAGARI_TABLE_II.values())
ratio_huggett = max(rounding_ratio(annual_rate, q, 5e-5, r, step / 2) for r, step, q in HUGGETT_TABLES.values())
r_pinned = max(0.005 * 2e-3 / (saving_rate(r - 1e-3) - saving_rate(r + 1e-3)) for r, _ in AIYAGARI_TABLE_II.values())
record("Transcription: Aiyagari s = delta*alpha/(r + delta), gap / rounding allowance", ratio_saving, 1.0,
       "consistent", "published")
record("Transcription: Huggett r = q^-6 - 1, gap / rounding allowance", ratio_huggett, 1.0, "consistent", "published")
print(f"Largest gap over the rounding allowance: Aiyagari's saving rates {ratio_saving:.2f}, Huggett's rates "
      f"{ratio_huggett:.2f}; the saving-rate identity confirms each printed rate to {r_pinned:.4f} pp")
assert ratio_saving <= 1.0 and ratio_huggett <= 1.0

# %% [markdown]
# ### 1. La cadena de Markov de Aiyagari (Tabla I)
#
# Aiyagari sigue a Deaton (1991) y a Tauchen (1986), y su documento de trabajo describe la cadena con exactitud (WP 502, nota 33): los estados del logaritmo del ingreso son $\{-3\sigma,\dots,3\sigma\}$, y las probabilidades de transición integran la densidad normal sobre intervalos con fronteras en $\pm\sigma/2$, $\pm3\sigma/2$ y $\pm5\sigma/2$. Es una cadena de Tauchen de tres desviaciones estándar incondicionales de ancho. La Tabla I reporta el coeficiente de variación y la correlación serial de la propia cadena. Son estadísticos del *nivel* del ingreso, por eso la correlación serial difiere entre los paneles $\sigma=0.2$ y $\sigma=0.4$. Reproducir cada entrada confirma, por tanto, la transcripción y nuestra implementación de la cadena documentada; los anchos 2.0, 2.5 y 3.5 se muestran solo para ver con qué precisión la Tabla I fija el ancho. Las últimas columnas comparan la cadena con el proceso que aproxima, cuyo c.v. en niveles, $\sqrt{e^{\sigma^2}-1}$, no depende de $\rho$.

# %%
rows = []
for (sigma, rho), (cv_pub, ac_pub) in AIYAGARI_TABLE_I.items():
    for width in (2.0, 2.5, 3.0, 3.5):
        log_grid, P = tauchen(7, rho, sigma * np.sqrt(1.0 - rho**2), m=width)
        cv, ac = level_moments(log_grid, P)
        rows.append({"sigma": sigma, "rho": rho, "grid width": width, "chain c.v.": cv, "chain rho": ac,
                     "published": f"{cv_pub}/{ac_pub}", "match": round(cv, 2) == cv_pub and round(ac, 2) == ac_pub,
                     "chain log s.d.": log_moments(log_grid, P)[0], "AR(1) c.v.": np.sqrt(np.exp(sigma**2) - 1.0)})
table_i = pd.DataFrame(rows)
table_i["c.v. ratio"] = table_i["chain c.v."] / table_i["AR(1) c.v."]
paper_chain = table_i[table_i["grid width"] == 3.0]
print(paper_chain.drop(columns="grid width").round(4).to_string(index=False))
matches = table_i.groupby("grid width")["match"].sum()
cv_excess = paper_chain.groupby("rho")["c.v. ratio"].mean()
print("Cells reproduced at the printed precision, by Tauchen grid width:", matches.to_dict())
print("Chain c.v. over the AR(1) c.v., by rho:", cv_excess.round(3).to_dict())
record("Aiyagari Table I: cells not reproduced, of 8", 8 - matches[3.0], 0, "exact", "published")
assert matches[3.0] == 8 and matches[[2.0, 2.5, 3.5]].max() < 8
assert cv_excess[0.9] > cv_excess[0.0] > 1.0      # the excess grows with persistence

# %% [markdown]
# ### 2. La Tabla II de Aiyagari: 24 equilibrios generales
#
# Cada celda resuelve el problema del hogar a una tasa de interés de prueba, calcula la distribución estacionaria de activos y compara el ahorro agregado con el capital que demanda la empresa, sobre la cadena documentada de siete estados. Nuestro error numérico sale de duplicar la malla del hogar en cuatro celdas, una tranquila y tres de alto riesgo, y extrapolar (Richardson; el método converge con orden dos), más el cambio que produce duplicar el histograma en una de ellas. Después, la brecha de cada celda se clasifica frente al mayor de esos errores. Las tres celdas con las tasas más bajas, una de ellas negativa, se resuelven otra vez con la función empaquetada `solve_aiyagari_continuous`. Es una implementación programada por separado del mismo método (su propio solucionador del hogar y sus propias mallas, el mismo código de la distribución), así que mide el error de implementación y de malla, no el del método.

# %%
table_ii = []
for (sigma, rho, mu), (r_pub, s_pub) in AIYAGARI_TABLE_II.items():
    log_grid, P = tauchen(7, rho, sigma * np.sqrt(1.0 - rho**2), m=3.0)
    r = 100.0 * aiyagari_rate(log_grid, P, mu)
    table_ii.append({"sigma": sigma, "rho": rho, "mu": mu, "r published": r_pub, "r replicated": r,
                     "gap (pp)": r - r_pub, "s published": s_pub, "s replicated": saving_rate(r)})
table_ii = pd.DataFrame(table_ii).set_index(["sigma", "rho", "mu"])

refine = {}
for cell in [(0.2, 0.0, 1), (0.4, 0.3, 5), (0.4, 0.6, 5), (0.4, 0.9, 5)]:
    sigma, rho, mu = cell
    log_grid, P = tauchen(7, rho, sigma * np.sqrt(1.0 - rho**2), m=3.0)
    r_fine = 100.0 * aiyagari_rate(log_grid, P, mu, n_a=500)                          # twice the household nodes
    refine[cell] = 4.0 / 3.0 * (r_fine - table_ii.loc[cell, "r replicated"])        # Richardson, second order
hist_cell = (0.4, 0.6, 5)
log_grid, P = tauchen(7, 0.6, 0.4 * np.sqrt(1.0 - 0.6**2), m=3.0)
hist_error = 100.0 * aiyagari_rate(log_grid, P, 5, n_hist=2400) - table_ii.loc[hist_cell, "r replicated"]
numerical_error = max(abs(v) for v in refine.values()) + abs(hist_error)
table_ii["beyond error"] = table_ii["gap (pp)"].abs() > numerical_error
n_beyond = int(table_ii["beyond error"].sum())
n_beyond_10 = int((table_ii["gap (pp)"].abs() > 10 * numerical_error).sum())
print(table_ii.round(4).to_string())
print("Household-grid error of the base grid (Richardson), pp:", ", ".join(f"{k} {v:+.4f}" for k, v in refine.items()))
print(f"Doubling the histogram moves {hist_cell} by {hist_error:+.5f} pp")
print(f"{n_beyond} of 24 gaps exceed our numerical error of {numerical_error:.4f} pp and {n_beyond_10} exceed ten "
      f"times it; within it: {', '.join(str(c) for c in table_ii.index[~table_ii['beyond error']])}")

wedge_pub = RTP - table_ii["r published"]
gap_wedge_corr = np.corrcoef(table_ii["gap (pp)"], wedge_pub)[0, 1]
wedge_rep = RTP - table_ii["r replicated"]
negative = table_ii.index[table_ii["gap (pp)"] < 0]
print(f"{len(negative)} gaps are negative, all at sigma = 0.2 or rho = 0; the correlation of the gap with the "
      f"published precautionary wedge is {gap_wedge_corr:.2f}")
print(f"Rate of time preference {RTP:.4f}%: replicated rates lie {wedge_rep.min():.4f} to {wedge_rep.max():.4f} pp "
      f"below it (published: {wedge_pub.min():.4f} to {wedge_pub.max():.4f} pp)")

packaged = {}
for cell in table_ii["r replicated"].nsmallest(3).index:
    sigma, rho, mu = cell
    log_grid, P = tauchen(7, rho, sigma * np.sqrt(1.0 - rho**2), m=3.0)
    eq = solve_aiyagari_continuous(beta=BETA, gamma=float(mu), alpha=ALPHA, delta=DELTA, P_z=P,
                                   z_grid=np.exp(log_grid), a_max=200.0, n_a=800)
    assert eq.converged      # household fixed point, stationary distribution and |K^s - K^d| < 1e-4, all at r*
    packaged[cell] = 100.0 * eq.r - table_ii.loc[cell, "r replicated"]
packaged_gap = max(abs(v) for v in packaged.values())
print("solve_aiyagari_continuous minus this notebook's solver:", ", ".join(f"{k} {v:+.4f} pp" for k, v in packaged.items()))


def falls_along(frame, column, level):
    """True if the rate falls as `level` rises, holding the other two parameters fixed."""
    others = [n for n in frame.index.names if n != level]
    return all(g.sort_index(level=level)[column].diff().dropna().lt(0).all() for _, g in frame.groupby(level=others))


orderings = {level: (falls_along(table_ii, "r published", level), falls_along(table_ii, "r replicated", level))
             for level in ("sigma", "rho", "mu")}
print("Rate falls with sigma, rho and mu (published, replicated):", orderings)
largest_gap = table_ii["gap (pp)"].abs().max()
violations = sum(not all(v) for v in orderings.values()) + int((table_ii["r published"] >= RTP).any())
record("Aiyagari Table II: orderings in sigma, rho, mu violated (of 3), or published r >= 1/beta - 1", violations, 0,
       "qualitative", "published")
record("Aiyagari Table II: cells whose |gap| exceeds our numerical error, of 24", n_beyond, "reported", "differs",
       "published")
record("Aiyagari Table II: largest |r - published|, pp", largest_gap, "reported", "differs", "published")
record("Numerical error: household grid (Richardson, 4 cells) plus histogram, pp", numerical_error, 0.01, "accuracy",
       "internal")
record("Packaged solve_aiyagari_continuous, three lowest-rate cells, pp", packaged_gap, 0.01, "accuracy", "internal")
assert violations == 0 and numerical_error < 0.01 and packaged_gap < 0.01
assert min(refine.values()) > 0.0                  # coarse household grids understate the rate
assert n_beyond >= 20 and largest_gap > 10 * numerical_error and gap_wedge_corr > 0.8
assert all(sigma == 0.2 or rho == 0.0 for sigma, rho, _ in negative)

# %% [markdown]
# ### 3. La cadena importa más que los decimales
#
# La sección 1 mostró que el c.v. en niveles de la cadena de siete estados supera al del proceso, y más con $\rho=0.9$ que con $\rho=0$. Aiyagari reconoció ese exceso (su nota 33: "for high values of $\sigma$ the Markov chain had a somewhat higher coefficient of variation", es decir, con valores altos de $\sigma$ la cadena tenía un coeficiente de variación algo mayor) y lo defendió (nota 35: un hogar de vida infinita necesita que la desviación estándar de sus ingresos se escale por un factor de alrededor de 1.2 para captar la variabilidad del ingreso permanente, algo que la cadena "tends to deliver automatically for the high value of $\sigma$", tiende a dar por sí sola con el valor alto de $\sigma$). Esa nota dice también que un $\rho$ mayor reduce el ajuste necesario, así que la defensa es más débil en la fila $\rho=0.9$, justo donde el exceso es mayor. La cadena de Rouwenhorst (1995) iguala exactamente la media, la varianza y la autocorrelación del logaritmo del ingreso con cualquier número de estados (Kopecky y Suen 2010); sus estados son binomiales y no normales, así que no iguala exactamente el c.v. en niveles. Resolver con ella las celdas con $\rho=0.9$ mide cuánto mueve la respuesta la discretización, y no el solucionador. La cadena de Tauchen de 27 estados que usó Kirkby (2023) sirve de referencia con muchos estados, y sus tasas publicadas con esa cadena son una comprobación independiente de nuestro solucionador: su estudio resolvió de nuevo el artículo con otro software, por iteración de la función de valor discretizada.

# %%
discretization = []
for sigma in (0.2, 0.4):
    for mu in (1, 3, 5):
        sigma_eps = sigma * np.sqrt(1.0 - 0.9**2)
        log_7, P_7 = rouwenhorst(7, 0.9, sigma_eps)
        log_27, P_27 = tauchen(27, 0.9, sigma_eps, m=3.0)
        discretization.append({"sigma": sigma, "mu": mu,
                               "published": AIYAGARI_TABLE_II[(sigma, 0.9, mu)][0],
                               "Tauchen 7": table_ii.loc[(sigma, 0.9, mu), "r replicated"],
                               "Rouwenhorst 7": 100.0 * aiyagari_rate(log_7, P_7, mu),
                               "Tauchen 27": 100.0 * aiyagari_rate(log_27, P_27, mu, n_hist=600, xtol=1e-7),
                               "Kirkby, Tauchen 27": KIRKBY_AIYAGARI_R[(sigma, mu)]})
discretization = pd.DataFrame(discretization).set_index(["sigma", "mu"])
print(discretization.round(4).to_string())
shift = (discretization["Rouwenhorst 7"] - discretization["Tauchen 7"]).abs().max()
chain_7_27 = (discretization["Tauchen 27"] - discretization["Tauchen 7"]).abs().max()
rouw_vs_27 = (discretization["Rouwenhorst 7"] - discretization["Tauchen 27"]).abs().max()
kirkby_r_gap = (discretization["Tauchen 27"] - discretization["Kirkby, Tauchen 27"]).abs().max()
corner_27 = discretization.loc[(0.4, 5), "Tauchen 27"]
print(f"Largest shift from Tauchen 7: to Rouwenhorst 7 {shift:.4f} pp, to Tauchen 27 {chain_7_27:.4f} pp; "
      f"Rouwenhorst 7 against Tauchen 27 {rouw_vs_27:.4f} pp; our Tauchen 27 against Kirkby's {kirkby_r_gap:.4f} pp "
      f"(bound: two steps of his rate grid, {2 * KIRKBY_R_STEP:.4f} pp)")
record("Chain: largest Tauchen-7 to Rouwenhorst-7 shift at rho = 0.9, pp", shift, "reported", "exceeds the gaps",
       "internal")
record("Chain: largest Rouwenhorst-7 minus Tauchen-27 at rho = 0.9, pp", rouw_vs_27, "reported", "7-state error",
       "internal")
record("Kirkby (VFI Toolkit): our Tauchen-27 rates minus his, 6 cells, pp", kirkby_r_gap, 2 * KIRKBY_R_STEP,
       "accuracy", "Kirkby")
assert shift > 3 * largest_gap and chain_7_27 > largest_gap and rouw_vs_27 < shift / 5
assert kirkby_r_gap <= 2 * KIRKBY_R_STEP

# %% [markdown]
# La figura principal pone lado a lado las dos fuentes de diferencia. A la izquierda, la brecha de cada celda frente a la tasa impresa se grafica contra esa tasa, con nuestro error numérico como una banda alrededor de cero. A la derecha, las celdas con $\rho=0.9$ comparan la tasa impresa con tres cadenas, y las tasas de Kirkby aparecen sobre las barras de 27 estados.

# %%
col = _nbstyle.palette(3)
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.5, 4.8))
ax1.axhspan(-numerical_error, numerical_error, color=_nbstyle.TINTA, alpha=_nbstyle.BANDA_ALPHA, lw=0,
            label="± our numerical error")
ax1.axhline(0.0, color=_nbstyle.SPINE, lw=0.8)
for sigma, marker, color in ((0.2, "o", col[0]), (0.4, "s", col[1])):
    part = table_ii.xs(sigma, level="sigma")
    ax1.plot(part["r published"], part["gap (pp)"], marker, color=color, mfc="none" if sigma == 0.2 else color,
             ms=7, linestyle="none", label=f"σ = {sigma}")
corner = table_ii.loc[(0.4, 0.9, 5)]
ax1.annotate("σ = 0.4, ρ = 0.9, μ = 5", (corner["r published"], corner["gap (pp)"]), textcoords="offset points",
             xytext=(10, -4), color=_nbstyle.TEXTO)
ax1.set(xlabel="Published net return (%)", ylabel="Replicated minus published (pp)", title="Table II, 24 cells")
ax1.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
x = np.arange(len(discretization))
names = ["published", "Tauchen 7", "Rouwenhorst 7", "Tauchen 27"]
for k, name in enumerate(names):
    ax2.bar(x + (k - 1.5) * 0.2, discretization[name], width=0.19, color=_nbstyle.BARRA_COLORES[k],
            hatch=_nbstyle.BARRA_HATCH[k], edgecolor=_nbstyle.TINTA, lw=0.6, label=name)
ax2.plot(x + 0.3, discretization["Kirkby, Tauchen 27"], "D", color=_nbstyle.TINTA, mfc=_nbstyle.FONDO, ms=6,
         linestyle="none", label="Kirkby (2023), Tauchen 27")
ax2.axhline(0.0, color=_nbstyle.SPINE, lw=0.8)
ax2.set_xticks(x, [f"σ = {s}\nμ = {m}" for s, m in discretization.index])
ax2.set(ylabel="Net return (%)", title="ρ = 0.9: published, and three chains")
ax2.set_ylim(top=1.45 * discretization.to_numpy().max())   # room for the legend above the bars
handles, labels = ax2.get_legend_handles_labels()
order = [labels.index(name) for name in names + ["Kirkby (2023), Tauchen 27"]]   # bars first, then the marker
ax2.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", ncol=2, frameon=True,
           facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 4. Las Tablas 1 y 2 de Huggett
#
# La economía de Huggett tiene seis períodos por año, $\beta=0.99322$, dotaciones de $1.0$ y $0.1$ y límites de crédito de $-2$ a $-8$ en unidades del bono (la dotación media de un año es de aproximadamente $5.3$). El precio $q$ es por período del modelo; las tasas anuales son $q^{-6}-1$. Las tablas se indexan por la aversión al riesgo $\mu$, el $\sigma$ de Huggett. Huggett detenía su búsqueda de precios cuando "excess demand for credit balances is within 0.0025 units of zero" (el exceso de demanda de saldos de crédito queda a menos de 0.0025 unidades de cero), y añade que "with this criterion, interest rates that are approximately market clearing vary by less than a tenth of one percent" (con este criterio, las tasas que vacían aproximadamente el mercado varían menos de una décima de uno por ciento) (p. 962). Su malla del hogar tenía "between 150 and 350 evenly spaced gridpoints" (entre 150 y 350 puntos equiespaciados), separados entre 0.03 y 0.1 unidades (p. 961). El código evalúa nuestro exceso de demanda en cada precio impreso y pone nuestros precios junto a los de Kirkby. Para la celda con la mayor brecha, `VFIProblem` resuelve la misma economía por iteración de la función de valor discreta en una malla uniforme casi tan fina como la más fina de Huggett: una comprobación interna con otro método.

# %%
huggett = []
for (mu, limit), (r_pub, _, q_pub) in HUGGETT_TABLES.items():
    q = huggett_price(mu, limit)
    huggett.append({"mu": mu, "credit limit": limit, "q published": q_pub, "q replicated": q,
                    "q Kirkby": KIRKBY_HUGGETT_Q[(mu, limit)], "r published": r_pub, "r replicated": annual_rate(q),
                    "balance at published q": huggett_balance(q_pub, mu, limit)})
huggett = pd.DataFrame(huggett).set_index(["mu", "credit limit"])
huggett["balance / 0.0025"] = huggett["balance at published q"].abs() / 0.0025
print(huggett.round(5).to_string())
rate_gaps = (huggett["r replicated"] - huggett["r published"]).abs()
print(f"At the printed prices our excess demand for credit balances is {huggett['balance / 0.0025'].min():.0f} to "
      f"{huggett['balance / 0.0025'].max():.0f} times Huggett's criterion, and the annual rates differ by "
      f"{rate_gaps.min():.2f} to {rate_gaps.max():.2f} pp")


def huggett_price_vfi(mu, limit, bracket, n_a=600, a_max=12.0):
    """The same equilibrium by discrete value function iteration on a uniform grid."""
    a_grid = np.linspace(limit, a_max, n_a)

    def payoff(ap, a, z, q, xp=np):
        c = a + xp.exp(z) - q * ap
        return xp.where(c > 0.0, (xp.maximum(c, 1e-12) ** (1.0 - mu) - 1.0) / (1.0 - mu), -np.inf)

    def mean_balance(q):
        problem = VFIProblem(a_grid=a_grid, z_grid=np.log(E_H), P_z=P_H, return_fn=payoff, beta=BETA_H,
                             params={"q": q}, options={"tol": 1e-9, "n_howard": 60, "max_iter": 20_000})
        dist = problem.stationary_distribution(problem.solve("numpy"))
        return float(np.sum(dist * a_grid[:, None]))

    return brentq(mean_balance, *bracket, xtol=1e-6), a_grid[1] - a_grid[0]


gaps_q = (huggett["q replicated"] - huggett["q published"]).abs()
worst = gaps_q.idxmax()
q_egm = huggett.loc[worst, "q replicated"]
q_vfi, vfi_spacing = huggett_price_vfi(*worst, bracket=(q_egm - 0.005, q_egm + 0.005))  # needs only a sign change
vfi_gap = abs(q_vfi - q_egm)
print(f"Largest gap at mu = {worst[0]}, limit = {worst[1]}: EGM q = {q_egm:.5f}, discrete VFI q = {q_vfi:.5f} "
      f"(grid spacing {vfi_spacing:.3f}), published {huggett.loc[worst, 'q published']}")

kirkby_q_gap = (huggett["q replicated"] - huggett["q Kirkby"]).abs()
kirkby_inner = kirkby_q_gap.drop(index=-8, level="credit limit").max()
kirkby_outer = kirkby_q_gap.xs(-8, level="credit limit").max()
truncated = {mu: huggett_price(mu, -8, x_max=KIRKBY_A_TOP * huggett.loc[(mu, -8), "q replicated"]) -
             huggett.loc[(mu, -8), "q replicated"] for mu in (1.5, 3.0)}
print(f"Our prices minus Kirkby's: at most {kirkby_inner:.1e} for limits -2 to -6 (his grid step plus rounding "
      f"{KIRKBY_Q_STEP + 5e-5:.1e}), {kirkby_outer:.1e} at -8; cutting our grid at a = {KIRKBY_A_TOP:.0f}, where "
      f"his stops, moves the -8 prices by {', '.join(f'{v:+.1e}' for v in truncated.values())}")

r_rep = huggett["r replicated"].unstack("credit limit")
r_pub = huggett["r published"].unstack("credit limit")
huggett_violations = (int(not (np.diff(r_rep.to_numpy(), axis=1) < 0).all()) + int(not (np.diff(r_pub.to_numpy(), axis=1) < 0).all())
                      + int(not (r_rep.loc[3.0] < r_rep.loc[1.5]).all()) + int(not (r_pub.loc[3.0] < r_pub.loc[1.5]).all())
                      + int(not (r_pub < annual_rate(BETA_H)).all().all()))
n_huggett_beyond = int((gaps_q > 5e-5 + vfi_gap).sum())
record("Huggett: orderings in the limit and mu violated, or published r >= 1/beta - 1", huggett_violations, 0,
       "qualitative", "published")
record("Huggett: largest |q - published|", gaps_q.max(), "reported", "differs", "published")
record("Huggett: smallest excess demand at a published price / 0.0025", huggett["balance / 0.0025"].min(),
       "reported", "not the tolerance", "published")
record("Kirkby (VFI Toolkit): |q - his q|, limits -2 to -6, 6 cells", kirkby_inner, KIRKBY_Q_STEP + 5e-5,
       "accuracy", "Kirkby")
record("Kirkby (VFI Toolkit): |q - his q| at limit -8, 2 cells", kirkby_outer, "reported", "his grid stops at a = 4",
       "Kirkby")
record("Discrete VFI minus EGM price, worst cell", vfi_gap, 1e-4, "accuracy", "internal")
assert huggett_violations == 0 and vfi_gap < 1e-4 and kirkby_inner <= KIRKBY_Q_STEP + 5e-5
assert huggett["balance / 0.0025"].min() > 1.0 and rate_gaps.min() > 0.1 and n_huggett_beyond == 8

# %%
fig, axes = _nbstyle.figura(1, 2, figsize=(11.5, 4.4))
kirkby_r = huggett["q Kirkby"].map(annual_rate).unstack("credit limit")
for ax, mu in zip(axes, (1.5, 3.0)):
    limits = r_rep.columns.to_numpy()
    ax.plot(limits, r_pub.loc[mu], "s", color=col[1], ms=8, mfc="none", label="Published")
    ax.plot(limits, r_rep.loc[mu], color=col[0], lw=2.0, marker="o", label="Replicated")
    ax.plot(limits, kirkby_r.loc[mu], "D", color=_nbstyle.TINTA, mfc=_nbstyle.FONDO, ms=5, linestyle="none",
            label="Kirkby (2023)")
    ax.axhline(annual_rate(BETA_H), color=_nbstyle.SPINE, lw=0.9, linestyle=":")
    ax.annotate("full insurance", (limits.max(), annual_rate(BETA_H)), textcoords="offset points", xytext=(-4, -12),
                ha="right", color=_nbstyle.TEXTO)
    ax.set(xlabel="Credit limit (bond units)", title=f"Huggett, relative risk aversion μ = {mu}")
axes[0].set_ylabel("Annual interest rate (%)")
axes[0].legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### Cuadro de verificación de la replicación
#
# Cada fila nombra su oráculo ("published" y "Kirkby" son independientes de puremacro; las filas "internal" son nuestras propias comprobaciones de precisión) y da el valor medido y la cota que debía cumplir, o "reported" cuando el tamaño de una diferencia es el hallazgo. Las líneas del veredicto se calculan a partir de las filas.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format}))
bounded = scorecard[scorecard["bound"] != "reported"]
assert (bounded["measured"] <= bounded["bound"].astype(float)).all()
huggett_wedge = annual_rate(BETA_H) - huggett["r replicated"]
print(f"\n{len(bounded)} checks with a bound pass. Table I is reproduced exactly and every published ordering holds. "
      f"{n_beyond} of 24 Aiyagari rates differ from the printed ones by more than our numerical error (by up to "
      f"{largest_gap:.2f} pp), and all {n_huggett_beyond} Huggett prices differ, by up to {gaps_q.max():.4f}. "
      f"Kirkby's independent solutions agree with ours to {kirkby_r_gap:.3f} pp and {kirkby_inner:.1e} in price. "
      f"Changing the chain moves Aiyagari's rates by up to {shift:.2f} pp.")
print(f"How far below the rate of time preference? Aiyagari, on his chain: {wedge_rep.min():.2f} to {wedge_rep.max():.2f} "
      f"pp below {RTP:.2f}%, and {RTP - corner_27:.2f} pp at the corner on the 27-state chain. "
      f"Huggett: {huggett_wedge.min():.2f} to {huggett_wedge.max():.2f} pp a year below {annual_rate(BETA_H):.2f}%.")

# %% [markdown]
# ## Lectura de los resultados
#
# **La respuesta a la pregunta.** Con la propia cadena de Aiyagari, el riesgo no asegurable deja la tasa de interés entre 0.02 y 4.26 pp por debajo de la tasa de preferencia temporal de 4.17%, desde la economía más tranquila hasta la más riesgosa, donde la tasa replicada es ligeramente negativa ($-0.0897\%$). Con una cadena más fina, la brecha más riesgosa es de 3.51 pp. En las economías de Huggett, donde los hogares solo pueden prestarse entre sí, la tasa anual queda entre 0.56 y 27.80 pp por debajo de 4.17%, y cuanto más estricto el límite de crédito, mayor la brecha.
#
# **La Tabla I** se reproduce celda por celda con la cadena que documenta Aiyagari, y las mallas de ancho 2.0, 2.5 y 3.5 fallan. Las columnas añadidas muestran lo que la tabla solo deja implícito: el c.v. en niveles de la cadena dividido por el del proceso es 1.039 con $\rho=0$ y 1.172 con $\rho=0.9$, para ambos valores de $\sigma$.
#
# **La estructura de la Tabla II** se reproduce. La tasa cae con el tamaño y la persistencia del riesgo y con la aversión al riesgo en cada una de las 24 celdas, y las tasas de ahorro la siguen. **La mayoría de sus decimales no se reproducen.** 20 de las 24 brechas superan nuestro error numérico de 0.0074 pp, y 8 superan diez veces ese error. El error tiene signo: las mallas del hogar gruesas subestiman la tasa. Entre las cuatro celdas dentro del error está $(0.4, 0, 3)$, que coincide en los cuatro decimales impresos. Las brechas no están dispersas al azar. Son negativas en las celdas tranquilas, positivas y crecientes hacia la esquina de alto riesgo, y siguen el tamaño del efecto precautorio (correlación 0.93). El documento de trabajo de Aiyagari menciona dos rasgos de su método que podrían dejar ese patrón, y ninguno se prueba aquí: la demanda de activos es una función lineal a trozos sobre 25 subintervalos (p. 27), y los activos medios salen de una sola serie simulada de 10,000 extracciones (pp. 27-28), aunque su nota 39 dice que con 20,000 extracciones los resultados cambiaron muy poco. El ejercicio avanzado de abajo mide el error de muestreo. La función empaquetada `solve_aiyagari_continuous`, una implementación programada por separado del mismo método, reproduce nuestras tres tasas más bajas con una diferencia de 0.0036 pp o menos.
#
# **La figura principal hace visible la segunda lección.** Con $\rho=0.9$, sustituir la cadena de Tauchen de siete estados por la de Rouwenhorst, que iguala exactamente la media, la varianza y la autocorrelación del logaritmo del ingreso, mueve la tasa de equilibrio hasta 0.81 pp, más de tres veces la mayor brecha entre la replicación y el artículo. Siete estados de Rouwenhorst quedan a 0.0612 pp de la cadena de Tauchen de 27 estados: bastan para ver el efecto de la cadena, no para fijar los decimales de la esquina. Con la cadena de 27 estados, nuestras tasas coinciden con las soluciones independientes de Kirkby con una diferencia de 0.019 pp. Su estudio de replicación dice que la esquina inferior derecha de la Tabla II de Aiyagari "contained numerical error" (contenía error numérico; Kirkby 2023, nota 3), y obtiene ahí 0.6806% frente al $-0.3456\%$ impreso. La mayor parte de esa diferencia es la cadena, en línea con su conclusión de que la discretización de los choques tiene "a large and unappreciated influence on results" (una influencia grande y poco reconocida en los resultados): con la cadena de siete estados de Aiyagari nuestra esquina es $-0.0897\%$, y con la de 27 estados es 0.6612%. Así, la esquina de alta persistencia de la Tabla II describe la cadena discretizada al menos tanto como el proceso AR(1). La conclusión cualitativa, que el riesgo no asegurable reduce la tasa de interés y eleva el ahorro, sobrevive a cada uno de estos cambios.
#
# **Las brechas de Huggett quedan sin explicar.** Su tolerancia no las explica: en los precios impresos, nuestro exceso de demanda de saldos de crédito es de 10 a 1990 veces su criterio de 0.0025, y las tasas anuales difieren entre 0.25 y 1.75 pp, cuando él dice que el criterio las mantiene dentro de una décima de punto. Tampoco vienen de nuestro método ni de nuestra malla. La solución independiente de Kirkby cae sobre nuestros precios, con diferencias de hasta 8.6e-05 para los límites $-2$ a $-6$, y la iteración de la función de valor discreta en una malla con separación de 0.027, cercana a la más fina de Huggett, reproduce el precio de nuestra peor celda con una diferencia de 1.02e-05. Con el límite $-8$, los precios de Kirkby quedan hasta 3.5e-04 por debajo de los nuestros. Su malla de activos termina en $a=4$, y cortar la nuestra ahí acerca nuestros precios a los suyos en 8.6e-05 y 1.1e-04, menos de un tercio de la diferencia. Aun así, la figura de Huggett muestra el mecanismo del límite de endeudamiento: límites de crédito más estrictos reducen la tasa libre de riesgo, y una mayor aversión al riesgo la reduce aún más. Como las tasas anuales capitalizan seis períodos del modelo, diferencias pequeñas en el precio por período se vuelven diferencias visibles en la tasa anual.

# %% [markdown]
# ## Tu turno
#
# La celda resuelve de nuevo la celda de la Tabla II con $\sigma=0.4$, $\rho=0.9$, $\mu=3$ en una cadena que tú eliges, y compara la cadena con el proceso log-normal que aproxima y con la Tabla I. Sus aserciones comprueban que el mercado de capital se vacía en la tasa obtenida, que la malla de activos no corta la distribución de la riqueza y, para Rouwenhorst, que los momentos del logaritmo son exactos mientras el c.v. en niveles queda por debajo del del proceso; cada una puede fallar. La cota $r<1/\beta-1$ no se afirma: la impone el intervalo del buscador de raíces, y una cadena sin equilibrio dentro de ese intervalo hace que `brentq` lance un error.
#
# 1. **Básico: la cadena frente al proceso.** Predice primero: ¿una cadena de siete estados exagera o subestima el c.v. en niveles del proceso? Ejecuta la celda por defecto, luego fija `chain = "rouwenhorst"` y prueba `n_states` = 3, 7 y 25. La desviación estándar y la autocorrelación del logaritmo son exactas con cualquier $N$; ¿por qué el c.v. en niveles sigue moviéndose con $N$, y desde qué lado se acerca a la fórmula cerrada? ¿Cuánto exagera el proceso la cadena de Tauchen de siete estados detrás del 0.49 de la Tabla I, y qué implica eso para la fila $\rho=0.9$ de la Tabla II? ¿Por qué Tauchen con un ancho fijo dejaría de converger cuando $N$ crece mucho?
# 2. **Intermedio: persistencia frente a dispersión.** Fija `chain = "tauchen"`, `n_states = 3` y `grid_width = 3.5`, e imprime `np.diag(P)`. El c.v. de la cadena es mayor que el de la celda por defecto; ¿por qué su tasa es mucho más alta? En una celda nueva, compara dos diseños con `rouwenhorst(7, rho, sigma_eps)` y `aiyagari_rate(..., 3)`: mantén la desviación estándar incondicional en 0.4 para $\rho\in\{0.6, 0.9, 0.99\}$, y luego mantén la desviación estándar de la innovación en $0.4\sqrt{1-0.81}$ para $\rho\in\{0.6, 0.9, 0.97\}$. ¿Qué diseño siguen las columnas $\rho$ de la Tabla II, y a cuál se parece la cadena de tres estados? Tus respuestas deben cumplir: con la cadena de tres estados, `cv > 0.4885`, `ac > 0.999` y `r_custom > 3.5`; en el primer diseño la tasa con 0.9 queda por debajo de las otras dos; en el segundo la tasa cae con $\rho$ y es negativa con 0.97.
# 3. **Intermedio: ¿cuánto crédito hace falta para una tasa libre de riesgo positiva?** Resuelve la economía de Huggett con `huggett_price` y `annual_rate` para los límites de crédito $-2, -3, -4, -6, -8, -10$ con $\mu=1.5$ y $3.0$ (12 soluciones, de menos de medio segundo cada una). Comprueba que los cuatro límites publicados caen sobre las curvas de la sección 4, y encuentra para cada $\mu$ el límite en el que la tasa anual cruza cero. ¿Por qué un crédito más holgado sube la tasa hacia `annual_rate(BETA_H)` sin llegar nunca a ella, y por qué el hogar más prudente necesita más margen de endeudamiento antes de que la tasa sea positiva? Tus respuestas deben cumplir: para cada $\mu$ la tasa sube al relajarse el límite y queda por debajo de `annual_rate(BETA_H)`; el cero está entre $-3$ y $-4$ con $\mu=1.5$ y entre $-4$ y $-6$ con $\mu=3$; la tasa con $\mu=3$ es menor que con $\mu=1.5$ en todos los límites.
# 4. **Avanzado: un presupuesto de errores para la brecha de la esquina.** Aiyagari simula la cadena una sola vez con 10,000 extracciones, las pasa por la demanda de activos y toma la media muestral como $Ea$ (WP 502, pp. 27-28; su nota 39 reporta cambios muy pequeños con 20,000 extracciones). Pruébalo en la esquina $\sigma=0.4$, $\rho=0.9$, $\mu=5$. (a) Recorre `grid_width` de 2.90 a 3.10 en pasos de 0.005, quédate con los anchos cuya cadena de siete estados reproduce las ocho celdas de la Tabla I, y resuelve de nuevo la esquina en ambos extremos. (b) En nuestra tasa de equilibrio, reconstruye la política con `solve_egm` sobre las mallas de `aiyagari_market`, simula 400 hogares independientes durante 10,000 y durante 20,000 períodos, y convierte el error de cada promedio temporal en un error de tasa con la pendiente de la oferta neta de capital (una diferencia centrada de `aiyagari_market` con $h=2\times10^{-4}$). Compara la desviación estándar con la brecha, con la dispersión de la cadena y con nuestro error de malla. ¿Por qué cae como $1/\sqrt{T}$, y por qué unos activos persistentes hacen que 10,000 extracciones valgan muchas menos observaciones independientes? (c) ¿Podrían errores de muestreo independientes y de media cero producir brechas negativas en las celdas más tranquilas y positivas y crecientes hacia la esquina? Tus respuestas deben cumplir: los anchos identificados rodean 3.0 en una banda de menos de 0.1; la dispersión de la esquina entre ellos es menor que 0.1 pp y que un tercio de la brecha; la desviación estándar con 20,000 extracciones está entre 0.6 y 0.85 veces la de 10,000; la brecha queda dentro de dos desviaciones estándar de 10,000 extracciones.

# %%
chain = "tauchen"   # ← change this: "tauchen" or "rouwenhorst"
n_states = 7        # ← change this: number of Markov states, 3 to 25
grid_width = 3.0    # ← change this: Tauchen half-width in unconditional s.d., 1.5 to 3.5 (Rouwenhorst ignores it)
assert chain in ("tauchen", "rouwenhorst") and 3 <= n_states <= 25 and 1.5 <= grid_width <= 3.5
sigma_u, rho_u, mu_u = 0.4, 0.9, 3               # the Table II cell: unconditional s.d., persistence, risk aversion
sigma_eps = sigma_u * np.sqrt(1.0 - rho_u**2)
if chain == "tauchen":
    log_grid, P = tauchen(n_states, rho_u, sigma_eps, m=grid_width)
else:
    log_grid, P = rouwenhorst(n_states, rho_u, sigma_eps)
log_sd, log_ac = log_moments(log_grid, P)
cv, ac = level_moments(log_grid, P)
cv_true = np.sqrt(np.exp(sigma_u**2) - 1.0)                                     # log-normal AR(1), closed form
ac_true = (np.exp(rho_u * sigma_u**2) - 1.0) / (np.exp(sigma_u**2) - 1.0)
r_custom = 100.0 * aiyagari_rate(log_grid, P, mu_u)
excess, demand, top_mass = aiyagari_market(log_grid, P, mu_u, r_custom / 100.0)
cv_pub, ac_pub = AIYAGARI_TABLE_I[(sigma_u, rho_u)]
width_note = f", width {grid_width}" if chain == "tauchen" else ""
print(f"{chain}, {n_states} states{width_note}: log s.d. {log_sd:.4f} and autocorrelation {log_ac:.4f} "
      f"(process {sigma_u}/{rho_u}); level c.v. {cv:.4f} and autocorrelation {ac:.4f} "
      f"(process {cv_true:.4f}/{ac_true:.4f}, Table I {cv_pub}/{ac_pub})")
print(f"Equilibrium net return {r_custom:.4f}% (published {AIYAGARI_TABLE_II[(sigma_u, rho_u, mu_u)][0]:.4f}%); "
      f"excess capital {excess / demand:.1e} of demand; mass in the top tenth of the asset grid {top_mass:.1e}")
assert abs(excess) < 1e-6 * demand            # the capital market clears at the rate brentq returned
assert top_mass < 1e-6                         # a_max = 200 does not cut off the wealth distribution
if chain == "rouwenhorst":                     # exact log moments at any N, on a binomial rather than normal grid
    assert abs(log_sd - sigma_u) < 1e-12 and abs(log_ac - rho_u) < 1e-12 and cv < cv_true

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# El cuaderno 01 presenta las economías de Aiyagari y Huggett con el solucionador de malla discreta, el 23 añade oferta de trabajo endógena, el 52 calcula transiciones tras un choque inesperado y el 56 resuelve la versión en tiempo continuo. La función `aiyagari_steady_state` del cuaderno 01 recibe la desviación estándar de la *innovación*: para reproducir con ella una celda de la Tabla II, pasa `sigma=σ*np.sqrt(1-ρ**2)`, `rho=ρ`, `n_z=7` y `gamma=μ`, amplía `r_bracket` para las celdas fuera de su rango por defecto (de 0.5% a 0.2 pp por debajo de $1/\beta-1$) y comprueba que `a_max` no corte la distribución de la riqueza. `docs/es/vfi_continuous_equilibrium.md` documenta `continuous_stationary_distribution` y `solve_aiyagari_continuous`, que empaqueta los mismos tres pasos; la sección 2 la ejecuta con `a_max=200` y `n_a=800`, y su indicador `converged` certifica el punto fijo del hogar, la distribución y el vaciado del mercado. Este cuaderno solo replica equilibrios estacionarios y no verifica los estadísticos distributivos de los artículos. Deja dos cosas abiertas: el origen de los precios impresos de Huggett, que ni su tolerancia ni una malla como la suya explican, y el origen del patrón en los decimales de Aiyagari, cuyos candidatos se nombran a partir de su texto pero no se prueban aquí.
