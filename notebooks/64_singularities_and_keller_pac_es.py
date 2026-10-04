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
# # Pliegues, jacobianos singulares y continuación de Keller
#
# **¿Cómo distinguimos si un solver de equilibrio falla en un pliegue, por un mal punto de partida o porque el equilibrio deja de existir, cuando se mueve un parámetro de política?**
#
# Cuatro experimentos, cada uno con insumos etiquetados por lo que son:
#
# | Experimento | Insumo | Oráculo |
# |---|---|---|
# | 1. Un pliegue de libro de texto | el residuo escrito a mano $F=(x_1^2-\lambda,\;x_2-x_1)$ | ramas en forma cerrada $x_1=\pm\sqrt\lambda$ (independiente, derivado a mano) |
# | 2. El modelo de comercio de puremacro | una matriz de contabilidad social de dos países y dos sectores construida a mano; los números son sintéticos | Newton contra la continuación de Keller (interno: dos rutas de puremacro); el umbral de Hawkins-Simon contra los valores propios de NumPy de una matriz construida a mano (interno: misma calibración) |
# | 3. El certificado de existencia a escala | la tabla insumo-producto 77x11 incluida en el paquete, un **conjunto de datos de prueba de regresión** del software, derivado de una exportación corrupta de la OCDE 2020 (`docs/es/ADVISORY.md`, entrada del 2026-09-22); no son datos de la OCDE y sus magnitudes no son estimaciones | las cotas de Collatz-Wielandt, que se certifican a sí mismas (interno) |
# | 4. Un paso lineal rígido | una matriz aleatoria de 77x77 con una coordenada rígida plantada (semilla 42) | el residuo de cada paso devuelto, recalculado (interno) |
#
# El ejercicio final usa la curva de Laffer, cuyo máximo $t^*=1/(1+\varepsilon)$ se conoce en forma cerrada.

# %% [markdown]
# ## El método en matemáticas
#
# Escribe las condiciones de equilibrio como $F(x,\lambda)=0$, $x\in\mathbb R^n$, con un parámetro escalar $\lambda$ (una trayectoria arancelaria, una meta de recaudación). Un **pliegue** (punto de retorno, o bifurcación silla-nodo) es una solución donde $F_x$ es singular y la rama da la vuelta en $\lambda$. La **continuación en el parámetro natural** fija $\lambda$ y aplica Newton en $x$, $x^{k+1}=x^k-F_x^{-1}F(x^k,\lambda)$, así que más allá de un pliegue no tiene raíz que encontrar. La **continuación por pseudo-longitud de arco de Keller (PAC)** convierte $\lambda$ en incógnita y avanza una distancia $\Delta s$ a lo largo de la tangente unitaria $t=(t_x,t_\lambda)$:
#
# $$\underbrace{\begin{pmatrix}F_x & F_\lambda\\ t_{x}^{\top} & t_{\lambda}\end{pmatrix}}_{\text{matriz orlada}}t^{\text{nueva}}=\begin{pmatrix}0\\1\end{pmatrix},\qquad (\hat x,\hat\lambda)=(x_k,\lambda_k)+\Delta s\,t\quad\text{(predictor)},$$
#
# $$\text{corrector: Newton sobre}\quad G(x,\lambda)=\begin{pmatrix}F(x,\lambda)\\ t_x^{\top}(x-x_k)+t_\lambda(\lambda-\lambda_k)-\Delta s\end{pmatrix}=0,\ \text{cuyo jacobiano es la matriz orlada.}$$
#
# En un pliegue simple $F_x$ pierde una dirección, pero $F_\lambda$ queda fuera de su imagen, de modo que la matriz orlada sigue siendo invertible mientras $t_\lambda$ pasa por cero: **un pliegue es un cambio de signo de $t_\lambda$**. En el ejemplo de juguete, $\det F_x=2x_1$ y $\kappa(F_x)\propto 1/|x_1|$ cerca del pliegue en $\lambda=0$.
#
# **Existencia.** Con insumos intermedios Leontief, los precios de costo unitario resuelven $p^\top=p^\top B_\tau+v^\top$ con $B_{\tau,ij}=a_{ij}\,m_{ij}/(1-t_j)$, donde $m_{ij}=1+\tau$ en los flujos transfronterizos, $t_j$ es el impuesto a la producción y $v>0$ reúne los costos unitarios de los factores. Existe un vector de precios positivo si y solo si $\rho(B_\tau)<1$ (Hawkins-Simon); entonces existe para todo $v>0$, pero nada en la condición dice que los precios de los factores que un equilibrio necesita sigan siendo positivos. Para cualquier $u>0$, $\min_i (B_\tau u)_i/u_i\le\rho(B_\tau)\le\max_i (B_\tau u)_i/u_i$ (Collatz-Wielandt): una cota superior menor que uno certifica la productividad, una cota inferior de al menos uno certifica el fracaso, y un intervalo que contiene al uno no prueba nada.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Piensa en $\lambda$ como una perilla de política y en $x$ como el equilibrio que produce. A lo largo de una rama suave, un giro pequeño de la perilla mueve poco el equilibrio, así que arrancar Newton desde la última solución funciona. En un pliegue la rama da la vuelta: más allá no hay un equilibrio cercano, y en el pliegue el jacobiano pierde una dirección, de modo que Newton pasa de convergencia cuadrática a lineal. PAC deja de tratar la perilla como dada y camina sobre la curva de soluciones; la columna $F_\lambda$ aporta la dirección que $F_x$ perdió. La curva de Laffer es el prototipo económico: ninguna tasa impositiva recauda más que el máximo, y una trayectoria de metas de recaudación crecientes da la vuelta en el máximo hacia el lado de impuestos altos. Pero no toda falla es un pliegue. Un solver puede fallar porque arranca demasiado lejos de la respuesta (cualquier continuación lo corrige), o porque el equilibrio deja de existir en una frontera donde un precio llega a cero. Un monitor de pliegues tiene que distinguir estos casos, y un certificado como el de Hawkins-Simon solo descarta algunos.

# %% [markdown]
# ## Código resuelto
#
# El preámbulo carga el estilo de las figuras y los solvers de comercio. El generador aleatorio solo lo usa el experimento 4.

# %%
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import brentq

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.data import load_icio_data
from puremacro.trade.solver import (
    check_hawkins_simon_viability,
    clamp_wage_displacement,
    solve_cyprus_manifold_step,
    solve_keller_pac,
    svd_clamped_newton_step,
)

rng = np.random.default_rng(42)

# %% [markdown]
# ### Experimento 1: un pliegue de libro de texto
#
# Dos funciones auxiliares sirven a todo el cuaderno: Newton con $\lambda$ fija, y PAC. `pac` registra la componente $\lambda$ de la tangente unitaria en cada nodo, incluido el inicial, para que la prueba del pliegue busque un cambio de signo y no un valor negativo.

# %%
def newton_fixed(F, Fx, x0, lam, max_iter=50, tol=1e-10):
    """Newton in x with lam held fixed. Returns the last iterate, success, and the path of x[0]."""
    x = np.array(x0, dtype=float)
    path = [x[0]]
    for _ in range(max_iter):
        f, J = F(x, lam), Fx(x, lam)
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(J))):
            return x, False, np.array(path)                      # el iterado salió del dominio de F
        if np.max(np.abs(f)) < tol:
            return x, True, np.array(path)
        try:
            x = x - la.solve(J, f)
        except la.LinAlgError:                                   # jacobiano exactamente singular
            return x, False, np.array(path)
        path.append(x[0])
    return x, bool(np.max(np.abs(F(x, lam))) < tol), np.array(path)


def pac(F, Fx, Flam, x0, lam0, ds, n_steps, lam_direction, until=None, tol=1e-12):
    """Keller pseudo-arclength continuation of F(x, lam) = 0 from the solution (x0, lam0)."""
    def bordered(z, t):
        x, lam = z[:-1], z[-1]
        return np.block([[Fx(x, lam), Flam(x, lam)[:, None]], [t[None, :]]])

    z = np.r_[x0, lam0]
    # Primera tangente: el vector nulo de [F_x, F_lam], orientado hacia donde debe moverse lam.
    t = la.null_space(np.column_stack([Fx(z[:-1], z[-1]), Flam(z[:-1], z[-1])]))[:, 0]
    t = t if np.sign(t[-1]) == np.sign(lam_direction) else -t
    rec = {"z": [z.copy()], "t_lam": [t[-1]], "cond_Fx": [np.linalg.cond(Fx(z[:-1], z[-1]))],
           "cond_bordered": [np.linalg.cond(bordered(z, t))], "max_F": [np.max(np.abs(F(z[:-1], z[-1])))]}
    for _ in range(n_steps):
        zk = z + ds * t                                          # predictor
        for _ in range(20):                                      # corrector
            g = np.r_[F(zk[:-1], zk[-1]), t @ (zk - z) - ds]
            if np.max(np.abs(g)) < tol:
                break
            zk = zk - la.solve(bordered(zk, t), g)
        t = la.solve(bordered(zk, t), np.eye(len(z))[-1])     # nueva tangente, t_vieja . t_nueva > 0
        t /= np.linalg.norm(t)
        z = zk
        rec["z"].append(z.copy())
        rec["t_lam"].append(t[-1])
        rec["cond_Fx"].append(np.linalg.cond(Fx(z[:-1], z[-1])))
        rec["cond_bordered"].append(np.linalg.cond(bordered(z, t)))
        rec["max_F"].append(np.max(np.abs(F(z[:-1], z[-1]))))
        if until is not None and until(z):
            break
    return {key: np.array(val) for key, val in rec.items()}


def F_toy(x, lam):
    return np.array([x[0] ** 2 - lam, x[1] - x[0]])


def Fx_toy(x, lam):
    return np.array([[2.0 * x[0], 0.0], [-1.0, 1.0]])


def Flam_toy(x, lam):
    return np.array([-1.0, 0.0])


# Continuación en el parámetro natural: bajar lam paso a paso, arrancando Newton en la última solución.
natural = []
x_warm = np.array([0.2, 0.2])
for lam in (0.03, 0.02, 0.01, 0.0, -0.01):
    x_new, ok, path = newton_fixed(F_toy, Fx_toy, x_warm, lam)
    natural.append({"lam": lam, "ok": ok, "iters": len(path) - 1, "x1": x_new[0],
                    "max_F": np.max(np.abs(F_toy(x_new, lam))), "path": path})
    if ok:
        x_warm = x_new
    print(f"lam = {lam:+.2f}: converged {ok!s:5}  iterations {len(path) - 1:2d}  "
          f"x1 = {x_new[0]:+.2e}  max|F| = {natural[-1]['max_F']:.1e}")

halving = natural[3]["path"][1:] / natural[3]["path"][:-1]
print(f"At lam = 0 all {len(halving)} Newton steps multiply x1 by {halving.mean():.6f}; "
      f"the state error {natural[3]['x1']:.1e} is the square root of the residual {natural[3]['max_F']:.1e}.")

assert [r["ok"] for r in natural] == [True, True, True, True, False], "Newton must succeed down to the fold and fail past it"
assert np.allclose(halving, 0.5, atol=1e-12), "at the singular root Newton is only linear, with ratio 1/2"
assert max(r["iters"] for r in natural[:3]) < natural[3]["iters"]

# %%
# PAC desde el mismo inicio, rumbo al pliegue (lam decreciente).
toy = pac(F_toy, Fx_toy, Flam_toy, np.array([0.2, 0.2]), 0.04, ds=0.08, n_steps=14, lam_direction=-1.0)
x1_pac, lam_pac = toy["z"][:, 0], toy["z"][:, -1]
sign_t = np.sign(toy["t_lam"])
flips = np.flatnonzero(np.diff(sign_t))
lower = x1_pac < 0
branch_err = np.max(np.abs(x1_pac[lower] + np.sqrt(lam_pac[lower])))

print("t_lambda at the nodes:", np.array2string(toy["t_lam"], precision=3, max_line_width=120))
assert sign_t[0] < 0 < sign_t[-1] and len(flips) == 1, f"t_lambda must change sign exactly once, not {len(flips)} times"
print(f"Sign changes of t_lambda: {len(flips)}, between nodes {flips[0]} and {flips[0] + 1} "
      f"(x1 from {x1_pac[flips[0]]:+.3f} to {x1_pac[flips[0] + 1]:+.3f}); smallest lambda visited {lam_pac.min():.1e}")
print(f"{lower.sum()} nodes on the lower branch, max|x1 + sqrt(lambda)| = {branch_err:.1e}; max|F| on the path {toy['max_F'].max():.1e}")
print(f"cond(F_x) at the nodes peaks at {toy['cond_Fx'].max():.3g}; the bordered matrix stays between "
      f"{toy['cond_bordered'].min():.3g} and {toy['cond_bordered'].max():.3g}")

# La prueba del pliegue: un cambio de signo (verificado arriba), en el cruce de x1 = 0, y el otro lado sobre la rama inferior.
assert x1_pac[flips[0]] > 0 > x1_pac[flips[0] + 1], "the sign change must coincide with x1 crossing zero"
assert lower.sum() >= 5 and branch_err < 1e-10, "the nodes past the fold must lie on x1 = -sqrt(lambda)"
assert lam_pac.min() >= 0.0 and toy["max_F"].max() < 1e-10

# %%
# Figura 1. Condicionamiento a lo largo de la rama analítica x = (s, s), lambda = s^2, en una malla que
# llega a |x1| = 1e-6: los 15 nodos de PAC son demasiado gruesos para mostrar la explosión por sí solos.
s_grid = np.r_[-np.logspace(np.log10(0.6), -6, 300), np.logspace(-6, np.log10(0.6), 300)]
cond_Fx_branch = np.array([np.linalg.cond(Fx_toy(np.array([s, s]), s * s)) for s in s_grid])
cond_bordered_branch = []
for s in s_grid:
    t_branch = np.array([1.0, 1.0, 2.0 * s]) / np.sqrt(2.0 + 4.0 * s * s)   # tangente unitaria de la rama
    J_b = np.block([[Fx_toy(np.array([s, s]), s * s), Flam_toy(None, None)[:, None]], [t_branch[None, :]]])
    cond_bordered_branch.append(np.linalg.cond(J_b))
cond_bordered_branch = np.array(cond_bordered_branch)
print(f"On the dense branch: max cond(F_x) = {cond_Fx_branch.max():.2e}, "
      f"bordered matrix between {cond_bordered_branch.min():.3g} and {cond_bordered_branch.max():.3g}")
assert cond_Fx_branch.max() > 1e5 and cond_bordered_branch.max() < 2.0

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
lam_line = np.linspace(0.0, lam_pac.max(), 300)
ax0.plot(lam_line, np.sqrt(lam_line), **_nbstyle.S1, label=r"upper branch $x_1=+\sqrt{\lambda}$")
ax0.plot(lam_line, -np.sqrt(lam_line), **_nbstyle.S2, label=r"lower branch $x_1=-\sqrt{\lambda}$")
ax0.plot(lam_pac, x1_pac, ls="none", marker="o", ms=5, mfc="none", mec=_nbstyle.TINTA, label="PAC nodes")
ok_runs = [r for r in natural if r["ok"]]
ax0.plot([r["lam"] for r in ok_runs], [r["x1"] for r in ok_runs], ls="none", marker="s", ms=5,
         color=_nbstyle.S3["color"], label=r"Newton at fixed $\lambda$")
failed = natural[-1]
ax0.plot(failed["lam"], failed["x1"], ls="none", marker="x", ms=8, mew=2, color=_nbstyle.TINTA,
         label=rf"last Newton iterate at $\lambda={failed['lam']:.2f}$ (no root)")
ax0.axvline(0.0, color=_nbstyle.SPINE, lw=0.8, ls=":")
ax0.set_xlabel(r"continuation parameter $\lambda$")
ax0.set_ylabel(r"$x_1$")
ax0.set_title("Solution branches of the toy fold")
ax0.set_ylim(-0.62, 1.05)
ax0.legend(loc="upper left", fontsize=8)

ax1.semilogy(s_grid, cond_Fx_branch, **_nbstyle.S1, label=r"$\kappa(F_x)$ on the branch")
ax1.semilogy(s_grid, cond_bordered_branch, **_nbstyle.S2, label=r"$\kappa$(bordered) on the branch")
ax1.plot(x1_pac, toy["cond_Fx"], ls="none", marker="o", ms=5, mfc="none", mec=_nbstyle.TINTA,
         label=r"$\kappa(F_x)$ at the PAC nodes")
ax1.set_xlabel(r"$x_1$ along the branch (fold at $x_1=0$)")
ax1.set_ylabel(r"2-norm condition number $\kappa$")
ax1.set_title("Conditioning near the fold")
ax1.legend(loc="upper right", fontsize=8)

# %% [markdown]
# ### Experimento 2: el modelo de comercio de puremacro
#
# Una matriz de contabilidad social construida a mano para dos países, A (`C00`) y B (`C01`), con dos sectores cada uno. Los números son sintéticos. Las cuentas de bienes cuadran; A importa más de lo que exporta, y la brecha se mantiene fija como ahorro externo.

# %%
nc_cge, ns_cge, nfd_cge = 2, 2, 3
data_cge = np.zeros((ns_cge * nc_cge + 3, ns_cge * nc_cge + nfd_cge * nc_cge))
data_cge[:4, :4] = np.array([[20.0, 10.0, 5.0, 2.0],
                             [8.0, 25.0, 2.0, 4.0],
                             [5.0, 5.0, 12.0, 18.0],
                             [10.0, 10.0, 18.0, 22.0]])
y_cge = np.array([100.0, 150.0, 120.0, 180.0])            # producción bruta por país-sector
va_fac = y_cge - data_cge[:4, :4].sum(axis=0) - 0.05 * y_cge
data_cge[4, :4] = 0.05 * y_cge                             # impuestos a la producción
data_cge[5, :4] = (2.0 / 3.0) * va_fac                     # las dos filas de factores
data_cge[6, :4] = (1.0 / 3.0) * va_fac
fd_rows = y_cge - data_cge[:4, :4].sum(axis=1)
home_bias, abroad = np.array([0.50, 0.25, 0.05]), np.array([0.10, 0.08, 0.02])
for i in range(4):
    split = np.r_[home_bias, abroad] if i < 2 else np.r_[abroad, home_bias]
    data_cge[i, 4:10] = fd_rows[i] * split
data_cge[4, 4:] = 0.02 * data_cge[:4, 4:].sum(axis=0)
calib_cge = calibrate_trade_model(data_cge, ns=ns_cge, nc=nc_cge, nfd=nfd_cge, validate=True)

exports_A = data_cge[0:2, [2, 3, 7, 8, 9]].sum()
imports_A = data_cge[2:4, [0, 1, 4, 5, 6]].sum()
print(f"A exports {exports_A:.1f} and imports {imports_A:.1f}; baseline foreign balances {calib_cge.invforT.ravel()}")

# %% [markdown]
# **Un pliegue anunciado que no existe.** Hasta la versión 4.3.0, los docstrings del solver, y este cuaderno, ubicaban en $\sigma=0.1238$ (la elasticidad de sustitución entre insumos intermedios) un pliegue en el que Newton estándar diverge. Resolvemos ahí un arancel de 25% por las dos vías.

# %%
sigma_claimed = 0.1238
pac_25 = solve_keller_pac(calib_cge, tau_target=0.25, sigma=sigma_claimed, tol=1e-9)
# Un arancel escalar de PAC también grava las importaciones de demanda final; por eso Newton recibe tau_fd.
newton_25 = solve_trade_equilibrium(calib_cge, tau=0.25, tau_fd=0.25, sigma=sigma_claimed, tol=1e-9)
gap_25 = float(np.max(np.abs(pac_25.x_sol - newton_25.x_sol)))
print(f"PAC   : converged {pac_25.converged} in {pac_25.metadata['pac_steps']} steps, max|F| = "
      f"{pac_25.metadata['max_residual']:.1e}, fold events {pac_25.metadata['fold_points']}, "
      f"raw detections {pac_25.metadata['fold_detections']}")
print(f"Newton: converged {newton_25.converged} in {newton_25.iterations} iterations from the no-tariff start, "
      f"max|F| = {np.max(np.abs(newton_25.residuals)):.1e}")
print(f"The two solutions differ by at most {gap_25:.1e}")

assert pac_25.converged and pac_25.metadata["fold_points"] == [] and pac_25.metadata["fold_detections"] == 0
assert pac_25.metadata["max_residual"] <= 1e-10, "PAC's terminal Newton polish should reach 1e-10"
assert newton_25.converged and gap_25 < 1e-8, "two routes to the same equilibrium must agree"

# %% [markdown]
# **Donde la continuación sí ayuda: un mal punto de partida.** Con insumos intermedios Leontief ($\sigma=0$), resolvemos aranceles mayores en frío (Newton desde la línea base sin aranceles) y con PAC, que sube el arancel desde cero.

# %%
sweep = []
for tau in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
    cold = solve_trade_equilibrium(calib_cge, tau=tau, tau_fd=tau, sigma=0.0, tol=1e-9)
    walk = solve_keller_pac(calib_cge, tau_target=tau, sigma=0.0, tol=1e-9)
    gap = float(np.max(np.abs(cold.x_sol - walk.x_sol))) if cold.converged else np.nan
    sweep.append({"tau": tau, "cold_ok": cold.converged, "cold_F": float(np.max(np.abs(cold.residuals))),
                  "pac_ok": walk.converged, "pac_events": len(walk.metadata["fold_points"]), "gap": gap})
    print(f"tariff {tau:.0%}: cold Newton converged {cold.converged!s:5} (max|F| {sweep[-1]['cold_F']:.1e}); "
          f"PAC converged {walk.converged} with {sweep[-1]['pac_events']} fold events")
first_cold_failure = next(row["tau"] for row in sweep if not row["cold_ok"])
print(f"First tariff in the sweep at which cold Newton fails: {first_cold_failure:.0%}")

assert sweep[0]["cold_ok"] and not sweep[-1]["cold_ok"], "cold Newton should work at 100% and fail at 150%"
assert all(row["pac_ok"] and row["pac_events"] == 0 for row in sweep), "PAC converges with no fold event"
assert all(row["gap"] < 1e-8 for row in sweep if row["cold_ok"]), "where both converge they agree"

# %% [markdown]
# **Donde el equilibrio deja de existir.** Seguimos el equilibrio mientras sube el arancel, arrancando Newton en la solución anterior (continuación en el parámetro natural, el arancel), y registramos el precio de los factores de A relativo al de B, que no depende del numerario. Después pedimos a PAC un arancel de 232% y leemos su monitor de pliegues.

# %%
def tariff_path(sigma, taus):
    """Warm-started Newton along a tariff grid; stops at the first failure. Returns taus and w_A / w_B."""
    x, rel = None, []
    for tau in taus:
        res = solve_trade_equilibrium(calib_cge, tau=tau, tau_fd=tau, sigma=sigma, x0=x, tol=1e-9)
        if not res.converged:
            break
        x = res.x_sol
        rel.append(res.w_sol.ravel()[0] / res.w_sol.ravel()[1])
    return np.asarray(taus[: len(rel)]), np.asarray(rel)


tau_leo, rel_leo = tariff_path(0.0, np.r_[np.arange(0.0, 2.25, 0.05), np.arange(2.25, 2.295, 0.01)])
tau_sub, rel_sub = tariff_path(1.0, np.arange(0.0, 3.001, 0.1))
tau_zero = np.polyval(np.polyfit(rel_leo[-4:], tau_leo[-4:], 1), 0.0)   # donde w_A / w_B llega a 0

with warnings.catch_warnings():         # silencia avisos de desbordamiento mientras log w_A va a -infinito
    warnings.simplefilter("ignore", RuntimeWarning)
    stall = solve_keller_pac(calib_cge, tau_target=2.32, sigma=0.0, tol=1e-9)
event = stall.metadata["fold_points"][0]
tau_stall = 2.32 * stall.metadata["final_lambda"]
subst = solve_keller_pac(calib_cge, tau_target=2.32, sigma=1.0, tol=1e-9)

print(f"sigma = 0: w_A/w_B = {rel_leo[0]:.3f} at 0%, {rel_leo[np.argmin(np.abs(tau_leo - 2.0))]:.3f} at 200%, "
      f"{rel_leo[-1]:.4f} at {tau_leo[-1]:.0%}; linear extrapolation reaches zero at a tariff of {tau_zero:.4f}")
print(f"PAC to 232%, sigma = 0: converged {stall.converged}, stalls at an effective tariff of {tau_stall:.4f}")
print(f"  fold monitor: {len(stall.metadata['fold_points'])} event of kind '{event['kind']}', "
      f"{event['n_detections']} detections (steps {event['first_step']}-{event['last_step']}), "
      f"lambda reversed: {event['lambda_reversed']}, smallest log factor price {event['min_log_factor_price']:.1f}")
print(f"PAC to 232%, sigma = 1: converged {subst.converged}, w_A/w_B = {subst.w_sol.ravel()[0] / subst.w_sol.ravel()[1]:.3f}, "
      f"fold events {len(subst.metadata['fold_points'])}")

assert (not stall.converged) and len(stall.metadata["fold_points"]) == 1
assert event["kind"] == "factor_price_boundary" and not event["lambda_reversed"], "a corner, not a fold"
assert abs(tau_zero - tau_stall) < 0.01, "the warm-started path and PAC must locate the same boundary"
assert subst.converged and subst.metadata["fold_points"] == [] and tau_sub[-1] > tau_stall

# %% [markdown]
# **Qué dice el certificado de existencia en esta tabla.** Buscamos el arancel en el que falla el certificado de Hawkins-Simon, por bisección sobre la verificación pública (lanza `ValueError` cuando un esquema no queda certificado; la función auxiliar lee el veredicto y las cotas en el mensaje). Lo comprobamos por una segunda ruta: los valores propios de NumPy de $B_\tau$ construida a mano a partir de la calibración.

# %%
def hawkins_simon(calib, tau):
    """('viable' | 'violated' | 'unresolved' | 'error', lower bound, upper bound) from the public check."""
    try:
        cert = check_hawkins_simon_viability(calib, tau=tau)
        return "viable", cert.cw_lower, cert.cw_upper
    except ValueError as err:
        msg = str(err)
        status = "violated" if "violates" in msg else "unresolved" if "unresolved" in msg else "error"
        found = re.findall(r"\[([-+0-9.eE]+), ([-+0-9.eE]+)\]", msg)
        low, high = (float(found[0][0]), float(found[0][1])) if found else (np.nan, np.nan)
        return status, low, high


def hs_threshold(calib, lo, hi, tol):
    """Bisect between a certified-viable and a certified-violated tariff."""
    assert hawkins_simon(calib, lo)[0] == "viable" and hawkins_simon(calib, hi)[0] == "violated"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        status = hawkins_simon(calib, mid)[0]
        assert status in ("viable", "violated"), f"certificate {status} at {mid}"
        lo, hi = (mid, hi) if status == "viable" else (lo, mid)
    return lo, hi


def B_by_hand(calib, tau):
    """B_tau = a * m / (1 - t) with m = 1 + tau across borders; taxes clipped as the public check does."""
    ns, nc = calib.n_sectors, calib.n_countries
    a = np.asarray(calib.a, dtype=float).reshape((ns * nc, ns * nc), order="F")
    tax = np.clip(np.asarray(calib.tax, dtype=float).flatten(order="F"), -0.9, 0.999)
    country = np.arange(ns * nc) // ns
    m = np.where(country[:, None] != country[None, :], 1.0 + tau, 1.0)
    return a * m / (1.0 - tax)[None, :]


hs_lo, hs_hi = hs_threshold(calib_cge, 0.0, 20.0, tol=1e-6)
tau_eig = brentq(lambda tau: np.max(np.abs(np.linalg.eigvals(B_by_hand(calib_cge, tau)))) - 1.0, 0.0, 20.0, xtol=1e-9)
status_stall, low_stall, high_stall = hawkins_simon(calib_cge, 2.32)
taus_cert = np.linspace(0.0, 10.0, 101)
rho_upper = np.array([check_hawkins_simon_viability(calib_cge, tau=t).cw_upper if t < hs_lo
                      else np.nan for t in taus_cert])
print(f"Certificate threshold on the 2x2 table: between {hs_lo:.6f} and {hs_hi:.6f}; eigenvalue route {tau_eig:.6f}")
print(f"At 232%, past the end of the equilibrium branch: {status_stall}, rho(B) in [{low_stall:.4f}, {high_stall:.4f}]")

assert abs(0.5 * (hs_lo + hs_hi) - tau_eig) < 1e-4, "the certificate and the eigenvalues must agree"
assert status_stall == "viable" and high_stall < 0.9, "Hawkins-Simon certifies prices past the end of the branch"

# %%
# Figura 2. Izquierda: precio relativo de los factores de A a lo largo del arancel. Derecha: el certificado en la misma tabla.
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
ax0.plot(tau_leo, rel_leo, **_nbstyle.S1, label=r"Leontief, $\sigma=0$")
ax0.plot(tau_sub, rel_sub, **_nbstyle.S2, label=r"unit elasticity, $\sigma=1$")
ax0.axvline(tau_stall, color=_nbstyle.SPINE, lw=1.0, ls=":", label=rf"PAC stalls ($\sigma=0$), {tau_stall:.2f}")
fail_rows = [row for row in sweep if not row["cold_ok"]]
ax0.plot([row["tau"] for row in fail_rows], np.interp([row["tau"] for row in fail_rows], tau_leo, rel_leo),
         ls="none", marker="x", ms=8, mew=2, color=_nbstyle.TINTA, label=r"cold Newton fails ($\sigma=0$)")
ax0.set_xlabel(r"tariff rate $\tau$ (1 = 100%)")
ax0.set_ylabel(r"factor price of A relative to B, $w_A/w_B$")
ax0.set_title("A's factor price along the tariff path")
ax0.set_ylim(bottom=0.0)
ax0.legend(loc="upper right", fontsize=8)

ax1.plot(taus_cert, rho_upper, **_nbstyle.S1, label=r"certified upper bound on $\rho(B_\tau)$")
ax1.axhline(1.0, color=_nbstyle.SPINE, lw=1.0, ls="-")
ax1.axvline(tau_stall, color=_nbstyle.SPINE, lw=1.0, ls=":")
ax1.axvline(tau_eig, color=_nbstyle.NOTA, lw=1.0, ls="--")
ax1.text(tau_stall + 0.15, 0.95, "equilibrium branch ends\n" rf"($\sigma=0$) at {tau_stall:.2f}",
         color=_nbstyle.TEXTO, fontsize=8, va="top")
ax1.text(tau_eig - 0.15, 0.55, f"certificate fails\nat {tau_eig:.2f}", color=_nbstyle.TEXTO, fontsize=8, ha="right")
ax1.set_xlabel(r"tariff rate $\tau$ (1 = 100%)")
ax1.set_ylabel(r"spectral radius bound")
ax1.set_title("Hawkins-Simon certificate, same 2x2 table")
ax1.legend(loc="lower right", fontsize=8)

# %% [markdown]
# ### Experimento 3: el certificado a escala
#
# La tabla 77x11 incluida en el paquete es un conjunto de datos de prueba de regresión, no datos de la OCDE (ver la tabla del inicio). Aquí solo sirve para ejercitar el certificado sobre una matriz de 847 filas. La verificación pública recorta las tasas del impuesto a la producción a $[-0.9, 0.999]$, así que las cotas corresponden a esa matriz recortada.

# %%
calib_icio = calibrate_trade_model(load_icio_data(source="legacy", sectors=11), ns=11, nc=77, nfd=3, validate=True)
codes = list(calib_icio.country_codes)
n_clipped = int(np.sum(np.asarray(calib_icio.tax) < -0.9))
print(f"{len(codes)} regions ({len(codes) - 1} economies and {codes[-1]}), "
      f"{calib_icio.n_sectors * len(codes)} country-sectors; {n_clipped} tax rates below -0.9 are clipped")
fixture = {tau: hawkins_simon(calib_icio, tau) for tau in (0.0, 0.25, 8.0)}
for tau, (status, low, high) in fixture.items():
    print(f"tariff {tau:4.0%}: {status:9s} rho(B) in [{low:.6f}, {high:.6f}]")
fix_lo, fix_hi = hs_threshold(calib_icio, 3.6, 3.7, tol=1e-4)
print(f"The certificate switches from viable to violated between tariffs of {fix_lo:.4f} and {fix_hi:.4f}")

assert fixture[0.25][0] == "viable" and fixture[0.25][2] < 1.0, "the upper bound, not an estimate, certifies"
assert fixture[8.0][0] == "violated" and fixture[8.0][1] >= 1.0, "800% must be a certified violation, not unresolved"

# %% [markdown]
# ### Experimento 4: ¿qué compra un paso de Newton acotado?
#
# `puremacro.trade.solver` incluye tres funciones auxiliares que acotan un paso de Newton para los precios de los factores en logaritmos (`svd_clamped_newton_step`, `clamp_wage_displacement`, `solve_cyprus_manifold_step`); los solvers de la biblioteca no las llaman. Una cota evita que una mala linealización lance el iterado lejos. Aquí la dificultad es el propio sistema lineal: una matriz simétrica aleatoria con una fila y una columna escaladas por $10^{-5}$. La coordenada rígida ocupa la posición de Chipre en la lista de 77 regiones porque `solve_cyprus_manifold_step` lleva el nombre de la pequeña economía que la motivó; la matriz es aleatoria y no dice nada sobre Chipre. Esa función elimina exactamente las otras 76 coordenadas, escala todo el paso hasta `max_disp` e informa el residuo del paso que devuelve.

# %%
k_stiff = codes.index("CYP")
n = len(codes)
A_rnd = rng.standard_normal((n, n))
S = A_rnd.T @ A_rnd + np.eye(n)
S[k_stiff, :] *= 1e-5
S[:, k_stiff] *= 1e-5
S[k_stiff, k_stiff] = 1e-6
rhs = rng.standard_normal(n)


def rel_residual(step):
    return np.linalg.norm(S @ step - rhs) / np.linalg.norm(rhs)


dw_exact = la.solve(S, rhs)
dw_block, res_block, conv_block, info = solve_cyprus_manifold_step(
    S, rhs, country_codes=codes, max_disp=0.30, return_info=True)
steps = {
    "exact solve": dw_exact,
    "SVD modal clamp, then scale to 0.30": svd_clamped_newton_step(S, -rhs, np.ones(n), np.ones(n), max_comp=20.0, max_disp=0.30),
    "block elimination, then scale to 0.30": dw_block,
    "clip each coordinate to 0.30": clamp_wage_displacement(dw_exact, 0.30),
    "no step": np.zeros(n),
}
print(f"cond(S) = {np.linalg.cond(S):.2e}; stiff coordinate {info['idx_cyp']} ({info['country']}); "
      f"the exact solve has relative residual {rel_residual(dw_exact):.1e}")
for name, step in steps.items():
    print(f"  {name:38s} max|step| = {np.max(np.abs(step)):9.3g}   ||S step - rhs|| / ||rhs|| = {rel_residual(step):.4f}")
cosine = dw_block @ dw_exact / (np.linalg.norm(dw_block) * np.linalg.norm(dw_exact))
print(f"solve_cyprus_manifold_step reports residual {res_block:.4g} and converged {conv_block}; "
      f"its step is the exact solve times {info['clamp_scale']:.3g} (cosine {cosine:.6f}); "
      f"before the clamp the residual was {info['direction_residual']:.1e}")
print(f"Capped at 0.30 per step, Newton would need about {np.max(np.abs(dw_exact)) / 0.30:.2e} steps to cover the exact step")

assert rel_residual(dw_exact) < 1e-12 and info["idx_cyp"] == k_stiff and info["country"] == "CYP"
assert abs(res_block - np.max(np.abs(S @ dw_block - rhs))) < 1e-12 and not conv_block, "reported = actual residual"
assert cosine > 1 - 1e-12 and info["direction_converged"]
assert 0.9 < rel_residual(steps["SVD modal clamp, then scale to 0.30"]) < 1.0
assert rel_residual(dw_block) > 0.999 and rel_residual(steps["clip each coordinate to 0.30"]) > 1.0

# %% [markdown]
# ## Lectura de los resultados
#
# - **El pliegue de juguete (figura 1).** Newton con $\lambda$ fija, arrancado desde la solución anterior, converge en $\lambda=0.03$, $0.02$ y $0.01$ en 3 o 4 iteraciones. En el pliegue, $\lambda=0$, necesita 14: cada paso multiplica $x_1$ exactamente por 0.5, de modo que el estado sigue desviado en 6.1e-06 cuando el residuo es 3.7e-11. En $\lambda=-0.01$ falla porque no hay raíz real. PAC, desde el mismo punto, rodea el pliegue: $t_\lambda$ cambia de signo una vez, entre los nodos 3 y 4, donde $x_1$ cruza el cero (+0.033 a -0.024), y los 11 nodos posteriores están sobre $x_1=-\sqrt\lambda$ con un error de 2.5e-14. El panel derecho explica por qué funciona. A lo largo de la rama, $\kappa(F_x)$ crece como $1/|x_1|$, hasta 1.00e+06 en $|x_1|=10^{-6}$, mientras la matriz orlada se mantiene entre 1.41 y 1.85. Los nodos de PAC nunca se acercan al pliegue más que $|x_1|=0.024$, así que $\kappa(F_x)$ en los nodos llega apenas a 41.9: una trayectoria gruesa puede saltarse una singularidad sin verla, y por eso la prueba es el signo de $t_\lambda$ y no el tamaño de $\kappa$.
# - **Un pliegue anunciado que no existe.** Con $\sigma=0.1238$ y un arancel de 25%, PAC no registra ningún evento de pliegue ni ninguna detección, Newton simple converge desde el punto sin aranceles en 8 iteraciones, y las dos soluciones difieren en a lo sumo 6.8e-10.
# - **Un mal punto de partida.** Con insumos Leontief, Newton en frío falla a partir de un arancel de 140% (max|F| 2.6e+01), mientras PAC alcanza todos los aranceles del barrido con 0 eventos de pliegue, y la trayectoria con arranque en caliente de la figura 2 también pasa el 140%. La falla depende de dónde arranca Newton, no de una singularidad, y cualquier continuación la corrige.
# - **Donde termina el equilibrio (figura 2, izquierda).** Con $\sigma=0$, el precio de los factores de A relativo al de B cae de 1.000 en 0% a 0.186 en 200% y a 0.0062 en 229%. Extrapolado linealmente llega a cero en un arancel de 2.3007, y PAC, al que se pide 232%, se detiene en 2.3016. Su monitor registra un evento de tipo `factor_price_boundary`: 55 detecciones, ninguna reversión de $\lambda$ y un logaritmo del precio de los factores que baja hasta -31.6. La rama de equilibrios termina donde el precio de los factores de A llega a cero, y más allá ningún solver encuentra un equilibrio con precios de los factores positivos. Es una esquina, no un pliegue, y el monitor acierta al no llamarlo punto de retorno. Con sustitución entre insumos ($\sigma=1$) el mismo arancel de 232% se resuelve sin eventos de pliegue, con $w_A/w_B$ = 0.647.
# - **Lo que el certificado puede y no puede decir (figura 2, derecha, y experimento 3).** En la tabla 2x2, el certificado de Hawkins-Simon se sostiene hasta un arancel entre 8.653800 y 8.653801; los valores propios de NumPy ubican $\rho(B_\tau)=1$ en 8.653814, un poco más arriba, porque el certificado exige una cota superior menor que $1-10^{-6}$. En 232%, más allá del final de la rama de equilibrios, certifica $\rho(B)$ = 0.5160. La condición es necesaria para precios Leontief positivos; está lejos de ser suficiente para un equilibrio con precios de los factores positivos. En la tabla de prueba de 847 filas, un arancel de 25% mueve las cotas de 0.696863 a 0.697149, el veredicto cambia entre aranceles de 3.6818 y 3.6819, y 800% es una violación certificada (ambas cotas en 1.564734), no un intervalo sin resolver. Estos números describen los datos de prueba de regresión, no la red de producción de la OCDE.
# - **El paso rígido.** La solución exacta tiene un residuo relativo de 4.6e-14, pero mueve la coordenada rígida en 8.52e+05. Cada paso acotado a 0.30 deja casi todo el lado derecho sin explicar: el recorte modal por SVD deja un residuo relativo de 0.9851, la eliminación por bloques seguida de un escalamiento uniforme deja 1.0000 (lo mismo que no moverse, porque ese paso es la solución exacta multiplicada por 3.52e-07), y recortar cada coordenada lo eleva a 16.0543. `solve_cyprus_manifold_step` ahora lo informa: residuo 3.208 y converged False para el paso que devuelve, mientras la dirección sin recortar tenía residuo 1.4e-13. Hasta la versión 4.3.0 informaba el residuo sin recortar con `converged=True`. Una cota limita un paso; no resuelve un sistema rígido, y a 0.30 por paso Newton necesitaría unos 2.84e+06 pasos. Una cota solo rinde cuando el modelo es no lineal y un paso largo saldría de la región donde vale la linealización, algo que esta prueba lineal no puede mostrar.

# %% [markdown]
# ## Tu turno
#
# Cambia la elasticidad de la base gravable $\varepsilon$. La recaudación es $R(t)=t(1-t)^{\varepsilon}$ para una tasa $t\in[0,1]$, así que el máximo de la curva de Laffer está en $t^*=1/(1+\varepsilon)$, una forma cerrada derivada a mano. La celda continúa la meta de recaudación $\bar R$ hacia arriba desde $t=0.05$ con la función `pac` del cuaderno y verifica tres cosas: la única vuelta de la trayectoria encierra a $t^*$, ningún nodo pide más que la recaudación máxima, y Newton para una meta 1% por encima del máximo no encuentra ninguna tasa en $[0,1]$.
#
# 1. **Básico.** Antes de ejecutar, predice $t^*$ para $\varepsilon=1$ y $\varepsilon=4$; luego ejecuta ambos casos. ¿Por qué el máximo de recaudación es un pliegue de la ecuación $R(t)=\bar R$, y qué registra el signo de $t_\lambda$ a lo largo de esta trayectoria?
# 2. **Intermedio.** En una celda nueva, ejecuta `newton_fixed(F_laffer, Fx_laffer, np.array([0.5 * t_star]), 0.99 * R_max)`. Ahora Newton converge: ¿de qué lado del máximo, si arranca en $t^*/2$, donde $R$ es cóncava? ¿Qué tasa recauda lo mismo del otro lado, y qué parte de la trayectoria de PAC la alcanza?
# 3. **Reto.** De vuelta en el experimento 2, ejecuta `tariff_path(0.5, np.arange(0.0, 4.0, 0.05))` y `solve_keller_pac(calib_cge, tau_target=3.5, sigma=0.5, tol=1e-9)`. ¿Más sustitución mueve el arancel en el que el precio de los factores de A llega a cero, y en qué dirección? ¿La nueva detención vuelve a ser un evento de tipo `factor_price_boundary` sin reversión de $\lambda$?

# %%
eps_base = 2.0   # ← cambia esto: elasticidad de la base gravable, de 0.5 a 4
assert 0.5 <= eps_base <= 4.0, "stay inside the advertised range 0.5 to 4"


def revenue(t):
    """Laffer revenue R(t) = t (1 - t)^eps; tax rates live in [0, 1] and are undefined outside."""
    return t * (1.0 - t) ** eps_base if 0.0 <= t <= 1.0 else np.nan


def F_laffer(x, R):
    return np.array([revenue(x[0]) - R])


def Fx_laffer(x, R):
    t = x[0]
    slope = (1.0 - t) ** eps_base - eps_base * t * (1.0 - t) ** (eps_base - 1.0) if 0.0 <= t < 1.0 else np.nan
    return np.array([[slope]])


def Flam_laffer(x, R):
    return np.array([-1.0])


t_star = 1.0 / (1.0 + eps_base)          # forma cerrada: R'(t) = 0
R_max = revenue(t_star)
laffer = pac(F_laffer, Fx_laffer, Flam_laffer, np.array([0.05]), revenue(0.05), ds=0.05, n_steps=60,
             lam_direction=+1.0, until=lambda z: z[0] > 0.9)
t_path, R_path = laffer["z"][:, 0], laffer["z"][:, 1]
turn = np.flatnonzero(np.diff(np.sign(laffer["t_lam"])))
assert len(turn) == 1, f"the path should turn exactly once, not {len(turn)} times"
_, above_ok, _ = newton_fixed(F_laffer, Fx_laffer, np.array([0.5 * t_star]), 1.01 * R_max)
print(f"Peak: t* = {t_star:.4f}, R_max = {R_max:.6f}; PAC turned between t = {t_path[turn[0]]:.3f} and {t_path[turn[0] + 1]:.3f}")
print(f"Largest revenue target on the path {R_path.max():.6f}; path ends at t = {t_path[-1]:.3f}, R = {R_path[-1]:.4f}")
print(f"Newton for a target 1% above the peak converged: {above_ok}")

assert t_path[turn[0]] < t_star < t_path[turn[0] + 1], "the turn must bracket the revenue peak"
assert R_path.max() <= R_max + 1e-12 and laffer["max_F"].max() < 1e-10
assert not above_ok, "no tax rate raises more than the peak revenue"

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# Los solvers de comercio usados aquí también sirven al cuaderno 63 (`63_trade_wars_and_nash_tariffs_es`), donde `solve_policy_equilibrium(method="auto")` termina su escalera de recuperación con la continuación de Keller (`docs/trade_policy.md`, en inglés); `docs/trade_accounting.md` enumera los solvers que comparten las mismas ecuaciones. Las filas "Keller PAC" y "MRIO productivity" de `docs/STRUCTURAL_VALIDATION_STATUS.md` (en inglés) dicen qué está validado: no se ha demostrado el recorrido de un pliegue económico del modelo de comercio, y la API de certificación de teoremas (`verify_theorems_1_to_4`) sigue en cuarentena. `docs/es/ADVISORY.md` registra la procedencia de la tabla incluida (2026-09-22) y la corrección del monitor de pliegues y de los residuos de los pasos (2026-09-30). Para trabajo empírico con insumo-producto, lee las tablas de la OCDE con `load_oecd_icio_granular` o `puremacro.trade.mrio.read_oecd_native`.
