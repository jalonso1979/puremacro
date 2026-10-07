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
# # Política monetaria óptima: una replicación de Clarida, Galí y Gertler (1999)
#
# **¿Cuánto mejor puede estabilizar la inflación y la brecha del producto un banco central que se compromete que uno que reoptimiza en cada período, y reproducen exactamente los solucionadores de política de puremacro las respuestas de manual?**
#
# Clarida, Galí y Gertler (1999), "The Science of Monetary Policy: A New Keynesian Perspective", *Journal of Economic Literature* 37(4), 1661-1707, derivan en forma cerrada la política óptima del modelo neokeynesiano canónico. Las formas cerradas son un objetivo de replicación estricto: cada número de abajo se compara con una fórmula, no con una corrida anterior de puremacro. No se usan datos y la calibración es ilustrativa.
#
# | Ficha de replicación | |
# |---|---|
# | Resultados replicados | Discreción óptima, compromiso dentro de una regla simple y compromiso desde la perspectiva atemporal |
# | Oráculo | Las formas cerradas del artículo, más la trayectoria de compromiso en forma cerrada derivada en Galí (2015, cap. 5) |
# | Datos | Ninguno; una calibración trimestral ilustrativa y luego 25 calibraciones aleatorias |
# | Solucionadores evaluados | `discretionary_policy` (iteración de Dennis 2007), `lq_commitment`, `ramsey_model` (condiciones de primer orden simbólicas) y `osr` |
# | Aceptación | Error absoluto menor que 1e-9 para choques de una desviación estándar; error relativo menor que 1e-6 cuando interviene una búsqueda sin derivadas |
# | Veredicto | Lo imprime el cuadro de verificación al final, a partir de los errores calculados |

# %% [markdown]
# ## El método en matemáticas
#
# La brecha del producto $x_t$ y la inflación $\pi_t$ siguen una curva IS y una curva de Phillips con choques de demanda y de costos,
# $$x_t=E_t x_{t+1}-\varphi\,(i_t-E_t\pi_{t+1})+g_t,\qquad \pi_t=\lambda x_t+\beta E_t\pi_{t+1}+u_t,$$
# donde $g_t=\mu g_{t-1}+\hat g_t$ y $u_t=\rho u_{t-1}+\hat u_t$. El banco central minimiza $E_t\sum_{j\ge 0}\beta^j(\alpha x_{t+j}^2+\pi_{t+j}^2)$. Los resultados que se reproducen son:
#
# - **Discreción.** $x_t=-\frac{\lambda}{\alpha}\pi_t$, de modo que $\pi_t=\alpha q\,u_t$ y $x_t=-\lambda q\,u_t$ con $q=1/[\lambda^2+\alpha(1-\beta\rho)]$. La regla implícita es $i_t=\gamma_\pi E_t\pi_{t+1}+g_t/\varphi$ con $\gamma_\pi=1+\frac{(1-\rho)\lambda}{\rho\varphi\alpha}>1$.
# - **Compromiso dentro de la regla $x_t=-\omega u_t$.** $\omega^c=\lambda/[\lambda^2+\alpha(1-\beta\rho)^2]$, la política discrecional de un banquero central cuyo peso sobre el producto es $\alpha(1-\beta\rho)<\alpha$.
# - **Compromiso desde la perspectiva atemporal.** $\pi_t=-\frac{\alpha}{\lambda}(x_t-x_{t-1})$, de modo que $x_t=\delta x_{t-1}-\frac{\lambda\delta}{\alpha(1-\delta\beta\rho)}u_t$ con $\delta=\frac{1-\sqrt{1-4\beta a^2}}{2a\beta}$ y $a=\frac{\alpha}{\alpha(1+\beta)+\lambda^2}$.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Un banco central que actúa con discreción toma como dada la inflación esperada, así que solo puede intercambiar la brecha del producto de hoy por la inflación de hoy: se inclina contra el viento en la proporción $\lambda/\alpha$. Un banco central que se compromete también puede prometer política futura. Prometer una brecha negativa más adelante reduce hoy la inflación esperada, lo que mejora la disyuntiva actual. Dentro de una regla simple, la promesa opera a través de la persistencia $\rho$ del choque de costos y equivale a designar un banquero central más averso a la inflación, "conservador" en el sentido de Rogoff (1985). El compromiso atemporal hace que la política dependa de la historia: la brecha sigue negativa después de que el choque se disipa, y el nivel de precios vuelve a su punto de partida. En este modelo los choques de demanda no implican disyuntiva, así que todo régimen los compensa por completo mediante la tasa de interés.

# %% [markdown]
# ## Código resuelto
#
# El modelo se escribe como texto al estilo de Dynare y se cierra con una regla de Taylor provisional, que cada solucionador de política elimina y sustituye por su propia política óptima. Cada choque tiene desviación estándar uno, así que las respuestas están en unidades del choque y las pérdidas en unidades de su varianza.

# %%
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.dsge import build_dynare, discretionary_policy, osr, ramsey_model

# Illustrative quarterly calibration. The CGG results are closed forms, so the checks
# do not depend on these values; section 5 repeats them at random calibrations.
CALIB = {"beta": 0.99, "phi": 1.0, "lam": 0.1, "alpha": 0.25, "rho": 0.5, "mu": 0.8}
TOL = 1e-9   # absolute tolerance for one-standard-deviation shocks
H = 40       # impulse-response horizon in quarters


def cgg_model(params, rule="i = phi_pi*pi + phi_x*x;"):
    """The CGG (1999) economy; `rule` closes it for the private sector."""
    values = {"phi_pi": 1.5, "phi_x": 0.5, **params}
    text = ("var x pi i g u;\nvarexo eps_g eps_u;\n"
            f"parameters {' '.join(values)};\n"
            + "".join(f"{k} = {float(v)!r};\n" for k, v in values.items())
            + "model;\n"
            "x = x(+1) - phi*(i - pi(+1)) + g;\n"   # IS curve
            "pi = lam*x + beta*pi(+1) + u;\n"       # Phillips curve
            f"{rule}\n"
            "g = mu*g(-1) + eps_g;\nu = rho*u(-1) + eps_u;\nend;\n"
            "shocks;\nvar eps_g; stderr 1;\nvar eps_u; stderr 1;\nend;\n")
    return build_dynare(text)


def cgg_closed_forms(beta, lam, alpha, rho, phi=1.0, **_):
    """CGG (1999) closed forms and the Galí (2015) commitment path coefficients."""
    q = 1.0 / (lam**2 + alpha * (1.0 - beta * rho))
    a = alpha / (alpha * (1.0 + beta) + lam**2)
    delta = (1.0 - np.sqrt(1.0 - 4.0 * beta * a**2)) / (2.0 * a * beta)
    c_u = -lam * delta / (alpha * (1.0 - delta * beta * rho))
    omega_c = lam / (lam**2 + alpha * (1.0 - beta * rho) ** 2)
    var_u = 1.0 / (1.0 - rho**2)
    # Unconditional loss E[pi^2] + alpha E[x^2] under each policy (unit shock variance).
    cov_ux = c_u * var_u / (1.0 - delta * rho)
    var_x = (c_u**2 * var_u + 2.0 * delta * c_u * rho * cov_ux) / (1.0 - delta**2)
    cov_x_lag = delta * var_x + c_u * rho * cov_ux
    # Conditional loss (1 - beta) E_0 sum_t beta^t (pi^2 + alpha x^2) from the steady state. Under
    # commitment both responses have the form a delta^j + b rho^j, so the sums are geometric series.
    geometric = lambda a, b: (a**2 / (1.0 - beta * delta**2) + 2.0 * a * b / (1.0 - beta * delta * rho)
                              + b**2 / (1.0 - beta * rho**2))
    x_delta, x_rho = c_u * delta / (delta - rho), -c_u * rho / (delta - rho)
    pi_delta, pi_rho = -(alpha / lam) * c_u * (delta - 1.0) / (delta - rho), -(alpha / lam) * c_u * (1.0 - rho) / (delta - rho)
    return {
        "q": q, "delta": delta, "c_u": c_u, "omega_c": omega_c,
        "gamma_pi": 1.0 + (1.0 - rho) * lam / (rho * phi * alpha) if rho > 0 else np.nan,
        "loss_disc": alpha * q**2 * (alpha + lam**2) * var_u,
        "loss_simple": (alpha * omega_c**2 + ((1.0 - lam * omega_c) / (1.0 - beta * rho)) ** 2) * var_u,
        "loss_comm": (alpha / lam) ** 2 * 2.0 * (var_x - cov_x_lag) + alpha * var_x,
        "cond_disc": alpha * q**2 * (alpha + lam**2) / (1.0 - beta * rho**2),
        "cond_comm": geometric(pi_delta, pi_rho) + alpha * geometric(x_delta, x_rho),
    }


def closed_form_paths(cf, rho, alpha, lam, horizon):
    """Responses to a one-standard-deviation cost-push shock, periods 0..horizon."""
    u = rho ** np.arange(horizon + 1)
    x_comm = np.zeros(horizon + 1)
    for t in range(horizon + 1):
        x_comm[t] = (cf["delta"] * x_comm[t - 1] if t else 0.0) + cf["c_u"] * u[t]
    return {"pi_disc": alpha * cf["q"] * u, "x_disc": -lam * cf["q"] * u,
            "x_comm": x_comm, "pi_comm": -(alpha / lam) * np.diff(x_comm, prepend=0.0)}


checks = []  # one row per replicated result


def record(result, error, tol=TOL):
    checks.append({"result": result, "max_error": float(error), "tolerance": tol})


model = cgg_model(CALIB)
cf = cgg_closed_forms(**CALIB)
paths = closed_form_paths(cf, CALIB["rho"], CALIB["alpha"], CALIB["lam"], H)
beta, lam, alpha, rho, phi = (CALIB[k] for k in ("beta", "lam", "alpha", "rho", "phi"))
weights = {"pi": 1.0, "x": alpha}
print(pd.DataFrame([CALIB], index=["illustrative value"]).to_string())
print("Variables:", model.variables, "| determinate under the placeholder rule:", model.is_determinate)
assert model.is_determinate

# %% [markdown]
# ### 1. Discreción
#
# `discretionary_policy` elimina la regla provisional e itera sobre la política de Markov perfecta (Dennis 2007). A lo largo de la respuesta a un choque de costos, la inflación y la brecha del producto deben ser proporcionales al choque, la condición de inclinarse contra el viento debe cumplirse en cada fecha y la tasa de interés debe ser igual a $\gamma_\pi E_t\pi_{t+1}$ (en una respuesta al impulso no llegan más choques, así que $E_t\pi_{t+1}=\pi_{t+1}$). Un choque de demanda debe dejar la inflación y la brecha en cero mientras la tasa se mueve en $g_t/\varphi$.

# %%
disc = discretionary_policy(model, target_vars=["pi", "x"], weights=weights, instruments="i",
                            beta=beta, tol=1e-12, max_iter=5000)
irf_d = disc.linear_model.irf("eps_u", horizon=H)
irf_dg = disc.linear_model.irf("eps_g", horizon=H)
pi_d, x_d, i_d = (irf_d[v].to_numpy() for v in ("pi", "x", "i"))

record("Discretion: inflation path pi = alpha q u", np.max(np.abs(pi_d - paths["pi_disc"])))
record("Discretion: output-gap path x = -lambda q u", np.max(np.abs(x_d - paths["x_disc"])))
record("Discretion: lean against the wind, x = -(lambda/alpha) pi", np.max(np.abs(x_d + lam / alpha * pi_d)))
record("Discretion: implied rule i = gamma_pi E pi(+1)", np.max(np.abs(i_d[:-1] - cf["gamma_pi"] * pi_d[1:])))
record("Discretion: demand shock fully offset", max(np.max(np.abs(irf_dg[v])) for v in ("x", "pi")))
record("Discretion: the rate tracks demand, i = g/phi",
       np.max(np.abs(irf_dg["i"].to_numpy() - irf_dg["g"].to_numpy() / phi)))
print(f"Dennis iteration converged: {disc.converged}, after {disc.iterations} iterations")
print(f"gamma_pi = {cf['gamma_pi']:.4f}: the implied rule raises the real rate when expected inflation rises")
print(pd.DataFrame(checks).to_string(index=False, float_format=lambda v: f"{v:.1e}"))
assert disc.converged and cf["gamma_pi"] > 1.0
assert all(c["max_error"] < c["tolerance"] for c in checks)

# %% [markdown]
# ### 2. Compromiso desde la perspectiva atemporal, por dos rutas independientes
#
# `discretionary_policy` también resuelve el problema de compromiso con `lq_commitment`, que apila el modelo con los multiplicadores de Lagrange del planificador y resuelve el sistema aumentado con el método QZ de Klein. `ramsey_model` sigue otra ruta: deriva simbólicamente el lagrangiano y resuelve las condiciones de primer orden resultantes. Ambas deben reproducir la trayectoria en forma cerrada. CGG resuelven el problema en dos etapas: eligen la inflación y la brecha sujetas a la curva de Phillips y luego obtienen la tasa de interés de la curva IS. Las condiciones de primer orden simbólicas muestran por qué: el multiplicador de la curva IS es idénticamente cero.

# %%
comm = disc.commitment_result
irf_c = comm.linear_model.irf("eps_u", horizon=H)
irf_cg = comm.linear_model.irf("eps_g", horizon=H)
pi_c, x_c, i_c = (irf_c[v].to_numpy() for v in ("pi", "x", "i"))
record("Commitment: output-gap path (closed form)", np.max(np.abs(x_c - paths["x_comm"])))
record("Commitment: inflation path (closed form)", np.max(np.abs(pi_c - paths["pi_comm"])))
record("Commitment: targeting rule pi = -(alpha/lambda)(x - x(-1))",
       np.max(np.abs(pi_c + alpha / lam * np.diff(x_c, prepend=0.0))))
record("Commitment: demand shock fully offset", max(np.max(np.abs(irf_cg[v])) for v in ("x", "pi")))

ramsey = ramsey_model(model, objective=f"{alpha} * x^2 + pi^2", planner_discount=beta, instruments="i")
foc = dict(zip(model.variables, ramsey.foc_nodes))


def coef(node, name, lead=0):
    """Coefficient of name(lead) in a linear first-order condition."""
    return float(node.diff(name, lead).simplify().value)


mult_is = next(m for m in ramsey.multipliers if np.isclose(coef(foc["i"], m), 1.0))  # dL/di = mult_IS
mult_pc = next(m for m in ramsey.multipliers if abs(coef(foc["x"], m) + lam) < 1e-12)  # enters dL/dx with -lambda
print("Symbolic first-order conditions, coefficients (zeta = Phillips-curve multiplier):")
print(pd.DataFrame({"dL/dx": [coef(foc["x"], "x"), coef(foc["x"], mult_pc), coef(foc["x"], mult_pc, -1)],
                    "dL/dpi": [coef(foc["pi"], "pi"), coef(foc["pi"], mult_pc), coef(foc["pi"], mult_pc, -1)]},
                   index=["own variable at t", "zeta_t", "zeta_{t-1}"]).to_string(float_format=lambda v: f"{v:+.4f}"))
irf_r = ramsey.irf(shock="eps_u", horizon=H)
record("Commitment: symbolic route = matrix route",
       max(np.max(np.abs(irf_r[v].to_numpy() - irf_c[v].to_numpy())) for v in ("x", "pi", "i")))
record("Commitment: IS-curve multiplier is zero", np.max(np.abs(irf_r[mult_is].to_numpy())))
price_d, price_c = np.cumsum(pi_d), np.cumsum(pi_c)
print(f"delta = {cf['delta']:.4f}; price level after {H} quarters: "
      f"discretion {price_d[-1]:+.4f}, commitment {price_c[-1]:+.4f}")
# 2 alpha x = lambda zeta and 2 pi = -(zeta - zeta(-1)) give the targeting rule.
assert np.isclose(coef(foc["x"], "x"), 2.0 * alpha) and np.isclose(coef(foc["pi"], "pi"), 2.0)
assert np.isclose(coef(foc["pi"], mult_pc), 1.0) and np.isclose(coef(foc["pi"], mult_pc, -1), -1.0)
assert abs(price_c[-1]) < 0.05 * abs(price_d[-1])
assert all(c["max_error"] < c["tolerance"] for c in checks)

# %% [markdown]
# La figura principal traza las respuestas de los solucionadores como líneas y las formas cerradas como círculos. El nivel de precios es la inflación acumulada: bajo discreción, lo pasado, pasado está, mientras que bajo compromiso el nivel de precios regresa.

# %%
col, dash = _nbstyle.palette(2), _nbstyle.styles(2)
quarters = np.arange(H + 1)
fig, axes = _nbstyle.figura(2, 2, figsize=(11.5, 7.6), sharex=True)
panels = [("Inflation", pi_d, pi_c, paths["pi_disc"], paths["pi_comm"]),
          ("Output gap", x_d, x_c, paths["x_disc"], paths["x_comm"]),
          ("Price level (cumulated inflation)", price_d, price_c,
           np.cumsum(paths["pi_disc"]), np.cumsum(paths["pi_comm"])),
          ("Nominal interest rate", i_d, i_c, None, None)]
for ax, (title, sol_d, sol_c, form_d, form_c) in zip(axes.flat, panels):
    ax.plot(quarters, sol_d, color=col[0], linestyle=dash[0], lw=2.0, label="Discretion (solver)")
    ax.plot(quarters, sol_c, color=col[1], linestyle=dash[1], lw=2.0, label="Commitment (solver)")
    if form_d is not None:
        ax.plot(quarters[::3], form_d[::3], "o", color=col[0], mfc="none", ms=5, label="Closed form")
        ax.plot(quarters[::3], form_c[::3], "o", color=col[1], mfc="none", ms=5)
    ax.axhline(0.0, color=_nbstyle.SPINE, lw=0.8)
    ax.set_title(title)
for ax in axes[:, 0]:
    ax.set_ylabel("Deviation (shock units)")
for ax in axes[1]:
    ax.set_xlabel("Quarters after a one-s.d. cost-push shock")
axes[0, 0].legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
fig.suptitle("Clarida, Galí and Gertler (1999): solver paths and closed forms")

# %% [markdown]
# ### 3. Compromiso dentro de una regla simple y la regla simple óptima
#
# El resultado de CGG sobre reglas simples dice que comprometerse con $x_t=-\omega u_t$ equivale a la discreción de un banquero central con peso sobre el producto $\alpha(1-\beta\rho)$, así que una solución discrecional con ese peso debe entregar $\omega^c$. `osr`, en cambio, busca sobre los coeficientes de una regla de instrumento, $i_t=g_t/\varphi+\phi_\pi\pi_t+\phi_x x_t$, que compensa los choques de demanda y responde a la inflación y a la brecha. Toda regla así implica $x_t\propto u_t$, de modo que lo mejor que puede lograr es la pérdida de la regla simple de CGG. Ese óptimo no es un punto sino una recta: todas las reglas con $\varphi(\phi_\pi-\rho)=c_0(1-\rho+\varphi\phi_x)$, donde $c_0=\omega^c(1-\beta\rho)/(1-\lambda\omega^c)$, implementan la misma asignación.

# %%
conservative = discretionary_policy(model, target_vars=["pi", "x"],
                                    weights={"pi": 1.0, "x": alpha * (1.0 - beta * rho)},
                                    instruments="i", beta=beta, tol=1e-12, max_iter=5000,
                                    compare_commitment=False)
x0_conservative = conservative.linear_model.irf("eps_u", horizon=1)["x"].iloc[0]
record("Simple rule: conservative central banker delivers -omega_c", abs(x0_conservative + cf["omega_c"]))

tracking = cgg_model(CALIB, rule="i = g/phi + phi_pi*pi + phi_x*x;")
best = osr(tracking, ["phi_pi", "phi_x"], weights, bounds={"phi_pi": (1.0, 10.0), "phi_x": (0.0, 10.0)})
x0_osr = best.optimal_model.irf("eps_u", horizon=1)["x"].iloc[0]
record("Simple rule: OSR loss = CGG simple-rule loss (relative)", abs(best.loss_opt / cf["loss_simple"] - 1.0), tol=1e-6)
record("Simple rule: OSR allocation x = -omega_c u (relative)", abs(x0_osr / cf["omega_c"] + 1.0), tol=1e-6)
c0 = cf["omega_c"] * (1.0 - beta * rho) / (1.0 - lam * cf["omega_c"])
phi_pi_ridge = rho + c0 * (1.0 - rho + phi * best.optimal_params["phi_x"]) / phi
print("OSR coefficients:", {k: round(v, 4) for k, v in best.optimal_params.items()},
      f"| ridge value of phi_pi at that phi_x: {phi_pi_ridge:.4f} | converged: {best.converged}")
print(f"omega_c = {cf['omega_c']:.4f} versus the discretionary response lambda*q = {lam * cf['q']:.4f}")
assert best.converged and abs(phi_pi_ridge - best.optimal_params["phi_pi"]) < 1e-3
assert all(c["max_error"] < c["tolerance"] for c in checks)

# %% [markdown]
# ### 4. ¿Qué tan grandes son las ganancias del compromiso, y con qué criterio?
#
# El atributo `loss` de cada resultado es la pérdida **incondicional** $E[\pi_t^2]+\alpha E[x_t^2]$, el promedio sobre la distribución estacionaria. El atributo `conditional_loss` es el criterio del propio planificador evaluado desde el estado estacionario, $(1-\beta)E_0\sum_t\beta^t(\pi_t^2+\alpha x_t^2)$. `stabilization_bias` compara pérdidas incondicionales por defecto y condicionales con `loss_criterion="conditional"`. Desde el estado estacionario, la solución de compromiso es la política plenamente óptima (de Ramsey), así que nunca puede perder frente a la discreción. Incondicionalmente sí puede: la regla atemporal sigue honrando promesas pasadas, lo que en promedio es costoso cuando el futuro se descuenta mucho (Jensen y McCallum 2002). Ambas pérdidas tienen aquí forma cerrada. Bajo compromiso las respuestas son sumas de dos sucesiones geométricas, $a\delta^j+b\rho^j$. Las pérdidas condicionales se contrastan además con sumas descontadas de respuestas al impulso al cuadrado. El panel izquierdo traza las fronteras de varianza de ambos regímenes al variar el peso sobre el producto. El panel derecho recalcula el cociente de pérdidas para distintos factores de descuento.

# %%
def conditional_loss(linear_model, weights, beta, horizon=4000):
    """(1 - beta) E_0 sum_t beta^t (loss_t) from discounted squared responses to unit shocks."""
    discount = beta ** np.arange(horizon + 1)
    total = 0.0
    for shock in linear_model.shocks:
        irf = linear_model.irf(shock, horizon=horizon)
        total += sum(w * np.sum(discount * irf[v].to_numpy() ** 2) for v, w in weights.items())
    return total


def variances(linear_model):
    """Unconditional variances of inflation and the output gap."""
    with warnings.catch_warnings():
        # Multipliers on constraints that never bind (the IS curve, the demand process) have
        # zero variance, so their variance decompositions are undefined: expected here.
        warnings.filterwarnings("ignore", message="forecast-error variance is zero")
        cov = linear_model.theoretical_moments().covariance
    return cov.loc["pi", "pi"], cov.loc["x", "x"]


record("Losses: discretion, unconditional (relative)", abs(disc.loss / cf["loss_disc"] - 1.0))
record("Losses: commitment, unconditional (relative)", abs(comm.loss / cf["loss_comm"] - 1.0))
record("Losses: discretion, conditional (relative)", abs(disc.conditional_loss / cf["cond_disc"] - 1.0))
record("Losses: commitment, conditional (relative)", abs(comm.conditional_loss / cf["cond_comm"] - 1.0))
record("Losses: conditional_loss against discounted squared responses (relative)",
       max(abs(r.conditional_loss / conditional_loss(r.linear_model, weights, beta) - 1.0) for r in (disc, comm)))
solved = {"discretion": (disc.loss, disc.conditional_loss),
          "optimal simple rule (OSR)": (best.loss_opt, conditional_loss(best.optimal_model, weights, beta)),
          "timeless commitment": (comm.loss, comm.conditional_loss)}
table = pd.DataFrame({"unconditional (.loss)": {k: v[0] for k, v in solved.items()},
                      "conditional, from the steady state": {k: v[1] for k, v in solved.items()}})
print(table.round(4).to_string())
disc_conditional = discretionary_policy(model, ["pi", "x"], weights, "i", beta=beta, tol=1e-12, max_iter=5000,
                                        loss_criterion="conditional")
print(f"stabilization_bias: {disc.stabilization_bias:.4f} by default (unconditional), "
      f"{disc_conditional.stabilization_bias:.4f} with loss_criterion='conditional'")

frontier = []
for alpha_k in np.geomspace(0.02, 2.0, 15):
    res_k = discretionary_policy(model, ["pi", "x"], {"pi": 1.0, "x": alpha_k}, "i",
                                 beta=beta, tol=1e-12, max_iter=20000)
    for regime, lm in (("discretion", res_k.linear_model), ("commitment", res_k.commitment_result.linear_model)):
        var_pi, var_x = variances(lm)
        frontier.append({"alpha": alpha_k, "regime": regime, "var_pi": var_pi, "var_x": var_x})
frontier = pd.DataFrame(frontier)

ratios = []
for beta_k in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995):
    res_k = discretionary_policy(cgg_model({**CALIB, "beta": beta_k}), ["pi", "x"], weights, "i",
                                 beta=beta_k, tol=1e-12, max_iter=50000)
    com_k = res_k.commitment_result
    ratios.append({"beta": beta_k, "converged": res_k.converged,
                   "unconditional": com_k.loss / res_k.loss,
                   "conditional": com_k.conditional_loss / res_k.conditional_loss})
ratios = pd.DataFrame(ratios)
print(ratios.round(4).to_string(index=False))
assert ratios["converged"].all() and (ratios["conditional"] < 1.0).all()
assert all(c["max_error"] < c["tolerance"] for c in checks)

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.5, 4.6))
for (regime, grp), color, ls in zip(frontier.groupby("regime", sort=False), col, dash):
    ax1.plot(grp["var_x"], grp["var_pi"], color=color, linestyle=ls, lw=2.0, label=f"{regime.capitalize()} frontier")
points = {"D": disc.linear_model, "S": best.optimal_model, "C": comm.linear_model}
for tag, lm in points.items():
    var_pi, var_x = variances(lm)
    ax1.plot(var_x, var_pi, "o", color=_nbstyle.TINTA, ms=6)
    ax1.annotate(tag, (var_x, var_pi), textcoords="offset points", xytext=(6, 6), color=_nbstyle.TINTA)
ax1.set(xscale="log", xlabel="Var(output gap)", ylabel="Var(inflation)",
        title="Frontiers as the output weight varies")
ax1.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.plot(ratios["beta"], ratios["unconditional"], color=col[0], linestyle=dash[0], lw=2.0, marker="o",
         label="Unconditional (library .loss)")
ax2.plot(ratios["beta"], ratios["conditional"], color=col[1], linestyle=dash[1], lw=2.0, marker="s",
         label="Conditional, from the steady state")
ax2.axhline(1.0, color=_nbstyle.SPINE, lw=0.8, linestyle=":")
ax2.set(xlabel="Discount factor β", ylabel="Loss: commitment / discretion",
        title="The ranking depends on the criterion")
ax2.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# En el panel izquierdo, D, S y C marcan la discreción, la regla simple óptima y el compromiso atemporal con el peso base. S está sobre la frontera discrecional porque la regla simple reproduce la discreción con un peso menor sobre el producto; solo el compromiso dependiente de la historia desplaza la frontera.
#
# ### 5. Robustez: las fórmulas valen lejos de la calibración
#
# Una replicación que solo se cumple en una calibración podría ser una coincidencia. El ciclo siguiente sortea 25 calibraciones aleatorias y compara con las formas cerradas las trayectorias y ambas pérdidas de la discreción y del compromiso, escalando el error de cada trayectoria por la respuesta más grande.

# %%
rng = np.random.default_rng(1999)
worst_path, worst_loss = 0.0, 0.0
for draw in range(25):
    p = {"beta": rng.uniform(0.9, 0.995), "phi": rng.uniform(0.3, 2.0), "lam": rng.uniform(0.02, 0.5),
         "alpha": rng.uniform(0.02, 1.0), "rho": rng.uniform(0.0, 0.9), "mu": rng.uniform(0.0, 0.9)}
    res_k = discretionary_policy(cgg_model(p), ["pi", "x"], {"pi": 1.0, "x": p["alpha"]}, "i",
                                 beta=p["beta"], tol=1e-12, max_iter=20000)
    cf_k = cgg_closed_forms(**p)
    path_k = closed_form_paths(cf_k, p["rho"], p["alpha"], p["lam"], H)
    r_d = res_k.linear_model.irf("eps_u", horizon=H)
    r_c = res_k.commitment_result.linear_model.irf("eps_u", horizon=H)
    scale = max(1.0, max(np.max(np.abs(v)) for v in path_k.values()))
    worst_path = max(worst_path, np.max(np.abs(r_d["pi"].to_numpy() - path_k["pi_disc"])) / scale,
                     np.max(np.abs(r_d["x"].to_numpy() - path_k["x_disc"])) / scale,
                     np.max(np.abs(r_c["pi"].to_numpy() - path_k["pi_comm"])) / scale,
                     np.max(np.abs(r_c["x"].to_numpy() - path_k["x_comm"])) / scale)
    worst_loss = max(worst_loss, abs(res_k.loss / cf_k["loss_disc"] - 1.0),
                     abs(res_k.commitment_result.loss / cf_k["loss_comm"] - 1.0),
                     abs(res_k.conditional_loss / cf_k["cond_disc"] - 1.0),
                     abs(res_k.commitment_result.conditional_loss / cf_k["cond_comm"] - 1.0))
    assert res_k.converged
record("Robustness: 25 random calibrations, paths (relative)", worst_path)
record("Robustness: 25 random calibrations, both losses (relative)", worst_loss)
print(f"Largest relative path error: {worst_path:.1e}; largest relative loss error: {worst_loss:.1e}")
assert all(c["max_error"] < c["tolerance"] for c in checks)

# %% [markdown]
# ### Cuadro de verificación de la replicación
#
# Todas las comprobaciones anteriores, con el error medido y la tolerancia que debía cumplir. La línea del veredicto se calcula a partir de la tabla.

# %%
scorecard = pd.DataFrame(checks)
scorecard["passed"] = scorecard["max_error"] < scorecard["tolerance"]
print(scorecard.to_string(index=False, formatters={"max_error": "{:.1e}".format, "tolerance": "{:.0e}".format}))
verdict = "reproduced" if scorecard["passed"].all() else "NOT reproduced"
print(f"\n{int(scorecard['passed'].sum())} of {len(scorecard)} checks pass: CGG (1999) results {verdict}.")
assert scorecard["passed"].all()

# %% [markdown]
# ## Lectura de los resultados
#
# Cada comparación con una forma cerrada en el cuadro está al nivel del redondeo, muy por dentro de su tolerancia, y se mantiene así en calibraciones aleatorias. Las dos filas de `osr` están limitadas por la búsqueda sin derivadas, por eso llevan una tolerancia relativa. Así, los cuatro solucionadores reproducen los resultados de CGG sobre discreción, regla simple y compromiso como enunciados analíticos, no solo en una calibración. Las dos rutas de compromiso (lagrangiano matricial y condiciones de primer orden simbólicas) coinciden entre sí al nivel del redondeo. Las condiciones simbólicas dan la regla de objetivos una vez que se elimina el multiplicador nulo de la curva IS.
#
# La figura principal muestra la economía. Tras un choque de costos, el banco central discrecional reparte el ajuste entre la inflación y la brecha, y el nivel de precios queda permanentemente más alto. El banco central comprometido acepta una pérdida inicial de producto menor, pero mantiene la brecha negativa por más tiempo, y el nivel de precios vuelve hacia su valor inicial. El panel de la tasa de interés es un resultado del modelo, no una recomendación de política: la trayectoria de $i_t$ bajo compromiso se deduce de la regla de objetivos y de la curva IS.
#
# La sección 4 es la advertencia. Con el criterio del planificador evaluado desde el estado estacionario, el compromiso le gana a la discreción en todos los factores de descuento de la tabla. Con el criterio incondicional, que usan `.loss` y el `stabilization_bias` por defecto, el orden se invierte con factores de descuento bajos. Un `stabilization_bias` positivo por defecto es, por lo tanto, una propiedad de la calibración, no un teorema. Con `loss_criterion="conditional"` nunca es negativo. El panel de fronteras separa las dos ideas de compromiso. La regla simple solo se mueve a lo largo de la frontera discrecional. Lo que la desplaza hacia adentro es la dependencia de la historia.

# %% [markdown]
# ## Tu turno
#
# Cambia la persistencia del choque de costos o el factor de descuento y vuelve a ejecutar la comparación. La aserción comprueba que los solucionadores siguen coincidiendo con las formas cerradas.
#
# 1. **Básico.** Con el valor por omisión `rho_custom = 0.0`, compara $\omega^c$ con la respuesta discrecional $\lambda q$. ¿Por qué comprometerse con una regla simple no gana nada cuando los choques de costos no tienen correlación serial, mientras que el compromiso atemporal todavía reduce la pérdida?
# 2. **Intermedio.** Fija `rho_custom = 0.9`. ¿Cómo cambian la brecha de pérdidas y el peso conservador $\alpha(1-\beta\rho)$, y por qué la persistencia hace más valioso el compromiso?
# 3. **Avanzado.** Mantén `rho_custom = 0.0` y fija `beta_custom = 0.5`. ¿Sigue siendo positivo `stabilization_bias`? Compáralo con el sesgo bajo `loss_criterion="conditional"` y usa las pérdidas condicionales para explicar por qué la política plenamente óptima sigue ganando desde el estado estacionario (Jensen y McCallum 2002).

# %%
rho_custom = 0.0    # ← change this: cost-push persistence, any value in [0, 0.95]
beta_custom = 0.99  # ← change this: discount factor, any value in [0.3, 0.995]
assert 0.0 <= rho_custom <= 0.95 and 0.3 <= beta_custom <= 0.995
custom = {**CALIB, "rho": rho_custom, "beta": beta_custom}
res_custom = discretionary_policy(cgg_model(custom), ["pi", "x"], weights, "i",
                                  beta=beta_custom, tol=1e-12, max_iter=50000)
cf_custom = cgg_closed_forms(**custom)
path_custom = closed_form_paths(cf_custom, rho_custom, alpha, lam, H)
err_custom = max(
    np.max(np.abs(res_custom.linear_model.irf("eps_u", horizon=H)["x"].to_numpy() - path_custom["x_disc"])),
    np.max(np.abs(res_custom.commitment_result.linear_model.irf("eps_u", horizon=H)["x"].to_numpy()
                  - path_custom["x_comm"])))
print(f"rho = {rho_custom}, beta = {beta_custom}: largest closed-form error {err_custom:.1e}")
print(f"omega_c = {cf_custom['omega_c']:.4f}; discretionary response lambda*q = {lam * cf_custom['q']:.4f}")
print(f"Unconditional loss: discretion {res_custom.loss:.4f}, timeless commitment "
      f"{res_custom.commitment_result.loss:.4f} (stabilization_bias {res_custom.stabilization_bias:+.4f})")
bias_conditional = res_custom.conditional_loss - res_custom.commitment_result.conditional_loss
print(f"Conditional loss from the steady state: discretion {res_custom.conditional_loss:.4f}, commitment "
      f"{res_custom.commitment_result.conditional_loss:.4f} (bias with loss_criterion='conditional' "
      f"{bias_conditional:+.4f})")
assert res_custom.converged and err_custom < TOL

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# El cuaderno 45 resuelve discreción y compromiso con un objetivo que genera sesgo inflacionario antes de pasar a DSGE-VAR y choques de noticias, y los cuadernos 41 y 57 añaden una cota inferior cero que se activa ocasionalmente. `discretionary_policy`, `lq_commitment` y `osr` están documentados en `docs/es/dsge_phase_c.md`; los cuatro solucionadores aceptan modelos construidos con `build_dynare` o `load_mod`, así que las mismas comprobaciones pueden aplicarse a modelos más grandes. Este cuaderno verifica la política lineal-cuadrática en un modelo de tres ecuaciones; no valida la política de Ramsey no lineal ni funciones de pérdida basadas en el bienestar.
