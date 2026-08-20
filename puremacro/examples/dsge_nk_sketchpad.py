"""Three-equation New Keynesian model, written as equations rather than matrices.

The textbook NK block — dynamic IS, Phillips curve, Taylor rule — solved
by handing ``puremacro.dsge.build`` the equilibrium conditions as a
Python function. No matrices are derived by hand: the Jacobians come
from complex-step differentiation, and ``klein_solve`` does the rest.

Three things this demonstrates:

1. **Level linearisation.** The NK block is already written in
   deviations from a zero steady state, so ``linearize="level"`` is the
   honest choice; ``build`` verifies that the supplied steady state
   really does solve the equations rather than taking it on trust.
2. **Both shock timings.** The policy shock enters the Taylor rule
   contemporaneously, so it moves everything at ``h=0`` and, being
   i.i.d. with no state behind it, is gone by ``h=1``. The demand shock
   drives a persistent state instead, so the natural rate itself is
   still zero at ``h=0`` and jumps at ``h=1`` — but the output gap and
   inflation move on impact anyway, because agents already know the
   higher natural rate is coming. Both jumps are the loading Klein calls
   ``L``, which returned zero for every model before 1.2.0.
3. **The Taylor principle as a solver outcome.** With ``phi_pi < 1`` the
   model has no unique stable solution, and ``build`` raises
   ``BlanchardKahnError`` rather than returning something plausible.

Run:
    python -m puremacro.examples.dsge_nk_sketchpad

Español
-------
Modelo neokeynesiano de tres ecuaciones, escrito como ecuaciones en vez
de matrices.

El bloque neokeynesiano de manual — IS dinámica, curva de Phillips y
regla de Taylor — resuelto entregando a ``puremacro.dsge.build`` las
condiciones de equilibrio como una función de Python. No se deriva
ninguna matriz a mano: los jacobianos provienen de la diferenciación de
paso complejo y ``klein_solve`` hace el resto.

Tres cosas que esto demuestra:

1. **Linealización en niveles.** El bloque NK ya está escrito en
   desviaciones respecto de un estado estacionario nulo, así que
   ``linearize="level"`` es la opción honesta; ``build`` verifica que el
   estado estacionario suministrado resuelve de verdad las ecuaciones en
   lugar de darlo por bueno.
2. **Ambos momentos del choque.** El choque de política entra de forma
   contemporánea en la regla de Taylor, así que mueve todo en ``h=0`` y,
   al ser i.i.d. y no tener un estado detrás, desaparece en ``h=1``. El
   choque de demanda impulsa en cambio un estado persistente, de modo que
   la propia tasa natural sigue en cero en ``h=0`` y salta en ``h=1``;
   aun así, la brecha del producto y la inflación se mueven en el impacto,
   porque los agentes ya saben que llega una tasa natural más alta. Ambos
   saltos son la carga que Klein denomina ``L``, que antes de 1.2.0
   devolvía cero para todos los modelos.
3. **El principio de Taylor como resultado del solver.** Con
   ``phi_pi < 1`` el modelo carece de solución estable única, y ``build``
   lanza ``BlanchardKahnError`` en lugar de devolver algo verosímil.

Ejecución:
    python -m puremacro.examples.dsge_nk_sketchpad
"""
from __future__ import annotations

from ..dsge import build
from ..dsge.klein import BlanchardKahnError

# Standard quarterly calibration (Galí, *Monetary Policy, Inflation and
# the Business Cycle*, ch. 3).
PARAMS = dict(
    beta=0.99,     # discount factor
    sigma=1.0,     # inverse intertemporal elasticity
    kappa=0.1275,  # slope of the Phillips curve
    phi_pi=1.5,    # Taylor coefficient on inflation
    phi_x=0.125,   # Taylor coefficient on the output gap
    rho_r=0.90,    # persistence of the natural rate
)

VARIABLES = ["rn", "x", "pi", "i"]
STATES = ["rn"]           # only the exogenous natural rate is predetermined
SHOCKS = ["eps_demand", "eps_policy"]


def nk_equations(xp, x, e, p):
    """E_t f(z_{t+1}, z_t, u_t) = 0 for the three-equation NK block."""
    return [
        # Natural rate: AR(1). The innovation moves the state into t+1.
        xp.rn - p.rho_r * x.rn - e.eps_demand,
        # Dynamic IS: x_t = E_t x_{t+1} - (1/sigma)(i_t - E_t pi_{t+1} - rn_t)
        xp.x - x.x - (x.i - xp.pi - x.rn) / p.sigma,
        # New Keynesian Phillips curve: pi_t = beta E_t pi_{t+1} + kappa x_t
        p.beta * xp.pi + p.kappa * x.x - x.pi,
        # Taylor rule with a contemporaneous policy shock.
        p.phi_pi * x.pi + p.phi_x * x.x + e.eps_policy - x.i,
    ]


def solve(**overrides):
    """Build and solve the model, with optional parameter overrides."""
    params = {**PARAMS, **overrides}
    return build(
        nk_equations,
        variables=VARIABLES,
        states=STATES,
        shocks=SHOCKS,
        params=params,
        # Everything is a deviation from a zero steady state, so the
        # model is linear already and there is nothing to solve for.
        steady_state={name: 0.0 for name in VARIABLES},
        linearize="level",
    )


def main() -> None:
    model = solve()
    print(model.summary())

    print("\nDecision rules (response to a one-unit natural-rate deviation):")
    print(model.policy().round(4).to_string())

    print("\nIRF to a 1pp policy shock (in the Taylor rule — hits at h=0, iid):")
    print(model.irf("eps_policy", horizon=4).round(4).to_string())

    print("\nIRF to a 1pp demand shock (rn moves at h=1; controls jump at h=0):")
    print(model.irf("eps_demand", horizon=4).round(4).to_string())

    # A tightening raises the nominal rate and lowers output and inflation.
    policy = model.irf("eps_policy", horizon=1)
    assert policy.loc[0, "i"] > 0, "a policy tightening should raise i"
    assert policy.loc[0, "x"] < 0 and policy.loc[0, "pi"] < 0

    print("\nSecond moments (sd of a 1pp demand shock, 2000 periods):")
    sim = model.simulate(periods=2000, sigma={"eps_demand": 0.01,
                                              "eps_policy": 0.0}, seed=0)
    print((sim.std() * 100).round(3).to_string())

    # The Taylor principle is not an assumption here — it is whether the
    # model has a unique stable solution at all.
    print("\nTaylor principle (phi_pi = 0.9):")
    try:
        solve(phi_pi=0.9)
    except BlanchardKahnError as exc:
        print(f"  BlanchardKahnError — {str(exc).split('.')[0]}")
    else:  # pragma: no cover - would mean the BK check stopped working
        print("  no error raised, which should not happen")


if __name__ == "__main__":
    main()
