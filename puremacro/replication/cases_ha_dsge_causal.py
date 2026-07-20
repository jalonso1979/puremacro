"""Replication cases — heterogeneous-agent / DSGE / causal family.

Worked example: Aiyagari (1994, QJE 109(3):659-684). The incomplete-markets
equilibrium interest rate is (i) below the representative-agent rate 1/beta-1
because precautionary saving raises aggregate capital, and (ii) decreasing in
the variance and persistence of idiosyncratic income risk. We reproduce those
headline results by solving the model at several calibrations. No external data.
"""
from __future__ import annotations

from typing import Any

from ._model import ReplicationCase, TargetKind, Tol

# Calibration shared across the cases; r_bracket=(0, 1/beta-1) brackets every
# equilibrium used here (the default bracket does not for low-rho).
_BETA = 0.96
_RA_RATE = 1.0 / _BETA - 1.0  # representative-agent rate ≈ 0.04167
_BRACKET = (0.0, 0.0416)
_BASE = dict(beta=_BETA, gamma=1.0, rho=0.9, sigma=0.2, n_z=5, n_a=120, a_max=80.0)


def _r(**overrides) -> float:
    from puremacro.vfi import aiyagari_steady_state

    kw: dict[str, Any] = dict(_BASE, r_bracket=_BRACKET)
    kw.update(overrides)
    return float(aiyagari_steady_state(**kw)["r"])


def _precautionary_wedge() -> dict:
    # 1/beta-1 - r > 0  (incomplete-markets r below the rep-agent rate)
    return {"wedge": _RA_RATE - _r()}


def _r_decreasing_in_sigma() -> dict:
    # r(low risk) - r(high risk) > 0
    return {"d_sigma": _r(sigma=0.2) - _r(sigma=0.4)}


def _r_decreasing_in_rho() -> dict:
    # r(low persistence) - r(high persistence) > 0
    return {"d_rho": _r(rho=0.3) - _r(rho=0.9)}


def _huggett_precautionary_wedge() -> dict:
    # Huggett (1993): in a pure-exchange bond economy the equilibrium risk-free
    # rate sits well below 1/beta-1 (precautionary bond demand). wedge > 0.
    from puremacro.vfi import huggett_steady_state

    r = float(
        huggett_steady_state(
            beta=_BETA, gamma=1.5, rho=0.9, sigma=0.2, n_z=7, n_a=150, r_bracket=_BRACKET
        )["r"]
    )
    return {"wedge": _RA_RATE - r}


# Smets-Wouters (2007) published specification: 7 structural shocks and 7
# observables (the choice that avoids stochastic singularity).
_SW07_SHOCKS = {"ea", "eb", "eg", "eqs", "em", "epinf", "ew"}


def _sw07_spec() -> dict:
    from puremacro.dsge.smets_wouters import SHOCK_NAMES
    from puremacro.dsge.sw07_estimate import OBSERVED_VARS

    names = set(SHOCK_NAMES)
    return {
        "n_shocks": float(len(SHOCK_NAMES)),
        "n_matching_published_shocks": float(len(names & _SW07_SHOCKS)),
        "n_observables": float(len(OBSERVED_VARS)),
    }


_PAPER = "Aiyagari (1994), 'Uninsured Idiosyncratic Risk and Aggregate Saving', QJE 109(3):659-684"

CASES: list[ReplicationCase] = [
    ReplicationCase(
        id="ha_dsge_causal.aiyagari_precautionary_wedge",
        family="ha_dsge_causal",
        paper=_PAPER,
        title="Aiyagari: equilibrium r below the representative-agent rate",
        title_es="Aiyagari: la r de equilibrio queda por debajo de la tasa del agente representativo",
        source="model",
        estimate=_precautionary_wedge,
        target={"wedge": +1},
        target_kind=TargetKind.SIGN,
        tol=Tol.COARSE,
        citation="Aiyagari (1994), §III: precautionary saving makes the equilibrium r < 1/beta-1.",
    ),
    ReplicationCase(
        id="ha_dsge_causal.aiyagari_r_decreasing_in_sigma",
        family="ha_dsge_causal",
        paper=_PAPER,
        title="Aiyagari: equilibrium r decreasing in income-risk variance",
        title_es="Aiyagari: la r de equilibrio decrece con la varianza del riesgo de ingreso",
        source="model",
        estimate=_r_decreasing_in_sigma,
        target={"d_sigma": +1},
        target_kind=TargetKind.SIGN,
        tol=Tol.COARSE,
        citation="Aiyagari (1994), Table II: r falls as the s.d. of the endowment shock rises.",
    ),
    ReplicationCase(
        id="ha_dsge_causal.aiyagari_r_decreasing_in_rho",
        family="ha_dsge_causal",
        paper=_PAPER,
        title="Aiyagari: equilibrium r decreasing in income-risk persistence",
        title_es="Aiyagari: la r de equilibrio decrece con la persistencia del riesgo de ingreso",
        source="model",
        estimate=_r_decreasing_in_rho,
        target={"d_rho": +1},
        target_kind=TargetKind.SIGN,
        tol=Tol.COARSE,
        citation="Aiyagari (1994), Table II: r falls as the persistence rho of the endowment shock rises.",
    ),
    ReplicationCase(
        id="ha_dsge_causal.huggett_precautionary_riskfree_rate",
        family="ha_dsge_causal",
        paper="Huggett (1993), 'The risk-free rate in heterogeneous-agent incomplete-insurance economies', JEDC 17(5-6):953-969",
        title="Huggett: equilibrium risk-free rate well below the rep-agent rate",
        title_es="Huggett: la tasa libre de riesgo de equilibrio muy por debajo de la tasa del agente representativo",
        source="model",
        estimate=_huggett_precautionary_wedge,
        target={"wedge": +1},
        target_kind=TargetKind.SIGN,
        tol=Tol.COARSE,
        citation="Huggett (1993), §3: precautionary demand for the bond drives the equilibrium risk-free rate below 1/beta-1.",
    ),
    ReplicationCase(
        id="ha_dsge_causal.sw07_seven_shocks_seven_observables",
        family="ha_dsge_causal",
        paper="Smets & Wouters (2007), 'Shocks and Frictions in US Business Cycles', AER 97(3):586-606",
        title="Smets-Wouters: the published 7-shock, 7-observable specification",
        title_es="Smets-Wouters: la especificacion publicada de 7 choques y 7 observables",
        source="model",
        estimate=_sw07_spec,
        target={"n_shocks": 7, "n_matching_published_shocks": 7, "n_observables": 7},
        target_kind=TargetKind.POINT,
        tol=Tol.TIGHT,
        citation="Smets & Wouters (2007), Table 1A/§II: 7 structural shocks (ea,eb,eg,eqs,em,epinf,ew) matched to 7 observables.",
    ),
]
