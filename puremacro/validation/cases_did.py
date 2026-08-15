"""Validation cases for the ``did`` subsystem (staggered DiD estimators).

Scope:
* ANALYTICAL: Callaway-Sant'Anna (2021) group-time ATT collapses analytically
  to the textbook manual 2x2 DiD in a balanced 2-group, 2-period design.
* INTERNAL: Sun-Abraham (2021) interaction-weighted estimator is algebraically
  identical to Callaway-Sant'Anna on the canonical 2-group, 2-period design.
* INTERNAL: Borusyak-Jaravel-Spiess (2024) imputation estimator recovers the
  exact 2x2 DiD treatment effect on the canonical design.

All imports of ``puremacro.did`` live inside compute callables to keep module
imports pyodide-light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._model import Mechanism, Tol, ValidationCase


def _did_canonical_2x2_data() -> pd.DataFrame:
    """Generate a balanced 2-group, 2-period panel.

    50 treated units (cohort g=1) and 50 never-treated units (cohort g=nan)
    over periods t in {0, 1}.
    """
    N = 100
    unit = np.repeat(np.arange(N), 2)
    time = np.tile([0, 1], N)
    treat_time = np.repeat([1.0] * 50 + [np.nan] * 50, 2)

    rng = np.random.default_rng(20260815)
    eps = rng.normal(0, 0.1, len(unit))
    # True treatment effect tau = 4.5
    y = 10.0 + 0.5 * unit + 2.0 * time + 4.5 * ((treat_time == 1.0) & (time == 1)) + eps
    return pd.DataFrame({"unit": unit, "time": time, "y": y, "treat_time": treat_time})


def _manual_2x2_did(df: pd.DataFrame) -> float:
    """Manual 2x2 sample difference-in-differences."""
    y_t1 = df[(df["treat_time"] == 1.0) & (df["time"] == 1)]["y"].mean()
    y_t0 = df[(df["treat_time"] == 1.0) & (df["time"] == 0)]["y"].mean()
    y_c1 = df[df["treat_time"].isna() & (df["time"] == 1)]["y"].mean()
    y_c0 = df[df["treat_time"].isna() & (df["time"] == 0)]["y"].mean()
    return float((y_t1 - y_t0) - (y_c1 - y_c0))


def _compute_cs_2x2() -> dict:
    from puremacro.did.callaway_santanna import callaway_santanna

    df = _did_canonical_2x2_data()
    res = callaway_santanna(df, unit="unit", time="time", outcome="y", treat_time="treat_time", n_boot=20, seed=42)
    return {"att": float(res.att_overall)}


def _compute_sa_2x2() -> dict:
    from puremacro.did.sun_abraham import sun_abraham

    df = _did_canonical_2x2_data()
    res = sun_abraham(df, unit="unit", time="time", outcome="y", treat_time="treat_time", n_boot=20, seed=42)
    return {"att": float(res.att_overall)}


def _compute_bjs_2x2() -> dict:
    from puremacro.did.borusyak_jaravel_spiess import borusyak_jaravel_spiess

    df = _did_canonical_2x2_data()
    res = borusyak_jaravel_spiess(df, unit="unit", time="time", outcome="y", treat_time="treat_time", n_boot=20, seed=42)
    return {"att": float(res.att_overall)}


def _did_canonical_3p_data() -> pd.DataFrame:
    """Generate a balanced panel with 2 pre-treatment and 1 post-treatment period."""
    N = 100
    unit = np.repeat(np.arange(N), 3)
    time = np.tile([0, 1, 2], N)
    treat_time = np.repeat([2.0] * 50 + [np.nan] * 50, 3)

    rng = np.random.default_rng(20260815)
    eps = rng.normal(0, 0.01, len(unit))
    y = 10.0 + 0.5 * unit + 1.2 * time + 3.5 * ((treat_time == 2.0) & (time == 2)) + eps
    return pd.DataFrame({"unit": unit, "time": time, "y": y, "treat_time": treat_time})


def _manual_3p_did(df: pd.DataFrame) -> float:
    y_t_post = df[(df["treat_time"] == 2.0) & (df["time"] == 2)]["y"].mean()
    y_t_pre = df[(df["treat_time"] == 2.0) & (df["time"] < 2)]["y"].mean()
    y_c_post = df[df["treat_time"].isna() & (df["time"] == 2)]["y"].mean()
    y_c_pre = df[df["treat_time"].isna() & (df["time"] < 2)]["y"].mean()
    return float((y_t_post - y_t_pre) - (y_c_post - y_c_pre))


def _compute_sdid_3p() -> dict:
    from puremacro.did.synthetic_did import synthetic_did

    df = _did_canonical_3p_data()
    res = synthetic_did(df, unit="unit", time="time", outcome="y", treat_time="treat_time", n_boot=20, seed=42)
    return {"att": float(res.tau)}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="did.callaway_santanna_recovers_canonical_2x2",
        subsystem="did",
        title="Callaway-Sant'Anna ATT recovers manual 2x2 difference-in-differences exactly",
        title_es=(
            "El ATT de Callaway-Sant'Anna recupera exactamente la diferencia en "
            "diferencias 2x2 manual"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_compute_cs_2x2,
        reference=lambda: {"att": _manual_2x2_did(_did_canonical_2x2_data())},
        tol=Tol.EXACT,
        citation=(
            "Callaway & Sant'Anna (2021, J. Econometrics 225:200-230): with one "
            "treatment cohort and a baseline period, ATT(g,t) collapses to the "
            "canonical 2x2 sample difference-in-differences."
        ),
    ),
    ValidationCase(
        id="did.sun_abraham_equals_callaway_santanna_2x2",
        subsystem="did",
        title="Sun-Abraham interaction-weighted ATT matches Callaway-Sant'Anna on 2x2 design",
        title_es=(
            "El ATT ponderado por interacción de Sun-Abraham coincide con "
            "Callaway-Sant'Anna en un diseño 2x2"
        ),
        mechanism=Mechanism.INTERNAL,
        compute=_compute_sa_2x2,
        reference=_compute_cs_2x2,
        tol=Tol.EXACT,
        citation=(
            "Sun & Abraham (2021, J. Econometrics 225:175-199): in a single-cohort "
            "2-period design, the interaction-weighted estimator and Callaway-Sant'Anna "
            "are algebraically identical."
        ),
    ),
    ValidationCase(
        id="did.borusyak_jaravel_spiess_recovers_2x2",
        subsystem="did",
        title="Borusyak-Jaravel-Spiess imputation ATT matches manual 2x2 DiD",
        title_es=(
            "El ATT de imputación de Borusyak-Jaravel-Spiess coincide con el DiD 2x2 manual"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_compute_bjs_2x2,
        reference=lambda: {"att": _manual_2x2_did(_did_canonical_2x2_data())},
        tol=Tol.TIGHT,
        citation=(
            "Borusyak, Jaravel & Spiess (2024, Review of Economic Studies 91:3253-3285): "
            "imputation of untreated potential outcomes under parallel trends recovers "
            "the 2x2 treatment effect."
        ),
    ),
    ValidationCase(
        id="did.synthetic_did_recovers_treatment_effect",
        subsystem="did",
        title="Synthetic Difference-in-Differences recovers true treatment effect",
        title_es="La Diferencia en Diferencias Sintética recupera el efecto de tratamiento verdadero",
        mechanism=Mechanism.INTERNAL,
        compute=_compute_sdid_3p,
        reference=lambda: {"att": _manual_3p_did(_did_canonical_3p_data())},
        tol=Tol.NUMERIC,
        citation=(
            "Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021, AER 111(12):4088-4118): "
            "Synthetic DiD reweights donor units and pre-periods to match the treated trajectory, "
            "recovering the average treatment effect on the treated."
        ),
    ),
]
