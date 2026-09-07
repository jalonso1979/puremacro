"""Validation cases for the ``did`` subsystem (staggered DiD estimators).

Scope:
* ANALYTICAL: Callaway-Sant'Anna (2021) group-time ATT collapses analytically
  to the textbook manual 2x2 DiD in a balanced 2-group, 2-period design.
* INTERNAL: Sun-Abraham (2021) interaction-weighted estimator is algebraically
  identical to Callaway-Sant'Anna on the canonical 2-group, 2-period design.
* INTERNAL: Borusyak-Jaravel-Spiess (2024) imputation estimator recovers the
  exact 2x2 DiD treatment effect on the canonical design.
* INTERNAL: ``spatial_did`` with every untreated unit beyond the outermost ring
  reduces coefficient-for-coefficient (and SE-for-SE) to
  ``lp._panel_helpers.two_way_fe_within`` on the plain treatment dummy; and its
  ``naive_att`` / ``naive_se`` equal that same two-way-FE fit on a pooled
  treatment dummy even when the rings are populated.
* ANALYTICAL: on a seeded planar DGP with a planted distance-decaying spillover,
  the ring coefficients recover the planted profile, and the ring-adjusted
  direct effect is closer to the truth than the contaminated naive DiD
  (QUALITATIVE, on the improvement metric).
* ANALYTICAL: on the same geography with the spillover switched off, the naive
  DiD is unbiased and every ring coefficient is insignificant (QUALITATIVE).

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


# ---------------------------------------------------------------------------
# Spatial DiD (puremacro.did.spatial_did): exposure-ring designs
# ---------------------------------------------------------------------------
#: Module-level DGP seed for every spatial-DiD case below.
_SDID_SEED = 20260906
#: Panel shape shared by the spatial-DiD DGPs: T periods, common adoption at T0.
_SDID_T, _SDID_T0, _SDID_N_TREATED = 8, 4, 10
#: Planted direct effect on a treated unit.
_SDID_DIRECT = 1.0
#: Untreated-unit offsets (in coordinate units) from their nearest treated unit,
#: three per band of ``(0, 25, 50, 100)`` plus three beyond-ring controls, and
#: the spillover planted at each offset.
_SDID_OFFSETS = np.array([8.0, 15.0, 22.0, 30.0, 40.0, 48.0,
                          60.0, 78.0, 95.0, 150.0, 250.0, 350.0])
_SDID_SPILLOVERS = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25,
                             0.10, 0.10, 0.10, 0.0, 0.0, 0.0])
_SDID_RINGS = (0.0, 25.0, 50.0, 100.0)


def _sdid_panel(x: np.ndarray, treated: np.ndarray, effect: np.ndarray,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A balanced two-way-FE panel on a line, with a planted post-treatment step.

    ``y_it = mu_i + tau_t + effect_i * 1{t >= T0} + eps_it``. Coordinates are
    one-dimensional (``y`` identically zero) so the euclidean nearest-treated
    distance is exactly ``|x_i - x_j|`` and the ring assignment is by
    construction, not by luck.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    ids = [f"u{i:03d}" for i in range(n)]
    coords = pd.DataFrame({"x": np.asarray(x, float), "y": np.zeros(n)}, index=ids)
    tt = pd.Series(np.where(treated, float(_SDID_T0), np.nan), index=ids)
    mu = rng.normal(0.0, 1.0, n)
    tau = rng.normal(0.0, 0.3, _SDID_T)
    rows = []
    for i, u in enumerate(ids):
        for t in range(_SDID_T):
            rows.append((u, t,
                         mu[i] + tau[t] + effect[i] * (t >= _SDID_T0)
                         + 0.05 * rng.normal(),
                         tt[u]))
    panel = pd.DataFrame(rows, columns=["unit", "time", "y", "treat_time"])
    return panel, coords


def _sdid_isolated_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """No spillover structure at all: every untreated unit is beyond ``rings[-1]``.

    Ten treated units 600 apart; four untreated units per treated unit at
    offsets 150/250/350/450, so the nearest-treated distance of every untreated
    unit is at least 150 > ``rings[-1] = 50``. Rings 1 and 2 are then empty,
    ``spatial_did`` drops them, and the design matrix is the single treatment
    dummy.
    """
    tx = np.arange(_SDID_N_TREATED) * 600.0
    ux = np.concatenate([tx[k] + np.array([150.0, 250.0, 350.0, 450.0])
                         for k in range(_SDID_N_TREATED)])
    x = np.concatenate([tx, ux])
    treated = np.zeros(len(x), dtype=bool)
    treated[:_SDID_N_TREATED] = True
    return _sdid_panel(x, treated, np.where(treated, _SDID_DIRECT, 0.0), _SDID_SEED)


def _sdid_ring_data(spillover: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ten treated units 1600 apart, each with 24 untreated neighbours.

    The neighbours sit at +/- ``_SDID_OFFSETS``, which populates every band of
    ``_SDID_RINGS`` with 60 units and leaves 60 beyond-ring controls. The
    treated units are 1600 apart and ``rings[-1] = 100``, so no treated unit is
    inside another's outermost ring: ring 0 is then the pure direct ATT rather
    than a treated-on-treated total. With ``spillover=False`` the same geography
    carries no spillover at all.
    """
    tx = np.arange(_SDID_N_TREATED) * 1600.0
    ux, ue = [], []
    for k in range(_SDID_N_TREATED):
        for sign in (-1.0, 1.0):
            ux.extend(tx[k] + sign * _SDID_OFFSETS)
            ue.extend(_SDID_SPILLOVERS if spillover
                      else np.zeros_like(_SDID_SPILLOVERS))
    x = np.concatenate([tx, np.array(ux)])
    treated = np.zeros(len(x), dtype=bool)
    treated[:_SDID_N_TREATED] = True
    effect = np.concatenate([np.full(_SDID_N_TREATED, _SDID_DIRECT), np.array(ue)])
    return _sdid_panel(x, treated, effect, _SDID_SEED + 1)


def _sdid_fit(panel: pd.DataFrame, coords: pd.DataFrame, rings):
    """``spatial_did`` on a seeded DGP, cluster-robust and without the
    finite-sample multiplier so the sandwich is comparable term for term with
    ``two_way_fe_within``."""
    import warnings

    from puremacro.did import spatial_did

    with warnings.catch_warnings():
        # An empty ring is dropped with a RuntimeWarning; that is the point of
        # the reduction case, not a problem to surface in the gallery.
        warnings.simplefilter("ignore", RuntimeWarning)
        return spatial_did(
            panel, coords=coords, metric="euclidean", rings=rings,
            cov_type="cluster", event_study=False, small_sample=False,
        )


def _sdid_pooled_twfe(panel: pd.DataFrame) -> dict:
    """Two-way-FE DiD on the pooled treatment dummy ``1{t >= g_i}``.

    This is the *short* regression behind the Frisch-Waugh decomposition, fitted
    by a different estimator (``lp._panel_helpers.two_way_fe_within``) on the
    same panel, with the same cluster-by-unit sandwich.
    """
    from puremacro.lp._panel_helpers import two_way_fe_within

    wide = panel.copy()
    wide["D"] = (wide["time"] >= wide["treat_time"].fillna(np.inf)).astype(float)
    wide = wide.set_index(["unit", "time"])
    return two_way_fe_within(wide, y_col="y", x_cols=["D"],
                             entity_level="unit", time_level="time")


def _compute_sdid_isolated() -> dict:
    res = _sdid_fit(*_sdid_isolated_data(), rings=(0.0, 25.0, 50.0))
    return {"n_coef": float(len(res.coef)),
            "direct_effect": float(res.direct_effect),
            "direct_se": float(res.ring_table["se"].iloc[0])}


def _reference_sdid_isolated() -> dict:
    fit = _sdid_pooled_twfe(_sdid_isolated_data()[0])
    return {"n_coef": 1.0,
            "direct_effect": float(fit["beta"][0]),
            "direct_se": float(fit["se"][0])}


def _compute_sdid_naive_leg() -> dict:
    res = _sdid_fit(*_sdid_ring_data(True), rings=_SDID_RINGS)
    return {"naive_att": float(res.naive_att), "naive_se": float(res.naive_se)}


def _reference_sdid_naive_leg() -> dict:
    fit = _sdid_pooled_twfe(_sdid_ring_data(True)[0])
    return {"naive_att": float(fit["beta"][0]), "naive_se": float(fit["se"][0])}


def _compute_sdid_ring_profile() -> dict:
    res = _sdid_fit(*_sdid_ring_data(True), rings=_SDID_RINGS)
    return {"coef": np.asarray(res.coef, dtype=float)}


def _compute_sdid_improvement() -> dict:
    res = _sdid_fit(*_sdid_ring_data(True), rings=_SDID_RINGS)
    naive_err = abs(float(res.naive_att) - _SDID_DIRECT)
    direct_err = abs(float(res.direct_effect) - _SDID_DIRECT)
    return {"improvement": naive_err - direct_err,
            "naive_downward_bias": _SDID_DIRECT - float(res.naive_att)}


def _compute_sdid_no_spillover() -> dict:
    res = _sdid_fit(*_sdid_ring_data(False), rings=_SDID_RINGS)
    spill = res.ring_table[(res.ring_table["ring"] >= 1)
                           & (res.ring_table["ring"] <= len(_SDID_RINGS) - 1)]
    return {
        # 0.05 minus the realised bias: >= 0 means "unbiased to within 0.05".
        "naive_accuracy": 0.05 - abs(float(res.naive_att) - _SDID_DIRECT),
        "direct_accuracy": 0.05 - abs(float(res.direct_effect) - _SDID_DIRECT),
        "min_ring_pvalue": float(spill["p"].min()),
        "ring_magnitude_slack": 0.05 - float(spill["effect"].abs().max()),
    }


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
    ValidationCase(
        id="did.spatial_did_beyond_outer_ring_equals_two_way_fe",
        subsystem="did",
        title=(
            "Spatial DiD with every unit beyond the outermost ring reduces exactly "
            "to the two-way-FE DiD"
        ),
        title_es=(
            "El DiD espacial con todas las unidades más allá del anillo exterior se "
            "reduce exactamente al DiD de efectos fijos bidireccionales"
        ),
        mechanism=Mechanism.INTERNAL,
        compute=_compute_sdid_isolated,
        reference=_reference_sdid_isolated,
        tol=Tol.TIGHT,
        citation=(
            "Butts, K. (2023), 'Difference-in-differences estimation with spatial "
            "spillovers', arXiv:2105.03737, eq. (5): with no untreated unit inside any "
            "spillover band the ring indicators vanish and the estimating equation "
            "collapses to y_it = mu_i + tau_t + delta_0 D_it, the canonical two-way-FE "
            "DiD of Ashenfelter & Card (1985)."
        ),
        notes=(
            "Ten treated units 600 apart, four untreated per treated at offsets "
            "150-450, rings=(0, 25, 50): rings 1 and 2 are empty and dropped. "
            "Reference is puremacro.lp._panel_helpers.two_way_fe_within on the plain "
            "treatment dummy — a different estimator, same panel. Compares the number "
            "of surviving coefficients, delta_0 and its cluster-by-unit standard error "
            "(small_sample=False makes the two sandwiches term-for-term identical)."
        ),
    ),
    ValidationCase(
        id="did.spatial_did_naive_att_equals_pooled_two_way_fe",
        subsystem="did",
        title=(
            "The naive (no-rings) leg of the spatial DiD equals a two-way-FE fit on the "
            "pooled treatment dummy"
        ),
        title_es=(
            "La rama ingenua (sin anillos) del DiD espacial coincide con una regresión "
            "de efectos fijos bidireccionales sobre la variable de tratamiento agrupada"
        ),
        mechanism=Mechanism.INTERNAL,
        compute=_compute_sdid_naive_leg,
        reference=_reference_sdid_naive_leg,
        tol=Tol.TIGHT,
        citation=(
            "Frisch, R. & Waugh, F.V. (1933), Econometrica 1(4), 387-401: the naive DiD "
            "that pools every untreated unit into the control group is the short "
            "regression of the ring specification, delta_naive = delta_0 + sum_r "
            "theta_r delta_r. Its point estimate and cluster-robust standard error must "
            "reproduce an independent two-way-FE fit on the same sample."
        ),
        notes=(
            "Fitted on the populated-ring DGP (60 units in each of the three bands), so "
            "naive_att = 0.761 is genuinely far from delta_0 = 0.974 and the check has "
            "content. Reference is two_way_fe_within on D_it = 1{t >= g_i}."
        ),
    ),
    ValidationCase(
        id="did.spatial_did_recovers_planted_ring_profile",
        subsystem="did",
        title="Spatial DiD recovers a planted distance-decaying spillover profile",
        title_es=(
            "El DiD espacial recupera un perfil de desbordamiento sembrado que decae "
            "con la distancia"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_compute_sdid_ring_profile,
        reference=lambda: {"coef": np.array([_SDID_DIRECT, 0.5, 0.25, 0.10])},
        tol=Tol.COARSE,
        citation=(
            "Clarke, D. (2017), 'Estimating difference-in-differences in the presence of "
            "spillovers', MPRA Paper 81604, and Berg, T., Reisinger, M. & Streitz, D. "
            "(2021), Journal of Financial Economics 142(3), 1109-1127: giving each "
            "distance band its own coefficient identifies the spillover profile against "
            "the beyond-ring comparison group."
        ),
        notes=(
            "Planted (delta_0, delta_1, delta_2, delta_3) = (1.0, 0.5, 0.25, 0.10); "
            "every reference value is at least 0.10, so Tol.COARSE's atol = 0.0 still "
            "leaves a workable relative budget. Treated units are 1600 apart against "
            "rings[-1] = 100, so no treated unit sits in another's outermost ring and "
            "delta_0 is the pure direct ATT rather than a treated-on-treated total."
        ),
    ),
    ValidationCase(
        id="did.spatial_did_ring_adjustment_beats_naive_under_spillover",
        subsystem="did",
        title=(
            "Under a positive spillover the ring-adjusted direct effect is closer to the "
            "truth than the naive DiD"
        ),
        title_es=(
            "Con desbordamiento positivo, el efecto directo ajustado por anillos está "
            "más cerca de la verdad que el DiD ingenuo"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_compute_sdid_improvement,
        reference=lambda: {"improvement": 0.10, "naive_downward_bias": 0.10},
        tol=Tol.QUALITATIVE,
        citation=(
            "Butts, K. (2023), arXiv:2105.03737, and Clarke, D. (2017), MPRA Paper "
            "81604: pooling spillover-exposed units into the control group biases the "
            "naive DiD by sum_r theta_r delta_r with theta_r < 0, so a positive "
            "spillover biases it downward and the ring-adjusted delta_0 is the estimand "
            "the naive number was aiming at."
        ),
        notes=(
            "A lower bound on an improvement metric, deliberately not a COARSE point "
            "comparison: |naive - truth| - |direct - truth| >= 0.10 (realised 0.213) and "
            "truth - naive >= 0.10 (realised 0.239), on the seeded DGP with truth = 1.0. "
            "A COARSE comparison against the ring coefficients would be the wrong "
            "instrument here — atol = 0.0 gives a near-zero reference a punitive budget."
        ),
    ),
    ValidationCase(
        id="did.spatial_did_no_spillover_leaves_naive_unbiased",
        subsystem="did",
        title=(
            "With no spillover the naive DiD is unbiased and every ring coefficient is "
            "insignificant"
        ),
        title_es=(
            "Sin desbordamiento, el DiD ingenuo es insesgado y todos los coeficientes "
            "de anillo son no significativos"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_compute_sdid_no_spillover,
        reference=lambda: {"naive_accuracy": 0.0, "direct_accuracy": 0.0,
                           "min_ring_pvalue": 0.05, "ring_magnitude_slack": 0.0},
        tol=Tol.QUALITATIVE,
        citation=(
            "Butts, K. (2023), arXiv:2105.03737: the ring design nests the canonical "
            "DiD. When delta_1 = ... = delta_R = 0 the contamination term sum_r theta_r "
            "delta_r vanishes, the naive estimate is unbiased for the ATT, and the ring "
            "coefficients are sampling noise around zero."
        ),
        notes=(
            "Same geography as the spillover cases with the spillover switched off. "
            "'naive_accuracy' / 'direct_accuracy' are 0.05 minus the realised bias "
            "(both realised 0.024), 'ring_magnitude_slack' is 0.05 minus the largest "
            "|delta_r| (realised 0.044), and the smallest ring p-value is 0.32 against "
            "a 0.05 floor — all four are lower bounds, which is what QUALITATIVE means."
        ),
    ),
]
