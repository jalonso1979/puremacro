"""Scale IRF results to standard pedagogical units.

The estimators in ``var_sm``, ``lp_sm`` and ``panel_lm`` return raw
coefficients in the natural units of the inputs:

- A Cholesky orthogonalised IRF is the response of ``y`` (in y's units)
  to a one-standard-deviation structural shock, by construction
  (Cholesky factor ``P`` rescales the unit-variance structural innovation
  to its sample-implied standard deviation).
- A Jordà LP coefficient β_h is the response of ``y`` to a one-unit
  change in ``x`` (in x's natural units).

For teaching, the conventional formulation reports IRFs as

    "% (or pp) response of y to a 1-SD shock to x, with 90 % bands."

This module converts the raw result-dicts to that formulation:

- ``y`` in logs ⇒ multiply by 100 to express the response in percent.
- ``y`` is a rate (e.g. ``urate``, ``d_log_cpi``) ⇒ leave untouched
  (the response is already in percentage points).
- For LP / panel-LP, multiply by ``sd(x)`` so the shock is 1-SD rather
  than 1 unit. For VAR-Cholesky, leave the coefficient alone (the
  Cholesky already encodes the 1-SD shock).
"""
from __future__ import annotations

import pandas as pd

# Outcome variables expressed in percentage points (no ×100).
RATE_OUTCOMES: frozenset[str] = frozenset({
    "urate", "urate_ifs_m", "d_log_cpi", "short_rate",
    "bci_m", "cci_m",
})


def is_rate_outcome(name: str) -> bool:
    """True when *name* is already in percentage-point units (no ×100)."""
    return name in RATE_OUTCOMES


def to_1sd_pct(
    result: dict,
    *,
    x_sd: float | None = None,
    outcome_is_log: bool | None = None,
    outcome_name: str | None = None,
) -> dict:
    """Rescale *result* to '% (or pp) response of y per 1-SD shock to x'.

    Parameters
    ----------
    result : dict
        Output of :func:`src.teaching.experiment.run_experiment`. Keys
        ``irf_point``, ``irf_lower``, ``irf_upper`` are scaled.
    x_sd : float, optional
        Sample standard deviation of the shock variable. Pass for LP and
        panel-LP. Pass ``None`` for ``var_cholesky`` — the Cholesky factor
        already produces a 1-SD structural shock IRF.
    outcome_is_log : bool, optional
        If ``True``, multiply by 100 to convert log-units to percent.
        If ``False``, leave the response in its native units (pp).
        If ``None``, infer from ``outcome_name`` via :func:`is_rate_outcome`.
    outcome_name : str, optional
        Used to infer ``outcome_is_log`` when not supplied; treated as
        log unless it appears in :data:`RATE_OUTCOMES`.

    Returns
    -------
    dict
        New result dict with scaled IRFs and two extra keys: ``shock_sd``
        (the SD applied) and ``y_units`` (``'%'`` or ``'pp'``).
    """
    if outcome_is_log is None:
        if outcome_name is None:
            raise ValueError("supply either outcome_is_log or outcome_name")
        outcome_is_log = not is_rate_outcome(outcome_name)

    pct_factor = 100.0 if outcome_is_log else 1.0
    sd_factor = 1.0 if x_sd is None else float(x_sd)
    factor = pct_factor * sd_factor

    out = dict(result)
    out["irf_point"] = result["irf_point"] * factor
    out["irf_lower"] = result["irf_lower"] * factor
    out["irf_upper"] = result["irf_upper"] * factor
    out["shock_sd"] = float(x_sd) if x_sd is not None else None
    out["y_units"] = "%" if outcome_is_log else "pp"
    return out


def peak_irf(result: dict, *, response_var: str | None = None) -> dict:
    """Return ``{h, beta, lo, hi}`` at the horizon of the maximum |β| response.

    Works on both single-Series IRFs (LP / panel-LP, where ``irf_point`` is
    a Series) and multi-column IRFs (VAR Cholesky, where ``irf_point`` is a
    DataFrame and ``response_var`` selects one column).
    """
    pt = result["irf_point"]
    if isinstance(pt, pd.DataFrame):
        if response_var is None:
            raise ValueError("response_var required for multi-column IRF")
        pt = pt[response_var]
        lo = result["irf_lower"][response_var]
        hi = result["irf_upper"][response_var]
    else:
        lo = result["irf_lower"]
        hi = result["irf_upper"]
    h_star = pt.abs().idxmax()
    return {"h": h_star, "beta": float(pt.loc[h_star]),
            "lo": float(lo.loc[h_star]), "hi": float(hi.loc[h_star])}


__all__ = ["to_1sd_pct", "peak_irf", "is_rate_outcome", "RATE_OUTCOMES"]
