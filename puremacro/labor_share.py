"""Gollin (2002) self-employed-adjusted labor share.

Gollin (2002) "Getting Income Shares Right" (JPE) documents that raw
compensation-of-employees / value-added (COE / VA) understates the labor
share because self-employed workers receive mixed income (B3G) that
conflates capital and labor returns.

This module implements **Adjustment 2** from Gollin (2002): impute the
self-employed as if each earned the same wage as the average employee.

Formula
-------
    alpha              = D11 / N_emp           (avg employee compensation)
    imputed_se_labor   = alpha * N_self
    ls_gollin          = (D11 + imputed_se_labor) / B1G

where:
    D11   = compensation_employees  (SNA national-accounts code)
    B1G   = value_added             (gross value added at basic prices)
    N_emp = employment_employees    (headcount of employees)
    N_self= employment_self         (headcount of self-employed / own-account)

The column ``mixed_income`` (B3G) is accepted in the input schema for
traceability and downstream use, but is **not used in this formula**.

Edge cases
----------
* If ``employment_employees == 0`` the per-employee wage is undefined;
  ``coe_per_employee`` is set to NaN so the result propagates NaN rather
  than raising ZeroDivisionError or producing inf.
* The result is clipped at 1.0.  Values > 1 can arise from measurement
  error; the clip prevents nonsensical shares from propagating.
"""
from __future__ import annotations

import pandas as pd


def gollin_adjusted_ls(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Gollin (2002) Adjustment-2 labor share.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ``compensation_employees``, ``value_added``, ``mixed_income``,
        ``employment_employees``, ``employment_self``.
        Additional columns (e.g. ``code``, ``qdate``) are passed through
        unchanged.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with one new column appended:

        ``ls_gollin`` — Gollin-adjusted labor share, clipped to [0, 1].
    """
    out = df.copy()

    # Replace zero employee counts with NaN to avoid division by zero.
    n_emp = out["employment_employees"].where(out["employment_employees"] != 0)

    coe_per_employee = out["compensation_employees"] / n_emp
    imputed_self_emp_labor = coe_per_employee * out["employment_self"]

    ls = (out["compensation_employees"] + imputed_self_emp_labor) / out["value_added"]
    out["ls_gollin"] = ls.clip(upper=1.0)

    return out


__all__ = ["gollin_adjusted_ls"]
