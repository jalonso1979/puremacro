"""Render LP and IRF results as Markdown / LaTeX tables ready to drop
into a paper or a Quarto / Jupyter notebook on iPad.

Workflow:
  1. Run a single-country Jordà LP on synthetic data.
  2. Wrap the result in ``LPResult`` / ``IRFResult`` dataclasses.
  3. Print a coefficient table + IRF table in Markdown.
  4. Print the same coefficient table in LaTeX.

Run:
    python -m puremacro.examples.report_workflow
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..lp.jorda import lp_hac
from ..reports import (
    coef_table, irf_to_markdown, IRFResult, LPResult,
    summary_to_markdown,
)


def _simulate(T: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(T); y = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal() * 0.5
        y[t] = 0.4 * y[t - 1] + 0.5 * x[t] + rng.standard_normal() * 0.4
    idx = pd.date_range("1980", periods=T, freq="Q")
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def main() -> None:
    df = _simulate()
    irf = lp_hac(df, y="y", x="x", horizons=range(0, 9), n_lags=2)
    res = LPResult(df=irf, method="lp_hac", n_obs=len(df))

    print("# LP regression — single country, Newey-West HAC")
    print()
    print(coef_table(
        res.df["beta"].values,
        res.df["se"].values,
        names=[f"β at h={int(h)}" for h in res.df["h"].values],
        digits=3, include_p=True,
    ))
    print()
    print("## Same result, IRF-form table:")
    print()
    point = res.df["beta"].values[:, None]   # (H+1, 1)
    lo = res.df["lo"].values[:, None]
    hi = res.df["hi"].values[:, None]
    print(irf_to_markdown(point, lo, hi, var_names=["y"]))
    print()

    print("## LaTeX version (paste into your manuscript):")
    print()
    print(coef_table(res.df["beta"].values, res.df["se"].values,
                      names=[f"h={int(h)}" for h in res.df["h"].values],
                      fmt="latex", digits=3, include_t=False,
                      include_p=False, include_ci=True))
    print()

    # Bundle multiple methods into a single comparison
    summary = {
        "method": res.method,
        "n_obs": res.n_obs,
        "alpha": res.alpha,
        "max_h": int(res.df["h"].max()),
        "peak_beta": float(res.df["beta"].iloc[
            int(np.argmax(res.df["beta"].abs().values))
        ]),
    }
    print(summary_to_markdown(summary, title="LP run summary"))


if __name__ == "__main__":
    main()
