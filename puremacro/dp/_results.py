"""Result object for puremacro.dp: one solution type whatever the back end."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


class DPSolution:
    """Solved dp model: value, policies, distribution, prices and aggregates.

    ``raw`` is the underlying ``VFISolution``, ``FiniteHorizonSolution`` or
    ``EGMHouseholdSolution`` (``method="egm"``; continuous ``c``/``aprime``) and
    ``equilibrium`` the ``EquilibriumResult`` when prices were solved for.
    Arrays are (n_states, n_shocks), with a leading age axis for finite horizons.
    """

    def __init__(self, compiled, raw, *, params: dict, distribution=None,
                 aggregates: dict | None = None, prices: dict | None = None,
                 equilibrium=None, method: str | None = None):
        self._c = compiled
        self.raw = raw
        self.model_name = compiled.model.name
        self.method = method or ("vfi-finite" if compiled.finite else "vfi")
        self.params = dict(params)
        self.distribution = distribution
        self.aggregates = dict(aggregates or {})
        self.prices = dict(prices or {})
        self.equilibrium = equilibrium

    @property
    def V(self) -> np.ndarray:
        return self.raw.V

    @property
    def grids(self) -> dict[str, np.ndarray]:
        """State and shock grids by name."""
        out = dict(zip(self._c.state_names, self._c.state_grids))
        out.update(zip(self._c.shock_names, self._c.z_grids))
        return out

    @property
    def P(self) -> np.ndarray:
        """Transition matrix of the (product) exogenous chain."""
        return self._c.P

    def policy(self, expr: str) -> np.ndarray:
        """Any model expression evaluated at the optimal choice, e.g. ``policy("c")``
        or ``policy("a(+1)")``; shape (n_states, n_shocks) or (T, n_states, n_shocks)."""
        from puremacro.dp._expr import parse

        c = self._c
        timed = c.state_names + c.shock_names + ([c.model._discrete[0]] if c.model._discrete else [])
        fn = c._household_function("_dp_policy", parse(expr, timed))
        if not c.finite:
            return c.evaluate(fn, self.raw, self.params)
        return np.stack([c.evaluate(fn, self.raw, self.params, age=j)
                         for j in range(self.raw.horizon)])

    def mean(self, expr: str):
        """Population mean of ``expr`` under the solved distribution: a float, or
        one value per age for a finite horizon (each cohort has mass 1)."""
        if self.distribution is None:
            raise ValueError("no distribution: solve with distribution=True")
        vals = self.policy(expr)
        if not self._c.finite:
            return float(np.sum(self.distribution * vals))
        return np.array([float(np.sum(self.distribution[j] * vals[j])) for j in range(vals.shape[0])])

    def summary(self) -> pd.DataFrame:
        rec = [("Model", self.model_name), ("Method", self.method),
               ("States", ", ".join(f"{k} ({g.size})" for k, g in zip(self._c.state_names, self._c.state_grids))),
               ("Shocks", ", ".join(f"{k} ({g.size})" for k, g in zip(self._c.shock_names, self._c.z_grids)) or "none")]
        if hasattr(self.raw, "n_iter"):
            rec.append(("Iterations", str(self.raw.n_iter)))
        for k, v in self.prices.items():
            rec.append((f"Price {k}", f"{v:.6g}"))
        for k, v in self.aggregates.items():
            rec.append((f"Aggregate {k}", f"{v:.6g}"))
        return pd.DataFrame(rec, columns=["Metric", "Value"]).set_index("Metric")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.summary(), **kwargs)

    def __repr__(self) -> str:
        extra = "".join(f", {k}={v:.6g}" for k, v in {**self.prices, **self.aggregates}.items())
        return f"DPSolution({self.model_name!r}, method={self.method!r}{extra})"

