"""Brock-Hommes affine-belief asset dynamics with lagged realized-profit selection.

Reference: Brock & Hommes (1998), JEDC 22, 1235-1274; explicit equations
(104)-(111) in Hommes, Heterogeneous Agent Models in Economics and Finance,
CeNDEF March 2005 manuscript, pp.48-50 (2006 Handbook chapter).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ._research import ResearchReport, finite, integer, readonly

__all__ = ["BeliefSimulationResult", "simulate_brock_hommes"]


@dataclass(frozen=True)
class BeliefSimulationResult(ResearchReport):
    names: tuple[str, ...]
    deviations: np.ndarray
    prices: np.ndarray
    fractions: np.ndarray
    forecasts: np.ndarray
    profits: np.ndarray
    fitness: np.ndarray
    shocks: np.ndarray
    specification_json: str

    def to_dataframe(self) -> pd.DataFrame:
        data = {"period": np.arange(len(self.deviations)), "deviation": self.deviations,
                "price": self.prices, "shock_rhs": self.shocks,
                "positive_price": self.prices > 0}
        for h,name in enumerate(self.names):
            data[f"fraction:{name}"] = self.fractions[:,h]
            data[f"forecast:{name}"] = self.forecasts[:,h]
            data[f"profit:{name}"] = self.profits[:,h]
            data[f"fitness:{name}"] = self.fitness[:,h]
        return pd.DataFrame(data)

    def plot(self, ax: Any = None, show: bool = False):
        from ._results import _new_ax, _finish
        fig, ax, given = _new_ax(ax)
        ax.plot(self.deviations)
        ax.axhline(0,color="0.5",lw=.7)
        ax.set(xlabel="period",ylabel="deviation from fundamental",title="Brock-Hommes belief dynamics")
        return _finish(fig, ax, given, show)


def simulate_brock_hommes(
    periods: int = 1000, *, slopes: Sequence[float] = (0.,1.2),
    biases: Sequence[float] = (0.,0.), costs: Sequence[float] | None = None,
    names: Sequence[str] | None = None, gross_rate: float = 1.01,
    intensity: float = 10., memory: float = 0., risk_scale: float = 1.,
    fundamental_value: float = 100., initial_deviations: Sequence[float] = (.1,.1,.1),
    initial_fitness: Sequence[float] | None = None, noise_std: float = 0.,
    shocks: Sequence[float] | None = None, seed: int = 0,
    max_abs_deviation: float = 1e6,
) -> BeliefSimulationResult:
    """Simulate the zero-outside-supply, constant-fundamental affine-rule model.

    f[h,t] = g[h]*x[t-1]+b[h]
    n[h,t] = softmax(intensity*U[h,t-1])
    R*x[t] = sum_h n[h,t]*f[h,t] + shock[t]
    profit[h,t] = (x[t]-R*x[t-1])*(g[h]*x[t-2]+b[h]-R*x[t-1])/risk_scale - cost[h]
    U[h,t] = profit[h,t] + memory*U[h,t-1]

    ``risk_scale`` is common a*sigma^2 (perceived payoff risk times aversion),
    not the simulated shock variance. ``shocks`` are RHS disturbances in eq.109,
    not dividend surprises or a path of changing fundamentals. Costs are an
    explicit net-profit extension; costs=0 and memory=0 recover eqs.109-111.
    A g=b=0 rule is a fundamentalist, not a perfect-foresight agent.

    Initial deviations are x[-3],x[-2],x[-1]. Default U[-1] is their realized
    profit, assuming U[-2]=0; provide initial_fitness to initialize accumulated
    memory differently. Saved fractions are used to clear the CURRENT price;
    saved fitness is updated AFTER that price. This timing is load-bearing.
    Negative implied prices are flagged in the frame and never clipped. Explosive
    deviations raise with the failing period; bounded output is not evidence of
    global stability, empirical fit, or a replication of a published chaotic path.
    """
    periods = integer(periods,"periods")
    seed = integer(seed,"seed",0)
    g,b = np.asarray(slopes,dtype=float),np.asarray(biases,dtype=float)
    if g.ndim != 1 or not len(g) or b.shape != g.shape or not np.isfinite(g).all() or not np.isfinite(b).all():
        raise ValueError("slopes and biases must be equal-sized finite vectors")
    h = len(g)
    c = np.zeros(h) if costs is None else np.asarray(costs,dtype=float)
    if c.shape != g.shape or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("costs must be a nonnegative finite vector matching slopes")
    labels = tuple(f"rule{i}" for i in range(h)) if names is None else tuple(names)
    if len(labels) != h or len(set(labels)) != h or not all(isinstance(s,str) and s for s in labels):
        raise ValueError("names must uniquely label every rule")
    R = finite(gross_rate,"gross_rate")
    beta = finite(intensity,"intensity",0)
    mem = finite(memory,"memory",0)
    risk = finite(risk_scale,"risk_scale",0)
    fundamental = finite(fundamental_value,"fundamental_value",0)
    std = finite(noise_std,"noise_std",0)
    limit = finite(max_abs_deviation,"max_abs_deviation",0)
    if R <= 1 or mem > 1 or risk == 0 or limit == 0:
        raise ValueError("require gross_rate>1, memory in [0,1], risk_scale>0, max_abs_deviation>0")
    initial = np.asarray(initial_deviations,dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all() or np.max(np.abs(initial)) > limit:
        raise ValueError("initial_deviations must contain three finite values within the divergence limit")
    if shocks is not None and std != 0:
        raise ValueError("supply shocks or noise_std, not both")
    innovations = np.random.default_rng(seed).normal(0,std,periods) if shocks is None else np.asarray(shocks,dtype=float)
    if innovations.shape != (periods,) or not np.isfinite(innovations).all():
        raise ValueError("shocks must have one finite value per period")
    with np.errstate(over="ignore",invalid="ignore"):
        u = ((initial[2]-R*initial[1])*(g*initial[0]+b-R*initial[1])/risk-c
             if initial_fitness is None else np.asarray(initial_fitness,dtype=float).copy())
    if u.shape != (h,) or not np.isfinite(u).all():
        raise ValueError("initial_fitness is nonfinite or has the wrong size")
    original_u = u.copy()
    x = np.empty(periods)
    fractions,forecasts,profits,fitness = (np.empty((periods,h)) for _ in range(4))
    x2,x1 = float(initial[1]),float(initial[2])
    for t in range(periods):
        # Saturation to exp(-inf)=0 is correct for extreme finite logit utilities.
        with np.errstate(over="ignore",under="ignore",invalid="ignore"):
            exponent = np.zeros(h) if beta == 0 else beta*(u-u.max())
            n = np.exp(exponent); n /= n.sum()
            f = g*x1+b
            current = float((n @ f+innovations[t])/R)
            profit = (current-R*x1)*(g*x2+b-R*x1)/risk-c
            updated = profit+mem*u
        if (not np.isfinite(current) or abs(current) > limit or not np.isfinite(updated).all()
                or not np.isfinite(f).all() or not np.isfinite(n).all()):
            raise RuntimeError(f"belief simulation diverged at period {t}; no clipped path returned")
        x[t] = current
        fractions[t],forecasts[t],profits[t],fitness[t] = n,f,profit,updated
        x2,x1,u = x1,current,updated
    prices = fundamental+x
    if not np.isfinite(prices).all():
        raise RuntimeError("implied prices overflowed")
    spec = {"model":"Brock-Hommes affine rules; constant fundamental; zero outside supply",
        "equations":"Hommes 2006 manuscript 104-111; explicit cost/memory extensions",
        "slopes":g.tolist(),"biases":b.tolist(),"costs":c.tolist(),"gross_rate":R,
        "intensity":beta,"memory":mem,"risk_scale":risk,"fundamental_value":fundamental,
        "implied_expected_dividend":(R-1)*fundamental,"initial_deviations":initial.tolist(),
        "initial_fitness":original_u.tolist(),"seed":seed,"noise_std":std,
        "shock_source":"explicit" if shocks is not None else "seeded_normal",
        "max_abs_deviation":limit}
    return BeliefSimulationResult(labels,readonly(x),readonly(prices),readonly(fractions),
        readonly(forecasts),readonly(profits),readonly(fitness),readonly(innovations),
        json.dumps(spec,sort_keys=True,allow_nan=False))
