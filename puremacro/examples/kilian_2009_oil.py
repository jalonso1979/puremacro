"""Kilian (2009 AER) "Not All Oil Price Shocks Are Alike" — recursive oil-market VAR with historical decomposition.

Replicates the three signature outputs of Kilian (2009), "Not All Oil Price
Shocks Are Alike: Disentangling Demand and Supply Shocks in the Crude Oil
Market", American Economic Review 99(3), 1053-1069, from a frozen FRED
snapshot (``puremacro/replication/data/kilian2009.csv``; regenerate with
``tools/gen_replication_data_kilian2009.py``). Runs fully offline.

Monthly 3-variable VAR in (Δlog oil production, real activity, real oil
price) with 24 lags as in the paper, identified recursively (Cholesky) into

1. **oil supply shocks**       (production innovations; reported as
   supply *disruptions*, i.e. sign-flipped),
2. **aggregate demand shocks** (global real activity innovations), and
3. **oil-specific demand shocks** (precautionary/oil-market-specific
   price innovations).

Signature outputs: (i) IRFs — supply shocks move the real price little,
demand shocks move it a lot; (ii) FEVD of the real price — demand shocks
dominate at all horizons; (iii) THE HEADLINE — the historical decomposition
of the real oil price (:func:`puremacro.var.irf.historical_decomp`), plotted
as stacked shock contributions over 1975-2007: demand, not supply, drove the
big price swings (1979-80 and 1990 surges = oil-specific/precautionary
demand; the 2003-07 surge = global aggregate demand).

Simplifications vs the paper (documented proxies, see the gen tool):
world crude-oil production is proxied by the US crude-oil production index
IPG211111S (the world EIA series is not on the no-key fredgraph endpoint);
the real price is WTI (WTISPLC) deflated by CPIAUCSL rather than the
refiners' acquisition cost of imported crude (RACIM-type ids 404 on
fredgraph); real activity is FRED's IGREA — Kilian's own index, 2019
corrected vintage. Estimation sample 1972:2-2007:12 (frozen), vs the
paper's 1973:2-2007:12.

Run:
    python -m puremacro.examples.kilian_2009_oil

Español
-------
Kilian (2009 AER) "Not All Oil Price Shocks Are Alike" — VAR recursivo del
mercado petrolero con descomposición histórica.

Replica los tres resultados distintivos de Kilian (2009, American Economic
Review 99(3), 1053-1069) a partir de un snapshot congelado de FRED (se
regenera con ``tools/gen_replication_data_kilian2009.py``). Corre
totalmente sin red.

VAR mensual de 3 variables (Δlog producción de crudo, actividad real,
precio real del crudo) con 24 rezagos como en el artículo, identificado de
forma recursiva (Cholesky) en: (1) choques de oferta (reportados como
*disrupciones* de oferta, con el signo invertido), (2) choques de demanda
agregada global, y (3) choques de demanda específica del petróleo
(precautoria).

Resultados distintivos: (i) IRFs — los choques de oferta mueven poco el
precio real, los de demanda lo mueven mucho; (ii) FEVD del precio real —
los choques de demanda dominan en todos los horizontes; (iii) EL TITULAR —
la descomposición histórica del precio real
(:func:`puremacro.var.irf.historical_decomp`), graficada como
contribuciones apiladas de los choques en 1975-2007: la demanda, no la
oferta, generó las grandes oscilaciones del precio (los picos de 1979-80 y
1990 = demanda precautoria; el auge 2003-07 = demanda agregada global).

Simplificaciones (proxies documentados): la producción mundial se aproxima
con el índice de producción de crudo de EE.UU. IPG211111S; el precio real
es WTI deflactado por el CPI en lugar del costo de adquisición de
refinadores importado; la actividad real es IGREA (el índice del propio
Kilian, vintage corregido 2019). Muestra 1972:2-2007:12 (congelada), vs
1973:2-2007:12 del artículo.

Ejecución:
    python -m puremacro.examples.kilian_2009_oil
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..replication._data import load_csv
from ..var.estimate import estimate_var
from ..var.identify.cholesky import cholesky_svar
from ..var.irf import fevd, historical_decomp


_SHOCKS = ("supply", "agg_demand", "oil_specific")
# Kilian's sign convention: report the supply shock as a supply DISRUPTION
# (production falls), so its responses are flipped. Demand shocks as-is.
_SIGN = np.array([-1.0, 1.0, 1.0])

_EPISODES = {
    "1979-80 crisis": ("1978-06-01", "1980-06-01"),
    "1986 collapse": ("1985-06-01", "1986-07-01"),
    "1990 Gulf war": ("1990-06-01", "1990-10-01"),
    "2003-07 surge": ("2002-12-01", "2007-12-01"),
}


def load_kilian_data() -> pd.DataFrame:
    """Frozen FRED snapshot -> columns ``dprod``, ``rea``, ``rpo`` (monthly).

    ``dprod`` = 100*Δlog(oil production proxy); ``rea`` = IGREA;
    ``rpo`` = demeaned 100*log(WTI/CPI) — the (log) real price of oil.
    """
    d = load_csv("kilian2009")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    dprod = 100.0 * np.log(d["oilprod"]).diff()
    rpo = 100.0 * np.log(d["wti"] / d["cpi"])
    out = pd.concat(
        {"dprod": dprod, "rea": d["igrea"], "rpo": rpo}, axis=1
    ).dropna()
    # Demean the real price on the estimation sample (after the diff row drops).
    out["rpo"] = out["rpo"] - out["rpo"].mean()
    return out


def run_kilian_replication(
    *,
    p: int = 24,
    horizon: int = 15,
    fevd_horizon: int = 60,
    n_boot: int = 100,
    seed: int = 0,
) -> dict:
    """Estimate the oil-market VAR and return IRFs, FEVD and the HD.

    ``p=24`` follows the paper (the frozen 1972-2007 sample supports it:
    431 obs vs 73 parameters per equation). Returns a dict with the
    :class:`CholeskySVARResult` (``irf``), the FEVD array (``fevd``), the
    historical-decomposition contributions to the real price
    (``hd_contrib``, sign-convention *not* applied: raw Cholesky shocks),
    the deterministic component (``hd_deterministic``), peak |IRF| of the
    real price per shock (``price_peaks``, Kilian sign convention), and
    per-episode contribution changes (``episodes``).
    """
    data = load_kilian_data()
    Y = data.to_numpy()

    irf_res = cholesky_svar(Y, p=p, horizon=horizon,
                            n_boot=n_boot, ci=0.9, seed=seed)

    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    B0 = np.linalg.cholesky(Sigma)
    fevd_arr = fevd(A_list, B0, fevd_horizon)

    hd = historical_decomp(A_list, B0, resid, init_y=Y[:p], intercept=c)
    idx = data.index[p:]
    contrib = pd.DataFrame(hd["shocks"][:, 2, :], index=idx,
                           columns=list(_SHOCKS))
    det = pd.Series(hd["deterministic"][:, 2], index=idx, name="deterministic")

    # Adding-up identity: y_t = deterministic_t + sum_j contribution_jt.
    recon = hd["deterministic"] + hd["shocks"].sum(axis=2)
    max_err = float(np.abs(recon - Y[p:]).max())
    if max_err > 1e-6:
        raise AssertionError(
            f"historical decomposition does not add up (max err {max_err:.2e})"
        )

    # Peak |response| of the real price (variable 2) per shock, over h<=horizon.
    price_irf = irf_res.irf_point[:, 2, :] * _SIGN[None, :]
    price_peaks = {s: float(np.abs(price_irf[:, j]).max())
                   for j, s in enumerate(_SHOCKS)}

    episodes = {}
    for label, (a, b) in _EPISODES.items():
        ch = contrib.loc[b] - contrib.loc[a]
        episodes[label] = {
            "total": float(data["rpo"].loc[b] - data["rpo"].loc[a]),
            **{s: float(ch[s]) for s in _SHOCKS},
        }

    return {
        "irf": irf_res,
        "fevd": fevd_arr,
        "hd_contrib": contrib,
        "hd_deterministic": det,
        "hd_max_err": max_err,
        "price_peaks": price_peaks,
        "price_irf": price_irf,
        "episodes": episodes,
        "data": data,
        "p": p,
        "horizon": horizon,
    }


def _print_summary(out: dict) -> None:
    data = out["data"]
    print("Kilian (2009 AER) oil-market VAR — supply vs demand shocks")
    print(f"  Sample: {data.index.min().date()}..{data.index.max().date()} "
          f"({len(data)} months), VAR({out['p']}), Cholesky "
          "[dprod, rea, rpo]")
    pk = out["price_peaks"]
    print("  (i) Peak |real-price IRF| by shock (h<=%d):" % out["horizon"])
    print(f"        supply disruption : {pk['supply']:.2f}%")
    print(f"        aggregate demand  : {pk['agg_demand']:.2f}%")
    print(f"        oil-specific dem. : {pk['oil_specific']:.2f}%")
    fe = out["fevd"]
    print("  (ii) FEVD of the real oil price (shares):")
    for h in (12, fe.shape[0] - 1):
        s = fe[h, 2, :]
        print(f"        h={h:<3d} supply {s[0]:.3f} | agg demand {s[1]:.3f} "
              f"| oil-specific {s[2]:.3f}")
    print("  (iii) HEADLINE — historical decomposition of the real price")
    print(f"        (adding-up max error {out['hd_max_err']:.1e}); "
          "change in contribution per episode, log points:")
    for label, ep in out["episodes"].items():
        print(f"        {label:<15s} total {ep['total']:+6.1f} | "
              f"supply {ep['supply']:+6.1f} | agg demand {ep['agg_demand']:+6.1f} | "
              f"oil-specific {ep['oil_specific']:+6.1f}")
    print("  -> demand shocks, not supply shocks, drove the big real-price "
          "swings (Kilian's headline).")


def _plot(out: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    H = out["horizon"]
    hs = np.arange(H + 1)
    irf_res = out["irf"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             gridspec_kw={"height_ratios": [1.0, 1.3]})

    # Panel 1: real-price IRFs to the three shocks (Kilian sign convention).
    ax = axes[0]
    styles = {"supply": ("0.0", "-"), "agg_demand": ("0.35", "--"),
              "oil_specific": ("0.6", "-.")}
    labels = {"supply": "oil supply disruption",
              "agg_demand": "aggregate demand",
              "oil_specific": "oil-specific demand"}
    for j, s in enumerate(_SHOCKS):
        sgn = _SIGN[j]
        lo = irf_res.irf_lower[:, 2, j] * sgn
        hi = irf_res.irf_upper[:, 2, j] * sgn
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        color, ls = styles[s]
        ax.plot(hs, out["price_irf"][:, j], color=color, ls=ls, lw=1.3,
                label=labels[s])
        ax.fill_between(hs, lo, hi, color=color, alpha=0.12)
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_title("Real oil price response to 1-SD shocks (90% bands)")
    ax.set_xlabel("months")
    ax.set_ylabel("% response")
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: stacked historical decomposition of the real price, 1975-2007.
    ax = axes[1]
    contrib = out["hd_contrib"].loc["1975-01-01":]
    total = contrib.sum(axis=1)
    x = contrib.index
    # Contribution of a shock is invariant to its sign normalisation, so the
    # HD legend says "oil supply" (not "disruption").
    hd_labels = {"supply": "oil supply", "agg_demand": "aggregate demand",
                 "oil_specific": "oil-specific demand"}
    shades = {"supply": "0.15", "agg_demand": "0.45", "oil_specific": "0.75"}
    pos_base = np.zeros(len(contrib))
    neg_base = np.zeros(len(contrib))
    width = 31.0  # days; monthly bars
    for s in _SHOCKS:
        v = contrib[s].to_numpy()
        bottom = np.where(v >= 0, pos_base, neg_base)
        ax.bar(x, v, bottom=bottom, width=width, color=shades[s],
               label=hd_labels[s], linewidth=0)
        pos_base += np.clip(v, 0, None)
        neg_base += np.clip(v, None, 0)
    ax.plot(x, total, color="0.0", lw=0.9,
            label="real price less deterministic")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_title("Historical decomposition of the real oil price, 1975-2007")
    ax.set_ylabel("log points")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "kilian_2009_oil.png", dpi=120, bbox_inches="tight")
    print(f"  Figure saved: {out_dir / 'kilian_2009_oil.png'}")


def main() -> None:
    """Run the replication, print the three signature outputs, assert the ranking."""
    out = run_kilian_replication()
    _print_summary(out)

    # Qualitative headline check: demand moves the price more than supply.
    pk = out["price_peaks"]
    if not pk["agg_demand"] > pk["supply"]:
        raise AssertionError(
            "expected peak |price IRF to aggregate demand| > peak |price IRF "
            f"to supply| (Kilian 2009); got {pk['agg_demand']:.2f} vs "
            f"{pk['supply']:.2f}"
        )

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        _plot(out, out_dir)


if __name__ == "__main__":
    main()
