"""Ramey-Zubairy (2018 JPE) spending multipliers — LP-IV with military news shocks on 126 years of US data.

Replicates the headline of Ramey & Zubairy (2018), "Government Spending
Multipliers in Good Times and in Bad: Evidence from US Historical
Data," Journal of Political Economy 126(2), 850-901, on a frozen
snapshot of their public historical quarterly dataset
(``puremacro/replication/data/rz2018.csv``, 1889Q1-2015Q4, from
Valerie Ramey's replication archive; regenerate with
``tools/gen_replication_data_rz2018.py``). Runs fully offline.

Three layers, in RZ's own baseline specification (Gordon-Krenn
transformation: real GDP ``y`` and real government purchases ``g``
both divided by trend potential GDP ``rgdp_pott6``, the military-news
instrument scaled by lagged nominal potential; 4 lags of ``y``, ``g``,
``newsy`` as controls; full 1890-2015 sample incl. WWII):

- **Linear cumulative multipliers** via :func:`puremacro.lp.iv.lp_iv`,
  one call per horizon in RZ's *one-step* form: LHS is the h-quarter
  forward window sum of ``y`` (a running sum fed to ``lp_iv`` so its
  long difference IS the window), the endogenous regressor is the
  matching window sum of ``g``, the instrument is ``newsy``. The
  coefficient is the cumulative multiplier itself. Headline: ~0.67 at
  2 years and ~0.71 at 4 years (RZ Table 1: 0.66 and 0.71).
- **State dependence** via :func:`puremacro.lp.state_dep.lp_state_dep`
  with RZ's slack indicator (lagged unemployment >= 6.5%,
  ``transition="threshold"``): per-horizon high/low-state responses of
  ``y`` and ``g`` to the news shock, multipliers as the ratio of
  cumulated coefficients (their two-step ``sumrecy/sumrecg``).
  Headline: slack-state multipliers are NOT above 1 — both states sit
  in the 0.6-0.8 range, RZ's central negative finding vs
  Auerbach-Gorodnichenko.
- **The pedagogical core — per-horizon weak-IV diagnostics** via
  :mod:`puremacro.inference.weak_iv`. The Olea-Pflueger (2013 JBES)
  effective F of the news first stage is horizon- and state-specific:
  strong in the slack state at short horizons (WWII lives there) but
  *collapsing* below the 23.1 cutoff by h=20, and never strong in the
  expansion state. The robust answer is Anderson-Rubin (1949)
  inversion: ``anderson_rubin_test`` + ``msw_bands`` (Montiel
  Olea-Stock-Watson 2021) give an honest 90% multiplier band that is
  visibly wider than the conventional +/-1.645 SE band.

Simplifications vs the paper: controls are not state-interacted in the
state-dependent LPs (RZ interact every control with the lagged state);
HAC bandwidth is h+1 (RZ use ``bw(auto)``); the AR/MSW comparison uses
the slack-state subsample directly rather than the interacted system.
Points move at the second decimal, not the headline.

Run:
    python -m puremacro.examples.ramey_zubairy_2018_multipliers

Español
-------
Multiplicadores de gasto de Ramey-Zubairy (2018 JPE) — LP-IV con
choques de noticias militares sobre 126 años de datos de EE.UU.

Replica el resultado principal de Ramey y Zubairy (2018, Journal of
Political Economy 126(2), 850-901) sobre un snapshot congelado de su
base histórica trimestral pública (1889T1-2015T4, del archivo de
replicación de Valerie Ramey; se regenera con
``tools/gen_replication_data_rz2018.py``). Corre totalmente sin red.

Tres capas, en la especificación base de RZ (transformación
Gordon-Krenn: PIB real y compras de gobierno divididos por el PIB
potencial de tendencia; instrumento de noticias militares escalado por
el potencial nominal rezagado; 4 rezagos de ``y``, ``g`` y ``newsy``
como controles; muestra completa 1890-2015 incl. la 2GM):

- **Multiplicadores acumulados lineales** con
  :func:`puremacro.lp.iv.lp_iv` en la forma de *un paso* de RZ: el
  coeficiente ES el multiplicador acumulado. Titular: ~0.67 a 2 años y
  ~0.71 a 4 años (RZ Tabla 1: 0.66 y 0.71).
- **Dependencia de estado** con
  :func:`puremacro.lp.state_dep.lp_state_dep` y el indicador de holgura
  de RZ (desempleo rezagado >= 6.5%): los multiplicadores en recesión
  NO superan 1 — ambos estados quedan en 0.6-0.8, el hallazgo central
  de RZ frente a Auerbach-Gorodnichenko.
- **El núcleo pedagógico — diagnósticos de instrumentos débiles por
  horizonte** con :mod:`puremacro.inference.weak_iv`: la F efectiva de
  Olea-Pflueger *se derrumba* en el estado de holgura al crecer el
  horizonte (y nunca es fuerte en expansión); la respuesta robusta es
  la inversión de Anderson-Rubin (``anderson_rubin_test`` +
  ``msw_bands``), cuya banda al 90% es visiblemente más ancha que la
  banda convencional de +/-1.645 EE.

Ejecución:
    python -m puremacro.examples.ramey_zubairy_2018_multipliers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.weak_iv import (
    anderson_rubin_test,
    msw_bands,
    olea_pflueger_f,
)
from ..lp.iv import lp_iv
from ..lp.state_dep import lp_state_dep
from ..replication._data import load_csv


_SLACK_THRESHOLD = 6.5   # RZ's fixed unemployment threshold (percent)
_OP_CUTOFF = 23.1        # Olea-Pflueger 5%-worst-case-bias cutoff, k=1
_N_LAGS = 4              # RZ's baseline lag order
_CTL = [f"{v}_l{k}" for k in range(1, _N_LAGS + 1)
        for v in ("y", "g", "newsy")]


def load_rz_data() -> pd.DataFrame:
    """Frozen RZ snapshot -> Gordon-Krenn-transformed quarterly frame.

    Columns: ``y`` = rgdp / rgdp_pott6; ``g`` = (ngov/pgdp) / rgdp_pott6;
    ``newsy`` = news / (lagged rgdp_pott6 x lagged pgdp); ``slack_l`` =
    1{unemp_{t-1} >= 6.5}; ``cumy`` / ``cumg`` running sums of y and g
    (so an h-quarter forward long difference equals the window sum);
    plus explicit lag-control columns ``{y,g,newsy}_l1..l4``.
    """
    d = load_csv("rz2018")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()

    y = d["rgdp"] / d["rgdp_pott6"]
    g = (d["ngov"] / d["pgdp"]) / d["rgdp_pott6"]
    newsy = d["news"] / (d["rgdp_pott6"].shift(1) * d["pgdp"].shift(1))
    slack_l = (d["unemp"] >= _SLACK_THRESHOLD).astype(float).shift(1)

    df = pd.DataFrame({"y": y, "g": g, "newsy": newsy, "slack_l": slack_l})
    df = df.dropna(subset=["y", "g", "newsy"])   # contiguous from 1890Q1
    for k in range(1, _N_LAGS + 1):
        for v in ("y", "g", "newsy"):
            df[f"{v}_l{k}"] = df[v].shift(k)
    df["cumy"] = df["y"].cumsum()
    df["cumg"] = df["g"].cumsum()
    return df


def _residualize(a: np.ndarray, X: np.ndarray) -> np.ndarray:
    b, *_ = np.linalg.lstsq(X, a, rcond=None)
    return a - X @ b


def _window_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Attach the h-quarter forward window sum of g (``cumg_h``) and of
    y (``dyw``) to a copy of ``df`` and drop unusable rows."""
    out = df.copy()
    out["cumg_h"] = df["cumg"].shift(-h) - df["cumg"].shift(1)
    out["dyw"] = df["cumy"].shift(-h) - df["cumy"].shift(1)
    return out.dropna(subset=["dyw", "cumg_h", "newsy", "slack_l"] + _CTL)


def run_rz_replication(
    *,
    horizon: int = 20,
    ar_horizon: int = 12,
    alpha: float = 0.10,
) -> dict:
    """Run all three layers; return the headline objects.

    Returns a dict with the one-step linear multiplier path (``linear``:
    DataFrame h/beta/se/lo/hi/first_stage_f, where ``beta`` IS the
    cumulative multiplier), the state-dependent multiplier paths
    (``m_slack``, ``m_expansion``: numpy arrays over h), the per-horizon
    Olea-Pflueger effective F by state (``f_eff``: DataFrame), and the
    conventional-vs-robust comparison at ``ar_horizon`` in the slack
    state (``robust``: dict).
    """
    df = load_rz_data()
    hs = list(range(horizon + 1))

    # ---- 1. Linear cumulative multipliers: one-step LP-IV per horizon.
    # lp_iv's LHS is cumy_{t+h} - cumy_{t-1} = sum_{j=0..h} y_{t+j}; the
    # endogenous column cumg_h already holds sum_{j=0..h} g_{t+j}; with
    # n_lags=0 the only controls are the explicit (predetermined) lag
    # columns, exactly RZ's L(1/4).(newsy y g) list.
    rows = []
    for h in hs:
        dfh = df.copy()
        dfh["cumg_h"] = df["cumg"].shift(-h) - df["cumg"].shift(1)
        r = lp_iv(dfh, y="cumy", x="cumg_h", z="newsy",
                  horizons=[h], n_lags=0, controls=_CTL, alpha=alpha)
        rows.append(r.iloc[0])
    linear = pd.DataFrame(rows).reset_index(drop=True)

    # ---- 2. State dependence: threshold LP on the slack indicator.
    # Reduced-form responses of y and of g to the news shock by state;
    # multipliers as ratios of cumulated coefficients (RZ's two-step).
    sub = df.dropna(subset=["slack_l"])
    ry = lp_state_dep(sub, y="y", x="newsy", state="slack_l",
                      horizons=hs, n_lags=_N_LAGS,
                      transition="threshold", threshold=0.0,
                      controls=["g_l1"], alpha=alpha)
    rg = lp_state_dep(sub, y="g", x="newsy", state="slack_l",
                      horizons=hs, n_lags=_N_LAGS,
                      transition="threshold", threshold=0.0,
                      controls=["y_l1"], alpha=alpha)
    m_slack = (ry["beta_H"].cumsum() / rg["beta_H"].cumsum()).to_numpy()
    m_expansion = (ry["beta_L"].cumsum() / rg["beta_L"].cumsum()).to_numpy()

    # ---- 3. Per-horizon Olea-Pflueger effective F of the first stage,
    # by state: residualize the window-sum of g and the instrument on
    # the controls within each subsample, then F_eff.
    f_rows = []
    for h in hs:
        dfh = _window_frame(df, h)
        rec: dict = {"h": h}
        for label, mask in [("slack", dfh["slack_l"] == 1.0),
                            ("expansion", dfh["slack_l"] == 0.0),
                            ("linear", dfh["slack_l"].notna())]:
            s = dfh[mask]
            X = np.column_stack(
                [np.ones(len(s))] + [s[c].to_numpy() for c in _CTL])
            xr = _residualize(s["cumg_h"].to_numpy(), X)
            zr = _residualize(s["newsy"].to_numpy(), X)
            rec[label] = olea_pflueger_f(xr, zr)
        f_rows.append(rec)
    f_eff = pd.DataFrame(f_rows)

    # ---- 4. Robust answer at ar_horizon in the slack state:
    # conventional 2SLS +/- z*se(HC0) band vs AR/MSW inversion band.
    s = _window_frame(df, ar_horizon)
    s = s[s["slack_l"] == 1.0]
    W = np.column_stack([s[c].to_numpy() for c in _CTL])
    yv = s["dyw"].to_numpy()
    xv = s["cumg_h"].to_numpy()
    zv = s["newsy"].to_numpy()
    X1 = np.column_stack([np.ones(len(s)), W])
    yr = _residualize(yv, X1)
    xr = _residualize(xv, X1)
    zr = _residualize(zv, X1)
    beta_iv = float(zr @ yr) / float(zr @ xr)
    u = yr - beta_iv * xr
    se_iv = float(np.sqrt(np.sum((zr * u) ** 2)) / abs(zr @ xr))
    from scipy.stats import norm
    zc = float(norm.ppf(1 - alpha / 2))
    grid = np.linspace(-3.0, 5.0, 801)
    msw_lo, msw_hi = msw_bands(grid, yv, xv, zv, alpha=alpha, controls=W)
    robust = {
        "h": ar_horizon,
        "n_slack": int(len(s)),
        "beta_2sls": beta_iv,
        "se_2sls": se_iv,
        "conv_lo": beta_iv - zc * se_iv,
        "conv_hi": beta_iv + zc * se_iv,
        "msw_lo": msw_lo,
        "msw_hi": msw_hi,
        "ar_at_point": anderson_rubin_test(beta_iv, yv, xv, zv, controls=W),
        "ar_at_zero": anderson_rubin_test(0.0, yv, xv, zv, controls=W),
    }

    return {
        "linear": linear,
        "m_slack": m_slack,
        "m_expansion": m_expansion,
        "f_eff": f_eff,
        "robust": robust,
        "horizon": horizon,
        "sample_start": df.index.min(),
        "sample_end": df.index.max(),
        "n_obs": len(df),
    }


def main() -> None:
    """Execute all layers, print the headline, assert RZ's signs/ranges."""
    out = run_rz_replication()
    lin, feff, rob = out["linear"], out["f_eff"], out["robust"]
    m2, m4 = float(lin.loc[8, "beta"]), float(lin.loc[16, "beta"])
    ms4, me4 = float(out["m_slack"][16]), float(out["m_expansion"][16])

    print("Ramey-Zubairy (2018 JPE) spending multipliers — LP-IV, news shocks")
    print(f"  Sample: {out['sample_start'].date()}..{out['sample_end'].date()}"
          f" ({out['n_obs']} quarters), {_N_LAGS} lags, slack = "
          f"unemp >= {_SLACK_THRESHOLD}%")
    print(f"  Linear cumulative multiplier: 2yr {m2:+.3f} "
          f"[90% {lin.loc[8, 'lo']:+.3f}, {lin.loc[8, 'hi']:+.3f}], "
          f"4yr {m4:+.3f} "
          f"[90% {lin.loc[16, 'lo']:+.3f}, {lin.loc[16, 'hi']:+.3f}]"
          f"   (RZ Table 1: 0.66, 0.71)")
    print(f"  State-dependent 4yr multiplier: slack {ms4:+.3f}, "
          f"expansion {me4:+.3f}  -> neither exceeds 1 (RZ's headline).")
    print("  Olea-Pflueger effective F (first stage of the window-sum of g "
          "on news | controls):")
    for h in (4, 8, 12, 16, 20):
        r = feff.loc[feff["h"] == h].iloc[0]
        print(f"    h={h:2d}: slack {r['slack']:7.1f}   "
              f"expansion {r['expansion']:6.1f}   linear {r['linear']:6.1f}"
              f"   (OP 5% cutoff {_OP_CUTOFF})")
    print(f"  -> F_eff COLLAPSES with the horizon in the slack state and is "
          f"never strong in expansion: conventional bands are suspect.")
    arp = rob["ar_at_point"]
    print(f"  Robust answer at h={rob['h']} in slack (n={rob['n_slack']}): "
          f"2SLS {rob['beta_2sls']:+.3f}")
    print(f"    conventional 90% band  [{rob['conv_lo']:+.3f}, "
          f"{rob['conv_hi']:+.3f}]  (width "
          f"{rob['conv_hi'] - rob['conv_lo']:.3f})")
    print(f"    AR/MSW  90% band       [{rob['msw_lo']:+.3f}, "
          f"{rob['msw_hi']:+.3f}]  (width "
          f"{rob['msw_hi'] - rob['msw_lo']:.3f})")
    print(f"    AR test at the 2SLS point: F={arp.stat:.2f}, "
          f"p={arp.p_value:.3f}; at beta=0: "
          f"p={rob['ar_at_zero'].p_value:.3f}")

    # Headline checks (loose: they must survive snapshot regeneration).
    if not (0.3 <= m2 <= 1.0 and 0.3 <= m4 <= 1.1):
        raise AssertionError(
            f"linear cumulative multipliers off RZ's range: "
            f"2yr {m2:+.3f}, 4yr {m4:+.3f} (expected ~0.6-0.8)")
    if not (0.2 <= ms4 <= 1.3 and 0.2 <= me4 <= 1.3):
        raise AssertionError(
            f"state-dependent 4yr multipliers off range: slack {ms4:+.3f}, "
            f"expansion {me4:+.3f}")
    f_slack_short = float(feff.loc[feff["h"] <= 8, "slack"].max())
    f_slack_long = float(feff.loc[feff["h"] == 20, "slack"].iloc[0])
    if not (f_slack_short > _OP_CUTOFF > f_slack_long):
        raise AssertionError(
            f"expected the slack-state effective F to collapse across "
            f"horizons (short {f_slack_short:.1f} > {_OP_CUTOFF} > "
            f"long {f_slack_long:.1f})")
    conv_w = rob["conv_hi"] - rob["conv_lo"]
    msw_w = rob["msw_hi"] - rob["msw_lo"]
    if not msw_w > 1.2 * conv_w:
        raise AssertionError(
            f"AR/MSW band ({msw_w:.3f}) should be clearly wider than the "
            f"conventional band ({conv_w:.3f}) under a weak first stage")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        hs = lin["h"].to_numpy()
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
        ax = axes[0, 0]
        ax.plot(hs, lin["beta"], color="0.0", lw=1.2)
        ax.fill_between(hs, lin["lo"], lin["hi"], color="0.8", alpha=0.8,
                        label="90% HAC band")
        ax.axhline(1.0, color="0.5", lw=0.6, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_title("Linear cumulative multiplier (one-step LP-IV)")
        ax.set_xlabel("horizon (quarters)")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[0, 1]
        ax.plot(hs, out["m_slack"], color="0.0", lw=1.2, label="slack")
        ax.plot(hs, out["m_expansion"], color="0.45", lw=1.2, ls="--",
                label="expansion")
        ax.axhline(1.0, color="0.5", lw=0.6, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_ylim(-0.5, 2.0)
        ax.set_title("State-dependent cumulative multiplier")
        ax.set_xlabel("horizon (quarters)")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, 0]
        ax.semilogy(feff["h"], feff["slack"], color="0.0", lw=1.2,
                    label="slack")
        ax.semilogy(feff["h"], feff["expansion"], color="0.45", lw=1.2,
                    ls="--", label="expansion")
        ax.semilogy(feff["h"], feff["linear"], color="0.7", lw=1.0,
                    ls=":", label="linear")
        ax.axhline(_OP_CUTOFF, color="0.3", lw=0.8, ls="-.",
                   label=f"OP cutoff {_OP_CUTOFF}")
        ax.set_title("Olea-Pflueger effective F, per horizon")
        ax.set_xlabel("horizon (quarters)")
        ax.legend(frameon=False, fontsize=7)
        ax = axes[1, 1]
        for i, (lo, hi, lab) in enumerate([
                (rob["conv_lo"], rob["conv_hi"], "conventional ±1.645 SE"),
                (rob["msw_lo"], rob["msw_hi"], "AR / MSW inversion")]):
            ax.plot([lo, hi], [i, i], lw=4,
                    color="0.55" if i == 0 else "0.0",
                    solid_capstyle="butt")
            ax.text(hi + 0.05, i, lab, va="center", fontsize=8)
        ax.plot(rob["beta_2sls"], 0, marker="o", color="0.0", ms=4)
        ax.plot(rob["beta_2sls"], 1, marker="o", color="0.0", ms=4)
        ax.axvline(0.0, color="0.5", lw=0.6)
        ax.axvline(1.0, color="0.5", lw=0.6, ls="--")
        ax.set_xlim(-0.5, 2.6)
        ax.set_ylim(-0.8, 1.8)
        ax.set_yticks([])
        ax.set_title(f"90% bands for the slack multiplier, h={rob['h']}")
        ax.set_xlabel("cumulative multiplier")
        fig.suptitle("Ramey-Zubairy (2018): multipliers, states, and "
                     "weak-IV honesty", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "ramey_zubairy_2018_multipliers.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: "
              f"{out_dir / 'ramey_zubairy_2018_multipliers.png'}")


if __name__ == "__main__":
    main()
