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

- **Linear cumulative multipliers** via one
  :func:`puremacro.lp.iv.lp_iv` call with ``cumulative=True``, RZ's
  *one-step* form: the outcome is the sum of ``y`` over the window
  t..t+h (net of ``y_{t-1}``), the endogenous regressor is the matching
  window sum of ``g``, and the instrument is ``newsy``, so the
  coefficient is the cumulative multiplier itself. RZ's "2-year" and
  "4-year" integrals are the 8- and 16-quarter windows, h = 7 and
  h = 15. Headline: 0.66 at 2 years and 0.71 at 4 years (RZ Table 1:
  0.66 and 0.71). The instrument is weak at both horizons (HAC
  effective F 16.0 and 12.0, below the 23.1 cutoff), so read them with
  the weak-instrument-robust Anderson-Rubin 90% sets: 0.57 to 0.83 at
  2 years and 0.65 to 0.81 at 4 years.
- **State dependence** via
  :func:`puremacro.lp.state_dep.lp_state_dep_iv` with
  ``cumulative=True``: RZ's one-step state-dependent multiplier, with
  HAC standard errors, on their slack indicator (lagged unemployment
  >= 6.5%, ``transition="threshold"``). Headline: slack-state
  multipliers are NOT above 1 — 0.60 at 2 years and 0.68 at 4 years
  with slack, 0.59 and 0.67 without, as in RZ Table 1 — RZ's central
  negative finding vs Auerbach-Gorodnichenko. Without slack the
  instrument is weak (F 7.3 and 10.8), and the Anderson-Rubin 90% sets
  are 0.44 to 0.82 at 2 years and 0.40 to 0.89 at 4 years.
- **The pedagogical core — per-horizon weak-IV diagnostics**, all with
  HAC statistics. The HAC first-stage F of the news instrument (with
  one instrument, the Montiel Olea-Pflueger 2013 effective F: ``mop_f``
  in the linear model, ``first_stage_f_H`` / ``first_stage_f_L`` by
  state; cutoff 23.1) is horizon- and state-specific: in the slack
  state it is weak on impact (h = 0, 1) and strong from h = 3 on (WWII
  lives there), while without slack, and in the linear model, it never
  clears the cutoff. The robust answer is the Anderson-Rubin (1949)
  set, inverted with the same HAC variance as the bands
  (``lp_iv(..., anderson_rubin=True)``). Without slack at 2 years
  (F 7.3) the 90% set, 0.44 to 0.82, is 1.28 times as wide as the
  conventional band, 0.45 to 0.74, and still excludes both 0 and 1.

Simplifications vs the paper: every statistic is HAC with bandwidth
h+1 (RZ use ``bw(auto)``), the effective F's and the Anderson-Rubin
sets as well as the standard errors, so the weak-IV lesson rests on the
same variance as the bands. The point estimates reproduce RZ's Table 1
to two decimals; the standard errors and p-values do not. The state
layer is RZ's one-step IV, so it carries standard errors and a test of
equal multipliers (``p_diff_HL``). The weak-IV comparison runs
:func:`lp_iv` on the quarters without slack (``newsy`` set to NaN in
the others) rather than the interacted system: it gives the same
estimate, with a slightly narrower conventional band. The
state-dependent LPs interact the constant and every control with the
lagged state, as RZ do (``interact="all"``, the default of
:func:`lp_state_dep_iv`). Pooling them (``interact="none"``) is not
innocuous: with the one-step :func:`lp_state_dep_iv` it erases the
slack/no-slack gap once World War II is excluded, and returns
explosive estimates on the post-1947 sample (GitHub issue #217).

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

- **Multiplicadores acumulados lineales** con una sola llamada a
  :func:`puremacro.lp.iv.lp_iv` con ``cumulative=True``, en la forma de
  *un paso* de RZ: el coeficiente ES el multiplicador acumulado. Las
  integrales "a 2 años" y "a 4 años" de RZ son las ventanas de 8 y 16
  trimestres, h = 7 y h = 15. Titular: 0.66 a 2 años y 0.71 a 4 años
  (RZ Tabla 1: 0.66 y 0.71). El instrumento es débil en ambos
  horizontes (F efectiva HAC 16.0 y 12.0, bajo el umbral de 23.1), así
  que hay que leerlos con los conjuntos de Anderson-Rubin al 90%,
  robustos a instrumentos débiles: de 0.57 a 0.83 a 2 años y de 0.65 a
  0.81 a 4 años.
- **Dependencia de estado** con
  :func:`puremacro.lp.state_dep.lp_state_dep_iv` y ``cumulative=True``:
  el multiplicador de un paso de RZ por estado, con errores estándar
  HAC, con el indicador de holgura de RZ (desempleo rezagado >= 6.5%):
  los multiplicadores con holgura NO superan 1 — 0.60 a 2 años y 0.68
  a 4 años con holgura, 0.59 y 0.67 sin ella, como en la Tabla 1 de
  RZ —, el hallazgo central de RZ frente a Auerbach-Gorodnichenko. Sin
  holgura el instrumento es débil (F 7.3 y 10.8), y los conjuntos de
  Anderson-Rubin al 90% van de 0.44 a 0.82 a 2 años y de 0.40 a 0.89
  a 4 años.
- **El núcleo pedagógico — diagnósticos de instrumentos débiles por
  horizonte**, todos con estadísticos HAC: la F HAC de la primera etapa
  (con un solo instrumento, la F efectiva de Montiel Olea-Pflueger;
  umbral 23.1) es débil con holgura en el impacto (h = 0, 1) y fuerte
  desde h = 3 (ahí vive la 2GM); sin holgura, y en el modelo lineal,
  nunca supera el umbral. La respuesta robusta es el conjunto de
  Anderson-Rubin, invertido con la misma varianza HAC que las bandas
  (``lp_iv(..., anderson_rubin=True)``): sin holgura a 2 años (F 7.3),
  el conjunto al 90%, de 0.44 a 0.82, mide 1.28 veces el ancho de la
  banda convencional, de 0.45 a 0.74, y aun así excluye el 0 y el 1.

Simplificaciones frente al artículo: todos los estadísticos son HAC
con ancho de banda h+1 (RZ usan ``bw(auto)``), tanto las F efectivas y
los conjuntos de Anderson-Rubin como los errores estándar, así que la
lección sobre instrumentos débiles descansa en la misma varianza que
las bandas. Las estimaciones puntuales reproducen la Tabla 1 de RZ a
dos decimales; los errores estándar y los valores p no. La capa de
estados es la IV de un paso de RZ, así que trae errores estándar y una
prueba de multiplicadores iguales (``p_diff_HL``). La comparación
robusta corre :func:`lp_iv` sobre los trimestres sin holgura
(``newsy`` en NaN en los demás) en lugar del sistema interactuado: da
la misma estimación, con una banda convencional algo más estrecha. Las
LP dependientes del estado interactúan la constante y cada control con
el estado rezagado, como RZ (``interact="all"``, el valor por omisión
de :func:`lp_state_dep_iv`). Agruparlos (``interact="none"``) no es
inocuo: con :func:`lp_state_dep_iv` de un paso borra la brecha entre
holgura y no holgura una vez excluida la Segunda Guerra Mundial, y da
estimaciones explosivas en la muestra posterior a 1947 (issue #217 de
GitHub).

Ejecución:
    python -m puremacro.examples.ramey_zubairy_2018_multipliers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..lp.iv import lp_iv
from ..lp.state_dep import lp_state_dep_iv
from ..replication._data import load_csv


_SLACK_THRESHOLD = 6.5   # RZ's fixed unemployment threshold (percent)
_OP_CUTOFF = 23.1        # Montiel Olea-Pflueger cutoff for one instrument (lp_iv's mop_cv_10)
_N_LAGS = 4              # RZ's baseline lag order
_HEADLINE = (7, 15)      # RZ's 2- and 4-year integrals: the windows t..t+7 and t..t+15
RZ_CONTROLS: list[str] = [f"{v}_l{k}" for k in range(1, _N_LAGS + 1)
                          for v in ("y", "g", "newsy")]
_CTL = RZ_CONTROLS       # old private name; GitHub issue #217's reproduction imports it


def load_rz_data() -> pd.DataFrame:
    """Frozen RZ snapshot -> Gordon-Krenn-transformed quarterly frame.

    Columns: ``y`` = rgdp / rgdp_pott6; ``g`` = (ngov/pgdp) / rgdp_pott6;
    ``newsy`` = news / (lagged rgdp_pott6 x lagged pgdp); ``slack_l`` =
    1{unemp_{t-1} >= 6.5}, NaN where unemp_{t-1} is missing; ``cumy`` /
    ``cumg`` running sums of y and g (so an h-quarter forward long
    difference equals the window sum, the manual one-step recipe); plus
    the lag-control columns ``RZ_CONTROLS`` = ``{y,g,newsy}_l1..l4``.
    The quarterly index has no gaps, as ``cumulative=True`` requires.
    """
    d = load_csv("rz2018")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()

    y = d["rgdp"] / d["rgdp_pott6"]
    g = (d["ngov"] / d["pgdp"]) / d["rgdp_pott6"]
    newsy = d["news"] / (d["rgdp_pott6"].shift(1) * d["pgdp"].shift(1))
    unemp = d["unemp"]
    # A missing rate stays missing: the comparison alone would read it as 0.
    slack_l = ((unemp >= _SLACK_THRESHOLD).astype(float)
               .where(unemp.notna()).shift(1))

    df = pd.DataFrame({"y": y, "g": g, "newsy": newsy, "slack_l": slack_l})
    df = df.dropna(subset=["y", "g", "newsy"])   # contiguous from 1890Q1
    for k in range(1, _N_LAGS + 1):
        for v in ("y", "g", "newsy"):
            df[f"{v}_l{k}"] = df[v].shift(k)
    df["cumy"] = df["y"].cumsum()
    df["cumg"] = df["g"].cumsum()
    return df


def run_rz_replication(
    *,
    horizon: int = 20,
    ar_horizon: int = 7,
    alpha: float = 0.10,
) -> dict:
    """Run all three layers; return the headline objects.

    Returns a dict with the one-step linear multiplier path (``linear``:
    the :func:`lp_iv` result indexed by h, with Anderson-Rubin sets,
    where ``beta`` IS the cumulative multiplier), the one-step
    state-dependent result (``state_dep``: the :func:`lp_state_dep_iv`
    result) and its multiplier paths (``m_slack``, ``m_expansion``: its
    ``beta_H`` and ``beta_L`` as numpy arrays over h), the per-horizon
    HAC effective F by state (``f_eff``: DataFrame with ``slack`` =
    ``first_stage_f_H``, ``expansion`` = ``first_stage_f_L`` and
    ``linear`` = ``mop_f``), the :func:`lp_iv` result on the expansion
    (no-slack) quarters alone at the headline horizons and
    ``ar_horizon`` (``expansion_iv``: indexed by h, with Anderson-Rubin
    sets), and the conventional-vs-robust comparison at ``ar_horizon``
    in the expansion state (``robust``: dict with ``h``, ``regime``,
    ``beta``, ``se``, ``conv_lo``, ``conv_hi``, ``ar_lo``, ``ar_hi``,
    ``ar_set_type``, ``f_eff``).
    """
    # Every call gets the full frame. The estimators skip the rows with a
    # missing state or instrument themselves; dropping rows would leave a
    # date gap, which cumulative=True rejects.
    df = load_rz_data()
    hs = list(range(horizon + 1))

    # ---- 1. Linear cumulative multipliers: RZ's one-step LP-IV. With
    # cumulative=True the outcome is sum_{j=0..h} (y_{t+j} - y_{t-1}) and
    # the endogenous regressor sum_{j=0..h} g_{t+j}; with n_lags=0 the only
    # controls are the explicit (predetermined) lag columns, exactly RZ's
    # L(1/4).(newsy y g) list, and y_{t-1} among them makes the coefficient
    # equal to RZ's sum-of-y form.
    linear = lp_iv(df, y="y", x="g", z="newsy", horizons=hs, n_lags=0,
                   controls=RZ_CONTROLS, alpha=alpha, cumulative=True,
                   anderson_rubin=True)

    # ---- 2. State dependence: RZ's one-step state-dependent LP-IV on the
    # lagged slack indicator (0/1, so threshold=0.5 splits it), every
    # coefficient regime-specific as in RZ (interact="all").
    state_dep = lp_state_dep_iv(df, y="y", x="g", z="newsy", state="slack_l",
                                horizons=hs, n_lags=0, transition="threshold",
                                threshold=0.5, controls=RZ_CONTROLS,
                                alpha=alpha, interact="all", cumulative=True)
    m_slack = state_dep["beta_H"].to_numpy()
    m_expansion = state_dep["beta_L"].to_numpy()

    # ---- 3. Per-horizon HAC effective F of the first stage (window sum
    # of g on news | controls): each state's own first stage, and the
    # linear model's Montiel Olea-Pflueger F.
    f_eff = pd.DataFrame({
        "h": hs,
        "slack": state_dep["first_stage_f_H"].to_numpy(),
        "expansion": state_dep["first_stage_f_L"].to_numpy(),
        "linear": linear["mop_f"].to_numpy(),
    })

    # ---- 4. Robust answer where the first stage is weak: the expansion
    # (no-slack) state at the headline horizons and ar_horizon. lp_iv on
    # the expansion quarters alone (newsy NaN elsewhere, every row kept)
    # reproduces beta_L and reports the conventional HAC band next to the
    # HAC Anderson-Rubin set.
    masked = df.copy()
    masked.loc[masked["slack_l"] != 0.0, "newsy"] = np.nan
    expansion_iv = lp_iv(masked, y="y", x="g", z="newsy",
                         horizons=sorted({ar_horizon, *_HEADLINE}), n_lags=0,
                         controls=RZ_CONTROLS, alpha=alpha, cumulative=True,
                         anderson_rubin=True)
    r = expansion_iv.loc[ar_horizon]
    robust = {
        "h": ar_horizon,
        "regime": "expansion",
        "beta": float(r["beta"]),
        "se": float(r["se"]),
        "conv_lo": float(r["lo"]),
        "conv_hi": float(r["hi"]),
        "ar_lo": float(r["ar_lo"]),
        "ar_hi": float(r["ar_hi"]),
        "ar_set_type": str(r["ar_set_type"]),
        "f_eff": float(r["mop_f"]),
    }

    return {
        "linear": linear,
        "state_dep": state_dep,
        "m_slack": m_slack,
        "m_expansion": m_expansion,
        "f_eff": f_eff,
        "expansion_iv": expansion_iv,
        "robust": robust,
        "horizon": horizon,
        "sample_start": df.index.min(),
        "sample_end": df.index.max(),
        "n_obs": len(df),
    }


def _weak_iv_note(f: float, ar: pd.Series) -> str:
    """Instrument strength and the Anderson-Rubin set, printed under a headline."""
    weak = f < _OP_CUTOFF
    ar_set = (f"[{ar['ar_lo']:+.3f}, {ar['ar_hi']:+.3f}]"
              if ar["ar_set_type"] == "bounded" else str(ar["ar_set_type"]))
    return (f"{'weak' if weak else 'strong'} instrument (F {f:.1f} "
            f"{'<' if weak else '>='} {_OP_CUTOFF}): Anderson-Rubin 90% {ar_set}")


def main() -> None:
    """Execute all layers, print the headline, assert RZ's signs/ranges."""
    out = run_rz_replication()
    lin, st = out["linear"], out["state_dep"]
    feff, rob = out["f_eff"], out["robust"]
    h2, h4 = _HEADLINE
    m2, m4 = float(lin.loc[h2, "beta"]), float(lin.loc[h4, "beta"])
    ms4, me4 = float(out["m_slack"][h4]), float(out["m_expansion"][h4])

    print("Ramey-Zubairy (2018 JPE) spending multipliers — one-step LP-IV, "
          "news shocks")
    print(f"  Sample: {out['sample_start'].date()}..{out['sample_end'].date()}"
          f" ({out['n_obs']} quarters), {_N_LAGS} lags, slack = "
          f"unemp(t-1) >= {_SLACK_THRESHOLD}%, expansion otherwise")
    print("  Linear cumulative multiplier (one-step LP-IV):")
    for h, label, rz in ((h2, "2yr", "0.66"), (h4, "4yr", "0.71")):
        r = lin.loc[h]
        print(f"    {label} (h={h:2d}): {r['beta']:+.3f} "
              f"[90% {r['lo']:+.3f}, {r['hi']:+.3f}]   (RZ Table 1: {rz})")
        print(f"      {_weak_iv_note(r['mop_f'], r)}")
    print("  State-dependent cumulative multiplier (one-step LP-IV):")
    for h, label, rz in ((h2, "2yr", "0.60, 0.59"), (h4, "4yr", "0.68, 0.67")):
        r, e = st.loc[h], out["expansion_iv"].loc[h]
        print(f"    {label} (h={h:2d}): slack {r['beta_H']:+.3f} "
              f"[90% {r['lo_H']:+.3f}, {r['hi_H']:+.3f}], expansion "
              f"{r['beta_L']:+.3f} [90% {r['lo_L']:+.3f}, {r['hi_L']:+.3f}]"
              f"   (RZ Table 1: {rz})")
        print(f"      expansion: {_weak_iv_note(r['first_stage_f_L'], e)}")
    print(f"  -> at 4 years neither exceeds 1 (slack {ms4:+.3f}, expansion "
          f"{me4:+.3f}): RZ's headline.")
    print("  HAC effective F of the first stage (window sum of g on news | "
          "controls):")
    for h in (0, 1, 3, h2, h4, out["horizon"]):
        f = feff.loc[feff["h"] == h].iloc[0]
        print(f"    h={h:2d}: slack {f['slack']:7.1f}   "
              f"expansion {f['expansion']:6.1f}   linear {f['linear']:6.1f}"
              f"   (cutoff {_OP_CUTOFF})")
    print("  -> weak on impact with slack, strong from h = 3, and never "
          "strong without slack or in the linear model.")
    conv_w = rob["conv_hi"] - rob["conv_lo"]
    ar_w = rob["ar_hi"] - rob["ar_lo"]
    print(f"  Robust answer at h={rob['h']} in {rob['regime']} (no slack), "
          f"where F = {rob['f_eff']:.1f}: estimate {rob['beta']:+.3f}")
    print(f"    conventional 90% HAC band   [{rob['conv_lo']:+.3f}, "
          f"{rob['conv_hi']:+.3f}]  (width {conv_w:.3f})")
    print(f"    Anderson-Rubin 90% HAC set  [{rob['ar_lo']:+.3f}, "
          f"{rob['ar_hi']:+.3f}]  (width {ar_w:.3f}, {rob['ar_set_type']})")
    print(f"  -> the robust set is {ar_w / conv_w:.2f} times as wide and "
          f"still excludes both 0 and 1.")

    # Headline checks (loose: they must survive snapshot regeneration).
    if not (0.3 <= m2 <= 1.0 and 0.3 <= m4 <= 1.1):
        raise AssertionError(
            f"linear cumulative multipliers off RZ's range: "
            f"2yr {m2:+.3f}, 4yr {m4:+.3f} (expected ~0.6-0.8)")
    if not (0.2 <= ms4 <= 1.3 and 0.2 <= me4 <= 1.3):
        raise AssertionError(
            f"state-dependent 4yr multipliers off range: slack {ms4:+.3f}, "
            f"expansion {me4:+.3f}")
    fh = feff.set_index("h")
    f_impact = float(fh.loc[[0, 1], "slack"].max())
    f_slack = float(fh.loc[3:, "slack"].min())
    f_expansion = float(fh["expansion"].max())
    f_linear = float(fh["linear"].max())
    if not (f_impact < _OP_CUTOFF < f_slack and f_expansion < _OP_CUTOFF
            and f_linear < _OP_CUTOFF):
        raise AssertionError(
            f"expected the effective F to be weak on impact with slack "
            f"(h <= 1: {f_impact:.1f}), strong from h = 3 (min {f_slack:.1f}),"
            f" and never strong in expansion (max {f_expansion:.1f}) or in "
            f"the linear model (max {f_linear:.1f}); cutoff {_OP_CUTOFF}")
    if not (rob["f_eff"] < _OP_CUTOFF and rob["ar_set_type"] == "bounded"
            and rob["conv_lo"] <= rob["beta"] <= rob["conv_hi"]
            and rob["ar_lo"] <= rob["beta"] <= rob["ar_hi"]):
        raise AssertionError(
            f"expected a weak first stage (F {rob['f_eff']:.1f}) with a "
            f"bounded Anderson-Rubin set ({rob['ar_set_type']}), both sets "
            f"containing the estimate {rob['beta']:+.3f}")
    if not ar_w > 1.2 * conv_w:
        raise AssertionError(
            f"Anderson-Rubin set ({ar_w:.3f}) should be clearly wider than "
            f"the conventional band ({conv_w:.3f}) under a weak first stage")
    if not (rob["ar_lo"] > 0.0 and rob["ar_hi"] < 1.0):
        raise AssertionError(
            f"the 90% Anderson-Rubin set [{rob['ar_lo']:+.3f}, "
            f"{rob['ar_hi']:+.3f}] should exclude both 0 and 1")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        hs = lin["h"].to_numpy()
        xlab = "horizon h (quarters; dotted: 2 and 4 years)"
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
        ax = axes[0, 0]
        ax.plot(hs, lin["beta"], color="0.0", lw=1.2)
        ax.fill_between(hs, lin["lo"], lin["hi"], color="0.8", alpha=0.8,
                        label="90% HAC band")
        ax.axhline(1.0, color="0.5", lw=0.6, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.6)
        for h in _HEADLINE:
            ax.axvline(h, color="0.6", lw=0.6, ls=":")
        ax.set_title("Linear cumulative multiplier (one-step LP-IV)")
        ax.set_xlabel(xlab)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[0, 1]
        ax.fill_between(hs, st["lo_H"], st["hi_H"], color="0.85", alpha=0.8,
                        label="slack, 90% HAC band")
        ax.plot(hs, out["m_slack"], color="0.0", lw=1.2, label="slack")
        ax.plot(hs, out["m_expansion"], color="0.45", lw=1.2, ls="--",
                label="expansion")
        ax.axhline(1.0, color="0.5", lw=0.6, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.6)
        for h in _HEADLINE:
            ax.axvline(h, color="0.6", lw=0.6, ls=":")
        ax.set_ylim(-0.5, 2.0)
        ax.set_title("State-dependent cumulative multiplier (one-step)")
        ax.set_xlabel(xlab)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, 0]
        ax.semilogy(feff["h"], feff["slack"], color="0.0", lw=1.2,
                    label="slack")
        ax.semilogy(feff["h"], feff["expansion"], color="0.45", lw=1.2,
                    ls="--", label="expansion")
        ax.semilogy(feff["h"], feff["linear"], color="0.7", lw=1.0,
                    ls=":", label="linear")
        ax.axhline(_OP_CUTOFF, color="0.3", lw=0.8, ls="-.",
                   label=f"cutoff {_OP_CUTOFF}")
        ax.set_title("HAC effective F of the first stage, per horizon")
        ax.set_xlabel("horizon h (quarters)")
        ax.legend(frameon=False, fontsize=7)
        ax = axes[1, 1]
        for i, (lo, hi, lab) in enumerate([
                (rob["conv_lo"], rob["conv_hi"], "conventional ±1.645 HAC SE"),
                (rob["ar_lo"], rob["ar_hi"], "Anderson-Rubin (HAC)")]):
            ax.plot([lo, hi], [i, i], lw=4,
                    color="0.55" if i == 0 else "0.0",
                    solid_capstyle="butt")
            ax.text(hi + 0.05, i, lab, va="center", fontsize=8)
        ax.plot(rob["beta"], 0, marker="o", color="0.0", ms=4)
        ax.plot(rob["beta"], 1, marker="o", color="0.0", ms=4)
        ax.axvline(0.0, color="0.5", lw=0.6)
        ax.axvline(1.0, color="0.5", lw=0.6, ls="--")
        ax.set_xlim(-0.25, 1.75)
        ax.set_ylim(-0.8, 1.8)
        ax.set_yticks([])
        ax.set_title(f"90% sets for the expansion multiplier, h={rob['h']} "
                     f"(F {rob['f_eff']:.1f})")
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
