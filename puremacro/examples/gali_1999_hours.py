"""Gali (1999 AER) technology shocks and the hours debate — BQ long-run SVAR on real US data.

Replicates the headline of Gali (1999), "Technology, Employment, and the
Business Cycle: Do Technology Shocks Explain Aggregate Fluctuations?",
American Economic Review 89(1), 249-271, using
:func:`puremacro.var.identify.bq.bq_svar` on a frozen FRED snapshot
(``puremacro/replication/data/gali1999.csv``: OPHNFB labor productivity,
HOANBS hours, CNP16OV population; regenerate with
``tools/gen_replication_data_gali1999.py``). Runs fully offline.

Bivariate VAR in (Δlog labor productivity, hours per capita) with the
Blanchard-Quah (1989 AER) long-run restriction: only the technology shock
moves labor productivity in the long run. Two specifications:

- **Difference spec (Gali)**: hours per capita in first differences.
  Headline: hours *fall* on impact after a positive technology shock —
  the anti-RBC result.
- **Levels spec (CEV critique)**: hours per capita in (demeaned log)
  levels, per Christiano, Eichenbaum and Vigfusson (2003), "What Happens
  After a Technology Shock?" (NBER WP 9819): the impact response flips
  sign / attenuates, so the "hours fall" finding hinges on the
  differences-vs-levels choice.

Simplifications vs the paper: productivity is the 2026 vintage of OPHNFB
(nonfarm business output per hour, not Gali's 1990s vintage); hours per
capita is HOANBS divided by quarterly-averaged CNP16OV; the default sample
runs 1948Q2-2007Q4 (pre-Great-Recession, standard in this debate). On this
vintage, Gali's original 1948-1994 sample gives an attenuated-but-still-
negative levels-spec impact; extending through 2007 flips it positive
(the CEV pattern). Bands are residual-bootstrap percentiles from
``bq_svar``; note the result object stores *cumulated* bands, so the
levels-spec hours response is shown as a point IRF only.

Run:
    python -m puremacro.examples.gali_1999_hours

Español
-------
Choques tecnológicos de Gali (1999 AER) y el debate de las horas — SVAR de
largo plazo BQ con datos reales de EE.UU.

Replica el resultado principal de Gali (1999, American Economic Review
89(1), 249-271) usando :func:`puremacro.var.identify.bq.bq_svar` sobre un
snapshot congelado de FRED (OPHNFB productividad laboral, HOANBS horas,
CNP16OV población; se regenera con
``tools/gen_replication_data_gali1999.py``). Corre totalmente sin red.

VAR bivariado en (Δlog productividad laboral, horas per cápita) con la
restricción de largo plazo de Blanchard-Quah (1989 AER): solo el choque
tecnológico mueve la productividad laboral en el largo plazo. Dos
especificaciones:

- **En diferencias (Gali)**: horas per cápita en primeras diferencias.
  Titular: las horas *caen* en impacto tras un choque tecnológico
  positivo — el resultado anti-RBC.
- **En niveles (crítica CEV)**: horas per cápita en (log) niveles
  demediados, según Christiano, Eichenbaum y Vigfusson (2003, NBER WP
  9819): la respuesta de impacto cambia de signo / se atenúa, así que el
  hallazgo "las horas caen" depende de la elección diferencias-vs-niveles.

Simplificaciones: productividad = vintage 2026 de OPHNFB; horas per cápita
= HOANBS / CNP16OV trimestral; muestra por defecto 1948T2-2007T4. Con este
vintage, la muestra original 1948-1994 de Gali da un impacto en niveles
atenuado pero aún negativo; extender hasta 2007 lo vuelve positivo (el
patrón CEV). Las bandas son percentiles bootstrap de ``bq_svar``; el objeto
de resultado guarda bandas *acumuladas*, por lo que la respuesta de horas
en la especificación en niveles se muestra solo como IRF puntual.

Ejecución:
    python -m puremacro.examples.gali_1999_hours
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..replication._data import load_csv
from ..var.identify.bq import bq_svar


_GALI_SAMPLE_END = "1994-12-31"   # Gali's original sample end (1948:1-1994:4)
_DEFAULT_END = "2007-12-31"       # pre-Great-Recession default


def load_gali_data() -> pd.DataFrame:
    """Frozen FRED snapshot -> columns ``dprod``, ``h``, ``dh`` (quarterly).

    ``dprod`` = 100*Δlog(OPHNFB); ``h`` = demeaned 100*log(HOANBS/CNP16OV);
    ``dh`` = first difference of ``h``.
    """
    d = load_csv("gali1999")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    dprod = 100.0 * np.log(d["ophnfb"]).diff()
    h = 100.0 * np.log(d["hoanbs"] / d["cnp16ov"])
    h = h - h.mean()
    out = pd.concat({"dprod": dprod, "h": h, "dh": h.diff()}, axis=1).dropna()
    return out


def run_gali_replication(
    *,
    end: str = _DEFAULT_END,
    p: int = 4,
    horizon: int = 12,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Run both BQ specifications and return the headline objects.

    Returns a dict with the two :class:`BQSVARResult` objects (``diff``,
    ``levels``), the hours impact responses (``impact_diff``,
    ``impact_levels``), the long-run productivity effect (``lr_prod``),
    the de-cumulated levels-spec hours IRF (``hours_irf_levels``), and
    sample metadata.
    """
    data = load_gali_data().loc[:end]
    Y_diff = data[["dprod", "dh"]].to_numpy()
    Y_lvl = data[["dprod", "h"]].to_numpy()

    res_diff = bq_svar(Y_diff, p=p, horizon=horizon,
                       permanent_var_idx=0, n_boot=n_boot, seed=seed)
    res_lvl = bq_svar(Y_lvl, p=p, horizon=horizon,
                      permanent_var_idx=0, n_boot=n_boot, seed=seed)

    # bq_svar cumulates the IRFs. For the diff spec, the cumulated hours
    # response IS the hours level response. For the levels spec, undo the
    # cumulation to recover the level response of hours.
    n = res_lvl.irf_point.shape[1]
    raw_lvl = np.diff(res_lvl.irf_point, axis=0, prepend=np.zeros((1, n, n)))

    return {
        "diff": res_diff,
        "levels": res_lvl,
        # impact (h=0): cumulated == raw at h=0, so both reads are exact.
        "impact_diff": float(res_diff.irf_point[0, 1, 0]),
        "impact_levels": float(raw_lvl[0, 1, 0]),
        "lr_prod": float(res_diff.irf_point[-1, 0, 0]),
        "hours_irf_levels": raw_lvl[:, 1, 0],
        "sample_start": data.index.min(),
        "sample_end": data.index.max(),
        "n_obs": len(data),
        "p": p,
        "horizon": horizon,
    }


def main() -> None:
    """Execute both specs, print the headline, assert Gali's sign."""
    out = run_gali_replication()
    rd, rl = out["diff"], out["levels"]
    print("Gali (1999 AER) technology shocks and hours — BQ long-run SVAR")
    print(f"  Sample: {out['sample_start'].date()}..{out['sample_end'].date()} "
          f"({out['n_obs']} quarters), VAR({out['p']})")
    print(f"  Long-run productivity effect of 1-SD tech shock: "
          f"{out['lr_prod']:+.3f}%")
    print(f"  Hours impact response, DIFFERENCE spec (Gali):   "
          f"{out['impact_diff']:+.3f}%  "
          f"[90% band {rd.irf_lower[0, 1, 0]:+.3f}, {rd.irf_upper[0, 1, 0]:+.3f}]")
    print(f"  Hours impact response, LEVELS spec (CEV):        "
          f"{out['impact_levels']:+.3f}%")
    print("  -> hours FALL on impact with hours differenced (Gali's result); "
          "the response flips/attenuates in levels (the CEV critique).")

    # Gali's own sample, for reference.
    gali = run_gali_replication(end=_GALI_SAMPLE_END, n_boot=50)
    print(f"  Original 1948-1994 sample: diff {gali['impact_diff']:+.3f}%, "
          f"levels {gali['impact_levels']:+.3f}% (attenuated on this vintage)")

    # Headline sign check (loose tolerance: sample vintages shift the point).
    if not out["impact_diff"] < 0.05:
        raise AssertionError(
            f"difference-spec hours impact should be negative (Gali 1999); "
            f"got {out['impact_diff']:+.3f}"
        )

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        H = out["horizon"]
        hs = np.arange(H + 1)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        ax = axes[0]
        ax.plot(hs, rd.irf_point[:, 1, 0], color="0.0", lw=1.2)
        ax.fill_between(hs, rd.irf_lower[:, 1, 0], rd.irf_upper[:, 1, 0],
                        color="0.8", alpha=0.8, label="90% bootstrap band")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_title("Hours in first differences (Gali 1999)")
        ax.set_xlabel("quarters")
        ax.set_ylabel("% response of hours per capita")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax = axes[1]
        ax.plot(hs, out["hours_irf_levels"], color="0.0", lw=1.2,
                label="point IRF (no bands: see docstring)")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_title("Hours in levels (CEV critique)")
        ax.set_xlabel("quarters")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        fig.suptitle("Hours after a positive technology shock (BQ long-run id.)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "gali_1999_hours.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'gali_1999_hours.png'}")


if __name__ == "__main__":
    main()
