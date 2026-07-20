"""Synthetic control on a simulated treatment.

Twelve donor units follow common growth + idiosyncratic noise; the
treated unit follows the same path until period T0, after which a
known constant treatment effect is added. Synthetic control should
build a convex combination of donors whose pre-period trajectory
matches the treated unit, then expose the post-period gap as the
treatment effect estimate. The placebo distribution gives a non-
parametric significance test.

Run:
    python -m puremacro.examples.synthetic_control_demo

Español
-------
Control sintético sobre un tratamiento simulado.

Doce unidades donantes siguen un crecimiento común más ruido idiosincrático;
la unidad tratada sigue la misma trayectoria hasta el período T0, tras el cual
se añade un efecto de tratamiento constante y conocido. El control sintético
construye una combinación convexa de donantes cuya trayectoria en el
pre-período coincide con la de la unidad tratada, y luego expone la brecha
en el post-período como estimación del efecto del tratamiento. La distribución
de placebos proporciona una prueba de significancia no paramétrica.

Ejecución:
    python -m puremacro.examples.synthetic_control_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..synthetic_control import synthetic_control


def _simulate(T: int = 30, T0: int = 20, n_donors: int = 12,
              effect: float = -2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(T) * 0.4)
    units = {}
    for i in range(n_donors):
        units[f"D{i:02d}"] = (
            common * rng.uniform(0.7, 1.3)
            + rng.standard_normal(T) * 0.5
            + rng.standard_normal()  # unit fixed effect
        )
    # Treated unit looks similar to a particular blend of donors plus its own noise
    blend_w = rng.dirichlet(np.ones(n_donors))
    treated = (np.array([units[f"D{i:02d}"] for i in range(n_donors)]).T
               @ blend_w
               + rng.standard_normal(T) * 0.3)
    treated[T0:] += effect  # treatment kicks in from T0 onward
    units["treated"] = treated
    idx = pd.date_range("1990", periods=T, freq="YE")
    df = pd.DataFrame(units, index=idx)
    return df, T0


def run_demo() -> dict:
    df, T0 = _simulate()
    pre = list(df.index[:T0])
    post = list(df.index[T0:])
    sc = synthetic_control(df, treated_unit="treated",
                            pre_period=pre, post_period=post,
                            placebo=True)
    return {"data": df, "T0": T0, "scm": sc}


def main() -> None:
    out = run_demo()
    sc = out["scm"]
    print("Synthetic control demo with known constant treatment effect = -2.0")
    print(f"  Pre-period RMSE: {sc.rmse_pre:.3f}")
    print()
    print("  Donor weights (top 5):")
    for unit, w in sc.weights.sort_values(ascending=False).head(5).items():
        print(f"    {unit}: {w:.3f}")
    print()
    eff = sc.treatment_effect
    print(f"  Mean post-treatment gap: {eff.mean():+.3f}  (true -2.0)")
    print(f"  First post-period gap:   {eff.iloc[0]:+.3f}")
    print(f"  Last  post-period gap:   {eff.iloc[-1]:+.3f}")

    if sc.placebo_gaps is not None:
        placebo = sc.placebo_gaps
        treated_mean = abs(eff.mean())
        placebo_means = placebo.mean(axis=0).abs()
        rank = int((placebo_means >= treated_mean).sum()) + 1
        n_units = len(placebo_means) + 1
        p_emp = rank / n_units
        print(f"  Placebo rank of |treated mean gap|: {rank} / {n_units}  "
              f"(empirical p ≈ {p_emp:.3f})")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.plot(sc.actual.index, sc.actual.values, color="0.0",
                lw=1.0, label="treated")
        ax.plot(sc.synthetic.index, sc.synthetic.values,
                color="0.0", lw=1.0, ls="--", label="synthetic")
        ax.axvline(out["data"].index[out["T0"]], color="0.5", lw=0.5)
        ax.set_xlabel("Year"); ax.set_ylabel("Outcome")
        ax.legend(frameon=False)
        ax.set_title("Synthetic control: treated vs counterfactual")
        fig.tight_layout()
        fig.savefig(out_dir / "synthetic_control.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'synthetic_control.png'}")


if __name__ == "__main__":
    main()
