"""SigmaObject volatility decomposition — Python equivalent of
``MASTER_VolatilityDecomposition_Labor.m``.

Demonstrates the headline metric of the sectoral / labor-volatility
track: the **covariance premium**

    cov_share(w) = (w'Σw - Σ_s w_s² σ_s²) / (w'Σw)

— the share of aggregate variance that comes from cross-sector
comovement rather than sector-specific volatility.

Synthetic four-sector panel; in real use ``sig`` and ``R`` come from
rolling-window estimates on the OECD QNA panel (see
``oecd_qna_rolling_volatility_macro_labor_uncertainty.csv`` in the
parent MAV workspace). The point of this script is to exhibit the API,
not to reproduce a specific empirical result.

Run:
    python -m puremacro.examples.sigma_decomposition

Español
-------
Descomposición de la volatilidad con `SigmaObject` — equivalente en Python de
``MASTER_VolatilityDecomposition_Labor.m``.

Muestra la métrica central del bloque sectorial / descomposición de la
volatilidad laboral: la **prima de covarianza**

    cov_share(w) = (w'Σw - Σ_s w_s² σ_s²) / (w'Σw)

— la fracción de la varianza agregada que proviene del
comovimiento entre sectores, en lugar de la volatilidad específica de cada uno.

Panel sintético de cuatro sectores; en un uso real, ``sig`` y ``R`` provienen de
estimaciones en ventana móvil sobre el panel de la OCDE QNA (véase
``oecd_qna_rolling_volatility_macro_labor_uncertainty.csv`` en el
espacio de trabajo MAV raíz). El propósito de este script es ilustrar la API,
no reproducir ningún resultado empírico concreto.

Ejecución:
    python -m puremacro.examples.sigma_decomposition
"""
from __future__ import annotations

import numpy as np

from ..volatility import SigmaObject


def main() -> None:
    rng = np.random.default_rng(2024)

    # 4-sector synthetic system. Sectors 1 and 2 share a common factor
    # (high cross-correlation); sectors 3 and 4 are roughly independent.
    sectors = ("manufacturing", "construction", "services", "agriculture")
    sig = np.array([0.025, 0.040, 0.012, 0.020])
    R = np.array([
        [1.00, 0.65, 0.10, 0.05],
        [0.65, 1.00, 0.15, 0.05],
        [0.10, 0.15, 1.00, 0.05],
        [0.05, 0.05, 0.05, 1.00],
    ])

    S = SigmaObject(
        sig=sig, R=R, sectors=sectors,
        code="DEMO", period="2010Q1-2019Q4",
    )

    # Compare two weighting schemes: sector-share weights vs uniform.
    w_share = np.array([0.25, 0.10, 0.55, 0.10])
    w_equal = np.ones(4) / 4

    print("=" * 64)
    print("SigmaObject volatility decomposition — synthetic 4-sector panel")
    print("=" * 64)
    print()
    print(f"sectors:           {sectors}")
    print(f"per-sector σ:      {sig}")
    print()
    print("Mean off-diagonal correlation:    "
            f"{S.mean_corr():.3f}")
    print("PC1 share (R):                    "
            f"{S.pc1_share('R'):.3f}")
    print()

    for label, w in (("share-weighted", w_share), ("equal-weighted", w_equal)):
        print(f"--- {label} aggregate ---")
        print(f"  σ_total            = {S.std(w):.5f}")
        print(f"  σ_diag (no comov.) = {np.sqrt(S.var_no_cov(w)):.5f}")
        print(f"  cov premium (var)  = {S.cov_premium_var(w):.6f}")
        print(f"  cov share          = {S.cov_share(w):.3f}")
        print()

    # Risk accounting: top three pairs by absolute contribution.
    M = S.pair_contrib(w_share)
    iu = np.triu_indices(4, k=1)
    pairs = sorted(
        ((abs(M[i, j]), i, j, M[i, j]) for i, j in zip(*iu)),
        reverse=True,
    )
    print("Top pairwise contributions (share-weighted):")
    for _, i, j, val in pairs[:3]:
        print(f"  {sectors[i]:>15s} × {sectors[j]:<15s}  {val:+.6e}")
    print()

    # One-row diagnostic table.
    summary = S.summary(w_share)
    print("Summary table (share-weighted):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
