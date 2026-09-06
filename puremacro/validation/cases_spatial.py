"""Validation cases for the ``spatial`` subsystem (autocorrelation diagnostics
and spatial HAC covariances).

Scope (five cases across the mechanisms that admit a SOUND, INDEPENDENT check):

* PACKAGE   ``morans_i`` / ``gearys_c`` on three fields over a 6 x 6 rook
            lattice vs esda (PySAL) ``Moran`` / ``Geary`` with a row-standardised
            weights matrix: the statistic, its expectation, the Cliff-Ord
            normality and randomisation variances and both z-scores. The
            reference is the frozen golden in ``goldens/spatial.json``; the
            live esda call lives only in the ``reference``-marked drift-guard.
* ANALYTICAL ``conley_cov`` at cutoff 0 collapses to the White (HC0) sandwich:
            the spatial kernel keeps only the diagonal score products.
* INTERNAL  ``conley_cov`` equals an explicit double loop over the Bartlett
            kernel; ``spatial_hac_panel_meat`` with every unit inside the
            cutoff equals the Driscoll-Kraay meat (the Hsiang 2010 space-time
            HAC reduces to Driscoll & Kraay 1998 when the spatial kernel is flat).

Pyodide-pure: numpy, pandas and puremacro only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._model import Mechanism, Tol, ValidationCase

LATTICE_SIDE = 6
CONLEY_CUTOFF = 3.0
FIELD_ORDER = ("checker", "gradient", "wave")


def lattice_neighbours(side: int = LATTICE_SIDE) -> dict:
    """Rook contiguity on a ``side x side`` lattice, units numbered row-major."""
    nb: dict = {}
    for i in range(side):
        for j in range(side):
            u = i * side + j
            nb[u] = []
            if i > 0:
                nb[u].append((i - 1) * side + j)
            if i < side - 1:
                nb[u].append((i + 1) * side + j)
            if j > 0:
                nb[u].append(i * side + j - 1)
            if j < side - 1:
                nb[u].append(i * side + j + 1)
    return nb


def lattice_demo_data() -> dict:
    """Three fields on the 6 x 6 rook lattice: a checkerboard (negative
    autocorrelation), a diagonal gradient (positive) and a sine wave."""
    side = LATTICE_SIDE
    fields = {
        "checker": np.array([(i + j) % 2 for i in range(side) for j in range(side)], dtype=float),
        "gradient": np.array([i + j for i in range(side) for j in range(side)], dtype=float),
        "wave": np.sin(np.arange(side * side) / 3.0),
    }
    return {"neighbours": lattice_neighbours(side), "fields": fields}


def conley_demo_data() -> dict:
    """Seeded planar cross-section (n=60) with OLS residuals for the Conley cases."""
    rng = np.random.default_rng(20260905)
    n = 60
    coords = rng.uniform(0.0, 10.0, (n, 2))
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"coords": coords, "X": X, "resid": y - X @ beta}


def spatial_panel_demo_data() -> dict:
    """Seeded balanced panel (6 units x 15 periods) with planar coordinates."""
    rng = np.random.default_rng(20260906)
    n_e, T = 6, 15
    X = np.column_stack([np.ones(n_e * T), rng.standard_normal(n_e * T)])
    return {
        "X": X,
        "resid": rng.standard_normal(n_e * T),
        "coords": pd.DataFrame(rng.uniform(0.0, 10.0, (n_e, 2)), index=np.arange(n_e), columns=["x", "y"]),
        "entities": np.repeat(np.arange(n_e), T),
        "times": np.tile(np.arange(T), n_e),
        "lags": 2,
    }


def _lattice_weights():
    from puremacro.spatial import contiguity_weights

    return contiguity_weights(lattice_demo_data()["neighbours"])


def _moran_stats() -> dict:
    from puremacro.spatial import morans_i

    W = _lattice_weights()
    fields = lattice_demo_data()["fields"]
    res = [morans_i(fields[k], W, n_perm=0) for k in FIELD_ORDER]
    return {
        "I": [r.I for r in res],
        "expected": [r.expected for r in res],
        "variance_norm": [r.variance_norm for r in res],
        "variance_rand": [r.variance_rand for r in res],
        "z_norm": [r.z_norm for r in res],
        "z_rand": [r.z_rand for r in res],
    }


def _geary_stats() -> dict:
    from puremacro.spatial import gearys_c

    W = _lattice_weights()
    fields = lattice_demo_data()["fields"]
    res = [gearys_c(fields[k], W, n_perm=0) for k in FIELD_ORDER]
    return {
        "C": [r.C for r in res],
        "variance_norm": [r.variance_norm for r in res],
        "variance_rand": [r.variance_rand for r in res],
        "z_norm": [r.z_norm for r in res],
        "z_rand": [r.z_rand for r in res],
    }


def _conley_bw0() -> dict:
    from puremacro.spatial import conley_cov

    d = conley_demo_data()
    return {"cov": conley_cov(d["X"], d["resid"], d["coords"], 0.0, metric="euclidean")}


def _white_hc0_reference() -> dict:
    d = conley_demo_data()
    X, u = d["X"], d["resid"]
    XtXi = np.linalg.inv(X.T @ X)
    U = X * u[:, None]
    return {"cov": XtXi @ (U.T @ U) @ XtXi}


def _conley_bartlett() -> dict:
    from puremacro.spatial import conley_cov

    d = conley_demo_data()
    return {"cov": conley_cov(d["X"], d["resid"], d["coords"], CONLEY_CUTOFF, kernel="bartlett", metric="euclidean")}


def _conley_double_loop_reference() -> dict:
    d = conley_demo_data()
    X, u, c = d["X"], d["resid"], d["coords"]
    n = len(u)
    D = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    K = np.where(D <= CONLEY_CUTOFF, 1.0 - D / CONLEY_CUTOFF, 0.0)
    U = X * u[:, None]
    S = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        for j in range(n):
            S += K[i, j] * np.outer(U[i], U[j])
    XtXi = np.linalg.inv(X.T @ X)
    return {"cov": XtXi @ S @ XtXi}


def _spatial_hac_flat_kernel() -> dict:
    from puremacro.spatial import spatial_hac_panel_meat

    d = spatial_panel_demo_data()
    meat = spatial_hac_panel_meat(
        d["X"], d["resid"], d["coords"], d["entities"], d["times"], 1e9, d["lags"],
        kernel="uniform", metric="euclidean",
    )
    return {"meat": meat}


def _driscoll_kraay_reference() -> dict:
    from puremacro.inference.dk import driscoll_kraay

    d = spatial_panel_demo_data()
    return {"meat": driscoll_kraay(d["X"] * d["resid"][:, None], d["times"], lags=d["lags"])}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="spatial.morans_i_vs_esda",
        subsystem="spatial",
        title="Moran's I, its Cliff-Ord variances and z-scores match esda (PySAL)",
        title_es="La I de Moran, sus varianzas de Cliff-Ord y sus z coinciden con esda (PySAL)",
        mechanism=Mechanism.PACKAGE,
        compute=_moran_stats,
        reference="spatial:morans_i_vs_esda",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "esda 2.10.0 (PySAL) Moran(x, w, permutations=0) with w.transform='r' "
            "on a 6 x 6 rook lattice; Cliff, A.D. & Ord, J.K. (1981), Spatial Processes, "
            "Pion, normality and randomisation moments."
        ),
        notes="Three fields (checkerboard, gradient, sine wave); statistic, EI, VI_norm, VI_rand, z_norm, z_rand.",
    ),
    ValidationCase(
        id="spatial.gearys_c_vs_esda",
        subsystem="spatial",
        title="Geary's C, its Cliff-Ord variances and z-scores match esda (PySAL)",
        title_es="La C de Geary, sus varianzas de Cliff-Ord y sus z coinciden con esda (PySAL)",
        mechanism=Mechanism.PACKAGE,
        compute=_geary_stats,
        reference="spatial:gearys_c_vs_esda",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "esda 2.10.0 (PySAL) Geary(x, w, permutations=0) with w.transform='r' "
            "on a 6 x 6 rook lattice; Geary, R.C. (1954), The Incorporated Statistician 5(3), 115-146."
        ),
        notes="Same three fields as the Moran case; C, VC_norm, VC_rand, z_norm, z_rand.",
    ),
    ValidationCase(
        id="spatial.conley_cutoff0_equals_hc0",
        subsystem="spatial",
        title="Conley spatial HAC at cutoff 0 equals the White (HC0) sandwich",
        title_es="El HAC espacial de Conley con radio 0 iguala el sándwich de White (HC0)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_conley_bw0,
        reference=_white_hc0_reference,
        tol=Tol.TIGHT,
        citation=(
            "Conley, T.G. (1999), 'GMM estimation with cross sectional dependence', "
            "Journal of Econometrics 92(1), 1-45: with a zero cutoff the kernel keeps only "
            "the own-observation score products, leaving White (1980) HC0."
        ),
    ),
    ValidationCase(
        id="spatial.conley_equals_double_loop",
        subsystem="spatial",
        title="Vectorised Conley covariance equals the explicit Bartlett double loop",
        title_es="La covarianza de Conley vectorizada iguala el doble bucle explícito con núcleo de Bartlett",
        mechanism=Mechanism.INTERNAL,
        compute=_conley_bartlett,
        reference=_conley_double_loop_reference,
        tol=Tol.TIGHT,
        citation=(
            "Conley, T.G. (1999), Journal of Econometrics 92(1), 1-45, eq. (3): "
            "sum_i sum_j K(d_ij) u_i u_j x_i x_j' with the Bartlett kernel max(0, 1 - d/cutoff)."
        ),
        notes="Planar Euclidean coordinates, cutoff 3.0, n=60, k=3.",
    ),
    ValidationCase(
        id="spatial.space_time_hac_flat_kernel_equals_driscoll_kraay",
        subsystem="spatial",
        title="Space-time HAC with every unit inside the cutoff equals the Driscoll-Kraay meat",
        title_es="El HAC espacio-temporal con todas las unidades dentro del radio iguala la matriz de Driscoll-Kraay",
        mechanism=Mechanism.INTERNAL,
        compute=_spatial_hac_flat_kernel,
        reference=_driscoll_kraay_reference,
        tol=Tol.TIGHT,
        citation=(
            "Hsiang, S.M. (2010), PNAS 107(35), 15367-15372, and Driscoll, J.C. & Kraay, A.C. "
            "(1998), Review of Economics and Statistics 80(4), 549-560: a uniform spatial kernel "
            "that covers every pair sums the scores across units, which is the Driscoll-Kraay "
            "cross-sectional aggregation."
        ),
        notes="Balanced 6 x 15 panel, uniform kernel, cutoff 1e9, 2 time lags.",
    ),
]
