"""Validation cases for the ``spatial`` subsystem (autocorrelation diagnostics,
spatial HAC covariances, the cross-section SAR/SEM/SDM/SLX family, the
fixed-effects spatial panel and spatial local projections).

Diagnostics and HAC (the original five cases):

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

Spatial regression models (:mod:`puremacro.spatial.models`):

* ANALYTICAL the concentrated log-likelihood of :func:`sar` and :func:`sem`
            evaluated at ``rho = 0`` / ``lambda = 0`` — the ``loglik_at_zero``
            attribute, produced by the *same* objective the optimiser maximised
            — equals the Gaussian OLS log-likelihood in closed form (the
            Jacobian ``ln|I - 0 W| = 0`` and ``SSR(0)`` is the OLS residual sum
            of squares).
* INTERNAL  :func:`sdm` is :func:`sar` on the augmented design ``[X, W X_d]``
            (coefficients, full covariance, ``rho``, ``sigma2``, log-likelihood);
            :func:`slx` is :func:`ols_spatial` on the same augmented design;
            the LeSage-Pace impacts of :func:`spatial_effects` equal a
            brute-force dense ``S_r(W) = (I - rho W)^-1 (I beta_r + W theta_r)``
            built with numpy alone.

Spatial panel and spatial local projections:

* INTERNAL  :func:`spatial_panel`'s ``loglik_null`` (the concentrated
            likelihood at ``rho = 0``) equals the Gaussian log-likelihood of the
            two-way within residuals from
            :func:`puremacro.lp._panel_helpers.two_way_fe_within` at the
            transformation divisor ``N* = (n-1)(T-1)``;
            :func:`spatial_lp` with ``spillover_orders=()`` reproduces
            :func:`puremacro.lp.panel_lp`.
* ANALYTICAL the Lee-Yu (2010) analytic bias correction under individual
            effects only is exactly ``(0_k, 0, -sigma^2/(T-1))``: it leaves
            ``beta`` and ``rho`` untouched and rescales ``sigma^2`` by
            ``T/(T-1)``.

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


# ---------------------------------------------------------------------------
# Cross-section SAR / SEM / SDM / SLX and the LeSage-Pace impacts
# ---------------------------------------------------------------------------
MODELS_SEED = 20260907
MODELS_RHO = 0.4

EFFECTS_RHO = 0.4
EFFECTS_NAMES = ("const", "x1", "x2")
EFFECTS_BETA = (0.7, 1.0, -0.5)
EFFECTS_THETA = (0.0, 0.3, -0.2)


def labelled_lattice_neighbours(side: int) -> dict:
    """:func:`lattice_neighbours` with string unit labels (panel entity ids)."""
    return {
        f"u{u:02d}": [f"u{v:02d}" for v in nb]
        for u, nb in lattice_neighbours(side).items()
    }


def models_demo_data() -> dict:
    """Seeded SDM cross-section on the 6 x 6 row-standardised rook lattice.

    ``y = (I - 0.4 W)^-1 (1 + X beta + W X theta + eps)`` with ``n = 36`` and
    two regressors. ``X_augmented`` is ``[X, W X]`` formed through the *same*
    sparse product ``W.W @ X`` that ``sdm`` / ``slx`` use internally, so the
    reduction cases compare identical designs rather than two spellings of one.
    """
    from puremacro.spatial import contiguity_weights

    W = contiguity_weights(lattice_neighbours(LATTICE_SIDE))
    n = W.n
    rng = np.random.default_rng(MODELS_SEED)
    X = rng.standard_normal((n, 2))
    WX = np.asarray(W.W @ X, dtype=float)
    A_inv = np.linalg.inv(np.eye(n) - MODELS_RHO * W.to_dense())
    y = A_inv @ (
        1.0 + X @ np.array([1.0, -0.5]) + WX @ np.array([0.3, 0.0])
        + rng.standard_normal(n)
    )
    return {"W": W, "W_dense": W.to_dense(), "X": X, "y": y,
            "X_augmented": np.column_stack([X, WX])}


def _sar_sem_loglik_at_zero() -> dict:
    from puremacro.spatial import sar, sem

    d = models_demo_data()
    return {
        "sar_loglik_at_zero": sar(d["y"], d["X"], d["W"]).loglik_at_zero,
        "sem_loglik_at_zero": sem(d["y"], d["X"], d["W"]).loglik_at_zero,
    }


def _gaussian_ols_loglik_reference() -> dict:
    """``-(n/2)(ln 2pi + 1) - (n/2) ln(e'e/n)`` from a plain numpy OLS fit."""
    d = models_demo_data()
    n = d["y"].shape[0]
    Xc = np.column_stack([np.ones(n), d["X"]])
    e = d["y"] - Xc @ np.linalg.lstsq(Xc, d["y"], rcond=None)[0]
    ll = -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(float(e @ e) / n))
    return {"sar_loglik_at_zero": ll, "sem_loglik_at_zero": ll}


def _sdm_fit() -> dict:
    from puremacro.spatial import sdm

    d = models_demo_data()
    res = sdm(d["y"], d["X"], d["W"])
    return {"params": res.params.to_numpy(dtype=float), "vcov": res.vcov,
            "rho": res.rho, "sigma2": res.sigma2, "loglik": res.loglik}


def _sar_on_augmented_design_reference() -> dict:
    from puremacro.spatial import sar

    d = models_demo_data()
    res = sar(d["y"], d["X_augmented"], d["W"])
    return {"params": res.params.to_numpy(dtype=float), "vcov": res.vcov,
            "rho": res.rho, "sigma2": res.sigma2, "loglik": res.loglik}


def _slx_fit() -> dict:
    from puremacro.spatial import slx

    d = models_demo_data()
    res = slx(d["y"], d["X"], d["W"])
    return {"params": res.params.to_numpy(dtype=float),
            "bse": res.bse.to_numpy(dtype=float),
            "sigma2": res.sigma2, "loglik": res.loglik, "r2": res.pseudo_r2}


def _ols_on_augmented_design_reference() -> dict:
    from puremacro.spatial import ols_spatial

    d = models_demo_data()
    res = ols_spatial(d["y"], d["X_augmented"], d["W"])
    return {"params": res.params.to_numpy(dtype=float),
            "bse": res.bse.to_numpy(dtype=float),
            "sigma2": res.sigma2, "loglik": res.loglik, "r2": res.r2}


def _lesage_pace_effects() -> dict:
    from puremacro.spatial import spatial_effects

    eff = spatial_effects(
        models_demo_data()["W"], np.asarray(EFFECTS_BETA), EFFECTS_RHO,
        theta=np.asarray(EFFECTS_THETA), names=EFFECTS_NAMES,
    )
    return {"direct": eff.direct, "indirect": eff.indirect, "total": eff.total}


def _dense_impact_matrix_reference() -> dict:
    """Brute-force ``S_r(W) = (I - rho W)^-1 (I beta_r + W theta_r)``.

    ``direct_r = tr(S_r)/n``, ``total_r = 1'S_r 1/n``, ``indirect = total -
    direct`` — the LeSage & Pace (2009, sec. 2.7) definitions, formed with a
    dense inverse and nothing from :mod:`puremacro.spatial.models`.
    """
    Wd = models_demo_data()["W_dense"]
    n = Wd.shape[0]
    A_inv = np.linalg.inv(np.eye(n) - EFFECTS_RHO * Wd)
    direct, total = [], []
    for r, name in enumerate(EFFECTS_NAMES):
        if name == "const":
            continue
        S = A_inv @ (EFFECTS_BETA[r] * np.eye(n) + EFFECTS_THETA[r] * Wd)
        direct.append(float(np.trace(S)) / n)
        total.append(float(S.sum()) / n)
    direct = np.asarray(direct)
    total = np.asarray(total)
    return {"direct": direct, "indirect": total - direct, "total": total}


# ---------------------------------------------------------------------------
# Fixed-effects spatial panel
# ---------------------------------------------------------------------------
PANEL_SIDE = 4
PANEL_T = 12
PANEL_LATTICE_SEED = 20260908
PANEL_RHO = 0.35


def panel_lattice_demo_data() -> dict:
    """Seeded balanced SAR panel: 4 x 4 rook lattice (16 units) x 12 periods.

    Both entity and time effects are in the DGP, plus two regressors, so the
    two-way and individual-effects routes are both exercised.
    """
    from puremacro.spatial import contiguity_weights

    W = contiguity_weights(labelled_lattice_neighbours(PANEL_SIDE))
    n, T = W.n, PANEL_T
    rng = np.random.default_rng(PANEL_LATTICE_SEED)
    S_inv = np.linalg.inv(np.eye(n) - PANEL_RHO * W.to_dense())
    xs = rng.standard_normal((n, T))
    zs = rng.standard_normal((n, T))
    mu = rng.standard_normal((n, 1))
    xi = rng.standard_normal((1, T))
    ys = S_inv @ (1.2 * xs - 0.6 * zs + mu + xi + 0.5 * rng.standard_normal((n, T)))
    index = pd.MultiIndex.from_product([list(W.ids), range(T)], names=["code", "date"])
    frame = pd.DataFrame(
        {"y": ys.ravel(), "x": xs.ravel(), "z": zs.ravel()}, index=index
    )
    return {"W": W, "frame": frame, "n": n, "T": T, "x_cols": ["x", "z"]}


def _spatial_panel_null_loglik() -> dict:
    from puremacro.spatial import spatial_panel

    d = panel_lattice_demo_data()
    res = spatial_panel(d["frame"], "y", d["x_cols"], d["W"], model="sar",
                        effects="two-way", method="transformation", impacts=False)
    return {"loglik_null": res.loglik_null, "n_eff": float(res.n_eff)}


def _two_way_fe_loglik_reference() -> dict:
    from puremacro.lp._panel_helpers import two_way_fe_within

    d = panel_lattice_demo_data()
    fe = two_way_fe_within(d["frame"], y_col="y", x_cols=d["x_cols"])
    u = np.asarray(fe["residuals"], dtype=float)
    n_eff = (d["n"] - 1) * (d["T"] - 1)
    ll = -0.5 * n_eff * (np.log(2.0 * np.pi) + 1.0 + np.log(float(u @ u) / n_eff))
    return {"loglik_null": ll, "n_eff": float(n_eff)}


def _panel_individual_effects_fit(bias_correction: str) -> dict:
    from puremacro.spatial import spatial_panel

    d = panel_lattice_demo_data()
    res = spatial_panel(d["frame"], "y", d["x_cols"], d["W"], model="sar",
                        effects="individual", method="direct",
                        bias_correction=bias_correction, impacts=False)
    return {"beta": res.beta.to_numpy(dtype=float), "rho": res.rho,
            "sigma2": res.sigma2}


def _lee_yu_corrected() -> dict:
    return _panel_individual_effects_fit("lee-yu")


def _lee_yu_closed_form_reference() -> dict:
    """``theta - I^-1 a = (beta, rho, sigma2 T/(T-1))`` under individual effects.

    The bias vector solves ``I b = a`` with ``a = (0_k, -tr(G), -n/(2 sigma^2))``
    and ``I[k, k+1] = (T-1) tr(G)/sigma^2``, ``I[k+1, k+1] = n(T-1)/(2 sigma^4)``,
    ``I[:k, k+1] = 0``. Substituting ``b = (0_k, 0, -sigma^2/(T-1))`` reproduces
    ``a`` row by row, so the correction touches ``sigma^2`` alone.
    """
    raw = _panel_individual_effects_fit("none")
    factor = PANEL_T / (PANEL_T - 1.0)
    return {"beta": raw["beta"], "rho": raw["rho"], "sigma2": raw["sigma2"] * factor}


# ---------------------------------------------------------------------------
# Spatial local projections
# ---------------------------------------------------------------------------
LP_SIDE = 4
LP_T = 26
LP_SEED = 20260909
LP_HORIZONS = (0, 1, 2, 3)
LP_LAGS = 1
LP_COV_TYPE = "cluster"


def lp_lattice_demo_data() -> dict:
    """Seeded 16-unit x 26-period panel with an own and a neighbour shock."""
    from puremacro.spatial import contiguity_weights

    W = contiguity_weights(labelled_lattice_neighbours(LP_SIDE))
    n, T = W.n, LP_T
    rng = np.random.default_rng(LP_SEED)
    Wd = W.to_dense()
    xs = rng.standard_normal((n, T))
    mu = rng.standard_normal((n, 1))
    ys = np.cumsum(
        0.8 * xs + 0.4 * (Wd @ xs) + mu + 0.5 * rng.standard_normal((n, T)), axis=1
    )
    index = pd.MultiIndex.from_product([list(W.ids), range(T)], names=["code", "date"])
    frame = pd.DataFrame({"y": ys.ravel(), "x": xs.ravel()}, index=index)
    return {"W": W, "frame": frame}


def _spatial_lp_without_spillover() -> dict:
    from puremacro.spatial import spatial_lp

    d = lp_lattice_demo_data()
    res = spatial_lp(d["frame"], "y", "x", d["W"], horizons=LP_HORIZONS,
                     n_lags=LP_LAGS, spillover_orders=(), cov_type=LP_COV_TYPE,
                     cumulative=False, cross_horizon=False)
    return {
        "beta": res["beta_direct"].to_numpy(dtype=float),
        "se": res["se_direct"].to_numpy(dtype=float),
        "lo": res["lo_direct"].to_numpy(dtype=float),
        "hi": res["hi_direct"].to_numpy(dtype=float),
    }


def _panel_lp_reference() -> dict:
    from puremacro.lp import panel_lp

    d = lp_lattice_demo_data()
    res = panel_lp(d["frame"], "y", "x", horizons=LP_HORIZONS, n_lags=LP_LAGS,
                   cov_type=LP_COV_TYPE)
    return {
        "beta": res["beta"].to_numpy(dtype=float),
        "se": res["se"].to_numpy(dtype=float),
        "lo": res["lo"].to_numpy(dtype=float),
        "hi": res["hi"].to_numpy(dtype=float),
    }


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
    ValidationCase(
        id="spatial.concentrated_loglik_at_zero_equals_ols",
        subsystem="spatial",
        title="The SAR and SEM concentrated log-likelihoods at zero equal the Gaussian OLS log-likelihood",
        title_es="Las log-verosimilitudes concentradas de SAR y SEM en cero igualan la log-verosimilitud gaussiana de MCO",
        mechanism=Mechanism.ANALYTICAL,
        compute=_sar_sem_loglik_at_zero,
        reference=_gaussian_ols_loglik_reference,
        tol=Tol.TIGHT,
        citation=(
            "Ord, J.K. (1975), 'Estimation methods for models of spatial interaction', "
            "JASA 70(349), 120-126, and Anselin, L. (1988), Spatial Econometrics, Kluwer, "
            "ch. 6: ln L_c(rho) = -(n/2)(ln 2pi + 1) + ln|I - rho W| - (n/2) ln(SSR(rho)/n), "
            "so at rho = 0 the Jacobian vanishes and SSR(0) is the OLS residual sum of squares."
        ),
        notes=(
            "Checks SpatialModelResult.loglik_at_zero for both the lag and the error "
            "concentration, each read off the objective the optimiser actually maximised, "
            "against a numpy lstsq OLS fit. 6 x 6 rook lattice, n = 36, k = 3."
        ),
    ),
    ValidationCase(
        id="spatial.sdm_equals_sar_on_augmented_design",
        subsystem="spatial",
        title="SDM equals SAR run on the augmented design [X, W X]: coefficients, covariance and log-likelihood",
        title_es="El SDM iguala al SAR estimado sobre el diseño ampliado [X, W X]: coeficientes, covarianza y log-verosimilitud",
        mechanism=Mechanism.INTERNAL,
        compute=_sdm_fit,
        reference=_sar_on_augmented_design_reference,
        tol=Tol.TIGHT,
        citation=(
            "LeSage, J.P. & Pace, R.K. (2009), Introduction to Spatial Econometrics, CRC, "
            "sec. 2.3: the spatial Durbin model y = rho W y + X beta + W X_d theta + eps is "
            "the spatial autoregressive model on Z = [X, W X_d], so the same concentrated "
            "likelihood and the same Anselin (1988, eq. 6.14) information matrix apply."
        ),
        notes=(
            "Compares the full parameter vector, the (k+2, k+2) covariance including the "
            "rho and sigma2 rows, rho, sigma2 and the maximised log-likelihood. The "
            "augmented design is built with the same sparse W @ X the estimator uses."
        ),
    ),
    ValidationCase(
        id="spatial.slx_equals_ols_on_augmented_design",
        subsystem="spatial",
        title="SLX equals ols_spatial on [X, W X] — it carries no spatial parameter",
        title_es="El SLX iguala a ols_spatial sobre [X, W X]: no lleva parámetro espacial",
        mechanism=Mechanism.INTERNAL,
        compute=_slx_fit,
        reference=_ols_on_augmented_design_reference,
        tol=Tol.TIGHT,
        citation=(
            "Halleck Vega, S. & Elhorst, J.P. (2015), 'The SLX model', Journal of Regional "
            "Science 55(3), 339-363, and LeSage & Pace (2009), sec. 2.4: the spatially "
            "lagged X specification has no endogenous spatial lag, so it is ordinary least "
            "squares on [X, W X_d] with the dof-corrected variance e'e/(n - k)."
        ),
        notes=(
            "Coefficients, standard errors, sigma2 and the log-likelihood, plus "
            "SpatialModelResult.pseudo_r2 against SpatialOLSResult.r2 (equal for an OLS "
            "fit with an intercept, where R^2 = corr(y, fitted)^2)."
        ),
    ),
    ValidationCase(
        id="spatial.effects_equal_dense_impact_matrix",
        subsystem="spatial",
        title="LeSage-Pace direct / indirect / total impacts equal the dense (I - rho W)^-1 (I beta_r + W theta_r)",
        title_es="Los impactos directo / indirecto / total de LeSage-Pace igualan la matriz densa (I - rho W)^-1 (I beta_r + W theta_r)",
        mechanism=Mechanism.INTERNAL,
        compute=_lesage_pace_effects,
        reference=_dense_impact_matrix_reference,
        tol=Tol.TIGHT,
        citation=(
            "LeSage, J.P. & Pace, R.K. (2009), Introduction to Spatial Econometrics, CRC, "
            "sec. 2.7: S_r(W) = (I - rho W)^-1 (I beta_r + W theta_r), with the average "
            "direct impact tr(S_r)/n, the average total impact 1'S_r 1/n and the indirect "
            "impact their difference."
        ),
        notes=(
            "spatial_effects reaches those summaries through four scalars (tr A^-1, "
            "tr(A^-1 W) from the spectrum; 1'A^-1 1, 1'A^-1 W 1 from the exact power "
            "series); the reference forms S_r densely with numpy. Fixed beta, theta and "
            "rho = 0.4, so nothing here depends on an estimator; the constant is dropped "
            "from both sides."
        ),
    ),
    ValidationCase(
        id="spatial.panel_null_loglik_equals_two_way_fe",
        subsystem="spatial",
        title="The spatial panel's log-likelihood at rho = 0 equals the two-way fixed-effects Gaussian log-likelihood",
        title_es="La log-verosimilitud del panel espacial en rho = 0 iguala la log-verosimilitud gaussiana de efectos fijos bidireccionales",
        mechanism=Mechanism.INTERNAL,
        compute=_spatial_panel_null_loglik,
        reference=_two_way_fe_loglik_reference,
        tol=Tol.TIGHT,
        citation=(
            "Lee, L.-F. & Yu, J. (2010), 'Estimation of spatial autoregressive panel data "
            "models with fixed effects', Journal of Econometrics 154(2), 165-185: the "
            "transformation approach runs on doubly demeaned data with the effective counts "
            "N* = (n-1)(T-1), so at rho = 0 the concentrated likelihood is the Gaussian "
            "log-likelihood of the two-way within residuals at that divisor."
        ),
        notes=(
            "Reference is puremacro.lp._panel_helpers.two_way_fe_within (a different "
            "estimator in the package, reached by iterative demeaning) plus the closed-form "
            "Gaussian log-likelihood. Also pins n_eff = (n-1)(T-1). 4 x 4 lattice, T = 12."
        ),
    ),
    ValidationCase(
        id="spatial.lee_yu_correction_rescales_sigma2_only",
        subsystem="spatial",
        title="Under individual effects the Lee-Yu correction leaves beta and rho untouched and rescales sigma^2 by T/(T-1)",
        title_es="Con efectos individuales la corrección de Lee-Yu deja intactos beta y rho y reescala sigma^2 por T/(T-1)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_lee_yu_corrected,
        reference=_lee_yu_closed_form_reference,
        tol=Tol.TIGHT,
        citation=(
            "Lee, L.-F. & Yu, J. (2010), Journal of Econometrics 154(2), 165-185, and Lee & "
            "Yu (2010), 'Some recent developments in spatial panel data models', Regional "
            "Science and Urban Economics 40(5), 255-271: theta_corrected = theta_hat - I^-1 a "
            "with a = (0_k, -tr(G), -n/(2 sigma^2)) solves to (0_k, 0, -sigma^2/(T-1)) when "
            "only the individual effects are concentrated out."
        ),
        notes=(
            "The reference is the SAME fit with bias_correction='none' pushed through the "
            "closed form; what is pinned is the correction, not the estimator. sigma_hat^2 = "
            "SSR/(nT) is rescaled to SSR/(n(T-1)) — exactly the transformation divisor — "
            "while every beta and rho entry is left alone. 4 x 4 lattice, T = 12."
        ),
    ),
    ValidationCase(
        id="spatial.spatial_lp_without_spillover_equals_panel_lp",
        subsystem="spatial",
        title="spatial_lp with no spillover order reproduces panel_lp horizon by horizon",
        title_es="spatial_lp sin órdenes de desbordamiento reproduce a panel_lp horizonte por horizonte",
        mechanism=Mechanism.INTERNAL,
        compute=_spatial_lp_without_spillover,
        reference=_panel_lp_reference,
        tol=Tol.EXACT,
        citation=(
            "Jorda, O. (2005), 'Estimation and inference of impulse responses by local "
            "projections', AER 95(1), 161-182: dropping the spatially lagged shock leaves "
            "the plain two-way fixed-effects panel local projection, and no finite-sample "
            "correction is applied on either path."
        ),
        notes=(
            "spillover_orders=() with cov_type='cluster'; compares beta, the "
            "cluster-by-entity standard error and both confidence-band endpoints at "
            "h = 0..3. 4 x 4 lattice, T = 26, one lag. EXACT because the two routes build "
            "the identical within-transformed design."
        ),
    ),
]
