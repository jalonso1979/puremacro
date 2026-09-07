"""Spatial local projections — ``puremacro.spatial.lp``.

The tests pin behaviour rather than restating the implementation: the exact
reduction to :func:`puremacro.lp.panel_lp`, closed-form covariance identities,
recovery of planted parameters on seeded DGPs, the invariances that make the
"direct / indirect" reading honest (in particular that ``total`` is NOT an
aggregate multiplier), and the adversarial geometries.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from puremacro.lp import panel_lp
from puremacro.lp._panel_helpers import two_way_fe_within
from puremacro.spatial import (
    SpatialWeights,
    contiguity_weights,
    economic_weights,
    knn_weights,
    spatial_hac_panel_meat,
)
from puremacro.spatial.lp import (
    SpatialLPResult,
    _focal_influence,
    _make_conley_cov_fn,
    _panel_score_grid,
    _space_time_cross_meat,
    higher_order_weights,
    spatial_lag_panel,
    spatial_lp,
)

IDS = [f"R{i:02d}" for i in range(12)]
T_DEFAULT = 60


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
def _coords(n: int = 12, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"lat": rng.uniform(30.0, 45.0, n), "lon": rng.uniform(-110.0, -75.0, n)},
        index=IDS[:n],
    )


def _panel(W, *, beta=0.5, gamma=0.3, sigma=0.4, T=T_DEFAULT, seed=0,
           kappa=0.0, with_control=False):
    """Level panel whose one-step change is ``beta x + gamma Wx (+ kappa xbar_t)``."""
    rng = np.random.default_rng(seed)
    ids = list(W.ids)
    n = len(ids)
    x = rng.standard_normal((n, T))
    wx = np.asarray(W.W @ x)
    e = sigma * rng.standard_normal((n, T))
    z = rng.standard_normal((n, T))
    dy = beta * x + gamma * wx + e + kappa * x.mean(axis=0)[None, :]
    if with_control:
        dy = dy + 0.25 * z
    y = np.cumsum(dy, axis=1)
    cols = {
        "code": np.repeat(ids, T),
        "date": np.tile(np.arange(T), n),
        "x": x.ravel(),
        "y": y.ravel(),
    }
    if with_control:
        cols["z"] = z.ravel()
    return pd.DataFrame(cols).set_index(["code", "date"])


@pytest.fixture(scope="module")
def knn3():
    return knn_weights(_coords(), 3)


@pytest.fixture(scope="module")
def demo(knn3):
    return _panel(knn3, seed=1, with_control=True)


def _zero_weights(ids):
    return SpatialWeights(sp.csr_matrix((len(ids), len(ids))), tuple(ids), "custom", False)


# ---------------------------------------------------------------------------
# An INDEPENDENT re-implementation of one horizon's regression.
#
# Nothing below calls ``spatial_lp``'s own helpers (``spatial_lag_panel``,
# ``_focal_influence``, ``_panel_score_grid``, ``_space_time_cross_meat``,
# ``_cross_horizon_cov``): the spatial lag is a per-period ``W @ x`` loop and
# the sandwiches are rebuilt from ``X_within`` / ``residuals`` with the
# package's public HAC primitives.  Tests that compare ``spatial_lp`` output
# against these are therefore not comparing the module to itself.
#
# OLS coefficients and their sandwich covariance are equivariant under column
# permutation, so the reference need only match the SET of regressors (and
# hence the dropna sample), not ``spatial_lp``'s internal column order.  The
# two focal columns are first here as they are there.
# ---------------------------------------------------------------------------
def _wx_by_loop(df, col, W):
    """Per-period spatial lag from an explicit ``W @ x`` loop."""
    ids = list(W.ids)
    vals = {}
    for t, block in df.groupby(level="date"):
        s = block.droplevel("date")[col].reindex(ids).to_numpy(dtype=float)
        for unit, v in zip(ids, np.asarray(W.W @ s)):
            vals[(unit, t)] = v
    return pd.Series([vals[k] for k in df.index], index=df.index)


def _reference_fit(df, W, h, n_lags, *, controls=(), lag_spillover=True):
    """``two_way_fe_within`` on horizon ``h`` of the spatial-LP design."""
    d = df.copy()
    d["__wx"] = _wx_by_loop(d, "x", W)
    cols = ["x", "__wx"]
    for L in range(1, n_lags + 1):
        g = d.groupby(level="code")
        d[f"__xL{L}"] = g["x"].shift(L)
        d[f"__yL{L}"] = g["y"].shift(L)
        cols += [f"__xL{L}", f"__yL{L}"]
        if lag_spillover:
            d[f"__wxL{L}"] = d.groupby(level="code")["__wx"].shift(L)
            cols.append(f"__wxL{L}")
        for c in controls:
            d[f"__{c}L{L}"] = g[c].shift(L)
            cols.append(f"__{c}L{L}")
    cols += list(controls)
    g = d.groupby(level="code")
    d["__dy"] = g["y"].shift(-h) - g["y"].shift(1)
    return two_way_fe_within(d[["__dy"] + cols].dropna(), y_col="__dy", x_cols=cols)


def _reference_cov(out, cov_type, *, coords=None, cutoff_km=None, time_lags=None):
    """The per-horizon sandwich, rebuilt outside ``puremacro.spatial.lp``."""
    A = out["XtX_inv"]
    if cov_type == "cluster":
        score = out["X_within"] * out["residuals"][:, None]
        keys = out["entity_keys"]
        S = np.zeros((score.shape[1], score.shape[1]))
        for e in np.unique(keys):
            s_e = score[keys == e].sum(axis=0)
            S += np.outer(s_e, s_e)
        return A @ S @ A
    T = len(np.unique(out["time_keys"]))
    L = int(time_lags) if time_lags is not None else max(
        1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    if cov_type == "driscoll-kraay":
        from puremacro.inference.dk import driscoll_kraay
        S = driscoll_kraay(out["X_within"] * out["residuals"][:, None],
                           out["time_keys"], lags=L)
    else:
        S = spatial_hac_panel_meat(out["X_within"], out["residuals"], coords,
                                   out["entity_keys"], out["time_keys"],
                                   cutoff_km, L)
    return A @ S @ A


def _cluster_combo_var(outs, sels):
    """Cluster-by-entity variance of ``sum_m l_m' beta^(m)``.

    ``sum_i (sum_m l_m' A^(m) X_i^(m)' u_i^(m))^2`` — the textbook clustered
    variance of a linear combination that spans several regressions, built
    from each horizon's own within output.
    """
    total = {}
    for out, l in zip(outs, sels):
        infl = (out["X_within"] @ (out["XtX_inv"] @ np.asarray(l, dtype=float))) \
            * out["residuals"]
        keys = out["entity_keys"]
        for e in np.unique(keys):
            total[e] = total.get(e, 0.0) + float(infl[keys == e].sum())
    return float(sum(v * v for v in total.values()))


# ---------------------------------------------------------------------------
# 1. Reduction to panel_lp (the BUILD_RULES proof obligation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cov_type, kw", [
    ("cluster", {}),
    ("driscoll-kraay", {}),
    ("conley", {"cutoff_km": 800.0}),
])
def test_zero_weights_reproduce_panel_lp_bit_for_bit(demo, cov_type, kw):
    """With no spillover regressor the estimator IS ``panel_lp``: identical
    column list, identical dropna set, identical ``two_way_fe_within`` call —
    so beta and se agree exactly, not merely to 1e-12.

    The reduction runs through ``spillover_orders=()`` rather than through a
    zero ``W`` at the default order, because a zero ``W`` with
    ``spillover_orders=(1,)`` raises instead of silently regressing on an
    all-zero column (pinned by the next test, which also checks the error
    names ``spillover_orders=()`` as the escape).  Both halves together are
    the BUILD_RULES obligation.
    """
    coords = _coords()
    if cov_type == "conley":
        kw = dict(kw, coords=coords)
    Wz = _zero_weights(IDS)
    got = spatial_lp(demo, "y", "x", Wz, horizons=range(0, 6), n_lags=2,
                     controls=["z"], spillover_orders=(), cov_type=cov_type,
                     cumulative=False, **kw)
    ref = panel_lp(demo, "y", "x", horizons=range(0, 6), n_lags=2,
                   controls=["z"], cov_type=cov_type, **kw)
    assert np.array_equal(got["beta_direct"].to_numpy(), ref["beta"].to_numpy())
    assert np.array_equal(got["se_direct"].to_numpy(), ref["se"].to_numpy())
    # indirect / total collapse onto direct when there is no spillover
    assert np.array_equal(got["beta_total"].to_numpy(), ref["beta"].to_numpy())
    assert np.allclose(got["beta_indirect"].to_numpy(), 0.0)


def test_zero_weights_with_a_spillover_order_raise_and_name_the_escape(demo):
    Wz = _zero_weights(IDS)
    with pytest.raises(ValueError, match=r"spillover_orders=\(\)"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spatial_lp(demo, "y", "x", Wz, horizons=[0, 1], n_lags=1, cumulative=False)


def test_a_single_island_does_not_raise(demo):
    """Some islands is a warning and a smaller scale, not an error."""
    nb = {IDS[i]: [IDS[i + 1]] for i in range(9)}   # chain over R00..R09
    W = contiguity_weights(nb, ids=IDS)             # R10 and R11 are islands
    with pytest.warns(RuntimeWarning, match="island"):
        res = spatial_lp(demo, "y", "x", W, horizons=[0, 1], n_lags=1, cumulative=False)
    assert np.all(np.isfinite(res["beta_indirect"].to_numpy()))
    assert res.weights_info[1]["n_islands"] == 2
    assert res.total_scale[1] == pytest.approx(10.0 / 12.0)


# ---------------------------------------------------------------------------
# 2. Recovery of planted parameters
# ---------------------------------------------------------------------------
def test_planted_beta_and_gamma_are_recovered(knn3):
    df = _panel(knn3, beta=0.5, gamma=0.3, sigma=0.4, T=200, seed=7)
    res = spatial_lp(df, "y", "x", knn3, horizons=range(0, 3), n_lags=2,
                     cov_type="driscoll-kraay")
    b = float(res["beta_direct"].iloc[0])
    g = float(res.spillover.query("h == 0")["gamma"].iloc[0])
    assert abs(b - 0.5) < 0.05
    assert abs(g - 0.3) < 0.05
    assert abs(b - 0.5) < 3 * float(res["se_direct"].iloc[0])
    assert abs(g - 0.3) < 3 * float(res.spillover.query("h == 0")["se"].iloc[0])


def test_noiseless_dgp_is_recovered_exactly_and_total_is_the_contrast(knn3):
    """No noise, no lags: the within regression fits perfectly, so beta and
    gamma come back to machine precision and ``total = beta + c1 * gamma``."""
    df = _panel(knn3, beta=0.5, gamma=0.3, sigma=0.0, T=40, seed=2)
    res = spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=0,
                     cov_type="cluster", cumulative=False)
    assert float(res["beta_direct"].iloc[0]) == pytest.approx(0.5, abs=1e-10)
    g = float(res.spillover["gamma"].iloc[0])
    assert g == pytest.approx(0.3, abs=1e-10)
    c1 = float(res.total_scale[1])
    assert float(res["beta_total"].iloc[0]) == pytest.approx(0.5 + c1 * g, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. The honest claim about `total` (G5)
# ---------------------------------------------------------------------------
def test_total_is_a_relative_contrast_not_an_aggregate_multiplier(knn3):
    """The time effect absorbs the aggregate response, so an arbitrary common
    component ``kappa * xbar_t`` in the DGP leaves every estimate unchanged.
    A `total` that measured the uniform-shock response could not be invariant
    to kappa (the true uniform response moves by kappa)."""
    a = spatial_lp(_panel(knn3, seed=3, kappa=0.0), "y", "x", knn3,
                   horizons=range(0, 3), n_lags=1, cumulative=False)
    b = spatial_lp(_panel(knn3, seed=3, kappa=5.0), "y", "x", knn3,
                   horizons=range(0, 3), n_lags=1, cumulative=False)
    for col in ("beta_direct", "beta_indirect", "beta_total", "se_total"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), atol=1e-10)
    text = a.summary()
    assert "RELATIVE TO THE PERIOD MEAN" in text
    assert "NOT the response to a uniform shock" in text


def test_no_joint_cross_horizon_test_ships(demo, knn3):
    """G6: nothing chi2-shaped may reach the user, by any route."""
    import puremacro.spatial.lp as mod

    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1)
    assert not hasattr(res, "joint_spillover")
    # no attribute, method, column or metadata key offers a joint/Wald p-value
    for name in dir(res):
        assert "joint" not in name.lower() and "wald" not in name.lower()
    for key in res.metadata:
        assert "joint" not in key.lower() and "wald" not in key.lower()
    for col in list(res.columns) + list(res.spillover.columns) + \
            list(res.cumulative.columns):
        assert "p_value" not in str(col) and "chi2" not in str(col)
    for name in dir(mod):
        assert "joint" not in name.lower() and "wald" not in name.lower()
    assert not any("joint" in n.lower() for n in mod.__all__)
    # the omission is stated, with the number that justifies it
    assert "joint" in res.summary().lower()
    doc = mod.__doc__
    assert "Deliberately omitted" in doc
    assert "0.313" in doc and "0.10" in doc


def test_the_shipped_per_horizon_test_reports_its_own_size_honestly(demo, knn3):
    """G6/G11: the contrast used to justify dropping the joint test may not be
    a single favourable horizon.  The measured per-horizon size varies with
    ``h`` and with ``cov_type``, so the docstring must qualify it and give the
    profile, and ``summary()`` must not quote the joint figure bare."""
    import puremacro.spatial.lp as mod

    doc = mod.__doc__
    omitted = doc.split("Deliberately omitted", 1)[1]
    assert "per-horizon z-test that DOES ship is not exact" in omitted
    # the per-horizon number is attributed to a horizon and a cov_type, and the
    # profile across horizons is given rather than one number
    assert "worst over h" in omitted
    assert "driscoll-kraay" in omitted and "conley" in omitted and "cluster" in omitted
    assert "0.110" in omitted and "0.172" in omitted        # the worst DK cells
    assert "N = 40" in omitted                              # more entities is not a fix
    # summary() states the contrast as a range, not as the single 0.080 figure
    text = spatial_lp(demo, "y", "x", knn3, horizons=[0, 1], n_lags=1,
                      cumulative=False).summary()
    assert "0.31" in text and "0.08-0.11" in text
    assert "0.080" not in text


# ---------------------------------------------------------------------------
# 4. Covariance identities
# ---------------------------------------------------------------------------
def test_total_se_carries_the_off_diagonal_covariance(demo, knn3):
    """``se_total`` is ``sqrt(c' V c)`` on an INDEPENDENTLY rebuilt Driscoll-
    Kraay sandwich — not merely on the block the estimator reported — and the
    cross term matters: adding the two SEs in quadrature is materially wrong."""
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 5), n_lags=2,
                     cov_type="driscoll-kraay")
    c = res.total_scale
    naive_gap = []
    for m, h in enumerate(res.horizon_values):
        out = _reference_fit(demo, knn3, int(h), 2)
        V = _reference_cov(out, "driscoll-kraay")[np.ix_([0, 1], [0, 1])]
        assert float(res["se_total"].iloc[m]) ** 2 == pytest.approx(
            float(c @ V @ c), rel=1e-10)
        naive = np.hypot(res["se_direct"].iloc[m], res["se_indirect"].iloc[m])
        naive_gap.append(abs(naive - res["se_total"].iloc[m]))
        # a sanity check on the reference itself: the off-diagonal is real
        assert abs(V[0, 1]) > 1e-8
    # the off-diagonal term is materially non-zero: adding the SEs is wrong
    assert max(naive_gap) > 1e-3


@pytest.mark.parametrize("cov_type, kw", [
    ("cluster", {}),
    ("driscoll-kraay", {}),
    ("conley", {"cutoff_km": 700.0}),
])
def test_vcov_diagonal_blocks_equal_the_reported_standard_errors(demo, knn3, cov_type, kw):
    """The diagonal blocks of ``.vcov`` (and the reported SEs) equal the
    sandwich of an independently rebuilt horizon regression.

    Comparing ``sqrt(diag(vcov))`` with ``res['se_direct']`` alone would be
    tautological — ``_cross_horizon_cov`` assigns the very block the SE came
    from — so the reference sandwich here is built outside the module, from
    ``two_way_fe_within`` output and the public HAC primitives.
    """
    ref_kw = {}
    if cov_type == "conley":
        kw = dict(kw, coords=_coords())
        ref_kw = {"coords": _coords(), "cutoff_km": 700.0}
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 5), n_lags=2,
                     cov_type=cov_type, **kw)
    nF = len(res.coef_names)
    d = np.sqrt(np.diag(res.vcov))
    for m, h in enumerate(res.horizon_values):
        out = _reference_fit(demo, knn3, int(h), 2)
        # the reference reproduces the point estimates as well
        assert float(res["beta_direct"].iloc[m]) == pytest.approx(
            float(out["beta"][0]), rel=1e-10)
        V = _reference_cov(out, cov_type, **ref_kw)
        se_ref = np.sqrt(np.diag(V))
        assert d[m * nF] == pytest.approx(float(se_ref[0]), rel=1e-10)
        assert d[m * nF + 1] == pytest.approx(float(se_ref[1]), rel=1e-10)
        assert float(res["se_direct"].iloc[m]) == pytest.approx(float(se_ref[0]), rel=1e-10)
        gam = res.spillover[res.spillover["h"] == h]
        assert float(gam["se"].iloc[0]) == pytest.approx(float(se_ref[1]), rel=1e-10)
        # ... and the off-diagonal of the block, which no reported SE exposes
        assert res.vcov[m * nF, m * nF + 1] == pytest.approx(float(V[0, 1]), rel=1e-10)


def test_conley_wide_uniform_cutoff_equals_driscoll_kraay(demo, knn3):
    coords = _coords()
    a = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1,
                   cov_type="conley", coords=coords, cutoff_km=1e9, kernel="uniform")
    b = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1,
                   cov_type="driscoll-kraay")
    np.testing.assert_allclose(a.vcov, b.vcov, atol=1e-10)
    np.testing.assert_allclose(a["se_total"].to_numpy(), b["se_total"].to_numpy(), atol=1e-12)
    # the Bartlett kernel at the same cutoff is NOT Driscoll-Kraay: K = 1 - d/cutoff
    # is 1 - 1e-6, not 1 — close, but not the identity the test above pins.
    c = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1,
                   cov_type="conley", coords=coords, cutoff_km=1e9, kernel="bartlett")
    gap = np.max(np.abs(c["se_total"].to_numpy() - b["se_total"].to_numpy()))
    assert 1e-12 < gap < 1e-5


@pytest.mark.parametrize("cov_type, kw", [
    ("driscoll-kraay", {}),
    ("conley", {"cutoff_km": 700.0}),
])
def test_explicit_time_lags_reach_every_bandwidth_and_the_sandwich(demo, knn3, cov_type, kw):
    """``time_lags=`` is not merely recorded: it must set the per-horizon
    bandwidth, the cross-horizon bandwidth, and the numbers."""
    ref_kw = {}
    if cov_type == "conley":
        kw = dict(kw, coords=_coords())
        ref_kw = {"coords": _coords(), "cutoff_km": 700.0}
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                     cov_type=cov_type, time_lags=6, **kw)
    assert res.cov_options["time_lags"] == 6
    assert set(res.cov_options["bandwidth_by_horizon"].values()) == {6}
    assert res.cov_options["cross_bandwidth"] == 6

    default = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                         cov_type=cov_type, **kw)
    T = int(default["n_periods"].iloc[0])
    auto = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    assert default.cov_options["time_lags"] is None
    assert set(default.cov_options["bandwidth_by_horizon"].values()) == {auto}
    assert auto != 6
    assert not np.allclose(res["se_direct"].to_numpy(),
                           default["se_direct"].to_numpy())

    for m, h in enumerate(res.horizon_values):
        out = _reference_fit(demo, knn3, int(h), 1)
        V6 = _reference_cov(out, cov_type, time_lags=6, **ref_kw)
        Vauto = _reference_cov(out, cov_type, **ref_kw)
        assert float(res["se_direct"].iloc[m]) == pytest.approx(
            float(np.sqrt(V6[0, 0])), rel=1e-10)
        assert float(default["se_direct"].iloc[m]) == pytest.approx(
            float(np.sqrt(Vauto[0, 0])), rel=1e-10)


def test_focal_influence_identity_is_the_sandwich(demo, knn3):
    """``A[F, :] S A[F, :]' = sum_{r,s} K(r,s) psi_r psi_s'`` — the identity the
    whole cross-horizon construction rests on."""
    coords = _coords()
    df = demo.copy()
    lag = spatial_lag_panel(df, "x", knn3, prefix="wx_")
    df["wx"] = lag["wx_x"]
    grouped = df.groupby(level="code")
    df["dy"] = grouped["y"].shift(-2) - grouped["y"].shift(1)
    df["xl"] = grouped["x"].shift(1)
    x_cols = ["x", "wx", "xl", "z"]
    sub = df[["dy"] + x_cols].dropna()
    out = two_way_fe_within(sub, y_col="dy", x_cols=x_cols)
    cov_fn = _make_conley_cov_fn(coords, cutoff_km=650.0, time_lags=3)
    V = cov_fn(out)[np.ix_([0, 1], [0, 1])]
    psi = _focal_influence(out, [0, 1])
    S = spatial_hac_panel_meat(psi, np.ones(psi.shape[0]), coords,
                               out["entity_keys"], out["time_keys"], 650.0, 3)
    np.testing.assert_allclose(S, V, atol=1e-15)


def test_space_time_cross_meat_reduces_to_the_panel_meat_and_to_hc0(demo, knn3):
    coords = _coords()
    df = demo.copy()
    df["wx"] = spatial_lag_panel(df, "x", knn3, prefix="wx_")["wx_x"]
    grouped = df.groupby(level="code")
    df["dy"] = grouped["y"].shift(-1) - grouped["y"].shift(1)
    sub = df[["dy", "x", "wx", "z"]].dropna()
    out = two_way_fe_within(sub, y_col="dy", x_cols=["x", "wx", "z"])
    psi = _focal_influence(out, [0, 1])
    entities = pd.Index(np.unique(out["entity_keys"]))
    periods = pd.Index(np.unique(out["time_keys"]))
    G = _panel_score_grid(psi, out["entity_keys"], out["time_keys"], entities, periods)

    from puremacro.spatial import kernel_matrix, pairwise_distances
    from puremacro.spatial.hac import _entity_coords
    K = kernel_matrix(pairwise_distances(_entity_coords(coords, np.asarray(entities))),
                      600.0, "bartlett")
    got = _space_time_cross_meat(G, G, K, 3)
    ref = spatial_hac_panel_meat(psi, np.ones(psi.shape[0]), coords,
                                 out["entity_keys"], out["time_keys"], 600.0, 3)
    np.testing.assert_allclose(got, ref, atol=1e-14)

    # K = I and L = 0 is the plain heteroskedasticity-robust (HC0) meat
    white = _space_time_cross_meat(G, G, np.eye(len(entities)), 0)
    np.testing.assert_allclose(white, psi.T @ psi, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Cumulative responses
# ---------------------------------------------------------------------------
def test_cumulative_points_are_the_running_sums_of_independent_fits(demo, knn3):
    """``cum_direct`` is the running sum of coefficients estimated OUTSIDE the
    module, so this pins the accumulated object, not ``np.cumsum`` of the
    module's own column."""
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 6), n_lags=2)
    c = res.total_scale
    ref_direct, ref_total = [], []
    for h in res.horizon_values:
        b = _reference_fit(demo, knn3, int(h), 2)["beta"]
        ref_direct.append(float(b[0]))
        ref_total.append(float(c[0] * b[0] + c[1] * b[1]))
    np.testing.assert_allclose(res["beta_direct"].to_numpy(), ref_direct, rtol=1e-10)
    np.testing.assert_allclose(res.cumulative["cum_direct"].to_numpy(),
                               np.cumsum(ref_direct), rtol=1e-10)
    np.testing.assert_allclose(res.cumulative["cum_total"].to_numpy(),
                               np.cumsum(ref_total), rtol=1e-10)
    assert res.cumulative["complete"].all()
    # and it really accumulates: the last row is not the last per-horizon point
    assert abs(float(res.cumulative["cum_direct"].iloc[-1])
               - float(res["beta_direct"].iloc[-1])) > 1e-3


def test_cumulative_se_is_the_quadratic_form_not_a_sum_of_variances(demo, knn3):
    """Under cluster errors the cumulative variance has a closed form —
    ``sum_i (sum_m l_m' A^(m) X_i' u_i)^2`` over the per-entity influence of
    every horizon — which is computed here from independent fits.  It pins the
    OFF-DIAGONAL blocks of ``.vcov``, which a ``r' vcov r`` re-run cannot.
    """
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 6), n_lags=2,
                     cov_type="cluster")
    c = res.total_scale
    outs = [_reference_fit(demo, knn3, int(h), 2) for h in res.horizon_values]
    m = len(res) - 1
    sels = []
    for out in outs[:m + 1]:
        l = np.zeros(out["X_within"].shape[1])
        l[:len(c)] = c
        sels.append(l)
    manual = float(np.sqrt(_cluster_combo_var(outs[:m + 1], sels)))
    assert float(res.cumulative["se_total"].iloc[m]) == pytest.approx(manual, rel=1e-9)
    # a single horizon is the ordinary clustered SE of direct + indirect
    one = float(np.sqrt(_cluster_combo_var(outs[:1], sels[:1])))
    assert float(res["se_total"].iloc[0]) == pytest.approx(one, rel=1e-9)
    # the overlapping horizons are positively correlated, so the honest band is
    # wider than the one that drops the cross-horizon terms
    per_h = np.sqrt(np.sum(res["se_total"].to_numpy() ** 2))
    assert manual > per_h


def test_a_negative_cumulative_variance_is_nan_and_warned():
    """The documented safeguard for a non-PSD cross-horizon ``C`` (the
    diagonal blocks are each horizon's own sandwich, the off-diagonals come
    from the common grid, so the assembly is not guaranteed PSD).  Built by
    hand because the estimator's own ``C`` is PSD on ordinary panels."""
    from puremacro.spatial.lp import _cumulative_frame, _label_selectors

    nF = 2
    coef = np.array([[0.5, 0.2], [0.4, 0.1]])
    label_sel = _label_selectors((1,), np.array([1.0, 1.0]), nF)
    C = np.eye(2 * nF) * 0.01
    C[0:nF, nF:2 * nF] = -1.0            # a cross-horizon block big enough to
    C[nF:2 * nF, 0:nF] = -1.0            # drive r' C r below zero
    with pytest.warns(RuntimeWarning, match="negative"):
        cum = _cumulative_frame([0, 1], coef, C, label_sel, nF, 1.645, "spatial_lp")
    # h = 0 only touches the PSD diagonal block
    assert float(cum["se_total"].iloc[0]) == pytest.approx(np.sqrt(0.02), rel=1e-12)
    # h = 1 picks up the negative cross term: SE and band are NaN, point survives
    assert np.isnan(cum["se_total"].iloc[1])
    assert np.isnan(cum["lo_total"].iloc[1]) and np.isnan(cum["hi_total"].iloc[1])
    assert float(cum["cum_total"].iloc[1]) == pytest.approx(1.2, abs=1e-12)
    assert np.isnan(cum["se_direct"].iloc[1])


def test_negative_cumulative_variance_surfaces_out_of_spatial_lp(demo, knn3, monkeypatch):
    """...and the warning and the NaN reach the caller, not just the helper."""
    import puremacro.spatial.lp as mod

    real = mod._cross_horizon_cov

    def flipped(blocks, psis, ent_keys, tim_keys, cov_kind, cov_options, nF):
        C, bw = real(blocks, psis, ent_keys, tim_keys, cov_kind, cov_options, nF)
        C = np.asarray(C, dtype=float).copy()
        for m in range(len(blocks)):
            for m2 in range(len(blocks)):
                if m != m2:
                    C[m * nF:(m + 1) * nF, m2 * nF:(m2 + 1) * nF] *= -50.0
        return C, bw

    monkeypatch.setattr(mod, "_cross_horizon_cov", flipped)
    with pytest.warns(RuntimeWarning, match="came out negative"):
        res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1)
    assert np.isnan(res.cumulative["se_total"].iloc[-1])
    assert np.isfinite(res.cumulative["cum_total"].iloc[-1])
    assert np.isfinite(res["se_total"].iloc[-1])       # per-horizon SEs untouched


def test_cumulative_requires_a_contiguous_grid_and_the_cross_horizon_block(demo, knn3):
    with pytest.raises(ValueError, match="contiguous horizon grid"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0, 4, 8], n_lags=1)
    with pytest.raises(ValueError, match="cross_horizon=True"):
        spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                   cross_horizon=False)
    res = spatial_lp(demo, "y", "x", knn3, horizons=[0, 4, 8], n_lags=1, cumulative=False)
    assert res.cumulative is None
    res2 = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                      cumulative=False, cross_horizon=False)
    assert res2.vcov is None and res2.cumulative is None


# ---------------------------------------------------------------------------
# 6. Higher-order weights (Anselin convention)
# ---------------------------------------------------------------------------
def test_higher_order_weights_supports_are_disjoint_and_rows_standardised(knn3):
    W2 = higher_order_weights(knn3, 2)
    assert not np.any(W2.W.diagonal())
    overlap = knn3.W.astype(bool).multiply(W2.W.astype(bool))
    assert overlap.nnz == 0, "W^(2) must not put mass back on first-order neighbours"
    rs = W2.row_sums()
    np.testing.assert_allclose(rs[rs > 0], 1.0, atol=1e-12)
    assert higher_order_weights(knn3, 1) is knn3
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            higher_order_weights(knn3, bad)


def test_higher_order_weights_honours_row_standardize(knn3):
    """G3: an explicit ``row_standardize=`` may not be silently dropped —
    including at ``order=1``, where the function is otherwise the identity."""
    raw = SpatialWeights(
        sp.csr_matrix(np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        ("a", "b", "c"), "custom", False)
    assert not raw.row_standardized
    np.testing.assert_allclose(raw.row_sums(), [2.0, 1.0, 1.0])

    std = higher_order_weights(raw, 1, row_standardize=True)
    assert std is not raw and std.row_standardized
    np.testing.assert_allclose(std.row_sums(), 1.0)
    # None, or a request that already holds, is the identity
    assert higher_order_weights(raw, 1) is raw
    assert higher_order_weights(raw, 1, row_standardize=False) is raw
    assert higher_order_weights(knn3, 1, row_standardize=True) is knn3
    # what cannot be honoured raises instead of being ignored
    with pytest.raises(ValueError, match="order=1"):
        higher_order_weights(knn3, 1, row_standardize=False)

    # order >= 2 honours it in both directions
    W2 = higher_order_weights(knn3, 2, row_standardize=False)
    assert not W2.row_standardized
    rs = W2.row_sums()
    assert not np.allclose(rs[rs > 0], 1.0)
    W2s = higher_order_weights(knn3, 2, row_standardize=True)
    rss = W2s.row_sums()
    np.testing.assert_allclose(rss[rss > 0], 1.0, atol=1e-12)
    assert W2s.W.nnz == W2.W.nnz          # same support, different scaling


def test_zero_lower_orders_false_keeps_the_first_ring_mass(knn3):
    W2 = higher_order_weights(knn3, 2, zero_lower_orders=False)
    overlap = knn3.W.astype(bool).multiply(W2.W.astype(bool))
    assert overlap.nnz > 0
    assert not np.any(W2.W.diagonal())      # the diagonal is zeroed either way


def test_every_spillover_order_gets_its_own_frame_label(demo, knn3):
    """BUILD_RULES: labels are ``direct``, ``indirect`` (``indirect2``, ... for
    the higher orders) and ``total``.  Each order's coefficient must be
    reachable through the LPResult layout — ``.point``, ``.labels``, ``.plot``
    — not only through the separate ``.spillover`` frame."""
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                     spillover_orders=(1, 2))
    assert res.labels == ["direct", "indirect", "indirect2", "indirect_all", "total"]
    expected = ["h"] + [f"{b}_{lab}" for lab in res.labels
                        for b in ("beta", "se", "t", "lo", "hi")] + \
        ["n_obs", "n_entities", "n_periods"]
    assert list(res.columns) == expected
    assert list(res.point.columns) == res.labels
    assert len(res.coef_names) == 3
    assert res.vcov.shape == (3 * 3, 3 * 3)
    assert set(res.spillover["order"]) == {1, 2}

    c = res.total_scale
    for m, h in enumerate(res.horizon_values):
        gam = res.spillover[res.spillover["h"] == h].sort_values("order")["gamma"].to_numpy()
        # each label carries its OWN order, scaled by that order's c_p
        assert float(res["beta_indirect"].iloc[m]) == pytest.approx(
            float(c[1] * gam[0]), abs=1e-12)
        assert float(res["beta_indirect2"].iloc[m]) == pytest.approx(
            float(c[2] * gam[1]), abs=1e-12)
        # and the aggregate is their sum, with `total` on top of `direct`
        assert float(res["beta_indirect_all"].iloc[m]) == pytest.approx(
            float(c[1] * gam[0] + c[2] * gam[1]), abs=1e-12)
        assert float(res["beta_total"].iloc[m]) == pytest.approx(
            float(res["beta_direct"].iloc[m] + res["beta_indirect_all"].iloc[m]),
            abs=1e-12)
        # the second ring is a real, separately estimated coefficient
        assert np.isfinite(res["se_indirect2"].iloc[m])
    assert not np.allclose(res["beta_indirect"].to_numpy(),
                           res["beta_indirect2"].to_numpy())
    # the cumulative frame and the summary carry the same labels
    for lab in res.labels:
        assert f"cum_{lab}" in res.cumulative.columns
        assert f"se_{lab}" in res.cumulative.columns
    assert "indirect2" in res.summary()


def test_indirect_labels_are_named_by_the_order_not_the_position(demo, knn3):
    """``spillover_orders=(1, 3)`` labels the second block ``indirect3``: the
    label names the RING, so a frame cannot silently mislabel order 3 as the
    second-order response."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                         spillover_orders=(1, 3), cumulative=False)
    assert res.labels == ["direct", "indirect", "indirect3", "indirect_all", "total"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        only2 = spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                           spillover_orders=(2,), cumulative=False)
    # a single order is already the aggregate: no `indirect_all` column
    assert only2.labels == ["direct", "indirect2", "total"]
    g = float(only2.spillover["gamma"].iloc[0])
    assert float(only2["beta_indirect2"].iloc[0]) == pytest.approx(
        float(only2.total_scale[1] * g), abs=1e-12)
    assert float(only2["beta_total"].iloc[0]) == pytest.approx(
        float(only2["beta_direct"].iloc[0] + only2["beta_indirect2"].iloc[0]), abs=1e-12)


def test_higher_order_plot_draws_one_panel_per_label(demo, knn3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1,
                     spillover_orders=(1, 2))
    fig = res.plot()
    assert len(fig.axes) == 5
    assert "order 2" in fig.axes[2].get_title()
    plt.close(fig)
    fig = res.plot(kind="cumulative", components=("indirect2",))
    assert len(fig.axes) == 1
    plt.close(fig)
    plt.close("all")


# ---------------------------------------------------------------------------
# 7. total_scale is resolved once and used everywhere
# ---------------------------------------------------------------------------
def test_total_scale_unit_versus_row_sum_on_an_islanded_w(demo):
    nb = {IDS[i]: [IDS[i + 1]] for i in range(9)}
    W = contiguity_weights(nb, ids=IDS)          # two islands -> mean row sum 10/12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row = spatial_lp(demo, "y", "x", W, horizons=range(0, 3), n_lags=1)
        unit = spatial_lp(demo, "y", "x", W, horizons=range(0, 3), n_lags=1,
                          total_scale="unit")
    assert row.total_scale[1] == pytest.approx(10 / 12)
    assert unit.total_scale[1] == 1.0
    gam = unit.spillover["gamma"].to_numpy()
    np.testing.assert_allclose(unit["beta_total"].to_numpy(),
                               unit["beta_direct"].to_numpy() + gam, atol=1e-12)
    assert not np.allclose(row["beta_total"].to_numpy(), unit["beta_total"].to_numpy())
    # the scale reaches the standard errors and the cumulative frame too
    assert not np.allclose(row["se_total"].to_numpy(), unit["se_total"].to_numpy())
    assert not np.allclose(row.cumulative["cum_indirect"].to_numpy(),
                           unit.cumulative["cum_indirect"].to_numpy())


def test_total_scale_accepts_a_float_and_a_per_order_sequence(demo, knn3):
    a = spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                   spillover_orders=(1, 2), total_scale=2.0, cumulative=False)
    np.testing.assert_allclose(a.total_scale, [1.0, 2.0, 2.0])
    b = spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                   spillover_orders=(1, 2), total_scale=[0.5, 0.25], cumulative=False)
    np.testing.assert_allclose(b.total_scale, [1.0, 0.5, 0.25])
    with pytest.raises(ValueError, match="total_scale"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                   spillover_orders=(1, 2), total_scale=[0.5], cumulative=False)


def test_non_row_standardised_weights_warn_about_the_reading(demo):
    nb = {IDS[i]: [IDS[i - 1], IDS[(i + 1) % 12]] for i in range(12)}
    W = contiguity_weights(nb, ids=IDS, row_standardize=False)   # raw counts, all 2
    W = SpatialWeights((sp.diags(np.linspace(1.0, 2.0, 12)) @ W.W).tocsr(),
                       W.ids, "contiguity", False)
    with pytest.warns(RuntimeWarning, match="row sums"):
        res = spatial_lp(demo, "y", "x", W, horizons=[0, 1], n_lags=1, cumulative=False)
    assert res.total_scale[1] == pytest.approx(float(W.row_sums().mean()))
    assert any("WEIGHTED SUM" in n for n in res.notes)


# ---------------------------------------------------------------------------
# 8. spatial_lag_panel
# ---------------------------------------------------------------------------
def test_spatial_lag_panel_matches_a_per_period_lag_loop_on_an_unbalanced_panel(knn3):
    """The unstack/restack round trip must not silently return all-NaN (the
    level order of ``.stack()`` is the reverse of the panel's)."""
    df = _panel(knn3, seed=4, T=15)
    df = df.drop(index=[(IDS[2], 3), (IDS[7], 11), (IDS[7], 12)])
    got = spatial_lag_panel(df, "x", knn3, prefix="W_")["W_x"]
    assert got.notna().any(), "the spatial lag came back empty"
    ref = {}
    for t, block in df.groupby(level="date"):
        s = block.droplevel("date")["x"].reindex(list(knn3.ids))
        lagged = pd.Series(np.asarray(knn3.W @ s.to_numpy(dtype=float)), index=list(knn3.ids))
        for unit, v in lagged.items():
            ref[(unit, t)] = v
    expect = pd.Series({k: ref[k] for k in df.index})
    np.testing.assert_allclose(got.to_numpy(), expect.to_numpy(), equal_nan=True)


def test_spatial_lag_panel_renormalize_is_the_observed_neighbour_mean():
    W = contiguity_weights({"a": ["b", "c"], "b": ["a"], "c": ["a"]})
    idx = pd.MultiIndex.from_product([["a", "b", "c"], [0, 1]], names=["code", "date"])
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}, index=idx)
    prop = spatial_lag_panel(df, "x", W)["W_x"].tolist()
    assert np.isnan(prop[0]) and prop[1] == pytest.approx(5.0)
    renorm = spatial_lag_panel(df, "x", W, unbalanced="renormalize")["W_x"].tolist()
    assert renorm[0] == pytest.approx(5.0)      # only c observed at t=0 -> its value
    assert renorm[1] == pytest.approx(0.5 * 4.0 + 0.5 * 6.0)
    with pytest.raises(ValueError, match="unbalanced"):
        spatial_lag_panel(df, "x", W, unbalanced="drop")


def test_spatial_lag_panel_checks_both_directions_of_the_id_match():
    """The docstring requires every entity to be an id AND every id to be an
    entity.  A panel entity with no row in ``W`` has no spatial lag at all;
    returning a silent all-NaN column for it would drop it from every
    downstream regression without saying so (G3)."""
    W = contiguity_weights({"a": ["b"], "b": ["a", "c"], "c": ["b"]})
    idx = pd.MultiIndex.from_product([["a", "b", "c", "d"], [0, 1]],
                                     names=["code", "date"])
    df = pd.DataFrame({"x": np.arange(8.0)}, index=idx)
    with pytest.raises(KeyError, match="absent from the weights"):
        spatial_lag_panel(df, "x", W)
    # the message names the offender
    with pytest.raises(KeyError, match="'d'"):
        spatial_lag_panel(df, "x", W)
    # the mirror direction still raises
    idx2 = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=["code", "date"])
    with pytest.raises(KeyError, match="absent from the panel"):
        spatial_lag_panel(pd.DataFrame({"x": np.arange(4.0)}, index=idx2), "x", W)
    # an exact match is fine
    idx3 = pd.MultiIndex.from_product([["a", "b", "c"], [0, 1]], names=["code", "date"])
    got = spatial_lag_panel(pd.DataFrame({"x": np.arange(6.0)}, index=idx3), "x", W)
    assert got["W_x"].notna().all()


def test_spatial_lag_panel_names_a_missing_column(knn3):
    """The panel shape, the MultiIndex and the id match are all checked with
    self-naming messages; an unknown ``columns=`` entry used to fall through to
    ``df[col]`` and surface as a bare ``KeyError: 'nope'`` with no mention of
    the function or the available columns."""
    idx = pd.MultiIndex.from_product([IDS, [0, 1]], names=["code", "date"])
    df = pd.DataFrame({"x": np.arange(float(len(idx))),
                       "z": np.arange(float(len(idx)))}, index=idx)
    with pytest.raises(KeyError) as ei:
        spatial_lag_panel(df, "nope", knn3)
    msg = str(ei.value)
    assert "spatial_lag_panel" in msg and "nope" in msg
    assert "'x'" in msg and "'z'" in msg          # the available columns
    # a sequence reports every offender at once, not just the first
    with pytest.raises(KeyError) as ei2:
        spatial_lag_panel(df, ["x", "nope", "alsonope"], knn3)
    assert "nope" in str(ei2.value) and "alsonope" in str(ei2.value)
    # the happy path is untouched
    assert list(spatial_lag_panel(df, ["x", "z"], knn3).columns) == ["W_x", "W_z"]


def test_higher_order_weights_names_a_non_numeric_order(knn3):
    """``float(order)`` on a non-numeric used to escape as a bare
    ``could not convert string to float`` / ``float() argument must be`` with
    no mention of ``higher_order_weights``."""
    for bad in ("two", None, [1, 2], object()):
        with pytest.raises(ValueError) as ei:
            higher_order_weights(knn3, bad)
        msg = str(ei.value)
        assert "higher_order_weights" in msg and "positive integer" in msg
    # numeric spellings that already worked still work
    assert higher_order_weights(knn3, 2.0).kind == higher_order_weights(knn3, 2).kind
    assert higher_order_weights(knn3, "1") is knn3


def test_islands_get_a_zero_lag_not_a_nan():
    W = contiguity_weights({"a": ["b"], "b": ["a"], "c": []}, ids=["a", "b", "c"])
    idx = pd.MultiIndex.from_product([["a", "b", "c"], [0]], names=["code", "date"])
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
    for policy in ("propagate", "renormalize"):
        got = spatial_lag_panel(df, "x", W, unbalanced=policy)["W_x"]
        assert got.loc[("c", 0)] == 0.0


def test_unbalanced_policy_reaches_the_estimator(knn3):
    df = _panel(knn3, seed=6, T=40)
    holes = [(IDS[0], t) for t in range(5, 9)]
    df = df.drop(index=holes)
    with pytest.warns(RuntimeWarning, match="incomplete neighbourhood"):
        a = spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=1, cumulative=False)
    b = spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=1, cumulative=False,
                   unbalanced="renormalize")
    assert int(b["n_obs"].iloc[0]) > int(a["n_obs"].iloc[0])
    with pytest.raises(ValueError, match="unbalanced"):
        spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=1, unbalanced="drop")


# ---------------------------------------------------------------------------
# 9. Specification flags
# ---------------------------------------------------------------------------
def test_flags_add_the_expected_regressors_without_moving_the_focal_block(demo, knn3):
    base = spatial_lp(demo, "y", "x", knn3, horizons=[1], n_lags=2, controls=["z"],
                      lag_spillover=False, cumulative=False)
    lagged = spatial_lp(demo, "y", "x", knn3, horizons=[1], n_lags=2, controls=["z"],
                        lag_spillover=True, cumulative=False)
    wy = spatial_lp(demo, "y", "x", knn3, horizons=[1], n_lags=2, controls=["z"],
                    lag_spillover=False, spatial_lag_y=True, cumulative=False)
    wc = spatial_lp(demo, "y", "x", knn3, horizons=[1], n_lags=2, controls=["z"],
                    lag_spillover=False, spatial_lag_controls=True, cumulative=False)
    n0 = base.spec["n_regressors"]
    assert lagged.spec["n_regressors"] == n0 + 2      # n_lags * n_orders
    assert wy.spec["n_regressors"] == n0 + 2
    assert wc.spec["n_regressors"] == n0 + 3          # n_lags lags + contemporaneous
    for r in (lagged, wy, wc):
        assert r.coef_names == base.coef_names
    assert float(lagged["beta_direct"].iloc[0]) != float(base["beta_direct"].iloc[0])


def test_spatial_lag_controls_equals_passing_the_lagged_column_as_a_control(demo, knn3):
    """OLS is invariant to column order, so building ``W z`` internally must
    agree with handing the same column in as an ordinary control."""
    df = demo.copy()
    df["wz"] = spatial_lag_panel(df, "z", knn3, prefix="W_")["W_z"]
    a = spatial_lp(demo, "y", "x", knn3, horizons=[0, 1], n_lags=1, controls=["z"],
                   spatial_lag_controls=True, cumulative=False)
    b = spatial_lp(df, "y", "x", knn3, horizons=[0, 1], n_lags=1, controls=["z", "wz"],
                   cumulative=False)
    np.testing.assert_allclose(a["beta_direct"].to_numpy(), b["beta_direct"].to_numpy(),
                               atol=1e-11)
    np.testing.assert_allclose(a["beta_indirect"].to_numpy(),
                               b["beta_indirect"].to_numpy(), atol=1e-11)


def test_spatial_flags_without_a_spillover_order_raise(demo, knn3):
    for flag in ("spatial_lag_y", "spatial_lag_controls"):
        with pytest.raises(ValueError, match="requires at least one spillover order"):
            spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1, controls=["z"],
                       spillover_orders=(), cumulative=False, **{flag: True})


# ---------------------------------------------------------------------------
# 10. Validation and degenerate shapes
# ---------------------------------------------------------------------------
def test_horizon_and_order_validation(demo, knn3):
    with pytest.raises(ValueError, match="strictly increasing"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0, 2, 1], n_lags=1, cumulative=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        spatial_lp(demo, "y", "x", knn3, horizons=[1, 1], n_lags=1, cumulative=False)
    with pytest.raises(ValueError, match="horizons is empty"):
        spatial_lp(demo, "y", "x", knn3, horizons=[], n_lags=1, cumulative=False)
    for bad in ((0,), (1, 1), (1.5,), (-2,)):
        with pytest.raises(ValueError, match="spillover_orders"):
            spatial_lp(demo, "y", "x", knn3, horizons=[0], n_lags=1,
                       spillover_orders=bad, cumulative=False)


def test_degenerate_panel_shapes(knn3):
    df = _panel(knn3, seed=8, T=6)
    one_period = df.loc[(slice(None), [0]), :]
    with pytest.raises(ValueError, match=r"1 period\(s\)"):
        spatial_lp(one_period, "y", "x", knn3, horizons=[0], n_lags=0, cumulative=False)
    one_entity = df.loc[([IDS[0]], slice(None)), :]
    with pytest.raises(ValueError, match=r"1 entit\(ies\) \(2 needed\)"):
        spatial_lp(one_entity, "y", "x", knn3, horizons=[0], n_lags=0, cumulative=False)
    dup = pd.concat([df, df.iloc[:4]])
    with pytest.raises(ValueError, match="duplicated"):
        spatial_lp(dup, "y", "x", knn3, horizons=[0], n_lags=0, cumulative=False)


def test_weights_ids_must_match_the_panel(demo, knn3):
    trimmed = demo.drop(index=IDS[-1], level="code")
    with pytest.raises(ValueError, match="rebuild the weights"):
        spatial_lp(trimmed, "y", "x", knn3, horizons=[0], n_lags=1, cumulative=False)
    small = knn_weights(_coords().iloc[:8], 3)
    with pytest.raises(KeyError, match="missing"):
        spatial_lp(demo, "y", "x", small, horizons=[0], n_lags=1, cumulative=False)


def test_generated_column_names_may_not_collide(demo, knn3):
    df = demo.copy()
    df["__W1_x__"] = 1.0
    with pytest.raises(ValueError, match="collide"):
        spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=1, cumulative=False)


def test_constant_regressor_raises_naming_the_design_columns(demo, knn3):
    df = demo.copy()
    df["x"] = 1.0
    with pytest.raises(np.linalg.LinAlgError, match="Column labels in design order"):
        spatial_lp(df, "y", "x", knn3, horizons=[0], n_lags=1, cumulative=False)


def test_complete_graph_is_perfectly_collinear_after_the_time_demeaning(knn3):
    """With a row-standardised complete graph, (Wx)_i - mean_i(Wx) = -x_i/(n-1)
    exactly, so the spillover column is not separable from the own shock."""
    ids = IDS[:5]
    flows = pd.DataFrame(1.0 - np.eye(5), index=ids, columns=ids)
    W = economic_weights(flows)
    df = _panel(W, seed=9, T=30)
    with pytest.raises(np.linalg.LinAlgError, match="singular within design"):
        spatial_lp(df, "y", "x", W, horizons=[0], n_lags=0, cumulative=False)


def test_weak_identification_is_warned_and_recorded():
    ids = [f"U{i}" for i in range(6)]
    rng = np.random.default_rng(11)
    flows = np.ones((6, 6)) + 0.03 * rng.random((6, 6))
    W = economic_weights(pd.DataFrame(flows, index=ids, columns=ids))
    df = _panel(W, gamma=0.0, seed=12, T=40)
    with pytest.warns(RuntimeWarning, match="weakly identified"):
        res = spatial_lp(df, "y", "x", W, horizons=[0, 1], n_lags=1,
                         cov_type="cluster", cumulative=False)
    assert np.all(np.abs(res.spillover["corr_own"].to_numpy()) > 0.999)
    assert res.spec["weak_id_corr"] == 0.999


def test_cov_keywords_that_the_cov_type_ignores_raise(demo, knn3):
    coords = _coords()
    with pytest.raises(ValueError, match="only for cov_type='conley'"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="cluster",
                   cutoff_km=500.0, cumulative=False)
    with pytest.raises(ValueError, match="only for cov_type='conley'"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="driscoll-kraay",
                   coords=coords, cumulative=False)
    with pytest.raises(ValueError, match="only for cov_type='conley'"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="dk",
                   metric="euclidean", cumulative=False)
    with pytest.raises(ValueError, match="time_lags"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="cluster",
                   time_lags=3, cumulative=False)
    with pytest.raises(ValueError, match="cov_type must be"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="hac", cumulative=False)
    with pytest.raises(ValueError, match="needs coords"):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov_type="conley",
                   cutoff_km=500.0, cumulative=False)


def test_no_kwargs_sink_and_the_2_0_aliases(demo, knn3):
    with pytest.raises(TypeError):
        spatial_lp(demo, "y", "x", knn3, horizons=[0], cov="cluster")
    a = spatial_lp(demo, "y", "x", knn3, horizon=2, lags=1, ci=0.95, cumulative=False)
    b = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 3), n_lags=1, alpha=0.05,
                   cumulative=False)
    np.testing.assert_allclose(a["lo_total"].to_numpy(), b["lo_total"].to_numpy(),
                               atol=1e-12)
    assert list(a["h"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 11. Adversarial geometries
# ---------------------------------------------------------------------------
def test_disconnected_graph_and_single_neighbour_chain(demo):
    pairs = {}
    for i in range(0, 12, 2):
        pairs[IDS[i]] = [IDS[i + 1]]
        pairs[IDS[i + 1]] = [IDS[i]]
    disconnected = contiguity_weights(pairs, ids=IDS)     # six 2-cliques
    res = spatial_lp(demo, "y", "x", disconnected, horizons=[0, 1], n_lags=1,
                     cumulative=False)
    assert np.all(np.isfinite(res["beta_indirect"].to_numpy()))
    assert res.weights_info[1]["n_islands"] == 0

    chain = {IDS[i]: [IDS[i + 1]] for i in range(11)}
    chain[IDS[11]] = [IDS[10]]
    Wc = contiguity_weights(chain, ids=IDS)
    res2 = spatial_lp(demo, "y", "x", Wc, horizons=[0, 1], n_lags=1, cumulative=False)
    assert np.all(np.isfinite(res2["se_total"].to_numpy()))


def test_duplicate_coordinates_are_estimable(demo):
    coords = _coords()
    coords.iloc[3] = coords.iloc[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        W = knn_weights(coords, 3)
        res = spatial_lp(demo, "y", "x", W, horizons=[0, 1], n_lags=1, cumulative=False,
                         cov_type="conley", coords=coords, cutoff_km=400.0)
    assert np.all(np.isfinite(res["se_total"].to_numpy()))


def test_an_entity_missing_for_the_last_periods(knn3):
    df = _panel(knn3, seed=13, T=40)
    df = df.drop(index=[(IDS[4], t) for t in range(30, 40)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_lp(df, "y", "x", knn3, horizons=range(0, 4), n_lags=1,
                         unbalanced="renormalize")
    assert np.all(np.isfinite(res["beta_total"].to_numpy()))
    assert np.all(res["n_entities"].to_numpy() == 12)


def test_a_genuinely_empty_horizon_warns_and_leaves_a_note(knn3):
    """The Notes promise a RuntimeWarning for a horizon whose sample is empty,
    not only for one the within transformation annihilates.  A silent NaN row
    with an unexplained ``incomplete`` flag would be documented behaviour that
    does not happen."""
    df = _panel(knn3, seed=15, T=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = spatial_lp(df, "y", "x", knn3, horizons=range(0, 10), n_lags=2)
    assert int(res["n_obs"].iloc[-1]) == 0            # h = 9 is genuinely empty
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    hits = [m for m in msgs if "horizon 9" in m and "no observations" in m]
    assert hits, f"the empty horizon warned nothing; got {msgs}"
    assert any("horizon 9" in n and "no observations" in n for n in res.notes)
    assert any("horizon 9" in line for line in res.summary().splitlines())
    assert not bool(res.cumulative["complete"].iloc[-1])


def test_an_empty_horizon_yields_nan_and_an_incomplete_cumulative(knn3):
    df = _panel(knn3, seed=14, T=12)
    with pytest.warns(RuntimeWarning, match="annihilates"):
        res = spatial_lp(df, "y", "x", knn3, horizons=range(0, 12), n_lags=2)
    assert np.isnan(res["beta_direct"].to_numpy()[-1])
    assert np.isnan(res["se_total"].to_numpy()[-1])
    assert int(res["n_obs"].iloc[-1]) == 0
    assert not bool(res.cumulative["complete"].iloc[-1])
    assert bool(res.cumulative["complete"].iloc[0])
    # the cumulative point stops accumulating once a horizon is skipped
    live = np.isfinite(res["beta_direct"].to_numpy())
    assert res.cumulative["cum_direct"].iloc[-1] == pytest.approx(
        float(np.nansum(res["beta_direct"].to_numpy()[live])), abs=1e-12)


# ---------------------------------------------------------------------------
# 12. Presentation contract
# ---------------------------------------------------------------------------
def test_presentation_contract(demo, knn3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1,
                     cov_type="conley", coords=_coords(), cutoff_km=600.0)
    assert isinstance(res, SpatialLPResult)
    expected = ["h"] + [f"{b}_{lab}" for lab in ("direct", "indirect", "total")
                        for b in ("beta", "se", "t", "lo", "hi")] + \
        ["n_obs", "n_entities", "n_periods"]
    assert list(res.columns) == expected
    assert len(expected) == 19
    assert res.labels == ["direct", "indirect", "total"]
    assert isinstance(res.point, pd.DataFrame) and list(res.point.columns) == res.labels

    text = res.summary()
    assert isinstance(text, str) and text
    for token in ("direct", "indirect", "Conley", "cumulative"):
        assert token in text

    frame = res.to_frame()
    assert type(frame) is pd.DataFrame and list(frame.columns) == expected
    md = res.to_markdown()
    assert md.startswith("|") and "---" in md
    assert "begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    fig = res.plot()
    assert len(fig.axes) == 3
    plt.close(fig)
    fig = res.plot(kind="overlay")
    assert len(fig.axes) == 1
    plt.close(fig)
    fig = res.plot(kind="cumulative", components=("total",))
    assert len(fig.axes) == 1
    plt.close(fig)
    _, ax = plt.subplots()
    fig = res.plot(components=("direct",), ax=ax)
    plt.close(fig)
    with pytest.raises(ValueError, match="single component"):
        _, ax2 = plt.subplots()
        res.plot(ax=ax2)
    with pytest.raises(ValueError, match="kind must be"):
        res.plot(kind="spaghetti")
    with pytest.raises(ValueError, match="components"):
        res.plot(components=("sideways",))
    plt.close("all")


def test_metadata_survives_slicing(demo, knn3):
    res = spatial_lp(demo, "y", "x", knn3, horizons=range(0, 4), n_lags=1)
    head = res.iloc[:2]
    assert isinstance(head, SpatialLPResult)
    assert head.cov_type == res.cov_type
    assert head.coef_names == res.coef_names
    meta = res.metadata
    assert meta["cov_type"] == "driscoll-kraay" and meta["n_lags"] == 1
    assert res.cov_options["bandwidth_by_horizon"][0] >= 1
    assert res.cov_options["cross_bandwidth"] >= 1
    assert set(res.weights_info[1]) == {
        "kind", "n_units", "n_islands", "islands", "row_standardized",
        "row_sum_mean", "row_sum_min", "row_sum_max"}
