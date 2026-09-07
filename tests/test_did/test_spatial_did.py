"""Behaviour pins for ``puremacro.did.spatial_did``.

The tests here are identities, invariances, reductions and planted-parameter
recoveries — things that break when the implementation is wrong rather than
when it merely changes shape:

* the Frisch-Waugh contamination identity, to machine precision, against an
  independently fitted single-regressor within model;
* the Conley covariance collapsing to HC0 at a zero cutoff and to
  Driscoll-Kraay under a flat kernel, both rebuilt by hand from
  ``two_way_fe_within``;
* the ring code at every band edge, and the rook-lattice hop counts;
* the three input routes (coordinates / ``RingAssignment`` / ``ring_col``)
  agreeing bit for bit;
* every adversarial case the design names: islands, a disconnected graph,
  duplicate coordinates, a single period, a constant regressor, an all-treated
  panel, an unbalanced panel.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import chi2 as scipy_chi2

from puremacro.did.spatial_did import (
    _MIN_UNITS_FOR_JOINT,
    RingAssignment,
    _sandwich,
    SpatialDiDResult,
    contiguity_rings,
    exposure_rings,
    spatial_did,
)
from puremacro.lp._panel_helpers import as_panel_index, two_way_fe_within
from puremacro.spatial.weights import SpatialWeights

RINGS = (0.0, 25.0, 50.0, 100.0)
#: `cutoff_km` has no default under cov_type='conley' (the Conley radius is
#: an assumption, not something `rings` implies), so every Conley call here
#: states it. 2 * rings[-1] is the radius the ring design itself implies.
CUT = 2 * RINGS[-1]
LEVELS = (1.0, 0.5, 0.25, 0.10, 0.0)


# ---------------------------------------------------------------------------
# Seeded DGPs. The ring assignment used to *generate* the data comes from
# `exposure_rings`, so the planted parameter is exactly the coefficient the
# estimator is meant to recover.
# ---------------------------------------------------------------------------
def make_panel(
    seed: int = 5,
    *,
    n: int = 90,
    T: int = 10,
    cohorts: tuple = (4,),
    n_treated: int = 9,
    sigma: float = 0.35,
    rings: tuple = RINGS,
    levels: tuple = LEVELS,
    box: float = 500.0,
    ring_timing: str = "already_treated",
    pre_trend_ring: int | None = None,
    pre_trend_slope: float = 0.0,
):
    """Two-way FE panel whose outcome is a step function of the ring code."""
    rng = np.random.default_rng(seed)
    ids = [f"u{i:03d}" for i in range(n)]
    xy = pd.DataFrame({"x": rng.uniform(0, box, n), "y": rng.uniform(0, box, n)},
                      index=ids)
    treat = {u: (float(cohorts[i % len(cohorts)]) if i < n_treated else np.nan)
             for i, u in enumerate(ids)}
    tt = pd.Series(treat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = exposure_rings(xy, treat_time=tt, times=list(range(T)), rings=rings,
                            metric="euclidean", ring_timing=ring_timing)
    code = ra.frame.pivot(index="unit", columns="time", values="ring").loc[
        ids, list(range(T))].to_numpy()
    lev = np.asarray(levels, dtype=float)
    mu = rng.normal(size=n)
    tau = rng.normal(size=T) * 0.2
    eps = sigma * rng.normal(size=(n, T))
    y = mu[:, None] + tau[None, :] + lev[code] + eps
    if pre_trend_ring is not None:
        entry = ra.frame.pivot(index="unit", columns="time",
                               values="entry_ring").loc[ids, list(range(T))]
        er = entry.to_numpy(dtype=float)[:, 0]
        hit = np.nan_to_num(er, nan=-1) == pre_trend_ring
        y = y + hit[:, None] * pre_trend_slope * np.arange(T)[None, :]
    rows = [{"unit": u, "time": t, "y": float(y[i, t]), "treat_time": tt[u]}
            for i, u in enumerate(ids) for t in range(T)]
    return pd.DataFrame(rows), xy, ra


def hand_within(df, xy, ra, ring_codes, *, outcome="y"):
    """Independently rebuilt within fit on the same ring dummies."""
    m = df.merge(ra.frame[["unit", "time", "ring"]], on=["unit", "time"])
    cols = []
    for r in ring_codes:
        m[f"R{r}"] = (m["ring"] == r).astype(float)
        cols.append(f"R{r}")
    panel, ent, tim = as_panel_index(m, entity_level="unit", time_level="time",
                                     unit_col="unit", time_col="time")
    return two_way_fe_within(panel, y_col=outcome, x_cols=cols,
                             entity_level=ent, time_level=tim)


@pytest.fixture(scope="module")
def base():
    return make_panel()


@pytest.fixture(scope="module")
def base_fit(base):
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)


# ---------------------------------------------------------------------------
# Ring construction
# ---------------------------------------------------------------------------
def test_ring_codes_band_edges():
    """`searchsorted(edges[1:], d, 'left') + 1` must close band 1 on both sides
    and every later band on the right only."""
    edges = np.array([0.0, 25.0, 50.0, 100.0])
    eps = 1e-9
    # One treated unit at the origin, probes at the exact edges.
    probes = [0.0, 25.0, 25.0 + eps, 50.0, 50.0 + eps, 100.0, 100.0 + eps, 400.0]
    ids = ["t"] + [f"p{i}" for i in range(len(probes))]
    xy = pd.DataFrame({"x": [0.0] + probes, "y": [0.0] * (len(probes) + 1)},
                      index=ids)
    tt = pd.Series({u: (0.0 if u == "t" else np.nan) for u in ids})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = exposure_rings(xy, treat_time=tt, times=[0, 1], rings=tuple(edges),
                            metric="euclidean")
    got = ra.frame[ra.frame.time == 1].set_index("unit")["ring"]
    assert got["t"] == 0
    assert [int(got[f"p{i}"]) for i in range(len(probes))] == [1, 1, 2, 2, 3, 3, 4, 4]


def test_ring_code_of_infinite_distance_is_the_control_ring():
    """Before anything nearby is treated the distance is +inf, which must land
    in the control code R+1 and not wrap round to ring 1."""
    df, xy, ra = make_panel(cohorts=(6,))
    early = ra.frame[ra.frame.time < 6]
    assert set(early["ring"].unique()) == {ra.n_rings + 1}
    assert np.isinf(early["distance"]).all()


def test_ring_edges_validation():
    xy = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}, index=["a", "b"])
    tt = {"a": 0.0, "b": np.nan}
    for bad in [(25.0, 50.0), (0.0, 50.0, 25.0), (0.0,), (0.0, -5.0, 10.0),
                (0.0, 50.0, np.nan)]:
        with pytest.raises(ValueError) as exc:
            exposure_rings(xy, treat_time=tt, times=[0, 1], rings=bad,
                           metric="euclidean")
        msg = str(exc.value)
        assert "exposure_rings" in msg and "rings" in msg


def test_nearest_treated_tie_rule_is_deterministic():
    """Two treated units exactly equidistant: the reported identity is the
    first in `ids` order, and nothing about the answer depends on input order."""
    xy = pd.DataFrame({"x": [-10.0, 10.0, 0.0], "y": [0.0, 0.0, 0.0]},
                      index=["A", "B", "mid"])
    tt = {"A": 0.0, "B": 0.0, "mid": np.nan}
    outs = []
    for _ in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ra = exposure_rings(xy, treat_time=tt, times=[0, 1], rings=RINGS,
                                metric="euclidean")
        outs.append(ra.frame.set_index(["unit", "time"]).loc[("mid", 1)])
    assert all(o["nearest_treated"] == "A" for o in outs)
    assert len({float(o["distance"]) for o in outs}) == 1
    xy2 = xy.loc[["B", "A", "mid"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra2 = exposure_rings(xy2, treat_time=tt, times=[0, 1], rings=RINGS,
                             metric="euclidean")
    both = ra2.frame.set_index(["unit", "time"]).loc[("mid", 1)]
    # The ring code cannot depend on tie order; only the reported identity can.
    assert int(both["ring"]) == int(outs[0]["ring"])
    assert float(both["distance"]) == pytest.approx(float(outs[0]["distance"]))


def test_already_treated_is_zero_before_any_treatment():
    """Under the default timing every ring indicator is identically zero before
    the first cohort switches on — that is what keeps the pre-period clean."""
    _df, xy, ra = make_panel(cohorts=(3, 7), n_treated=8)
    piv = ra.frame.pivot(index="unit", columns="time", values="ring")
    assert (piv.loc[:, :2] == ra.n_rings + 1).to_numpy().all()
    assert (piv.loc[:, 3:] <= ra.n_rings).to_numpy().any()


def test_ever_treated_uses_the_nearest_ever_treated_unit():
    """The two timings differ exactly where the design says they do: a unit that
    is far from the early cohort and close to the late one is coded in the outer
    band meanwhile under 'already_treated', and stays in the control group under
    'ever_treated' until its own nearest unit switches on."""
    xy = pd.DataFrame(
        {"x": [0.0, 40.0, 50.0, 1000.0], "y": [0.0, 0.0, 0.0, 0.0]},
        index=["early", "probe", "late", "far"])
    tt = {"early": 1.0, "late": 5.0, "probe": np.nan, "far": np.nan}
    times = list(range(8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        at = exposure_rings(xy, treat_time=tt, times=times, rings=RINGS,
                            metric="euclidean")
        et = exposure_rings(xy, treat_time=tt, times=times, rings=RINGS,
                            metric="euclidean", ring_timing="ever_treated")
    a = at.frame[at.frame.unit == "probe"].set_index("time")["ring"]
    e = et.frame[et.frame.unit == "probe"].set_index("time")["ring"]
    assert list(a[[0, 1, 4, 5, 7]]) == [4, 2, 2, 1, 1]      # 40 km, then 10 km
    assert list(e[[0, 1, 4, 5, 7]]) == [4, 4, 4, 1, 1]      # nearest is `late`


def test_ring_code_is_monotone_under_already_treated():
    """The set of already-treated units only grows, so a unit walks inward: the
    code is weakly decreasing, and `n_switchers` counts the strict decreases."""
    _df, _xy, ra = make_panel(cohorts=(3, 6, 8), n_treated=15, seed=11)
    piv = ra.frame.pivot(index="unit", columns="time", values="ring")
    arr = piv.to_numpy()
    assert np.all(np.diff(arr, axis=1) <= 0)
    exposed = arr <= ra.n_rings
    strict = 0
    for i in range(arr.shape[0]):
        if not exposed[i].any():
            continue
        first = int(np.argmax(exposed[i]))
        tail = arr[i, first:]
        if np.any(tail[tail <= ra.n_rings] != arr[i, first]):
            strict += 1
    assert ra.n_switchers == strict > 0


def test_duplicate_coordinates_land_in_ring_one_and_warn():
    xy = pd.DataFrame({"x": [0.0, 0.0, 300.0], "y": [0.0, 0.0, 0.0]},
                      index=["t", "twin", "far"])
    tt = {"t": 0.0, "twin": np.nan, "far": np.nan}
    with pytest.warns(RuntimeWarning, match="distance exactly 0"):
        ra = exposure_rings(xy, treat_time=tt, times=[0, 1], rings=RINGS,
                            metric="euclidean")
    row = ra.frame.set_index(["unit", "time"]).loc[("twin", 1)]
    assert int(row["ring"]) == 1 and float(row["distance"]) == 0.0


def test_ring_assignment_counts_partition_the_grid():
    """`n_unit_periods` partitions every (unit, period) cell; `n_units` counts
    units *ever* in a ring and so need not sum to N."""
    _df, _xy, ra = make_panel(cohorts=(3, 7), n_treated=12, seed=3)
    assert int(ra.counts["n_unit_periods"].sum()) == len(ra.frame)
    assert int(ra.counts["n_units"].sum()) >= len(ra.ids)


# ---------------------------------------------------------------------------
# contiguity_rings
# ---------------------------------------------------------------------------
def _rook_lattice(g: int = 8, with_island: bool = True):
    ids = [f"c{i}_{j}" for i in range(g) for j in range(g)]
    pos = {f"c{i}_{j}": (i, j) for i in range(g) for j in range(g)}
    if with_island:
        ids = ids + ["isl1", "isl2"]
    idx = {u: k for k, u in enumerate(ids)}
    rows, cols = [], []
    for u, (i, j) in pos.items():
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < g and 0 <= b < g:
                rows.append(idx[u])
                cols.append(idx[f"c{a}_{b}"])
    if with_island:
        rows += [idx["isl1"], idx["isl2"]]
        cols += [idx["isl2"], idx["isl1"]]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids),) * 2)
    return SpatialWeights(A, ids=tuple(ids)), pos


def test_contiguity_rings_are_manhattan_hops_on_a_rook_lattice():
    W, pos = _rook_lattice()
    tt = {u: (1.0 if u == "c3_3" else np.nan) for u in W.ids}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = contiguity_rings(W, treat_time=tt, times=[0, 1], orders=3)
    got = ra.frame[ra.frame.time == 1].set_index("unit")["ring"]
    for u, (i, j) in pos.items():
        man = abs(i - 3) + abs(j - 3)
        assert int(got[u]) == (0 if man == 0 else min(man, 4)), u
    assert ra.metric == "graph" and ra.distance_unit == "hops"


def test_disconnected_component_lands_in_the_control_ring():
    W, _pos = _rook_lattice()
    tt = {u: (1.0 if u == "c3_3" else np.nan) for u in W.ids}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = contiguity_rings(W, treat_time=tt, times=[0, 1], orders=3)
    got = ra.frame[ra.frame.time == 1].set_index("unit")
    assert int(got.loc["isl1", "ring"]) == 4
    assert np.isinf(float(got.loc["isl2", "distance"]))
    assert ra.n_islands == 2


def test_islands_warn_and_orders_is_capped():
    ids = tuple(f"n{i}" for i in range(4))
    A = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                                [0, 1, 0, 0], [0, 0, 0, 0]], float))
    W = SpatialWeights(A, ids=ids)
    tt = {"n0": 1.0, "n1": np.nan, "n2": np.nan, "n3": np.nan}
    with pytest.warns(RuntimeWarning, match="island"):
        ra = contiguity_rings(W, treat_time=tt, times=[0, 1], orders=2)
    assert int(ra.frame.set_index(["unit", "time"]).loc[("n3", 1), "ring"]) == 3
    with pytest.raises(ValueError, match="orders must lie in 1..10"):
        contiguity_rings(W, treat_time=tt, times=[0, 1], orders=11)
    with pytest.raises(ValueError, match="contiguity_rings: W must be"):
        contiguity_rings(A, treat_time=tt, times=[0, 1])


def test_graph_assignment_forbids_conley():
    """Hop counts have no kilometre scale, so a graph assignment may not reach
    the Conley kernel — the critique's `metric='graph'` hole."""
    W, _pos = _rook_lattice(g=6, with_island=False)
    ids = list(W.ids)
    tt = pd.Series({u: (4.0 if u in ("c2_2", "c3_3") else np.nan) for u in ids})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = contiguity_rings(W, treat_time=tt, times=list(range(8)), orders=2)
    code = ra.frame.pivot(index="unit", columns="time", values="ring").loc[
        ids, list(range(8))].to_numpy()
    rng = np.random.default_rng(0)
    lev = np.array([1.0, 0.4, 0.15, 0.0])
    y = (rng.normal(size=(len(ids), 1)) + lev[code]
         + 0.3 * rng.normal(size=(len(ids), 8)))
    df = pd.DataFrame([{"unit": u, "time": t, "y": float(y[i, t]),
                        "treat_time": tt[u]}
                       for i, u in enumerate(ids) for t in range(8)])
    with pytest.raises(ValueError, match="cov_type='cluster'"):
        spatial_did(df, assignment=ra, cutoff_km=CUT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, assignment=ra, cov_type="cluster")
    assert res.metric == "graph" and res.distance_unit == "hops"
    assert res.cutoff_km is None and res.n_treated_clusters is None
    txt = res.summary()
    assert "cluster-robust by unit" in txt and "hops" in txt
    assert "hop" in " ".join(res.ring_table["label"])
    assert res.to_markdown()
    with pytest.raises(ValueError, match="meaningless when assignment"):
        spatial_did(df, assignment=ra, cov_type="cluster", metric="euclidean")


# ---------------------------------------------------------------------------
# The Frisch-Waugh decomposition
# ---------------------------------------------------------------------------
def test_fwl_contamination_identity_is_exact(base, base_fit):
    """naive = direct + theta @ spillovers, and `naive_att` equals an
    independently fitted single-regressor within model."""
    df, xy, ra = base
    res = base_fit
    implied = res.direct_effect + float(res.theta @ res.coef[1:])
    assert abs(res.naive_att - implied) < 1e-12
    assert abs(res.contamination - (res.naive_att - res.direct_effect)) < 1e-12
    solo = hand_within(df, xy, ra, [0])
    assert abs(float(solo["beta"][0]) - res.naive_att) < 1e-10
    long = hand_within(df, xy, ra, list(res.ring_codes))
    assert np.abs(np.asarray(long["beta"]) - res.coef).max() < 1e-12


def test_theta_is_negative_so_spillovers_bias_the_naive_estimate_down(base_fit):
    res = base_fit
    assert res.theta.shape == (len(res.coef) - 1,)
    assert np.all(res.theta < 0)
    assert np.all(res.coef[1:] > 0)
    assert res.contamination < 0 < res.direct_effect
    assert res.naive_att < res.direct_effect


def test_empty_spillover_rings_reduce_exactly_to_the_twfe_did():
    """With every spillover band empty the design matrix is literally the
    canonical two-way-FE DiD regressor, so the correction vanishes exactly."""
    df, xy, ra = make_panel(rings=(0.0, 1e-9, 2e-9),
                            levels=(1.0, 0.0, 0.0, 0.0))
    with pytest.warns(RuntimeWarning):
        res = spatial_did(df, coords=xy, metric="euclidean",
                          rings=(0.0, 1e-9, 2e-9), cutoff_km=CUT)
    assert res.dropped_rings == (1, 2)
    assert res.ring_codes == (0,)
    assert res.theta.shape == (0,)
    assert res.contamination == 0.0
    assert res.direct_effect == res.naive_att
    solo = hand_within(df, xy, ra, [0])
    assert abs(float(solo["beta"][0]) - res.direct_effect) < 1e-12


def test_naive_att_reports_the_estimation_sample(base):
    """`naive_att` is the naive DiD *on the sample actually fitted*; when rows
    are dropped that is not the untrimmed panel, and `n_obs_dropped` says so."""
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        trimmed = spatial_did(df, coords=xy, metric="euclidean",
                              min_units_per_ring=6, cutoff_km=CUT)
    assert full.n_obs_dropped == 0
    assert trimmed.n_obs_dropped > 0
    assert trimmed.n_obs + trimmed.n_obs_dropped == full.n_obs
    # The identity still holds inside the trimmed projection.
    implied = trimmed.direct_effect + float(trimmed.theta @ trimmed.coef[1:])
    assert abs(trimmed.naive_att - implied) < 1e-12


# ---------------------------------------------------------------------------
# Recovery of a planted parameter
# ---------------------------------------------------------------------------
def test_known_spillover_is_recovered_within_three_standard_errors():
    df, xy, _ra = make_panel(n=200, T=12, n_treated=20, sigma=0.35, seed=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    truth = np.array(LEVELS[:len(res.coef)])
    se = res.ring_table.loc[~res.ring_table["dropped"], "se"].to_numpy()[:len(truth)]
    assert np.all(np.abs(res.coef - truth) <= 3.0 * se), (res.coef, se)


def test_the_ring_estimator_beats_the_naive_did_on_the_direct_effect():
    """The improvement metric the critique asked for in place of a point
    comparison at Tol.COARSE: |naive - 1| - |direct - 1| is comfortably
    positive, i.e. the rings buy a materially better ATT."""
    improvements = []
    for seed in (17, 23, 31, 37, 43):
        df, xy, _ra = make_panel(n=200, T=12, n_treated=20, sigma=0.35, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        improvements.append(abs(res.naive_att - 1.0) - abs(res.direct_effect - 1.0))
    assert min(improvements) > 0.05, improvements
    assert float(np.mean(improvements)) >= 0.10, improvements


def test_no_spillover_dgp_leaves_the_naive_did_unbiased():
    """When the truth has no spillover the ring correction must not invent one:
    naive and direct agree, and the joint no-spillover test does not reject."""
    df, xy, _ra = make_panel(n=200, T=12, n_treated=20, sigma=0.35, seed=41,
                             levels=(1.0, 0.0, 0.0, 0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert abs(res.direct_effect - 1.0) < 3 * res.direct_se
    assert abs(res.naive_att - 1.0) < 3 * res.naive_se
    assert abs(res.contamination) < 3 * res.contamination_se
    p = float(res.tests.set_index("test").loc["no_spillover", "p"])
    assert p > 0.05


# ---------------------------------------------------------------------------
# Covariance identities
# ---------------------------------------------------------------------------
def test_conley_cutoff_zero_equals_hc0(base):
    """`kernel_matrix` returns the identity at cutoff 0, so the meat collapses
    to sum_it x_it x_it' u_it^2 — White's HC0 over the panel rows."""
    df, xy, ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=0.0,
                          time_lags=0, small_sample=False)
    fit = hand_within(df, xy, ra, list(res.ring_codes))
    X, u, B = fit["X_within"], fit["residuals"], fit["XtX_inv"]
    hc0 = B @ (X.T @ (X * u[:, None] ** 2)) @ B
    assert np.abs(hc0 - res.vcov).max() < 1e-12
    assert res.dof_factor == 1.0


def test_conley_flat_kernel_equals_driscoll_kraay(base):
    """A cutoff above every distance makes the spatial kernel all-ones, so the
    meat is exactly the Driscoll-Kraay long-run covariance of the summed
    scores."""
    from puremacro.inference.dk import driscoll_kraay

    df, xy, ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=1e9,
                          time_lags=2, kernel="uniform", small_sample=False)
    fit = hand_within(df, xy, ra, list(res.ring_codes))
    X, u, B = fit["X_within"], fit["residuals"], fit["XtX_inv"]
    S = driscoll_kraay(X * u[:, None], fit["time_keys"], lags=2)
    assert np.abs(B @ S @ B - res.vcov).max() < 1e-10


def test_conley_refuses_to_guess_the_cutoff(base):
    """BUILD_RULES: `cutoff_km=None` under cov_type='conley' raises. The radius
    is an assumption about the error field, and the old default (2*rings[-1])
    silently made it for the user. The error must name the argument, the mode
    and the value it declines to assume."""
    df, xy, _ra = base
    with pytest.raises(ValueError) as exc:
        spatial_did(df, coords=xy, metric="euclidean")
    msg = str(exc.value)
    assert "spatial_did" in msg and "cutoff_km" in msg and "conley" in msg
    assert "200" in msg                        # 2 * rings[-1], as a suggestion
    # ... and under cov_type='cluster' the same omission is correct, not an
    # error, because there is no kernel to give a radius to.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clu = spatial_did(df, coords=xy, metric="euclidean", cov_type="cluster")
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert clu.cutoff_km is None
    assert res.cutoff_km == pytest.approx(2 * RINGS[-1])


def test_cutoff_below_ring_warning(base):
    df, xy, _ra = base
    with pytest.warns(RuntimeWarning, match="below rings"):
        spatial_did(df, coords=xy, metric="euclidean", cutoff_km=40.0)
    # cutoff 0 is an explicit opt-out (HC0), not a mistake: it must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning) as exc:
            spatial_did(df, coords=xy, metric="euclidean", cutoff_km=0.0,
                        time_lags=0)
        assert "below rings" not in str(exc.value)


def test_time_lag_rules(base):
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert res.time_lags == max(1, int(np.floor(4 * (10 / 100) ** (2 / 9))))
    with pytest.raises(ValueError, match="time_lags"):
        spatial_did(df, coords=xy, metric="euclidean", time_lags=10, cutoff_km=CUT)
    with pytest.raises(ValueError, match="time_lags must be non-negative"):
        spatial_did(df, coords=xy, metric="euclidean", time_lags=-1, cutoff_km=CUT)
    short, xy_s, _ = make_panel(T=4, cohorts=(2,))
    with pytest.warns(RuntimeWarning, match="time_lags is forced"):
        res_s = spatial_did(short, coords=xy_s, metric="euclidean", cutoff_km=CUT)
    assert res_s.time_lags == 0


def test_indefinite_meat_is_flagged_and_clipping_is_conservative():
    """The two-dimensional kernel is not positive definite. Clipping projects
    the meat onto the PSD cone, which weakly INCREASES every variance — the
    inequality the critique found reversed in the design."""
    df, xy, _ra = make_panel(n=80, seed=5)
    kw = dict(coords=xy, metric="euclidean", kernel="uniform", cutoff_km=300.0)
    with pytest.warns(RuntimeWarning, match="indefinite"):
        warn = spatial_did(df, psd_adjust="warn", **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clip = spatial_did(df, psd_adjust="clip", **kw)
        none = spatial_did(df, psd_adjust="none", **kw)
    assert warn.meat_min_eig < 0
    se_w = np.sqrt(np.diag(warn.vcov))
    se_c = np.sqrt(np.diag(clip.vcov))
    assert np.all(se_c >= se_w - 1e-12)
    assert np.any(se_c > se_w + 1e-8)
    assert np.array_equal(warn.coef, clip.coef)
    assert np.array_equal(warn.vcov, none.vcov)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning) as exc:
            spatial_did(df, psd_adjust="none", **kw)
        assert "indefinite" not in str(exc.value)


def test_cluster_discards_the_spatial_covariance_that_conley_keeps():
    """Under a spatially correlated error field the cross-unit score products
    are real. Cluster-by-unit drops every one of them and reports a smaller
    standard error; Conley keeps them. Asserted as a band over twelve seeded
    draws, not a point, so the claim is about the estimator and not the seed."""
    rng = np.random.default_rng(9)
    n, T, field_km = 150, 8, 300.0
    xy = pd.DataFrame({"x": rng.uniform(0, 400, n), "y": rng.uniform(0, 400, n)},
                      index=[f"u{i:03d}" for i in range(n)])
    tt = pd.Series({u: (4.0 if i < 15 else np.nan)
                    for i, u in enumerate(xy.index)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = exposure_rings(xy, treat_time=tt, times=list(range(T)),
                            rings=RINGS, metric="euclidean")
    code = ra.frame.pivot(index="unit", columns="time", values="ring").loc[
        list(xy.index), list(range(T))].to_numpy()
    d = np.sqrt(((xy.to_numpy()[:, None, :] - xy.to_numpy()[None, :, :]) ** 2
                 ).sum(-1))
    L = np.linalg.cholesky(np.exp(-d / field_km) + 1e-8 * np.eye(n))
    lev = np.array(LEVELS)
    ratios = []
    for rep in range(12):
        g = np.random.default_rng(1000 + rep)
        y = g.normal(size=(n, 1)) + lev[code] + 0.5 * (L @ g.normal(size=(n, T)))
        df = pd.DataFrame([{"unit": u, "time": t, "y": float(y[i, t]),
                            "treat_time": tt[u]}
                           for i, u in enumerate(xy.index) for t in range(T)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            con = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
            clu = spatial_did(df, coords=xy, metric="euclidean",
                              cov_type="cluster")
        assert np.array_equal(con.coef, clu.coef)
        assert con.n_treated_clusters is not None
        assert clu.n_treated_clusters is None
        ratios.append(con.direct_se / clu.direct_se)
    ratios = np.asarray(ratios)
    assert float(ratios.mean()) > 1.25, ratios
    assert int((ratios > 1.0).sum()) >= 10, ratios


def test_cluster_col_and_conley_argument_gating(base):
    df, xy, _ra = base
    with pytest.raises(ValueError, match="cluster_col is meaningless"):
        spatial_did(df, coords=xy, metric="euclidean", cluster_col="unit", cutoff_km=CUT)
    with pytest.raises(ValueError, match="meaningless for cov_type='cluster'"):
        spatial_did(df, coords=xy, metric="euclidean", cov_type="cluster",
                    time_lags=2)
    codes = _ra.frame.set_index(["unit", "time"])["ring"]
    with_col = df.assign(
        rc=codes.loc[list(zip(df["unit"], df["time"]))].to_numpy())
    with pytest.raises(ValueError, match="cov_type='conley' needs unit"):
        spatial_did(with_col, ring_col="rc", cutoff_km=CUT)


def test_cluster_col_groups_units_together(base):
    """A coarser cluster variable must change the SE but not the point
    estimates — proof that `cluster_col` is actually wired to the meat."""
    df, xy, _ra = base
    grp = df.assign(region=[u[-1] for u in df["unit"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        by_unit = spatial_did(df, coords=xy, metric="euclidean",
                              cov_type="cluster")
        by_region = spatial_did(grp, coords=xy, metric="euclidean",
                                cov_type="cluster", cluster_col="region")
    assert np.array_equal(by_unit.coef, by_region.coef)
    assert not np.allclose(by_unit.vcov, by_region.vcov)


# ---------------------------------------------------------------------------
# Tests table and the outer-ring contrast
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wide_fit():
    """A design every ring of which clears `_MIN_UNITS_FOR_JOINT`, so the joint
    Wald rows are actually reported and can be checked against hand algebra."""
    df, xy, ra = make_panel(n=200, T=12, n_treated=20, seed=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    carried = res.ring_table.set_index("ring")["n_units"]
    assert all(int(carried.loc[r]) >= _MIN_UNITS_FOR_JOINT
               for r in res.ring_codes), carried
    return df, xy, ra, res


def test_no_spillover_wald_matches_the_hand_built_quadratic_form(wide_fit):
    _df, _xy, _ra, res = wide_fit
    row = res.tests.set_index("test").loc["no_spillover"]
    idx = [j for j, r in enumerate(res.ring_codes) if r >= 1]
    b = res.coef[idx]
    V = res.vcov[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V, rcond=1e-10, hermitian=True) @ b)
    assert bool(row["underpowered"]) is False
    assert int(row["df"]) == len(idx)
    assert float(row["stat"]) == pytest.approx(stat, rel=1e-10)
    assert float(row["p"]) == pytest.approx(
        float(scipy_chi2.sf(stat, len(idx))), rel=1e-10)


def test_tests_table_carries_every_family_with_the_right_degrees_of_freedom(
        wide_fit):
    """The four Wald families the design names, plus one dynamics row per ring,
    and each dynamics test has (post cells - 1) restrictions. Every reported
    statistic is checked against chi2.sf of its own stat and df, which is a
    claim that can fail — `p in [0, 1]` is not."""
    _df, _xy, _ra, res = wide_fit
    names = list(res.tests["test"])
    assert names[:3] == ["outer_ring", "no_spillover", "equal_rings"]
    dyn = res.tests[res.tests["test"].str.startswith("dynamics_ring")]
    assert len(dyn) == len(res.att_by_ring_es)
    for _, row in dyn.iterrows():
        r = int(str(row["test"]).replace("dynamics_ring", ""))
        n_post = int(res.att_by_ring_es.set_index("ring").loc[r, "n_post_cells"])
        assert int(row["df"]) == n_post - 1
    assert int(res.tests.set_index("test").loc["equal_rings", "df"]) == \
        len(res.ring_codes) - 2
    # Nothing is gated in this design, and every p is exactly chi2.sf(stat, df).
    assert not res.tests["underpowered"].any()
    for _, row in res.tests.iterrows():
        assert float(row["p"]) == pytest.approx(
            float(scipy_chi2.sf(float(row["stat"]), int(row["df"]))),
            rel=1e-10, abs=1e-300)
    for _, row in res.pretrend.iterrows():
        assert int(row["df"]) == int(row["n_pre"])
        assert float(row["p"]) == pytest.approx(
            float(scipy_chi2.sf(float(row["stat"]), int(row["df"]))),
            rel=1e-10, abs=1e-300)
    # G6: the all-rings joint pre-trend row does not ship.
    assert -1 not in set(res.pretrend["ring"])
    assert "joint" not in set(res.pretrend["label"])


def test_outer_ring_contrast_bound_and_share(base_fit):
    res = base_fit
    j = len(res.coef) - 1
    z = 1.959963984540054
    assert res.outer_ring_contrast_bound == pytest.approx(
        abs(res.coef[j]) + z * np.sqrt(res.vcov[j, j]), rel=1e-10)
    assert res.outer_ring_contrast_share == pytest.approx(
        res.outer_ring_contrast_bound / abs(res.direct_effect), rel=1e-10)
    assert "CONTRAST" in res.summary()
    assert "common to both is not identified" in res.summary().lower()


def test_outer_ring_test_rejects_when_the_spillover_reaches_the_outer_band():
    quiet, xy_q, _ = make_panel(n=200, T=12, n_treated=20, seed=17,
                                levels=(1.0, 0.5, 0.25, 0.0, 0.0))
    loud, xy_l, _ = make_panel(n=200, T=12, n_treated=20, seed=17,
                               levels=(1.0, 0.8, 0.7, 0.6, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_quiet = spatial_did(quiet, coords=xy_q, metric="euclidean", cutoff_km=CUT)
        r_loud = spatial_did(loud, coords=xy_l, metric="euclidean", cutoff_km=CUT)
    assert float(r_quiet.tests.set_index("test").loc["outer_ring", "p"]) > 0.10
    assert float(r_loud.tests.set_index("test").loc["outer_ring", "p"]) < 0.01
    assert abs(r_quiet.coef[-1]) < 2 * np.sqrt(r_quiet.vcov[-1, -1])
    assert r_quiet.outer_ring_contrast_bound < 0.40 * abs(r_quiet.direct_effect)


def test_outer_ring_share_is_nan_below_the_documented_floor(monkeypatch, base):
    """The share is a ratio to |direct_effect|; below the documented floor it is
    reported as NaN rather than as an inf-shaped number."""
    import importlib

    # The package __init__ re-exports the FUNCTION under the module's name
    # (house style, cf. did.callaway_santanna), so import the module itself.
    mod = importlib.import_module("puremacro.did.spatial_did")

    df, xy, _ra = base
    monkeypatch.setattr(mod, "_SHARE_FLOOR", 1e6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = mod.spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert np.isfinite(res.outer_ring_contrast_bound)
    assert np.isnan(res.outer_ring_contrast_share)
    assert "CONTRAST" in res.summary()


def _hand_total_weights(res):
    """N_0 = ever-treated units; N_r = never-treated units entering at ring r."""
    assign = res.assignment
    ever = assign.frame.groupby("unit", sort=False)["treated"].any()
    first = assign.frame.dropna(subset=["entry_ring"]).groupby(
        "unit", sort=False).head(1).set_index("unit")["entry_ring"].astype(int)
    n0 = int(ever.sum())
    return np.array([1.0 if r == 0
                     else int(((first == r) & ~first.index.map(ever)).sum()) / n0
                     for r in res.ring_codes]), n0, ever, first


def test_total_effect_is_per_treated_unit_and_counts_each_unit_once():
    df, xy, _ra = make_panel(cohorts=(3, 7), n_treated=12, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    a, n0, ever, first = _hand_total_weights(res)
    assert n0 == res.n_treated_units
    assert res.total_effect == pytest.approx(float(a @ res.coef), rel=1e-10)
    assert res.total_effect_se == pytest.approx(
        float(np.sqrt(a @ res.vcov @ a)), rel=1e-10)
    # Every exposed unit is counted exactly once: treated in N_0, never-treated
    # in the N_r of its entry ring.
    n_spill = sum(int(((first == r) & ~first.index.map(ever)).sum())
                  for r in range(1, len(res.ring_edges)))
    exposed = int(first.index.size)
    assert n0 + n_spill == exposed


def test_total_effect_denominator_is_treated_units_not_entry_ring_zero():
    """The defect the audit found. A late cohort's treated unit that spent its
    pre-period inside an EARLIER cohort's ring has entry ring 1, not 0. The old
    convention left it out of N_0 while still counting it in a spillover N_r,
    inflating every ratio; here that inflates the aggregate by ~2.1x."""
    rng = np.random.default_rng(11)
    n, T = 150, 12
    ids = [f"u{i:03d}" for i in range(n)]
    x = rng.uniform(0, 400, n)
    y = rng.uniform(0, 400, n)
    x[0], y[0] = 50.0, 50.0                    # early cohort
    x[1], y[1] = 300.0, 300.0                  # early cohort
    x[2], y[2] = 60.0, 50.0                    # late cohort, 10 units from u000
    x[3], y[3] = 312.0, 300.0                  # late cohort, 12 units from u001
    xy = pd.DataFrame({"x": x, "y": y}, index=ids)
    tt = pd.Series({u: (3.0 if i < 2 else (7.0 if i < 4 else np.nan))
                    for i, u in enumerate(ids)})
    yv = (rng.normal(size=n)[:, None] + 0.2 * rng.normal(size=T)[None, :]
          + 0.3 * rng.normal(size=(n, T)))
    df = pd.DataFrame([{"unit": u, "time": t, "y": float(yv[i, t]),
                        "treat_time": tt[u]}
                       for i, u in enumerate(ids) for t in range(T)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    a, n0, ever, first = _hand_total_weights(res)
    assert res.n_treated_units == 4
    assert int((first == 0).sum()) == 2        # only the early cohort enters at 0
    assert n0 == 4                             # ...but four units are treated
    assert res.total_effect == pytest.approx(float(a @ res.coef), rel=1e-10)
    # The superseded convention (divide by #entry-ring-0, count treated units
    # in the spillover numerators) is a different, much larger number.
    old = np.array([1.0 if r == 0 else int((first == r).sum()) / 2
                    for r in res.ring_codes])
    assert abs(float(old @ res.coef)) > 1.9 * abs(res.total_effect)


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------
def test_event_study_has_no_reference_cell_and_recovers_the_step(base_fit):
    res = base_fit
    es = res.event_study
    assert not (es["event_time"] == -1).any()
    for r, truth in zip(res.ring_codes, LEVELS):
        post = es[(es["ring"] == r) & (es["event_time"] >= 0)]
        assert np.all(np.abs(post["att"] - truth) < 4 * post["se"])
        pre = es[(es["ring"] == r) & (es["event_time"] < 0)]
        assert np.all(np.abs(pre["att"]) < 4 * pre["se"])


def test_sunab_equals_twfe_with_a_single_cohort(base):
    """The cohort interaction is vacuous when there is one cohort, so the two
    routes must agree to machine precision."""
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean",
                        event_study_estimator="twfe", cutoff_km=CUT)
        b = spatial_did(df, coords=xy, metric="euclidean",
                        event_study_estimator="sunab", cutoff_km=CUT)
    m = a.event_study.merge(b.event_study, on=["ring", "event_time"],
                            suffixes=("_t", "_s"))
    assert len(m) == len(a.event_study)
    assert np.abs(m["att_t"] - m["att_s"]).max() < 1e-10
    assert np.abs(m["se_t"] - m["se_s"]).max() < 1e-10
    assert a.event_study_estimator == "twfe"
    assert b.event_study_estimator == "sunab"


def test_auto_routes_multi_cohort_panels_to_sunab(base_fit):
    df, xy, _ra = make_panel(cohorts=(3, 7), n_treated=12, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert res.n_cohorts > 1 and res.event_study_estimator == "sunab"
    assert base_fit.n_cohorts == 1 and base_fit.event_study_estimator == "twfe"


def test_sunab_aggregation_is_linear_in_the_cell_coefficients():
    """Rebuild delta_{r,e} from scratch: fit the (ring, cohort, event-time)
    cell regression independently, form the count-weighted aggregation matrix
    A by hand, and check that A @ beta and sqrt(diag(A V A')) reproduce the
    reported `att` and `se`. Under cutoff_km=0 / time_lags=0 /
    small_sample=False the module's Conley meat is HC0, which is rebuildable
    here, so the covariance claim is checked and not just asserted."""
    df, xy, ra = make_panel(cohorts=(3, 7), n_treated=14, seed=21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=0.0,
                          time_lags=0, small_sample=False,
                          event_study_estimator="sunab")
    es = res.event_study
    f = ra.frame
    first = f.dropna(subset=["entry_time"]).groupby(
        "unit", sort=False).head(1).set_index("unit")
    m = df.copy()
    m["a"] = m["unit"].map(first["entry_time"])
    m["er"] = m["unit"].map(first["entry_ring"].astype("Float64")).astype(float)
    m["cur"] = f.set_index(["unit", "time"])["ring"].loc[
        list(zip(m["unit"], m["time"]))].to_numpy()
    m["e"] = m["time"] - m["a"]
    R = ra.n_rings
    exposed = m["a"].notna().to_numpy()
    e = m["e"].to_numpy(float)
    cur = m["cur"].to_numpy(int)
    esr = np.where(exposed & (e >= 0) & (cur >= 0) & (cur <= R), cur,
                   m["er"].to_numpy(float))
    coh = np.where(exposed, m["a"].to_numpy(float), -1.0)
    keep_c = {c for c in m["a"].dropna().unique()
              if ((m["a"] == c) & (m["e"] == -1)).any()}
    row_ok = m["a"].isna().to_numpy() | m["a"].isin(keep_c).to_numpy()
    active = row_ok & exposed & np.isfinite(e) & (e != -1)
    kept = set(res.ring_codes)
    cells = sorted({(int(r), int(c), int(ee))
                    for r, c, ee in zip(esr[active], coh[active], e[active])
                    if int(r) in kept})
    sub = m.loc[row_ok, ["unit", "time", "y"]].copy()
    counts = {}
    names = []
    for k, (r, c, ee) in enumerate(cells):
        msk = active & (esr == r) & (e == ee) & (coh == c)
        nm = f"D_{r}_{c}_{ee}"
        sub[nm] = msk[row_ok].astype(float)
        counts[k] = float(msk.sum())
        names.append(nm)
    assert all(counts[k] > 0 for k in counts)
    panel, ent, tim = as_panel_index(sub, entity_level="unit",
                                     time_level="time", unit_col="unit",
                                     time_col="time")
    fit = two_way_fe_within(panel, y_col="y", x_cols=names, entity_level=ent,
                            time_level=tim)
    Xw, u, B = fit["X_within"], fit["residuals"], fit["XtX_inv"]
    V = B @ (Xw.T @ (Xw * u[:, None] ** 2)) @ B
    beta = np.asarray(fit["beta"], float)

    out_cells = sorted({(r, ee) for (r, _c, ee) in cells})
    A = np.zeros((len(out_cells), len(cells)))
    for i, (r, ee) in enumerate(out_cells):
        idx = [k for k, (rr, _c, e2) in enumerate(cells) if rr == r and e2 == ee]
        tot = sum(counts[k] for k in idx)
        for k in idx:
            A[i, k] = counts[k] / tot
    # A is a genuine aggregation here, not a relabelling.
    assert len(cells) > len(out_cells)
    assert int((A > 0).sum(axis=1).max()) > 1
    assert np.allclose(A.sum(axis=1), 1.0)

    delta = A @ beta
    V_agg = A @ V @ A.T
    se = np.sqrt(np.diag(V_agg))
    got = es.set_index(["ring", "event_time"])
    assert len(out_cells) == len(es)
    for i, (r, ee) in enumerate(out_cells):
        assert float(got.loc[(r, ee), "att"]) == pytest.approx(
            float(delta[i]), rel=1e-10, abs=1e-12)
        assert float(got.loc[(r, ee), "se"]) == pytest.approx(
            float(se[i]), rel=1e-9, abs=1e-12)
    # The same rebuilt V_agg has to reproduce `att_by_ring_es`, which
    # aggregates a second time across event times — and there the off-diagonal
    # terms are worth roughly a factor of two, so an implementation that summed
    # variances instead of carrying A V A' would be caught here.
    att = res.att_by_ring_es.set_index("ring")
    for r in sorted({rr for rr, _ee in out_cells}):
        post = [i for i, (rr, ee) in enumerate(out_cells) if rr == r and ee >= 0]
        if not post:
            continue
        w = np.array([float(got.loc[out_cells[i], "n_obs"]) for i in post])
        w = w / w.sum()
        Bv = np.zeros(len(out_cells))
        Bv[post] = w
        assert float(att.loc[r, "att"]) == pytest.approx(
            float(Bv @ delta), rel=1e-10, abs=1e-12)
        full = float(np.sqrt(Bv @ V_agg @ Bv))
        assert float(att.loc[r, "se"]) == pytest.approx(full, rel=1e-9)
        diag_only = float(np.sqrt((Bv ** 2) @ np.diag(V_agg)))
        assert full > 1.3 * diag_only, (r, full, diag_only)


def test_att_by_ring_es_is_a_count_weighted_average_of_the_dynamic_path(base_fit):
    res = base_fit
    es = res.event_study
    for _, row in res.att_by_ring_es.iterrows():
        post = es[(es["ring"] == row["ring"]) & (es["event_time"] >= 0)]
        w = post["n_obs"].to_numpy(float)
        w = w / w.sum()
        assert float(row["att"]) == pytest.approx(
            float(w @ post["att"].to_numpy(float)), rel=1e-10)
        assert int(row["n_post_cells"]) == len(post)


def test_event_study_uses_the_current_ring_and_agrees_on_a_switcher_free_panel():
    """The critique's fix: cells are indexed by the CURRENT ring, so a unit that
    migrates inward stops dragging its old band's path with it. On a panel with
    no switchers the current-ring and entry-ring definitions coincide, which is
    what makes the two readings comparable at all."""
    df, xy, ra = make_panel(cohorts=(4,), n_treated=9, seed=5)
    assert ra.n_switchers == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    # Entry-ring cells, built by hand, reproduce the reported event study.
    entry = ra.frame.dropna(subset=["entry_ring"]).groupby("unit").head(1)
    entry = entry.set_index("unit")["entry_ring"].astype(int)
    m = df.copy()
    m["er"] = m["unit"].map(entry)
    m["ev"] = m["unit"].map(
        ra.frame.dropna(subset=["entry_time"]).groupby("unit").head(1)
        .set_index("unit")["entry_time"])
    m["e"] = m["time"] - m["ev"]
    cols = []
    for r in res.ring_codes:
        for e in sorted(res.event_study.loc[res.event_study["ring"] == r,
                                            "event_time"]):
            nm = f"D{r}_{e}"
            m[nm] = ((m["er"] == r) & (m["e"] == e)).astype(float)
            cols.append(nm)
    panel, ent, tim = as_panel_index(m, entity_level="unit", time_level="time",
                                     unit_col="unit", time_col="time")
    fit = two_way_fe_within(panel, y_col="y", x_cols=cols,
                            entity_level=ent, time_level=tim)
    assert np.abs(np.asarray(fit["beta"])
                  - res.event_study["att"].to_numpy(float)).max() < 1e-10


def test_event_study_current_ring_does_not_inherit_a_switcher_drift():
    """The assertion that catches the entry-ring bookkeeping bug the critique
    described. On a staggered panel where units migrate inward, an event study
    indexed by the ENTRY ring makes the outer band's dynamic path drift upward
    for a purely mechanical reason — the cells fill with rows whose current
    exposure is now ring 1 or 2. Indexing by the CURRENT ring removes the drift,
    and the true effect really is constant in event time here."""
    df, xy, ra = make_panel(cohorts=(3, 6, 9), n_treated=24, n=200, T=13,
                            seed=29, sigma=0.35)
    assert ra.n_switchers > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)

    first = ra.frame.dropna(subset=["entry_ring"]).groupby("unit").head(1)
    first = first.set_index("unit")
    m = df.copy()
    m["er"] = m["unit"].map(first["entry_ring"].astype("Float64")).astype(float)
    m["e"] = m["time"] - m["unit"].map(first["entry_time"])
    cols, keys = [], []
    for r in res.ring_codes:
        for e in range(-3, 10):
            if e == -1:
                continue
            v = ((m["er"] == r) & (m["e"] == e)).astype(float)
            if v.any():
                nm = f"D{r}_{e}"
                m[nm] = v
                cols.append(nm)
                keys.append((r, e))
    panel, ent, tim = as_panel_index(m, entity_level="unit", time_level="time",
                                     unit_col="unit", time_col="time")
    fit = two_way_fe_within(panel, y_col="y", x_cols=cols, entity_level=ent,
                            time_level=tim)
    entry_path = dict(zip(keys, np.asarray(fit["beta"], dtype=float)))

    def slope(pairs):
        e = np.array([p[0] for p in pairs], float)
        b = np.array([p[1] for p in pairs], float)
        return float(np.polyfit(e, b, 1)[0])

    outer = max(res.ring_codes)
    es = res.event_study
    cur = es[(es["ring"] == outer) & (es["event_time"] >= 0)]
    cur_pairs = list(zip(cur["event_time"], cur["att"]))
    ent_pairs = [(e, v) for (r, e), v in entry_path.items()
                 if r == outer and e >= 0]
    s_entry = slope(sorted(ent_pairs))
    s_cur = slope(sorted(cur_pairs))
    assert s_entry > 0.015, s_entry              # the mechanical drift
    # The current-ring path carries no such upward drift (here it is faintly
    # negative); the signed comparison is the claim, since the bug is a drift
    # TOWARDS the inner bands.
    assert s_cur < 0.5 * s_entry, (s_cur, s_entry)
    ent_sorted = [v for _e, v in sorted(ent_pairs)]
    assert ent_sorted[-1] - ent_sorted[0] > 0.15, ent_sorted
    # The static coefficient (the headline) is unaffected and on target.
    assert abs(res.coef[-1] - LEVELS[outer]) < 3 * np.sqrt(res.vcov[-1, -1])


def test_event_window_bins_the_endpoints(base):
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean",
                          event_window=(-2, 2), cutoff_km=CUT)
    es = res.event_study
    assert set(es["event_time"].unique()) == {-2, 0, 1, 2}
    binned = es[(es["ring"] == 0) & (es["event_time"] == 2)]
    unbinned_n = len(df[df["unit"].isin(
        res.assignment.frame.loc[res.assignment.frame["entry_ring"] == 0,
                                 "unit"].unique())]) // 10
    assert int(binned["n_obs"].iloc[0]) > unbinned_n
    with pytest.raises(ValueError, match="event_window must be"):
        spatial_did(df, coords=xy, metric="euclidean", event_window=(1, 3), cutoff_km=CUT)


def test_max_params_guard(base):
    df, xy, _ra = base
    with pytest.raises(ValueError, match="max_params"):
        spatial_did(df, coords=xy, metric="euclidean", max_params=5, cutoff_km=CUT)


def test_event_study_estimator_is_rejected_when_there_is_no_event_study(base):
    df, xy, _ra = base
    with pytest.raises(ValueError, match="meaningless with event_study=False"):
        spatial_did(df, coords=xy, metric="euclidean", event_study=False,
                    event_study_estimator="twfe", cutoff_km=CUT)


def test_event_time_is_measured_in_period_positions_not_raw_dates(base):
    """A panel indexed by Timestamps must give the same integer event times as
    the same panel indexed by 0..T-1."""
    df, xy, _ra = base
    stamps = pd.date_range("2000-01-01", periods=df["time"].nunique(), freq="YS")
    dated = df.copy()
    dated["time"] = dated["time"].map(dict(enumerate(stamps)))
    dated["treat_time"] = dated["treat_time"].map(
        lambda g: pd.NaT if pd.isna(g) else stamps[int(g)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        b = spatial_did(dated, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert np.abs(a.coef - b.coef).max() < 1e-12
    m = a.event_study.merge(b.event_study, on=["ring", "event_time"],
                            suffixes=("_i", "_d"))
    assert len(m) == len(a.event_study)
    assert np.abs(m["att_i"] - m["att_d"]).max() < 1e-12


# ---------------------------------------------------------------------------
# Pre-trends and placebo
# ---------------------------------------------------------------------------
def test_pretrend_is_quiet_under_parallel_trends_and_fires_on_a_planted_one():
    df, xy, _ra = make_panel(n=200, T=12, n_treated=20, seed=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clean = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert (clean.pretrend["p"] > 0.05).all(), clean.pretrend

    trend, xy_t, _ = make_panel(n=200, T=12, n_treated=20, seed=17,
                                pre_trend_ring=1, pre_trend_slope=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = spatial_did(trend, coords=xy_t, metric="euclidean", cutoff_km=CUT)
    pt = bad.pretrend.set_index("ring")
    assert float(pt.loc[1, "p"]) < 0.01
    assert float(pt.loc[3, "p"]) > 0.05
    # The all-rings joint row does not ship (G6); a Holm step-down over the
    # per-ring rows is the joint statement, and it still fires on ring 1.
    assert -1 not in set(bad.pretrend["ring"])
    ps = np.sort(bad.pretrend["p"].to_numpy(float))
    m = len(ps)
    holm = np.maximum.accumulate(ps * (m - np.arange(m)))
    assert float(holm[0]) < 0.01


def test_placebo_is_flat_under_parallel_trends():
    df, xy, _ra = make_panel(n=200, T=12, n_treated=20, seed=17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", placebo_periods=2, cutoff_km=CUT)
    pl = res.placebo
    assert pl is not None and len(pl) == len(res.ring_codes)
    assert np.all(np.abs(pl["effect"]) <= 2.5 * pl["se"]), pl


def test_placebo_drops_short_cohorts_and_degrades_to_none():
    df, xy, _ra = make_panel(cohorts=(2, 7), n_treated=12, seed=3)
    with pytest.warns(RuntimeWarning, match="fewer than"):
        res = spatial_did(df, coords=xy, metric="euclidean", placebo_periods=2, cutoff_km=CUT)
    assert res.placebo is not None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res2 = spatial_did(df, coords=xy, metric="euclidean",
                           placebo_periods=9, cutoff_km=CUT)
    assert res2.placebo is None
    assert any("placebo" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# Routes, gates and degenerate panels
# ---------------------------------------------------------------------------
def test_the_three_routes_agree_bit_for_bit(base):
    df, xy, ra = base
    codes = ra.frame.set_index(["unit", "time"])["ring"]
    with_col = df.assign(
        rc=codes.loc[list(zip(df["unit"], df["time"]))].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        b = spatial_did(df, assignment=ra, coords=xy, cutoff_km=CUT)
        c = spatial_did(with_col, ring_col="rc", coords=xy, metric="euclidean", cutoff_km=CUT)
    for other in (b, c):
        assert np.array_equal(a.coef, other.coef)
        assert np.array_equal(a.vcov, other.vcov)
        assert a.ring_codes == other.ring_codes
    assert b.assignment is ra
    assert c.assignment is None


def test_ring_col_validation(base):
    df, xy, ra = base
    codes = ra.frame.set_index(["unit", "time"])["ring"]
    good = df.assign(rc=codes.loc[list(zip(df["unit"], df["time"]))].to_numpy())
    with pytest.raises(ValueError, match=r"outside 0\.\.4"):
        spatial_did(good.assign(rc=good["rc"] + 3), ring_col="rc", coords=xy,
                    metric="euclidean", cutoff_km=CUT)
    with pytest.raises(ValueError, match="integer ring codes"):
        spatial_did(good.assign(rc=good["rc"] + 0.5), ring_col="rc", coords=xy,
                    metric="euclidean", cutoff_km=CUT)
    with pytest.raises(ValueError, match="cov_type='conley' needs unit"):
        spatial_did(good, ring_col="rc", cutoff_km=CUT)
    with pytest.raises(ValueError, match="not a column"):
        spatial_did(df, ring_col="nope", coords=xy, metric="euclidean", cutoff_km=CUT)


def test_no_clean_control_group_raises_before_any_linalg_error(base):
    """The modal failure: with every unit inside the outermost band the ring
    dummies sum to a function of the period alone. It must be caught by name,
    not surface as an opaque singular X'X."""
    df, xy, _ra = base
    with pytest.raises(ValueError) as exc:
        spatial_did(df, coords=xy, metric="euclidean", rings=(0.0, 10_000.0), cutoff_km=CUT)
    msg = str(exc.value)
    assert "spatial_did" in msg and "control" in msg and "rings" in msg
    assert "singular" not in msg


def test_include_not_yet_treated_false_keeps_delta_zero_identified():
    """The blocking bug: the flag must drop only not-yet-treated units sitting
    *inside a ring*, never the pre-period of a treated unit."""
    df, xy, _ra = make_panel(cohorts=(3, 7), n_treated=14, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        on = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        off = spatial_did(df, coords=xy, metric="euclidean",
                          include_not_yet_treated_in_rings=False, cutoff_km=CUT)
    assert np.isfinite(off.direct_effect)
    assert np.isfinite(off.direct_se) and off.direct_se > 0
    assert off.include_not_yet_treated_in_rings is False
    # The rows removed are EXACTLY the not-yet-treated units' in-ring rows —
    # an exact count, not a tolerance. Every pre-treatment row of a treated
    # unit survives, which is what keeps delta_0 identified.
    ring = _ra.frame.pivot(index="unit", columns="time", values="ring")
    ring = ring.loc[list(xy.index), sorted(df["time"].unique())].to_numpy()
    g = (df.drop_duplicates("unit").set_index("unit")["treat_time"]
         .reindex(list(xy.index)).to_numpy(float))
    t = np.asarray(sorted(df["time"].unique()), dtype=float)
    not_yet = np.isfinite(g)[:, None] & (t[None, :] < np.nan_to_num(g, nan=-1e18)[:, None])
    expected = int((not_yet & (ring >= 1) & (ring <= _ra.n_rings)).sum())
    assert expected > 0
    assert off.n_obs_dropped == expected
    assert on.n_obs_dropped == 0
    assert off.n_obs + off.n_obs_dropped == on.n_obs
    # Treated units keep their whole pre-period: ring 0's unit-period count is
    # untouched, and delta_0 still recovers the planted 1.0.
    n0_on = int(on.ring_table.set_index("ring").loc[0, "n_obs"])
    n0_off = int(off.ring_table.set_index("ring").loc[0, "n_obs"])
    assert n0_on == n0_off
    assert abs(off.direct_effect - LEVELS[0]) < 3 * off.direct_se
    assert abs(on.direct_effect - LEVELS[0]) < 3 * on.direct_se
    # ... and the two fits are within sampling noise of each other.
    sd = float(np.hypot(off.direct_se, on.direct_se))
    assert abs(off.direct_effect - on.direct_effect) < 3 * sd


def test_no_pre_period_at_all_raises_by_name():
    """Every exposed unit exposed in the first observed period: the unit fixed
    effect absorbs every ring dummy, and the message must say so rather than
    letting `inv_xtx` speak."""
    df, xy, _ra = make_panel(cohorts=(0,), n_treated=9, seed=5)
    with pytest.raises(ValueError) as exc:
        spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    msg = str(exc.value)
    assert "spatial_did" in msg and "pre-period" in msg


def test_a_ring_constant_within_every_carrier_raises_by_name():
    """The blocking gate BUILD_RULES demanded, reached for real. Ring 1 is
    carried by exactly one unit, which sits 10 units from a cohort treated in
    the FIRST observed period, so its ring-1 dummy is on in every period and
    the unit fixed effect absorbs it. A second, later cohort keeps the separate
    'no pre-period' gate from firing first, so this test cannot pass by
    accident on the wrong error — the previous version of it did exactly that.
    """
    ids = ["A", "B", "C", "D", "E", "F"]
    xy = pd.DataFrame(
        {"x": [0.0, 10.0, 5_000.0, 12_000.0, 30_000.0, 60_000.0],
         "y": [0.0] * 6}, index=ids)
    tt = pd.Series({"A": 0.0, "C": 3.0, "B": np.nan, "D": np.nan,
                    "E": np.nan, "F": np.nan})[ids]
    rng = np.random.default_rng(0)
    T = 6
    rows = [{"unit": u, "time": t,
             "y": float(rng.normal() + 0.1 * t + 0.05 * rng.normal()),
             "treat_time": tt[u]}
            for u in ids for t in range(T)]
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = exposure_rings(xy, treat_time=tt, times=list(range(T)),
                            rings=RINGS, metric="euclidean")
    # B really is in ring 1 in every single period — that is the design.
    b = ra.frame[ra.frame["unit"] == "B"]["ring"].to_numpy()
    assert set(b.tolist()) == {1}
    # ... and the 'no pre-period' gate must NOT be the one that fires: C enters
    # at t = 3, so not every exposed unit is exposed at t = 0.
    with pytest.raises(ValueError) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    msg = str(exc.value)
    assert "spatial_did" in msg
    assert "ring-1 indicator" in msg
    assert "constant within every unit that carries it" in msg
    assert "delta_1 is not identified" in msg
    assert "pre-period" not in msg
    assert "singular" not in msg
    # The same gate names the flag when the flag is what removed the variation.
    df2, xy2, _ = make_panel(cohorts=(3, 7), n_treated=14, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = spatial_did(df2, coords=xy2, metric="euclidean", cutoff_km=CUT,
                         include_not_yet_treated_in_rings=False)
    assert np.all(np.isfinite(ok.coef))


def test_min_units_per_ring_drops_rows_not_columns(base):
    """The tiny ring's rows leave the sample entirely; they are never moved into
    the omitted category, which would contaminate the control group. The proof
    is that the surviving coefficients equal a hand-built fit on exactly the
    reduced sample."""
    df, xy, ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        cut = spatial_did(df, coords=xy, metric="euclidean",
                          min_units_per_ring=6, cutoff_km=CUT)
    dropped = set(full.ring_codes) - set(cut.ring_codes)
    assert dropped
    assert all(bool(cut.ring_table.set_index("ring").loc[r, "dropped"])
               for r in dropped)
    m = df.merge(ra.frame[["unit", "time", "ring"]], on=["unit", "time"])
    m = m[~m["ring"].isin(dropped)]
    cols = []
    for r in cut.ring_codes:
        m[f"R{r}"] = (m["ring"] == r).astype(float)
        cols.append(f"R{r}")
    panel, ent, tim = as_panel_index(m, entity_level="unit", time_level="time",
                                     unit_col="unit", time_col="time")
    fit = two_way_fe_within(panel, y_col="y", x_cols=cols, entity_level=ent,
                            time_level=tim)
    assert np.abs(np.asarray(fit["beta"]) - cut.coef).max() < 1e-12
    assert cut.n_obs == int(fit["n_obs"])


def test_unbalanced_panel_and_a_constant_outcome(base):
    df, xy, _ra = base
    rng = np.random.default_rng(7)
    thin = df[rng.random(len(df)) > 0.05]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(thin, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert res.n_obs == len(thin)
    assert np.all(np.isfinite(res.coef))
    flat = df.assign(y=1.0)
    with pytest.raises(ValueError, match="no variation in the estimation"):
        spatial_did(flat, coords=xy, metric="euclidean", cutoff_km=CUT)


def test_degenerate_panels_raise_by_name(base):
    df, xy, _ra = base
    cases = [
        (df[df["time"] == 0], {}, "at least two periods"),
        (df.assign(treat_time=np.nan), {}, "no treated units"),
        (df.assign(treat_time=1.0), {}, "no clean control group"),
    ]
    for frame, kw, needle in cases:
        with pytest.raises(ValueError) as exc:
            spatial_did(frame, coords=xy, metric="euclidean", cutoff_km=CUT,
                        **kw)
        assert "spatial_did" in str(exc.value)
        assert needle in str(exc.value)
    bad = df.copy()
    bad.loc[bad.index[0], "treat_time"] = 99.0
    with pytest.raises(ValueError, match="constant within a unit"):
        spatial_did(bad, coords=xy, metric="euclidean", cutoff_km=CUT)
    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        spatial_did(dup, coords=xy, metric="euclidean", cutoff_km=CUT)
    with pytest.raises(KeyError, match="missing"):
        spatial_did(df, coords=xy.iloc[:-3], metric="euclidean", cutoff_km=CUT)


def test_a_non_finite_outcome_is_refused_by_name(base):
    """NaN *and* inf. Both used to slip through: `.notna()` dropped the NaN
    rows silently (and `n_obs_dropped` stayed 0 because the drop happened
    before it was computed), while an inf sailed past `.notna()` into the HAC
    and died there with a bare LAPACK `LinAlgError`. `spatial_panel` and
    `var.gvar` both refuse a non-finite column by name; so does this."""
    df, xy, _ra = base
    for bad in (np.nan, np.inf, -np.inf):
        frame = df.copy()
        frame.loc[frame.index[:3], "y"] = bad
        with pytest.raises(ValueError) as exc:
            spatial_did(frame, coords=xy, metric="euclidean", cutoff_km=CUT)
        msg = str(exc.value)
        assert "spatial_did" in msg          # the function names itself
        assert "'y'" in msg                  # and the column
        assert "3 non-finite" in msg         # and the count
    with pytest.raises(ValueError, match="is not numeric"):
        spatial_did(df.assign(y="not a number"), coords=xy, metric="euclidean",
                    cutoff_km=CUT)


def test_n_obs_dropped_accounts_for_every_input_row(base):
    """`n_obs + n_obs_dropped` is the number of rows handed in, and
    `summary()` prints the shortfall. Missing *rows* are still fine -- the
    panel may be unbalanced -- they simply never enter either count."""
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT,
                          include_not_yet_treated_in_rings=False,
                          min_units_per_ring=8)
    assert res.n_obs_dropped > 0
    assert res.n_obs + res.n_obs_dropped == len(df)
    assert f"({res.n_obs_dropped} dropped)" in res.summary()

    thin = df.drop(df.index[:7])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res_thin = spatial_did(thin, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert res_thin.n_obs == len(thin)
    assert res_thin.n_obs_dropped == 0


def test_a_zero_sandwich_variance_is_blanked_like_spatial_panel():
    """A zero variance is not a standard error. `spatial_panel` calls a
    non-positive sandwich diagonal bad (`nan`); this module used to accept
    exactly 0 and return `se = 0`, which makes `t` infinite. A perfect fit
    (zero residuals) sets the cluster meat, and so the whole sandwich, to
    exactly zero -- the boundary case."""
    rng = np.random.default_rng(0)
    n, k = 40, 2
    X = rng.normal(size=(n, k))
    fit = {
        "X_within": X,
        "residuals": np.zeros(n),
        "XtX_inv": np.linalg.inv(X.T @ X),
        "entity_keys": np.repeat(np.arange(20), 2),
        "time_keys": np.tile(np.arange(2), 20),
    }
    with pytest.warns(RuntimeWarning, match="non-positive"):
        cov = _sandwich(fit, cov_type="cluster", coords_df=None, cutoff_km=None,
                        time_lags=0, kernel="bartlett", metric="euclidean",
                        cluster_keys=None, small_sample=False,
                        psd_adjust="warn", func="spatial_did")
    assert np.allclose(np.diag(cov["vcov"]), 0.0)
    assert cov["n_negative_variance"] == k
    assert np.all(np.isnan(cov["se"]))


def test_unknown_kwarg_raises_typeerror(base):
    df, xy, _ra = base
    with pytest.raises(TypeError):
        spatial_did(df, coords=xy, metric="euclidean", ringz=(0.0, 25.0), cutoff_km=CUT)


def test_choice_arguments_list_the_valid_values(base):
    df, xy, _ra = base
    for kw, needle in [
        (dict(event_study_estimator="ols"), "'auto', 'twfe', 'sunab'"),
        (dict(ring_timing="nearest"), "'already_treated', 'ever_treated'"),
        (dict(cov_type="hc3"), "'conley', 'cluster'"),
        (dict(kernel="parzen"), "'bartlett', 'uniform'"),
        (dict(psd_adjust="fix"), "'none', 'warn', 'clip'"),
        (dict(alpha=0.0), "alpha must lie in (0, 1)"),
    ]:
        with pytest.raises(ValueError) as exc:
            spatial_did(df, coords=xy, **{"metric": "euclidean", **kw})
        assert needle in str(exc.value)
    with pytest.raises(ValueError) as exc:
        spatial_did(df, coords=xy, metric="manhattan", cutoff_km=CUT)
    assert "'haversine', 'euclidean'" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        spatial_did(df, coords=xy, metric="graph", cutoff_km=CUT)
    assert "contiguity_rings" in str(exc.value)


def test_bare_array_coords_are_refused(base):
    """Aligning an (n, 2) array to the panel's unit ids means guessing the row
    order; a wrong guess silently corrupts every distance."""
    df, xy, _ra = base
    with pytest.raises(ValueError, match="bare array"):
        spatial_did(df, coords=xy.to_numpy(), metric="euclidean", cutoff_km=CUT)


def test_small_rings_trigger_the_joint_test_warning(base):
    """Joint Wald statistics on a ring carried by a handful of units are badly
    over-sized; the module must say so rather than print a confident p-value."""
    df, xy, _ra = base
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert any("JOINT Wald" in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------
def test_headline_is_always_the_static_twfe_fit(base):
    """`event_study_estimator` must not silently move the headline: the ring
    table, the decomposition and every Wald test come from one static fit."""
    df, xy, _ra = make_panel(cohorts=(3, 7), n_treated=14, seed=21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean",
                        event_study_estimator="twfe", cutoff_km=CUT)
        b = spatial_did(df, coords=xy, metric="euclidean",
                        event_study_estimator="sunab", cutoff_km=CUT)
        c = spatial_did(df, coords=xy, metric="euclidean", event_study=False, cutoff_km=CUT)
    for other in (b, c):
        assert np.array_equal(a.coef, other.coef)
        assert np.array_equal(a.vcov, other.vcov)
        assert a.naive_att == other.naive_att
        assert a.static_estimator == other.static_estimator == "twfe"
    assert "STATIC two-way-FE fit" in a.summary()
    assert "Goodman-Bacon" in a.summary()          # n_cohorts > 1


def test_frozen_and_presentation_contract(base):
    df, xy, ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", placebo_periods=2, cutoff_km=CUT)
    for obj in (res, ra):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, "metric", "nope")
    assert isinstance(res.summary(), str) and res.summary()
    assert isinstance(ra.summary(), str) and ra.summary()
    available = [w for w in ("rings", "event_study", "att_by_ring_es", "tests",
                             "pretrend", "placebo")
                 if getattr(res, {"rings": "ring_table"}.get(w, w)) is not None]
    for which in available:
        assert isinstance(res.to_frame(which), pd.DataFrame)
        for fmt in (res.to_markdown, res.to_latex, res.to_typst):
            assert fmt(which)
    for fmt in (ra.to_markdown, ra.to_latex, ra.to_typst):
        assert fmt("counts") and fmt("assignment")
    with pytest.raises(ValueError, match="which must be"):
        ra.to_frame("nope")


def test_unavailable_tables_raise_and_list_what_exists(base):
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", event_study=False, cutoff_km=CUT)
    for which in ("event_study", "att_by_ring_es", "pretrend", "placebo"):
        with pytest.raises(ValueError) as exc:
            res.to_frame(which)
        assert "available tables are" in str(exc.value)
        assert "'rings'" in str(exc.value)
    with pytest.raises(ValueError, match="which must be one of"):
        res.to_frame("nope")


def test_result_frames_are_private_copies(base):
    df, xy, _ra = base
    mutable = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(mutable, coords=xy, metric="euclidean", cutoff_km=CUT)
    before = res.ring_table.copy()
    mutable.loc[mutable.index[0], "y"] = 1e9
    handed = res.to_frame()
    handed.loc[0, "effect"] = -999.0
    assert res.ring_table.equals(before)


def test_ring_assignment_dtypes_and_unit_frame():
    _df, _xy, ra = make_panel(cohorts=(4,), n_treated=9, seed=5)
    assert ra.frame["ring"].dtype == np.int64
    assert str(ra.frame["entry_ring"].dtype) == "Int64"
    assert ra.frame["entry_ring"].isna().any()
    assert ra.is_static and ra.n_switchers == 0
    uf = ra.unit_frame()
    assert len(uf) == len(ra.ids)
    _d2, _x2, ra2 = make_panel(cohorts=(3, 6, 8), n_treated=15, seed=11)
    assert not ra2.is_static
    with pytest.raises(ValueError, match="time-varying"):
        ra2.unit_frame()


def test_plot_returns_figures(base):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, xy, ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        bare = spatial_did(df, coords=xy, metric="euclidean", event_study=False, cutoff_km=CUT)
    for fig in (res.plot(), res.plot(show_event_study=False), ra.plot(),
                bare.plot()):
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    fig, ax = plt.subplots()
    assert res.plot(ax=ax) is fig
    plt.close(fig)
    with pytest.raises(ValueError, match="carries no event study"):
        bare.plot(show_event_study=True)


def test_determinism_and_row_order_invariance(base):
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean", placebo_periods=1, cutoff_km=CUT)
        b = spatial_did(df, coords=xy, metric="euclidean", placebo_periods=1, cutoff_km=CUT)
        c = spatial_did(df.sample(frac=1.0, random_state=3), coords=xy,
                        metric="euclidean", placebo_periods=1, cutoff_km=CUT)
    for other in (b, c):
        assert np.array_equal(a.coef, other.coef)
        assert np.array_equal(a.vcov, other.vcov)
        pd.testing.assert_frame_equal(a.ring_table, other.ring_table)
        pd.testing.assert_frame_equal(a.event_study, other.event_study)
        pd.testing.assert_frame_equal(a.placebo, other.placebo)
        pd.testing.assert_frame_equal(a.tests, other.tests)


def test_multiindex_input_is_accepted(base):
    df, xy, _ra = base
    idxed = df.set_index(["unit", "time"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
        b = spatial_did(idxed, coords=xy, metric="euclidean", cutoff_km=CUT)
    assert np.array_equal(a.coef, b.coef)


def test_module_docstring_carries_the_mandatory_caveats():
    import importlib

    # The package __init__ re-exports the FUNCTION under the module's name
    # (house style, cf. did.callaway_santanna), so import the module itself.
    mod = importlib.import_module("puremacro.did.spatial_did")

    doc = mod.__doc__
    for needle in ("Deliberately omitted", "increasing-domain",
                   "contrast", "estimator='cs'", "0.515",
                   "joint pre-trend Wald"):
        assert needle in doc, needle
    assert mod.__all__ == ["spatial_did", "exposure_rings", "contiguity_rings",
                           "SpatialDiDResult", "RingAssignment"]
    assert issubclass(SpatialDiDResult, object)
    assert issubclass(RingAssignment, object)


@pytest.mark.mechanism_control
def test_import_is_pyodide_clean():
    """Importing the module must not drag in the heavy statistics stack, and
    matplotlib must stay inside `plot()`."""
    src = Path(spatial_did.__code__.co_filename).read_text(encoding="utf-8")
    for line in src.splitlines():
        if "import matplotlib" in line:
            assert line.startswith("        import matplotlib"), line
    code = (
        "import sys, importlib;"
        "importlib.import_module('puremacro.did.spatial_did');"
        "banned=[m for m in ('statsmodels','linearmodels','arch','geopandas',"
        "'libpysal','pysal') if m in sys.modules];"
        "print(banned)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         cwd=Path(__file__).resolve().parents[2],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------------------
# G6: joint Wald statistics are gated, and the gate is where the size breaks
# ---------------------------------------------------------------------------
def _null_panel(seed, n=90, T=10, n_treated=9, box=500.0, t0=4):
    """Pure two-way-FE noise: no treatment effect, no spillover, no pre-trend.
    Every rejection under this DGP is a size distortion."""
    rng = np.random.default_rng(seed)
    ids = [f"u{i:03d}" for i in range(n)]
    xy = pd.DataFrame({"x": rng.uniform(0, box, n), "y": rng.uniform(0, box, n)},
                      index=ids)
    tt = pd.Series({u: (float(t0) if i < n_treated else np.nan)
                    for i, u in enumerate(ids)})
    y = (rng.normal(size=n)[:, None] + 0.2 * rng.normal(size=T)[None, :]
         + 0.35 * rng.normal(size=(n, T)))
    df = pd.DataFrame([{"unit": u, "time": t, "y": float(y[i, t]),
                        "treat_time": tt[u]}
                       for i, u in enumerate(ids) for t in range(T)])
    return df, xy


def test_joint_wald_rows_are_blanked_when_a_ring_is_too_small(base_fit):
    """G6 made operational. The base design carries rings 0 and 1 on 9 and 5
    units; every joint restriction touching them comes back NaN with
    underpowered=True, while the per-coefficient rows — which ARE correctly
    sized — are untouched, and the rings that clear the gate still report."""
    res = base_fit
    carried = res.ring_table.set_index("ring")["n_units"]
    tiny = {r for r in res.ring_codes if int(carried.loc[r]) < _MIN_UNITS_FOR_JOINT}
    big = {r for r in res.ring_codes if int(carried.loc[r]) >= _MIN_UNITS_FOR_JOINT}
    assert tiny and big, carried                      # the design exercises both

    t = res.tests.set_index("test")
    # A single-coefficient Wald is correctly sized on a tiny ring: never gated.
    assert bool(t.loc["outer_ring", "underpowered"]) is False
    assert np.isfinite(float(t.loc["outer_ring", "p"]))
    # no_spillover / equal_rings span every spillover ring, so they are gated.
    for nm in ("no_spillover", "equal_rings"):
        assert bool(t.loc[nm, "underpowered"]) is True
        assert np.isnan(float(t.loc[nm, "stat"]))
        assert np.isnan(float(t.loc[nm, "p"]))
        assert int(t.loc[nm, "df"]) > 0               # the df is still reported
    for r in tiny:
        assert bool(t.loc[f"dynamics_ring{r}", "underpowered"]) is True
        assert np.isnan(float(t.loc[f"dynamics_ring{r}", "p"]))
    for r in big:
        assert bool(t.loc[f"dynamics_ring{r}", "underpowered"]) is False
        assert np.isfinite(float(t.loc[f"dynamics_ring{r}", "p"]))
    pt = res.pretrend.set_index("ring")
    for r in tiny:
        assert bool(pt.loc[r, "underpowered"]) is True and np.isnan(pt.loc[r, "p"])
    for r in big:
        assert bool(pt.loc[r, "underpowered"]) is False
        assert np.isfinite(float(pt.loc[r, "p"]))
    # Per-coefficient inference survives intact on the very same tiny rings.
    keep = res.ring_table[~res.ring_table["dropped"]].iloc[:len(res.ring_codes)]
    assert np.all(np.isfinite(keep["se"].to_numpy(float)))
    assert np.all(np.isfinite(keep["p"].to_numpy(float)))
    txt = res.summary()
    assert "NOT REPORTED" in txt and "underpowered=True" in txt


def test_the_gate_warns_and_names_the_rings(base):
    df, xy, _ra = base
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    msgs = [str(w.message) for w in rec if "JOINT Wald" in str(w.message)]
    assert msgs, [str(w.message) for w in rec]
    assert "underpowered=True" in msgs[0]
    assert f"fewer than {_MIN_UNITS_FOR_JOINT} units" in msgs[0]


def test_the_gated_joint_statistic_really_is_oversized():
    """The measurement the module docstring reports, run in-tree so the claim
    cannot rot. On a seeded NULL design with tiny rings, the no-spillover Wald
    — rebuilt by hand from `coef` and `vcov`, because the module refuses to
    print it — rejects far above its nominal 0.10. That is why it is gated."""
    ps, gated = [], 0
    for s in range(120):
        df, xy = _null_panel(20000 + s, n=90, T=10, n_treated=8, box=800.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = spatial_did(df, coords=xy, metric="euclidean",
                              cov_type="cluster")
        idx = [j for j, r in enumerate(res.ring_codes) if r >= 1]
        b = res.coef[idx]
        V = res.vcov[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.pinv(V, rcond=1e-10, hermitian=True) @ b)
        ps.append(float(scipy_chi2.sf(stat, len(idx))))
        gated += int(bool(res.tests.set_index("test").loc["no_spillover",
                                                          "underpowered"]))
    rej = float((np.asarray(ps) < 0.10).mean())
    assert rej > 0.25, rej                 # measured 0.453 over these 120 seeds
    assert gated >= 0.90 * len(ps), (gated, len(ps))   # the module refuses it


def test_the_joint_statistics_that_do_ship_are_approximately_sized():
    """The other half of the claim: once every ring clears the gate, the joint
    rows the module DOES report reject near their nominal level on a seeded
    null. Bounds are loose enough for Monte-Carlo noise and far below the
    0.26-0.45 the ungated small design produces."""
    ns, pre, any_gated = [], [], 0
    for s in range(80):
        df, xy = _null_panel(3000 + s, n=200, T=10, n_treated=22, box=500.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = spatial_did(df, coords=xy, metric="euclidean",
                              cov_type="cluster")
        any_gated += int(bool(res.tests["underpowered"].any()
                              or res.pretrend["underpowered"].any()))
        v = float(res.tests.set_index("test").loc["no_spillover", "p"])
        if np.isfinite(v):
            ns.append(v)
        pre.extend(float(x) for x in res.pretrend["p"] if np.isfinite(x))
    assert any_gated == 0, any_gated       # this design never trips the gate
    assert len(ns) >= 75 and len(pre) >= 300
    r_ns = float((np.asarray(ns) < 0.10).mean())
    r_pre = float((np.asarray(pre) < 0.10).mean())
    assert 0.01 <= r_ns <= 0.22, r_ns      # measured 0.10
    assert 0.02 <= r_pre <= 0.26, r_pre    # measured 0.17


# ---------------------------------------------------------------------------
# G3: keywords that do nothing in the chosen mode raise
# ---------------------------------------------------------------------------
def test_keywords_meaningless_for_the_chosen_mode_raise(base):
    df, xy, ra = base
    for kw, needles in [
        (dict(event_study=False, event_window=(-2, 2)),
         ("event_window", "event_study=False")),
        (dict(event_study=False, max_params=1),
         ("max_params", "event_study=False")),
        (dict(event_study=False, event_study_estimator="twfe"),
         ("event_study_estimator", "event_study=False")),
    ]:
        with pytest.raises(ValueError) as exc:
            spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT, **kw)
        msg = str(exc.value)
        assert "spatial_did" in msg
        for needle in needles:
            assert needle in msg, (needle, msg)
    # psd_adjust cannot fire under a cluster meat: it is a sum of outer
    # products and PSD by construction.
    for adj in ("clip", "none"):
        with pytest.raises(ValueError) as exc:
            spatial_did(df, coords=xy, metric="euclidean", cov_type="cluster",
                        psd_adjust=adj)
        assert "psd_adjust" in str(exc.value)
        assert "positive semi-definite" in str(exc.value)
    # ... but the default is accepted, so the gate is about the argument and
    # not about cov_type='cluster' itself.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = spatial_did(df, coords=xy, metric="euclidean", cov_type="cluster",
                         psd_adjust="warn")
    assert ok.psd_adjust == "warn"
    # And the same keywords are fine when the mode actually reads them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        live = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT,
                           event_window=(-2, 2), max_params=400)
    assert set(live.event_study["event_time"]) == {-2, 0, 1, 2}


def test_ring_col_route_refuses_to_echo_keywords_that_did_nothing(base):
    """G3 + G11: on the ring_col route the codes are the caller's, so nothing
    computes a timing and — under cov_type='cluster' — nothing computes a
    distance either. Those keywords raise instead of being echoed back on the
    result as if they had built the rings."""
    df, xy, ra = base
    codes = ra.frame.set_index(["unit", "time"])["ring"]
    with_col = df.assign(
        rc=codes.loc[list(zip(df["unit"], df["time"]))].to_numpy())
    for kw in (dict(ring_timing="ever_treated", cov_type="cluster"),
               dict(ring_timing="already_treated", cov_type="cluster"),
               dict(metric="haversine", cov_type="cluster"),
               dict(metric="euclidean", cov_type="cluster"),
               dict(ring_timing="ever_treated", cutoff_km=CUT)):
        with pytest.raises(ValueError) as exc:
            spatial_did(with_col, ring_col="rc", coords=xy, **kw)
        assert "ring_col" in str(exc.value)
        assert "meaningless" in str(exc.value)
    # metric IS read by the Conley kernel, so it stays legal there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        conley = spatial_did(with_col, ring_col="rc", coords=xy,
                             metric="euclidean", cutoff_km=CUT)
        clu = spatial_did(with_col, ring_col="rc", cov_type="cluster")
    for res in (conley, clu):
        assert res.ring_timing == "unknown"
        assert res.distance_unit == "unknown"
        # ... and the band labels carry no invented unit.
        assert "km" not in " ".join(res.ring_labels)
        assert "units" not in " ".join(res.ring_labels)
        assert "0-25" in " ".join(res.ring_labels)
    # An assignment fixes all three, so supplying any of them raises even at
    # its documented default value.
    for kw in (dict(metric="haversine"), dict(ring_timing="already_treated")):
        with pytest.raises(ValueError, match="meaningless when assignment"):
            spatial_did(df, assignment=ra, coords=xy, cutoff_km=CUT, **kw)


# ---------------------------------------------------------------------------
# RingAssignment.unit_frame
# ---------------------------------------------------------------------------
def test_unit_frame_reports_ever_treated_and_control_distances():
    """`frame['treated']` is the time-varying 1{t >= g_i}; collapsing it on the
    first period would call every staggered unit untreated while the same row
    says ring 0. And a never-exposed unit's distance is the number a user needs
    to defend the outermost cut point, so it must not be NaN."""
    xy = pd.DataFrame({"x": [0.0, 10.0, 40.0, 150.0, 400.0], "y": [0.0] * 5},
                      index=list("abcde"))
    tt = {"a": 1, "b": None, "c": None, "d": None, "e": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = exposure_rings(xy, treat_time=tt, times=[0, 1, 2], rings=RINGS,
                            metric="euclidean")
    uf = ra.unit_frame().set_index("unit")
    assert len(uf) == len(ra.ids)
    # 'a' is treated at t=1, i.e. after the first observed period.
    assert bool(uf.loc["a", "treated"]) is True
    assert int(uf.loc["a", "ring"]) == 0 and uf.loc["a", "ring_label"] == "treated"
    assert not uf.loc[uf.index != "a", "treated"].any()
    assert uf["treated"].dtype == bool
    # Ever-treated agrees with the long frame's own any().
    ever = ra.frame.groupby("unit", sort=False)["treated"].any()
    assert uf["treated"].reindex(ever.index).equals(ever.rename("treated"))
    # Never-exposed units carry their real last-period distance, not NaN.
    assert float(uf.loc["d", "distance"]) == pytest.approx(150.0)
    assert float(uf.loc["e", "distance"]) == pytest.approx(400.0)
    assert uf["distance"].notna().all()
    assert int(uf.loc["b", "ring"]) == 1 and int(uf.loc["c", "ring"]) == 2
    assert int(uf.loc["d", "ring"]) == ra.n_rings + 1
    # A treated unit exposed from the very first period is still ever-treated.
    tt0 = {"a": 0, "b": None, "c": None, "d": None, "e": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra0 = exposure_rings(xy, treat_time=tt0, times=[0, 1, 2], rings=RINGS,
                             metric="euclidean")
    assert bool(ra0.unit_frame().set_index("unit").loc["a", "treated"]) is True


# ---------------------------------------------------------------------------
# Placebo table column meanings
# ---------------------------------------------------------------------------
def test_placebo_n_units_and_n_obs_mean_what_ring_table_means(base):
    """`placebo` advertises the columns of `ring_table`; `n_units` must
    therefore be units carrying that ring, not the sample-wide unit count
    repeated on every row."""
    df, xy, _ra = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT,
                          placebo_periods=2)
    pl = res.placebo
    assert pl is not None
    n_panel_units = res.n_units
    assert pl["n_units"].nunique() > 1, pl        # not one constant repeated
    assert (pl["n_units"] < n_panel_units).all()
    assert (pl["n_obs"] < res.n_obs).all()
    # Ring 0 is carried by exactly the treated units, whatever the placebo
    # shift; the outer rings by more units than the inner ones here.
    assert int(pl.set_index("ring").loc[0, "n_units"]) == res.n_treated_units
    ring_units = res.ring_table.set_index("ring")["n_units"]
    for r in res.ring_codes:
        assert int(pl.set_index("ring").loc[r, "n_units"]) <= int(ring_units.loc[r])
    assert (pl["n_units"] >= 1).all()
    assert (pl["n_obs"] >= pl["n_units"]).all()


# ---------------------------------------------------------------------------
# Event-study cell counts
# ---------------------------------------------------------------------------
def test_event_study_counts_exclude_a_cohort_dropped_for_want_of_a_reference():
    """`n_obs` / `n_cohorts` are counted over the rows in the event-study
    regression, not over every estimation row: a cohort dropped for having no
    e = -1 cell must not turn up in the cell sizes."""
    df, xy, ra = make_panel(cohorts=(0, 4), n=120, T=10, n_treated=12, seed=5)
    with pytest.warns(RuntimeWarning, match="no e = -1 reference"):
        res = spatial_did(df, coords=xy, metric="euclidean", cutoff_km=CUT)
    f = ra.frame
    first = f.dropna(subset=["entry_time"]).groupby("unit", sort=False).head(1)
    first = first.set_index("unit")
    m = df.copy()
    m["a"] = m["unit"].map(first["entry_time"])
    m["e"] = m["time"] - m["a"]
    m["ring_now"] = f.set_index(["unit", "time"])["ring"].loc[
        list(zip(m["unit"], m["time"]))].to_numpy()
    m["er"] = m["unit"].map(first["entry_ring"].astype("Float64")).astype(float)
    m["esr"] = np.where((m["e"] >= 0) & (m["ring_now"] <= ra.n_rings),
                        m["ring_now"], m["er"])
    kept = {c for c in m["a"].dropna().unique()
            if ((m["a"] == c) & (m["e"] == -1)).any()}
    assert len(kept) == 1                       # the t=0 cohort has no e = -1
    differs = 0
    for _, row in res.event_study.iterrows():
        r, e = int(row["ring"]), int(row["event_time"])
        cell = (m["esr"] == r) & (m["e"] == e)
        assert int(row["n_obs"]) == int((cell & m["a"].isin(kept)).sum())
        assert int(row["n_cohorts"]) == int(
            m.loc[cell & m["a"].isin(kept), "a"].nunique())
        differs += int(int(cell.sum()) != int((cell & m["a"].isin(kept)).sum()))
    assert differs > 0                          # the masking is load-bearing


# ---------------------------------------------------------------------------
# Breaking the fixtures' circularity
# ---------------------------------------------------------------------------
def _brute_force_codes(coords, treat_pos, T, edges):
    """Nearest ALREADY-treated distance and band code by triple loop.

    Deliberately written from the definition — for each unit and period, scan
    every unit treated at or before that period — with no shared code path with
    `exposure_rings`. `make_panel` builds its DGP by calling `exposure_rings`,
    so without an independent assignment like this every planted-parameter test
    in this file would still pass if the ring codes were systematically wrong.
    """
    xy = coords.to_numpy(float)
    n = len(xy)
    R = len(edges) - 1
    code = np.empty((n, T), dtype=int)
    dist = np.empty((n, T), dtype=float)
    for t in range(T):
        for i in range(n):
            if treat_pos[i] is not None and treat_pos[i] <= t:
                code[i, t], dist[i, t] = 0, 0.0
                continue
            best = np.inf
            for j in range(n):
                if treat_pos[j] is None or treat_pos[j] > t:
                    continue
                d = float(np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
                best = min(best, d)
            dist[i, t] = best
            if not np.isfinite(best):
                code[i, t] = R + 1
            else:
                band = R + 1
                for r in range(1, R + 1):
                    lo, hi = edges[r - 1], edges[r]
                    if (best <= hi) if r == 1 else (lo < best <= hi):
                        band = r
                        break
                code[i, t] = band
    return code, dist


def test_exposure_rings_matches_a_brute_force_assignment_on_random_panels():
    edges = list(RINGS)
    T = 6
    for seed in range(25):
        rng = np.random.default_rng(2000 + seed)
        n = int(rng.integers(8, 26))
        ids = [f"u{i:02d}" for i in range(n)]
        xy = pd.DataFrame({"x": rng.uniform(0, 220, n),
                           "y": rng.uniform(0, 220, n)}, index=ids)
        n_tr = int(rng.integers(1, max(2, n // 3)))
        pos = [int(rng.integers(0, T)) if i < n_tr else None for i in range(n)]
        tt = pd.Series({u: (float(pos[i]) if pos[i] is not None else np.nan)
                        for i, u in enumerate(ids)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ra = exposure_rings(xy, treat_time=tt, times=list(range(T)),
                                rings=RINGS, metric="euclidean")
        got = ra.frame.pivot(index="unit", columns="time",
                             values="ring").loc[ids, list(range(T))].to_numpy()
        gotd = ra.frame.pivot(index="unit", columns="time",
                              values="distance").loc[
                                  ids, list(range(T))].to_numpy()
        want, wantd = _brute_force_codes(xy, pos, T, edges)
        assert np.array_equal(got, want), seed
        assert np.allclose(gotd, wantd, equal_nan=True), seed


def test_contiguity_rings_matches_a_hand_written_bfs_on_random_graphs():
    for seed in range(15):
        rng = np.random.default_rng(4000 + seed)
        n = int(rng.integers(6, 16))
        ids = tuple(f"n{i}" for i in range(n))
        M = (rng.random((n, n)) < 0.25).astype(float)
        M = np.triu(M, 1)
        M = M + M.T
        W = SpatialWeights(sp.csr_matrix(M), ids=ids)
        tr = int(rng.integers(0, n))
        tt = {u: (0.0 if k == tr else np.nan) for k, u in enumerate(ids)}
        orders = 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ra = contiguity_rings(W, treat_time=tt, times=[0, 1], orders=orders)
        # Hand-written breadth-first search from the single treated node.
        adj = [set(np.flatnonzero(M[i] > 0).tolist()) for i in range(n)]
        hop = {tr: 0}
        frontier = [tr]
        while frontier:
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if b not in hop:
                        hop[b] = hop[a] + 1
                        nxt.append(b)
            frontier = nxt
        got = ra.frame[ra.frame.time == 1].set_index("unit")["ring"]
        for k, u in enumerate(ids):
            h = hop.get(k)
            want = (orders + 1 if h is None else
                    (0 if h == 0 else min(h, orders + 1)))
            assert int(got[u]) == want, (seed, u, h)


def test_static_coefficients_match_an_explicit_lsdv_regression(base, base_fit):
    """`hand_within` reuses the package's own two-way projection, so it cannot
    catch a bug inside it. This rebuilds the fit the long way — full unit and
    time dummy matrices, one least-squares solve — and demands the same
    numbers."""
    df, xy, ra = base
    res = base_fit
    m = df.merge(ra.frame[["unit", "time", "ring"]], on=["unit", "time"])
    m = m.sort_values(["unit", "time"], kind="mergesort").reset_index(drop=True)
    units = sorted(m["unit"].unique())
    times = sorted(m["time"].unique())
    D_u = np.zeros((len(m), len(units)))
    D_u[np.arange(len(m)), [units.index(u) for u in m["unit"]]] = 1.0
    D_t = np.zeros((len(m), len(times)))
    D_t[np.arange(len(m)), [times.index(t) for t in m["time"]]] = 1.0
    Rg = np.column_stack([(m["ring"] == r).to_numpy(float)
                          for r in res.ring_codes])
    # Drop one time dummy for the unit-dummy collinearity.
    X = np.column_stack([Rg, D_u, D_t[:, 1:]])
    beta, *_ = np.linalg.lstsq(X, m["y"].to_numpy(float), rcond=None)
    assert np.abs(beta[:len(res.ring_codes)] - res.coef).max() < 1e-9
    # The naive DiD is the same regression with the spillover columns removed.
    X0 = np.column_stack([Rg[:, :1], D_u, D_t[:, 1:]])
    b0, *_ = np.linalg.lstsq(X0, m["y"].to_numpy(float), rcond=None)
    assert abs(float(b0[0]) - res.naive_att) < 1e-9
