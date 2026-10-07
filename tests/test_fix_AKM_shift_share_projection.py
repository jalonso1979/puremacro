"""AKM shift-share standard errors: the hat-X projection and sector clusters.

``shift_share_iv``'s ``se_akm`` must be the IV standard error of Adão, Kolesár
and Morales (2019): eq. (39) with AKM's ``Xhat = (W'W)^{-1} W' z''`` of
eq. (28) (the control-partialled instrument regressed on the shares, Remark 5)
and, with sector clusters, eq. (40). Up to 4.3.0 it residualised the raw
shocks on a share-weighted constant instead, which is the same thing only when
the intercept is the sole control and each unit's shares sum to one.

The reference below, ``_ivreg_ss_akm``, is an independent numpy transcription
of ``ivreg_ss.fit`` in Kolesár's R package ShiftShareSE 1.1.0 (``R/iv.R``):

    hX <- lm.wfit(y=ddX, x=W, w=w)$coefficients
    cR <- hX*drop(crossprod(wgt * resid, W))
    cR <- tapply(cR, factor(sector_cvar), sum)        # when sector_cvar given
    se.akm <- sqrt(sum(cR^2) / RX^2)

The network-marked test at the end pins the package vignette's printed ADH
numbers (vignette of 23 April 2022, p. 2): estimate -0.7742267, AKM standard
error 0.2403730 with 3-digit SIC sector clusters.
"""
from __future__ import annotations

import io
import os
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.bartik import shift_share_iv


def _wls(yv, X, w):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)
    return yv - X @ coef, coef


def _ivreg_ss_akm(y1, y2, X, W, Z, w=None, sector_cvar=None):
    """ShiftShareSE 1.1.0 ``ivreg_ss.fit``: estimate and AKM standard error."""
    w = np.ones(len(y1)) if w is None else np.asarray(w, dtype=float)
    ddX, _ = _wls(X, Z, w)
    ddY1, _ = _wls(y1, Z, w)
    ddY2, _ = _wls(y2, Z, w)
    _, hX = _wls(ddX, W, w)
    RX = np.sum(w * ddY2 * ddX)
    beta = np.sum(w * ddY1 * ddX) / RX
    resid = ddY1 - beta * ddY2
    cR = hX * ((w * resid) @ W)
    if sector_cvar is not None:
        cR = pd.Series(cR).groupby(np.asarray(sector_cvar)).sum().to_numpy()
    return beta, float(np.sqrt(np.sum(cR ** 2) / RX ** 2)), hX


def _dgp(seed, n=500, K=30):
    """Incomplete shares and a unit-level control correlated with the instrument."""
    rng = np.random.default_rng(seed)
    S = rng.dirichlet(np.full(K, 0.5), size=n) * rng.uniform(0.3, 1.0, size=(n, 1))
    g = rng.standard_normal(K)
    z = S @ g
    c1 = z / z.std() + rng.standard_normal(n)
    x = z + 0.5 * c1 + rng.standard_normal(n)
    y = x + 0.3 * c1 + S @ rng.standard_normal(K) + 0.5 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "x": x, "c1": c1, "ssum": S.sum(axis=1), "wt": rng.uniform(0.5, 2.0, n)})
    return df, S, g


def _Z(df, controls):
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in controls])


@pytest.mark.parametrize("controls,weights", [(["c1"], None), (["c1", "ssum"], "wt"), ([], "wt"), ([], None)])
def test_akm_se_is_the_hat_x_projection_of_eq_28_39(controls, weights):
    df, S, g = _dgp(0)
    res = shift_share_iv(df, "y", "x", S, g, controls=controls, weights=weights)
    w = None if weights is None else df[weights].to_numpy()
    beta, se, hX = _ivreg_ss_akm(df["y"].to_numpy(), df["x"].to_numpy(), S @ g, S, _Z(df, controls), w)
    assert res.beta == pytest.approx(beta, rel=1e-10)
    assert res.se_akm == pytest.approx(se, rel=1e-9)
    np.testing.assert_allclose(res.shocks_residualized.to_numpy(), hX, rtol=1e-8, atol=1e-10)
    assert res.akm_shocks == "projection" and res.n_sector_clusters is None


def test_old_residualised_shocks_differ_with_a_regional_control():
    """The defect: with a unit-level control the pre-fix formula (still available
    as akm_shocks='residualized') is far from AKM's (here almost twice as large)."""
    df, S, g = _dgp(0)
    new = shift_share_iv(df, "y", "x", S, g, controls=["c1"])
    old = shift_share_iv(df, "y", "x", S, g, controls=["c1"], akm_shocks="residualized")
    _, se, _ = _ivreg_ss_akm(df["y"].to_numpy(), df["x"].to_numpy(), S @ g, S, _Z(df, ["c1"]))
    assert new.se_akm == pytest.approx(se, rel=1e-9)
    assert new.se_akm == pytest.approx(0.21195415, abs=5e-8)
    assert old.se_akm == pytest.approx(0.41167943, abs=5e-8)
    assert old.akm_shocks == "residualized" and old.beta == new.beta


def test_intercept_only_with_complete_shares_is_unchanged():
    """With only the intercept and shares summing to one, Xhat_k equals the shock
    minus its share-weighted mean, so the fix leaves these values unchanged (the
    docs/spatial.md section 4 example: se_akm 0.14838290482386557)."""
    rng = np.random.default_rng(3)
    n_regions, n_industries = 300, 25
    shares = pd.DataFrame(rng.dirichlet(np.full(n_industries, 0.5), size=n_regions),
                          columns=[f"ind{k:02d}" for k in range(n_industries)])
    shocks = pd.Series(rng.standard_normal(n_industries), index=shares.columns)
    exposure = shares.to_numpy() @ shocks.to_numpy()
    employment_growth = exposure + rng.standard_normal(n_regions)
    industry_confounder = shares.to_numpy() @ rng.standard_normal(n_industries)
    wage_growth = 0.8 * employment_growth + industry_confounder + 0.5 * rng.standard_normal(n_regions)
    df = pd.DataFrame({"wage_growth": wage_growth, "employment_growth": employment_growth})
    new = shift_share_iv(df, "wage_growth", "employment_growth", shares, shocks)
    old = shift_share_iv(df, "wage_growth", "employment_growth", shares, shocks, akm_shocks="residualized")
    assert new.se_akm == pytest.approx(0.14838290482386557, rel=1e-10)
    assert new.se_akm == pytest.approx(old.se_akm, rel=1e-10)
    np.testing.assert_allclose(new.shocks_residualized, old.shocks_residualized, atol=1e-10)
    wt = rng.uniform(0.5, 2.0, n_regions)
    a = shift_share_iv(df, "wage_growth", "employment_growth", shares, shocks, weights=wt)
    b = shift_share_iv(df, "wage_growth", "employment_growth", shares, shocks, weights=wt, akm_shocks="residualized")
    assert a.se_akm == pytest.approx(b.se_akm, rel=1e-10)


def test_sector_clusters_follow_eq_40():
    df, S, g = _dgp(1)
    K = S.shape[1]
    labels = np.arange(K) % 7
    res = shift_share_iv(df, "y", "x", S, g, controls=["c1", "ssum"], weights="wt", sector_clusters=labels)
    _, se, _ = _ivreg_ss_akm(df["y"].to_numpy(), df["x"].to_numpy(), S @ g, S, _Z(df, ["c1", "ssum"]),
                             df["wt"].to_numpy(), labels)
    assert res.se_akm == pytest.approx(se, rel=1e-9)
    assert res.n_sector_clusters == 7 and "7 sector clusters" in res.summary()
    unclustered = shift_share_iv(df, "y", "x", S, g, controls=["c1", "ssum"], weights="wt")
    singletons = shift_share_iv(df, "y", "x", S, g, controls=["c1", "ssum"], weights="wt",
                                sector_clusters=[f"s{k}" for k in range(K)])
    assert singletons.se_akm == pytest.approx(unclustered.se_akm, rel=1e-12)
    assert res.se_akm != pytest.approx(unclustered.se_akm, rel=1e-3)
    one = shift_share_iv(df, "y", "x", S, g, controls=["c1", "ssum"], weights="wt", sector_clusters=np.zeros(K))
    assert one.beta == unclustered.beta
    # labelled sectors: a Series of clusters is aligned on the share columns
    cols = [f"k{k}" for k in range(K)]
    shares = pd.DataFrame(S, index=df.index, columns=cols)
    shocks = pd.Series(g, index=cols)
    cl = pd.Series(labels, index=cols).iloc[::-1]
    aligned = shift_share_iv(df, "y", "x", shares, shocks, controls=["c1", "ssum"], weights="wt", sector_clusters=cl)
    assert aligned.se_akm == pytest.approx(res.se_akm, rel=1e-12)
    # clustering also applies to the sector-level residualisation alternative
    alt = shift_share_iv(df, "y", "x", S, g, controls=["c1", "ssum"], weights="wt",
                         akm_shocks="residualized", sector_clusters=labels)
    assert alt.n_sector_clusters == 7 and np.isfinite(alt.se_akm)


def test_collinear_sectors_are_dropped_left_to_right_like_shiftsharese():
    df, S, g = _dgp(2, n=300, K=8)
    S = S.copy()
    S[:, 7] = 0.5 * (S[:, 1] + S[:, 2])            # sector 7 is a mix of sectors 1 and 2
    S[:, 0] = S[:, 3] + S[:, 4]                    # 0 = 3 + 4: the scan keeps 0 and 3, drops 4
    y, x, z = df["y"].to_numpy(), df["x"].to_numpy(), S @ g
    with pytest.warns(RuntimeWarning, match="collinear"):
        res = shift_share_iv(df, "y", "x", S, g, controls=["c1"])
    dropped = np.flatnonzero(res.shocks_residualized.isna().to_numpy())
    assert list(dropped) == [4, 7]
    keep = np.setdiff1d(np.arange(8), dropped)
    beta, se, hX = _ivreg_ss_akm(y, x, z, S[:, keep], _Z(df, ["c1"]))
    assert res.beta == pytest.approx(beta, rel=1e-10)
    assert res.se_akm == pytest.approx(se, rel=1e-9)
    np.testing.assert_allclose(res.shocks_residualized.to_numpy()[keep], hX, rtol=1e-7)


def test_unexposed_sector_is_dropped_silently():
    df, S, g = _dgp(3, n=200, K=6)
    S = S.copy()
    S[:, 2] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = shift_share_iv(df, "y", "x", S, g, controls=["c1"])
    assert np.isnan(res.shocks_residualized.iloc[2])
    keep = [0, 1, 3, 4, 5]
    _, se, _ = _ivreg_ss_akm(df["y"].to_numpy(), df["x"].to_numpy(), S @ g, S[:, keep], _Z(df, ["c1"]))
    assert res.se_akm == pytest.approx(se, rel=1e-9)


def test_more_sectors_than_units_keeps_at_most_n_sectors():
    df, S, g = _dgp(4, n=25, K=40)
    with pytest.warns(RuntimeWarning, match="rank 25 < 40"):
        res = shift_share_iv(df, "y", "x", S, g)
    assert int(res.shocks_residualized.notna().sum()) == 25
    assert np.isfinite(res.se_akm) and res.se_akm > 0


def test_instrument_override_reproduces_an_external_instrument():
    df, S, g = _dgp(5)
    rng = np.random.default_rng(55)
    z_pub = np.round(S @ g, 3) + 1e-4 * rng.standard_normal(len(df))   # published at limited precision
    df = df.assign(z_pub=z_pub)
    res = shift_share_iv(df, "y", "x", S, g, controls=["c1"], instrument="z_pub", sector_clusters=np.arange(30) // 3)
    beta, se, _ = _ivreg_ss_akm(df["y"].to_numpy(), df["x"].to_numpy(), z_pub, S, _Z(df, ["c1"]),
                                sector_cvar=np.arange(30) // 3)
    assert res.beta == pytest.approx(beta, rel=1e-10)
    assert res.se_akm == pytest.approx(se, rel=1e-9)
    assert res.rotemberg_weights.sum() == pytest.approx(1.0, abs=1e-10)
    same = shift_share_iv(df, "y", "x", S, g, controls=["c1"], instrument=S @ g)
    base = shift_share_iv(df, "y", "x", S, g, controls=["c1"])
    assert same.beta == base.beta and same.se_akm == base.se_akm
    with pytest.raises(ValueError, match="instrument"):
        shift_share_iv(df, "y", "x", S, g, instrument=z_pub[:-1])


def test_weights_scale_does_not_matter():
    df, S, g = _dgp(6)
    a = shift_share_iv(df, "y", "x", S, g, controls=["c1"], weights=df["wt"].to_numpy())
    b = shift_share_iv(df, "y", "x", S, g, controls=["c1"], weights=7.0 * df["wt"].to_numpy())
    assert a.se_akm == pytest.approx(b.se_akm, rel=1e-10) and a.beta == pytest.approx(b.beta, rel=1e-12)


def test_option_validation():
    df, S, g = _dgp(7, n=100, K=10)
    Q = np.random.default_rng(0).standard_normal((10, 1))
    auto = shift_share_iv(df, "y", "x", S, g, shock_controls=Q)
    assert auto.akm_shocks == "residualized"
    with pytest.raises(ValueError, match="shock_controls"):
        shift_share_iv(df, "y", "x", S, g, shock_controls=Q, akm_shocks="projection")
    with pytest.raises(ValueError, match="akm_shocks"):
        shift_share_iv(df, "y", "x", S, g, akm_shocks="bhj")
    with pytest.raises(ValueError, match="sector_clusters has 9"):
        shift_share_iv(df, "y", "x", S, g, sector_clusters=np.arange(9))
    with pytest.raises(ValueError, match="missing labels"):
        shift_share_iv(df, "y", "x", S, g, sector_clusters=[1.0] * 9 + [np.nan])
    with pytest.raises(KeyError):
        shift_share_iv(df, "y", "x", S, g, sector_clusters=pd.Series(np.arange(9), index=range(9)))


# --------------------------------------------------------------------------
# Published oracle: ShiftShareSE vignette, ADH (Autor, Dorn and Hanson 2013)
# --------------------------------------------------------------------------

_CRAN = (
    "https://cran.r-project.org/src/contrib/ShiftShareSE_1.1.0.tar.gz",
    "https://cran.r-project.org/src/contrib/Archive/ShiftShareSE/ShiftShareSE_1.1.0.tar.gz",
)


def _adh_rda(tmp_path: Path) -> Path:
    """ADH.rda from $PUREMACRO_SHIFTSHARESE_ADH (the .rda or the package
    tarball) or downloaded from CRAN into ``tmp_path``. The GPL-3 data are
    never stored in the repository."""
    local = os.environ.get("PUREMACRO_SHIFTSHARESE_ADH")
    blob = None
    if local:
        p = Path(local)
        if p.suffix == ".rda":
            return p
        blob = p.read_bytes()
    else:
        for url in _CRAN:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "puremacro-tests"})
                blob = urllib.request.urlopen(req, timeout=60).read()
                break
            except Exception:  # pragma: no cover - network dependent
                continue
        if blob is None:  # pragma: no cover
            pytest.skip("could not download ShiftShareSE 1.1.0 from CRAN")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        data = tf.extractfile("ShiftShareSE/data/ADH.rda").read()
    out = tmp_path / "ADH.rda"
    out.write_bytes(data)
    return out


@pytest.mark.network
@pytest.mark.replication
def test_adh_reproduces_shiftsharese_vignette(tmp_path):
    rdata = pytest.importorskip("rdata")
    adh = rdata.conversion.convert(rdata.parser.parse_file(_adh_rda(tmp_path)))["ADH"]
    reg = pd.DataFrame(adh["reg"])
    W = np.asarray(adh["W"], dtype=float)                       # 1444 x 770 CZ-period x 4-digit SIC shares
    sic3 = np.floor(np.asarray(adh["sic"], dtype=float) / 10)   # sector_cvar = floor(ADH$sic/10)
    num = ["t2", "l_shind_manuf_cbp", "l_sh_popedu_c", "l_sh_popfborn", "l_sh_empl_f",
           "l_sh_routine33", "l_task_outsource"]
    df = pd.DataFrame({c: reg[c].astype(float).to_numpy() for c in ["d_sh_empl", "shock", "IV", "weights"] + num})
    div = pd.get_dummies(reg["division"].astype(str), prefix="div", drop_first=True).astype(float)
    df = pd.concat([df, div.reset_index(drop=True)], axis=1)
    ctrls = num + list(div.columns)
    # The package ships the regional instrument, not the shocks: recover the
    # shocks for the Rotemberg weights and pass the instrument itself.
    g, *_ = np.linalg.lstsq(W, df["IV"].to_numpy(), rcond=None)
    kw = dict(controls=ctrls, weights="weights", instrument="IV")
    res = shift_share_iv(df, "d_sh_empl", "shock", W, g, sector_clusters=sic3, **kw)
    # vignette p. 2: "Estimate: -0.7742267" and "AKM 0.2403730 1.277718e-03 -1.2453492 -0.3031041"
    assert res.beta == pytest.approx(-0.7742267, abs=5e-8)
    assert res.se_akm == pytest.approx(0.2403730, abs=5e-8)
    assert res.ci_lower == pytest.approx(-1.2453492, abs=5e-8)
    assert res.ci_upper == pytest.approx(-0.3031041, abs=5e-8)
    assert res.p_value == pytest.approx(1.277718e-03, rel=5e-7)
    # "EHW 0.1647892": HC0; puremacro's se_robust is HC1 (n / (n - k), k = 17)
    n, k = len(df), len(ctrls) + 2
    assert res.se_robust * np.sqrt((n - k) / n) == pytest.approx(0.1647892, abs=5e-8)
    assert res.n_sector_clusters == 136
    # without sector clusters (eq. 39) and with the pre-fix formula
    assert shift_share_iv(df, "d_sh_empl", "shock", W, g, **kw).se_akm == pytest.approx(0.2101180, abs=5e-8)
    old = shift_share_iv(df, "d_sh_empl", "shock", W, g, akm_shocks="residualized", **kw)
    assert old.se_akm == pytest.approx(0.1816483, abs=1e-6)
