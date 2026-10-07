"""SurveyDesign / MicroFrame estimators against hand-computed formulas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import (
    MicroFrame, SurveyDesign, replicate_variance, rubin_combine,
)


def _acs_like(n=400, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(5, 50, n)
    reps = w[:, None] * rng.uniform(0.5, 1.5, (n, 80))
    df = pd.DataFrame(reps, columns=[f"PWGTP{r}" for r in range(1, 81)])
    df.insert(0, "PWGTP", w)
    df["WAGP"] = rng.lognormal(10, 1, n)
    df["SEX"] = rng.integers(1, 3, n)
    return df


def _manual_sdr_mean(df, col):
    theta = np.average(df[col], weights=df["PWGTP"])
    reps = np.array([np.average(df[col], weights=df[f"PWGTP{r}"]) for r in range(1, 81)])
    return theta, np.sqrt(4 / 80 * np.sum((reps - theta) ** 2))


def test_acs_sdr_mean_matches_the_census_formula():
    df = _acs_like()
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    res = mf.mean("WAGP")
    theta, se = _manual_sdr_mean(df, "WAGP")
    assert res.loc[0, "estimate"] == pytest.approx(theta, rel=1e-12)
    assert res.loc[0, "se"] == pytest.approx(se, rel=1e-12)
    assert res.loc[0, "n"] == len(df)


def test_total_and_by_groups():
    df = _acs_like()
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    res = mf.total("WAGP", by="SEX")
    assert list(res["SEX"]) == [1, 2]
    for _, row in res.iterrows():
        sub = df[df["SEX"] == row["SEX"]]
        theta = (sub["WAGP"] * sub["PWGTP"]).sum()
        reps = np.array([(sub["WAGP"] * sub[f"PWGTP{r}"]).sum() for r in range(1, 81)])
        assert row["estimate"] == pytest.approx(theta)
        assert row["se"] == pytest.approx(np.sqrt(0.05 * ((reps - theta) ** 2).sum()))


def test_missing_values_are_dropped_from_the_domain():
    df = _acs_like()
    df.loc[:9, "WAGP"] = np.nan
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    res = mf.mean("WAGP")
    theta, se = _manual_sdr_mean(df.dropna(subset=["WAGP"]), "WAGP")
    assert res.loc[0, "n"] == len(df) - 10
    assert res.loc[0, "estimate"] == pytest.approx(theta)
    assert res.loc[0, "se"] == pytest.approx(se)


def test_weighted_quantile_is_the_inverse_weighted_cdf():
    df = pd.DataFrame({"w": [1.0, 1.0, 1.0, 1.0], "x": [4.0, 1.0, 3.0, 2.0]})
    mf = MicroFrame(df, SurveyDesign.weights_only("w"), "person", "t", "x")
    assert mf.quantile("x", 0.5).loc[0, "estimate"] == 2.0
    assert mf.quantile("x", 0.51).loc[0, "estimate"] == 3.0
    df["w"] = [10.0, 1.0, 1.0, 1.0]          # x=4 carries 10/13 of the weight
    mf = MicroFrame(df, SurveyDesign.weights_only("w"), "person", "t", "x")
    assert mf.quantile("x", 0.5).loc[0, "estimate"] == 4.0


def test_share_sums_to_one_within_groups():
    df = _acs_like()
    df["G"] = np.where(df["WAGP"] > df["WAGP"].median(), "hi", "lo")
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    res = mf.share("SEX", by="G")
    assert list(res.columns) == ["G", "SEX", "estimate", "se", "n", "implicates"]
    sums = res.groupby("G")["estimate"].sum()
    assert np.allclose(sums, 1.0)
    assert (res["se"] > 0).all()


def test_weights_only_design_reports_nan_se_not_a_wrong_number():
    df = _acs_like()
    mf = MicroFrame(df, SurveyDesign.weights_only("PWGTP"), "person", "t", "x")
    res = mf.mean("WAGP")
    assert np.isnan(res.loc[0, "se"])
    assert res.loc[0, "estimate"] == pytest.approx(np.average(df["WAGP"], weights=df["PWGTP"]))


def test_implicates_are_combined_with_rubins_rules():
    rng = np.random.default_rng(3)
    n, R = 50, 20
    w = rng.uniform(1, 3, n)
    reps = w[:, None] * rng.uniform(0.6, 1.4, (n, R))
    frames, ests, vars_ = [], [], []
    design = SurveyDesign(weight="w", replicate_weights=tuple(f"r{k}" for k in range(R)),
                          method="bootstrap", scale=1 / (R - 1), implicate="imp")
    for i in range(1, 6):
        x = rng.normal(10 + 0.3 * i, 2, n)
        f = pd.DataFrame(reps, columns=list(design.replicate_weights))
        f["w"], f["x"], f["imp"] = w, x, i
        frames.append(f)
        theta = np.average(x, weights=w)
        rt = np.array([np.average(x, weights=reps[:, k]) for k in range(R)])
        ests.append(theta)
        vars_.append(replicate_variance(theta, rt, scale=1 / (R - 1)))
    mf = MicroFrame(pd.concat(frames, ignore_index=True), design, "household", "t", "x")
    res = mf.mean("x")
    est, var = rubin_combine(ests, vars_)
    b = np.var(ests, ddof=1)
    assert var == pytest.approx(np.mean(vars_) + 1.2 * b)
    assert res.loc[0, "estimate"] == pytest.approx(est)
    assert res.loc[0, "se"] == pytest.approx(np.sqrt(var))
    assert res.loc[0, "implicates"] == 5


def test_non_mse_centres_on_the_replicate_mean():
    reps = np.array([1.0, 2.0, 3.0])
    assert replicate_variance(10.0, reps, scale=1.0, mse=False) == pytest.approx(2.0)
    assert replicate_variance(2.0, reps, scale=1.0, mse=True) == pytest.approx(2.0)


def test_design_validation():
    with pytest.raises(ValueError, match="needs replicate_weights"):
        SurveyDesign(weight="w", method="sdr")
    with pytest.raises(ValueError, match="not in"):
        SurveyDesign(weight="w", method="linearised")
    with pytest.raises(ValueError, match="needs a psu"):
        SurveyDesign(weight="w", method="taylor")
    with pytest.raises(ValueError, match="design columns absent"):
        MicroFrame(pd.DataFrame({"x": [1]}), SurveyDesign.acs_pums(), "person", "t", "x")


def test_presets_carry_documented_constants():
    acs = SurveyDesign.acs_pums("household")
    assert acs.weight == "WGTP" and len(acs.replicate_weights) == 80
    assert acs.scale == pytest.approx(0.05) and acs.mse
    scf = SurveyDesign.scf()
    assert not scf.mse and scf.variance_implicate == 1
    assert len(scf.replicate_weights) == 999 and scf.scale == pytest.approx(1 / 998)
    assert scf.implicate == "implicate"


def test_variables_excludes_design_columns():
    df = _acs_like()
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    assert mf.variables == ["WAGP", "SEX"]


# ---------------------------------------------------------------------------
# Taylor linearisation (strata + PSU designs)
# ---------------------------------------------------------------------------
def _clustered(seed=0, lonely=False):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(6):
        for i in range(1 if (lonely and h == 5) else 3):
            for _ in range(4):
                rows.append({"h": h, "psu": h * 10 + i, "w": rng.uniform(50, 150),
                             "y": rng.lognormal(3, 0.6), "g": rng.integers(0, 2)})
    return pd.DataFrame(rows)


def _taylor_frame(df):
    design = SurveyDesign(weight="w", method="taylor", strata="h", psu="psu")
    return MicroFrame(df, design, "household", "t", "x")


def _psu_formula(u, df):
    """Σ_h n_h/(n_h-1) Σ_i (t_hi - mean t_h)², written out longhand."""
    t = df.assign(u=u).groupby(["h", "psu"])["u"].sum()
    v = 0.0
    for _, th in t.groupby(level=0):
        n = len(th)
        v += n / (n - 1) * ((th - th.mean()) ** 2).sum()
    return v


def test_taylor_total_and_mean_match_the_textbook_formula():
    df = _clustered()
    mf = _taylor_frame(df)
    tot = mf.total("y").iloc[0]
    assert tot["estimate"] == pytest.approx((df.w * df.y).sum())
    assert tot["se"] == pytest.approx(np.sqrt(_psu_formula(df.w * df.y, df)))
    mean = mf.mean("y").iloc[0]
    theta = (df.w * df.y).sum() / df.w.sum()
    z = df.w * (df.y - theta) / df.w.sum()
    assert mean["estimate"] == pytest.approx(theta)
    assert mean["se"] == pytest.approx(np.sqrt(_psu_formula(z, df)))


def test_taylor_domains_keep_every_psu():
    df = _clustered(1)
    by = _taylor_frame(df).mean("y", by="g")
    for g, row in by.set_index("g").iterrows():
        d = df.g == g
        theta = (df.w * df.y)[d].sum() / df.w[d].sum()
        z = np.where(d, df.w * (df.y - theta) / df.w[d].sum(), 0.0)
        assert row["estimate"] == pytest.approx(theta)
        assert row["se"] == pytest.approx(np.sqrt(_psu_formula(z, df)))
        # Subsetting first (the wrong way) gives a different answer.
        sub = _taylor_frame(df[d].reset_index(drop=True)).mean("y").iloc[0]
        assert sub["estimate"] == pytest.approx(theta)


def test_taylor_single_psu_stratum_is_centred_on_the_grand_mean():
    df = _clustered(2, lonely=True)
    u = df.w * df.y
    t = df.assign(u=u).groupby(["h", "psu"])["u"].sum()
    v = 0.0
    for h, th in t.groupby(level=0):
        if len(th) == 1:
            v += float((th.iloc[0] - t.mean()) ** 2)
        else:
            v += len(th) / (len(th) - 1) * ((th - th.mean()) ** 2).sum()
    se = _taylor_frame(df).total("y").iloc[0]["se"]
    assert np.isfinite(se) and se == pytest.approx(np.sqrt(v))


def test_taylor_quantile_uses_woodruff_and_is_positive():
    df = _clustered(3)
    mf = _taylor_frame(df)
    med = mf.quantile("y", 0.5).iloc[0]
    assert med["estimate"] == pytest.approx(
        mf.quantile("y", 0.5).iloc[0]["estimate"])
    assert 0 < med["se"] < df.y.std()
    assert mf.quantile("y", 0.0).iloc[0]["estimate"] == df.y.min()


def test_taylor_share_and_missing_values():
    df = _clustered(4)
    df.loc[[0, 5], "y"] = np.nan
    mf = _taylor_frame(df)
    res = mf.mean("y").iloc[0]
    assert res["n"] == len(df) - 2 and np.isfinite(res["se"])
    sh = mf.share("g")
    assert sh["estimate"].sum() == pytest.approx(1.0)
    assert (sh["se"] > 0).all()


def test_enigh_preset():
    d = SurveyDesign.enigh()
    assert (d.weight, d.strata, d.psu, d.method) == ("factor", "est_dis", "upm", "taylor")
    assert d.columns() == ["factor", "est_dis", "upm"]


def test_cps_month_out_of_range_is_rejected():
    from puremacro.fetch.micro import census
    for bad in (0, -1, 13):
        with pytest.raises(ValueError, match="not in 1..12"):
            census.fetch_cps_basic(2024, bad, ["PEMLR"], api_key="k")
