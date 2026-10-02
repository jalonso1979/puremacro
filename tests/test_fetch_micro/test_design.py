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
        SurveyDesign(weight="w", method="taylor")
    with pytest.raises(ValueError, match="design columns absent"):
        MicroFrame(pd.DataFrame({"x": [1]}), SurveyDesign.acs_pums(), "person", "t", "x")


def test_presets_carry_documented_constants():
    acs = SurveyDesign.acs_pums("household")
    assert acs.weight == "WGTP" and len(acs.replicate_weights) == 80
    assert acs.scale == pytest.approx(0.05) and acs.mse
    scf = SurveyDesign.scf()
    assert len(scf.replicate_weights) == 999 and scf.scale == pytest.approx(1 / 998)
    assert scf.implicate == "implicate"


def test_variables_excludes_design_columns():
    df = _acs_like()
    mf = MicroFrame(df, SurveyDesign.acs_pums(), "person", "t", "x")
    assert mf.variables == ["WAGP", "SEX"]
