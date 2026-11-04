import copy
import json
import warnings
import numpy as np, pandas as pd, pytest
import puremacro.examples.first_brief_fiscal as fb


# rows kept by RZ's war-exclusion rule on the bundled data: a row drops when its outcome window, shock or lagged controls touch 1941Q3-1945Q4
RZ_RULE_ROWS = [(7, 464), (15, 448), (19, 440), (20, 438)]


@pytest.fixture(scope="module")
def computed():
    return fb.compute_brief(exclude=fb.WWII)          # (record, paths)


def test_parse_exclude_valid():
    assert fb.parse_exclude("1941Q3:1945Q4") == (pd.Period("1941Q3", "Q"), pd.Period("1945Q4", "Q"))


@pytest.mark.parametrize("text", ["1941Q3", "1945Q4:1941Q3", "1941-07:1945-12", "", "1941Q5:1945Q4"])
def test_parse_exclude_rejects(text):
    with pytest.raises(ValueError):
        fb.parse_exclude(text)


def test_exclude_period_masks_y_g_newsy_and_rebuilds_lags():
    df = fb.load_brief_data()
    m = fb.exclude_period(df, fb.parse_exclude(fb.WWII))
    q = df.index.to_period("Q")
    inside = (q >= "1941Q3") & (q <= "1945Q4")
    assert len(m) == len(df) and m.loc[inside, ["y", "g", "newsy"]].isna().all().all()
    pd.testing.assert_frame_equal(m.loc[~inside, ["y", "g", "newsy"]], df.loc[~inside, ["y", "g", "newsy"]])
    after = df.index[q == "1946Q1"][0]
    assert np.isnan(m.loc[after, "y_l1"]) and np.isnan(m.loc[after, "newsy_l1"])   # lags rebuilt from masked series
    assert m.loc[after, "y"] == df.loc[after, "y"]
    assert "cumy" not in m and "cumg" not in m                                       # stale running sums dropped


def test_exclude_period_rejects_period_outside_sample():
    with pytest.raises(ValueError, match="does not overlap"):
        fb.exclude_period(fb.load_brief_data(), fb.parse_exclude("1700Q1:1701Q4"))


@pytest.mark.parametrize(("h", "n"), RZ_RULE_ROWS)   # (h, n): rows kept at h = 7, 15, 19, 20 under RZ's war-exclusion rule
def test_exclusion_selects_rz_rows(h, n):
    from puremacro.examples.ramey_zubairy_2018_multipliers import RZ_CONTROLS
    from puremacro.lp._common import cumulative_windows
    m = fb.exclude_period(fb.load_brief_data(), fb.parse_exclude(fb.WWII))
    yw, xw = cumulative_windows(m["y"], m["g"], h)
    used = pd.concat([yw, xw, m["newsy"], m[RZ_CONTROLS]], axis=1).notna().all(axis=1)
    assert int(used.sum()) == n


def test_concentration_finds_a_planted_spike():
    df = fb.load_brief_data().copy()
    spike = df.index[300]
    df.loc[spike, "newsy"] = df["newsy"].abs().max() * 50
    regime = float(df.loc[spike, "slack_l"])
    out = fb.concentration(df, regime, 8)
    assert out["share"] > 0.9 and out["date"] == str(spike.to_period("Q"))


def test_weak_flags_follow_the_cutoff(computed):
    record, _ = computed
    cutoff = record["specification"]["weak_cutoff"]
    assert abs(cutoff - 23.11) < 0.01
    for sample in ("full", "experiment"):
        for cells in record[sample]["horizons"].values():
            for name in ("linear", "slack", "no_slack"):
                assert cells[name]["weak"] == (cells[name]["f_eff"] < cutoff)
            assert cells["difference"]["reliable"] == (not cells["slack"]["weak"] and not cells["no_slack"]["weak"])


def test_record_and_paths_shape(computed):
    record, paths = computed
    assert set(record) == {"puremacro_version", "data", "specification", "full", "experiment"}
    assert record["experiment"]["exclude"] == ["1941Q3", "1945Q4"]
    assert set(record["full"]["horizons"]) == {"7", "15"}
    assert list(paths.groupby("sample").size()) == [21, 21]
    assert json.loads(json.dumps(record)) == record                  # JSON-ready, no NaN or inf


def test_default_run_has_no_experiment_and_keeps_estimator_warnings(monkeypatch):
    real = fb.estimate

    def noisy(df):
        warnings.warn("horizon h=3 could not be estimated", RuntimeWarning)
        return real(df)

    monkeypatch.setattr(fb, "estimate", noisy)                       # looked up at call time
    with pytest.warns(RuntimeWarning, match="could not be estimated"):  # re-issued after capture
        record, paths = fb.compute_brief()
    assert record["experiment"] is None and list(paths["sample"]) == ["full"] * 21
    assert fb._LAST_WARNINGS == ["full: RuntimeWarning: horizon h=3 could not be estimated"]
