"""Synthetic tests for ``splice_sources`` and ``to_long``.

Every frame is built inline: quarterly dates, one or two countries, and
sources whose relationship is known exactly, so each assertion checks a
rule of the splice rather than a number from a provider.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.longpanel import splice_sources, to_long


def _src(source, values, start, *, code="ESP", variable="gdp", freq="QS"):
    dates = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"code": code, "date": dates, "variable": variable,
                         "value": np.asarray(values, dtype=float),
                         "source": source})


def _growth(n, start=100.0, g=0.01):
    return start * (1 + g) ** np.arange(n)


def test_the_primary_is_never_overwritten():
    new = _src("a", _growth(20), "2000-01-01")
    old = _src("b", _growth(40, 50.0), "1995-01-01")       # overlaps 2000-2004
    panel, prov, seams = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1})
    kept = panel[panel["date"] >= "2000-01-01"]
    assert (kept["source"] == "a").all()
    np.testing.assert_array_equal(kept["value"].to_numpy(), new["value"].to_numpy())
    ext = panel[panel["date"] < "2000-01-01"]
    assert len(ext) == 20 and (ext["source"] == "b").all()
    assert prov.loc[0, "extended_from"] == "b" and prov.loc[0, "n_extended"] == 20
    assert seams.loc[0, "method"] == "ratio"


def test_an_interior_gap_is_never_filled():
    new = _src("a", _growth(20), "2000-01-01")
    new = new.drop(index=[8, 9])                         # 2002Q1-Q2 missing
    old = _src("b", _growth(40, 50.0), "1995-01-01")
    panel, prov, _ = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1})
    gap = pd.to_datetime(["2002-01-01", "2002-04-01"])
    assert not panel["date"].isin(gap).any()
    assert prov.loc[0, "n"] == 18 + 20


def test_a_thousandfold_units_change_is_absorbed_with_pow10_3():
    lvl = _growth(40)
    new = _src("a", lvl[20:] * 1000.0, "2000-01-01")       # millions
    old = _src("b", lvl[:24], "1995-01-01")                # billions
    panel, _, seams = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1})
    s = seams.iloc[0]
    assert s["method"] == "ratio" and s["pow10"] == 3 and bool(s["stable"])
    assert s["ratio"] == pytest.approx(1000.0)
    np.testing.assert_allclose(panel["value"].to_numpy(), lvl * 1000.0)
    assert s["note"] == "units differ by 10^3; residual x1.0000"


def test_the_pow10_note_reports_a_residual_level_shift():
    lvl = _growth(40)
    new = _src("a", lvl[20:] * 1020.0, "2000-01-01")       # 10^3 and a 2% break
    old = _src("b", lvl[:24], "1995-01-01")
    _, _, seams = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1})
    s = seams.iloc[0]
    assert s["pow10"] == 3 and s["ratio"] == pytest.approx(1020.0)
    assert s["note"] == "units differ by 10^3; residual x1.0200"


def test_unused_categories_and_all_missing_sources_change_nothing():
    rows = []
    for code in ["ESP", "USA"]:
        for var in ["gdp", "inv"]:
            rows += [_src("a", _growth(20), "2000-01-01", code=code, variable=var),
                     _src("b", _growth(40, 50.0), "1995-01-01", code=code, variable=var)]
    plain = pd.concat(rows, ignore_index=True)
    # an all-NaN source that is in neither rank nor anything else
    ghost = _src("x", [np.nan] * 6, "1990-01-01", variable="gdp")
    rank, kinds = {"a": 0, "b": 1}, {"gdp": "flow", "inv": "flow"}
    want = splice_sources(plain, rank, kinds=kinds)
    cat = pd.concat([plain, ghost], ignore_index=True)
    cat["source"] = pd.Categorical(cat["source"], categories=["zzz", "b", "x", "a"])
    cat["variable"] = pd.Categorical(cat["variable"],
                                     categories=["inv", "unused_var", "gdp"])
    cat["code"] = pd.Categorical(cat["code"], categories=["USA", "FRA", "ESP"])
    got = splice_sources(cat, rank, kinds=kinds)
    for w, g in zip(want, got):
        pd.testing.assert_frame_equal(w, g)


def test_a_rate_with_a_one_point_gap_is_rejected():
    u = 8 + np.sin(np.arange(40) / 3)
    new = _src("a", u[20:], "2000-01-01", variable="urate")
    old = _src("b", u[:24] - 1.0, "1995-01-01", variable="urate")
    panel, prov, seams = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1},
                                        kinds={"urate": "rate"})
    s = seams.iloc[0]
    assert s["method"] == "rejected" and s["ratio"] == pytest.approx(1.0)
    assert (panel["source"] == "a").all() and len(panel) == 20
    assert prov.loc[0, "extended_from"] == "" and prov.loc[0, "n_extended"] == 0
    # within tolerance the same rows are appended unscaled
    old_ok = _src("b", u[:24] - 0.1, "1995-01-01", variable="urate")
    panel2, _, seams2 = splice_sources(pd.concat([new, old_ok]), {"a": 0, "b": 1},
                                       kinds={"urate": "rate"})
    assert seams2.loc[0, "method"] == "level" and len(panel2) == 40
    ext = panel2[panel2["source"] == "b"]["value"].to_numpy()
    np.testing.assert_allclose(ext, u[:20] - 0.1)


def test_too_short_an_overlap_is_rejected_with_note_overlap():
    new = _src("a", _growth(20), "2000-01-01")
    old = _src("b", _growth(22, 50.0), "1995-01-01")       # overlaps 2 quarters
    panel, _, seams = splice_sources(pd.concat([new, old]), {"a": 0, "b": 1})
    assert seams.loc[0, "method"] == "rejected" and seams.loc[0, "note"] == "overlap"
    assert seams.loc[0, "overlap_n"] == 2 and len(panel) == 20


def test_chains_extend_through_successive_sources():
    lvl = _growth(60)
    a = _src("a", lvl[40:], "2000-01-01")
    b = _src("b", lvl[20:44] * 2.0, "1995-01-01")
    c = _src("c", lvl[:24] * 4.0, "1990-01-01")
    panel, prov, seams = splice_sources(pd.concat([c, b, a]), {"a": 0, "b": 1, "c": 2})
    np.testing.assert_allclose(panel["value"].to_numpy(), lvl)
    assert list(seams["older"]) == ["b", "c"] and list(seams["newer"]) == ["a", "b"]
    assert prov.loc[0, "extended_from"] == "b;c" and prov.loc[0, "n"] == 60


def test_prefer_overrides_rank():
    a = _src("a", _growth(20), "2000-01-01")
    b = _src("b", _growth(20) * 1.5, "2000-01-01")
    long = pd.concat([a, b])
    _, prov, _ = splice_sources(long, {"a": 0, "b": 1})
    assert prov.loc[0, "source"] == "a"
    panel, prov, _ = splice_sources(long, {"a": 0, "b": 1}, prefer={"gdp": ["b"]})
    assert prov.loc[0, "source"] == "b" and (panel["source"] == "b").all()
    assert prov.loc[0, "alternatives"] == "a:2000Q1-2004Q4"


def test_ties_are_deterministic():
    # equal rank: more observations wins, then later last date, then name
    a = _src("a", _growth(20), "2000-01-01")
    b = _src("b", _growth(24), "2000-01-01")
    c = _src("c", _growth(24), "2001-01-01")
    d = _src("d", _growth(24), "2001-01-01")
    rank = dict.fromkeys("abcd", 0)
    long = pd.concat([a, b, c, d])
    picks = set()
    for seed in range(5):
        shuffled = long.sample(frac=1.0, random_state=seed)
        shuffled["source"] = pd.Categorical(
            shuffled["source"], categories=list(np.random.default_rng(seed).permutation(list("dcba"))))
        _, prov, _ = splice_sources(shuffled, rank)
        picks.add(prov.loc[0, "source"])
    assert picks == {"c"}
    _, prov, _ = splice_sources(pd.concat([c, d]), rank)
    assert prov.loc[0, "source"] == "c"


def test_output_is_sorted_and_has_no_duplicates():
    parts = []
    for code in ["USA", "ESP", "JPN"]:
        for var in ["inv", "gdp"]:
            parts += [_src("a", _growth(20), "2000-01-01", code=code, variable=var),
                      _src("b", _growth(40), "1995-01-01", code=code, variable=var)]
    long = pd.concat(parts).sample(frac=1.0, random_state=0)
    long["code"] = long["code"].astype("category")
    panel, prov, seams = splice_sources(long, {"a": 0, "b": 1})
    key = ["code", "date", "variable"]
    assert not panel.duplicated(key).any()
    pd.testing.assert_frame_equal(panel, panel.sort_values(key, ignore_index=True))
    assert list(prov[["code", "variable"]].itertuples(index=False, name=None)) == [
        ("ESP", "gdp"), ("ESP", "inv"), ("JPN", "gdp"), ("JPN", "inv"),
        ("USA", "gdp"), ("USA", "inv")]
    assert list(prov.columns) == ["freq", "code", "variable", "source", "first",
                                  "last", "n", "alternatives", "extended_from",
                                  "n_extended", "ref_year"]
    assert list(seams.columns) == ["freq", "code", "variable", "date", "older",
                                   "newer", "overlap_n", "ratio", "ratio_drift",
                                   "ratio_min", "ratio_max", "stable", "pow10",
                                   "method", "note"]
    assert (prov["freq"] == "Q").all() and len(seams) == 6


def test_to_long_reshapes_drops_missing_and_keeps_attrs():
    idx = pd.MultiIndex.from_product(
        [["ESP", "USA"], pd.date_range("2000-01-01", periods=3, freq="QS")],
        names=["code", "date"])
    wide = pd.DataFrame({"gdp": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                         "cons_hh": [0.5] * 6, "src_gdp": ["x"] * 6}, index=idx)
    wide.attrs["meta"] = ({"variable": "gdp"},)
    out = to_long(wide)
    assert list(out.columns) == ["code", "date", "variable", "value"]
    assert len(out) == 11 and not out["value"].isna().any()
    assert out.attrs["meta"] == ({"variable": "gdp"},)
    assert out.iloc[0].tolist() == ["ESP", pd.Timestamp("2000-01-01"), "cons_hh", 0.5]
    assert to_long(wide, keep_attrs=False).attrs == {}
    yearly = wide.copy()
    yearly.index = pd.MultiIndex.from_product([["ESP", "USA"], [2000, 2001, 2002]],
                                              names=["code", "year"])
    assert to_long(yearly)["date"].iloc[-1] == pd.Timestamp("2002-01-01")
    quarterly = wide.copy()
    quarterly.index = pd.MultiIndex.from_product(
        [["ESP", "USA"], pd.period_range("2000Q1", periods=3, freq="Q")],
        names=["code", "date"])
    pd.testing.assert_frame_equal(to_long(quarterly), out)
    # round trip into splice_sources
    panel, _, _ = splice_sources(out.assign(source="a"), {"a": 0}, min_obs=1)
    assert len(panel) == 11


def test_performance_on_one_and_a_half_million_rows():
    """Printed, not asserted: the target is under 3 s."""
    rng = np.random.default_rng(0)
    codes = [f"C{i:03d}" for i in range(200)]
    variables = [f"v{j:02d}" for j in range(25)]
    dates = pd.date_range("1960-01-01", periods=260, freq="QS")
    frames = []
    for src, (lo, hi) in {"a": (120, 260), "b": (40, 160), "c": (0, 60)}.items():
        n_dates = hi - lo
        c = np.repeat(codes, len(variables) * n_dates)
        v = np.tile(np.repeat(variables, n_dates), len(codes))
        d = np.tile(dates[lo:hi].to_numpy(), len(codes) * len(variables))
        frames.append(pd.DataFrame({"code": c, "date": d, "variable": v,
                                    "value": rng.lognormal(size=c.size),
                                    "source": src}))
    long = pd.concat(frames, ignore_index=True)
    long = long.sample(n=min(1_500_000, len(long)), random_state=1)
    for col in ("code", "variable", "source"):
        long[col] = long[col].astype("category")
    kinds = {v: ("rate" if i % 5 == 0 else "flow") for i, v in enumerate(variables)}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel, prov, seams = splice_sources(long, {"a": 0, "b": 1, "c": 2},
                                            kinds=kinds)
    dt = time.perf_counter() - t0
    print(f"\nsplice_sources: {len(long):,} rows -> {len(panel):,} rows, "
          f"{len(prov):,} pairs, {len(seams):,} seams in {dt:.2f} s")
    assert len(prov) == len(codes) * len(variables)
