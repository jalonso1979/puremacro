"""Census Microdata API provider against a fake endpoint (offline)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import census

KEY = "SECRETKEY0123456789"


class FakeCensus:
    """Serves a synthetic person file, honouring get=, for= and the key."""

    def __init__(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        serial = [f"2023HU{1000 + i // 3:07d}" for i in range(n)]
        cols = {"SERIALNO": serial,
                "SPORDER": [str(i % 3 + 1) for i in range(n)],
                "AGEP": [str(a) for a in rng.integers(18, 90, n)],
                "WAGP": [str(a) for a in rng.integers(0, 200000, n)],
                "ADJINC": ["1019518"] * n}
        for stem in ("PWGTP", "WGTP"):
            base = rng.integers(5, 60, n)
            cols[stem] = [str(b) for b in base]
            for r in range(1, 81):
                cols[f"{stem}{r}"] = [str(max(0, b + d)) for b, d in
                                      zip(base, rng.integers(-5, 6, n))]
        self.table = pd.DataFrame(cols)
        self.calls = []

    def __call__(self, url, *, params=None, secret_params=None, refresh=False,
                 timeout=60, headers=None):
        self.calls.append({"url": url, "params": dict(params or {}),
                           "secret": dict(secret_params or {})})
        assert secret_params == {"key": KEY}
        assert "key" not in (params or {})
        get = params["get"].split(",")
        assert len(get) <= census.MAX_VARIABLES
        # shuffle row order per call, as a real server may
        t = self.table.sample(frac=1, random_state=len(self.calls))
        rows = [get + (["state"] if "for" in params else [])]
        for _, r in t.iterrows():
            rows.append([r[c] for c in get] + (["11"] if "for" in params else []))
        return json.dumps(rows).encode()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    f = FakeCensus()
    monkeypatch.setattr(census._http, "cached_get", f)
    return f


def test_acs_pums_chunks_past_the_variable_cap_and_rejoins(fake):
    mf = census.fetch_acs_pums(2023, ["AGEP", "WAGP", "ADJINC"], geography="state:11")
    # 3 analysis + 81 weight columns, 48 per call after the 2 ids
    assert len(fake.calls) == 2
    assert all(c["params"]["for"] == "state:11" for c in fake.calls)
    assert len(mf.data) == 30
    t = fake.table.set_index(["SERIALNO", "SPORDER"])
    got = mf.data.assign(SPORDER=mf.data["SPORDER"].astype(str)).set_index(["SERIALNO", "SPORDER"])
    for col in ("AGEP", "WAGP", "PWGTP", "PWGTP80"):
        assert (got[col].astype(int) == t.loc[got.index, col].astype(int)).all()
    assert not pd.api.types.is_numeric_dtype(mf.data["SERIALNO"])
    assert mf.data["WAGP"].dtype.kind in "if"
    assert mf.design.method == "sdr" and mf.source == "census.acs_pums"
    assert KEY not in mf.query
    assert "state" in mf.data.columns


def test_acs_pums_household_unit_keeps_one_record_per_unit(fake):
    mf = census.fetch_acs_pums(2023, ["WAGP"], unit="household")
    assert len(mf.data) == 10
    assert (mf.data["SPORDER"] == 1).all()
    assert mf.design.weight == "WGTP"
    res = mf.mean("WAGP")
    assert np.isfinite(res.loc[0, "se"])


def test_predicates_are_passed_through(fake):
    census.fetch_acs_pums(2023, ["WAGP"], where={"AGEP": (25, 64), "SEX": 2,
                                                  "ST": ["06", "36"]},
                          replicate_weights=False)
    p = fake.calls[0]["params"]
    assert p["AGEP"] == "25:64" and p["SEX"] == "2" and p["ST"] == "06,36"
    assert len(fake.calls) == 1


def test_missing_key_raises_before_any_request(monkeypatch, tmp_path):
    from puremacro import credentials
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "default_config_path", lambda: tmp_path / "none.toml")
    calls = []
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: calls.append(1))
    with pytest.raises(credentials.MissingCredentialError):
        census.fetch_acs_pums(2023, ["WAGP"])
    assert not calls


def test_empty_204_response_gives_an_empty_frame(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: b"")
    df = census.fetch_records("u", ["A"], ["SERIALNO", "SPORDER"])
    assert df.empty and list(df.columns) == ["SERIALNO", "SPORDER", "A"]


def test_chunks_with_different_record_sets_are_refused(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    replies = iter([
        json.dumps([["ID", "A"], ["1", "x"], ["2", "y"]]).encode(),
        json.dumps([["ID", "B"], ["1", "x"]]).encode(),
    ])
    monkeypatch.setattr(census._http, "cached_get", lambda *a, **k: next(replies))
    monkeypatch.setattr(census, "MAX_VARIABLES", 2)
    with pytest.raises(ValueError, match="different record sets"):
        census.fetch_records("u", ["A", "B"], ["ID"], )


def test_cps_basic_is_weights_only(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", KEY)
    seen = {}

    def fake(url, *, params, secret_params, **k):
        seen["url"] = url
        get = params["get"].split(",")
        rows = [get, ["0001", "11", "1", "1", "2500.5"], ["0001", "11", "2", "2", "2400.0"]]
        return json.dumps(rows).encode()

    monkeypatch.setattr(census._http, "cached_get", fake)
    mf = census.fetch_cps_basic(2024, 1, ["PEMLR"])
    assert seen["url"].endswith("/2024/cps/basic/jan")
    assert mf.design.weight == "PWCMPWGT" and not mf.design.has_variance
    assert mf.data["HRHHID"].tolist() == ["0001", "0001"]
    assert np.isnan(mf.mean("PEMLR").loc[0, "se"])


def test_chunk_layout():
    ch = census._chunks(["A", "B"], [f"v{i}" for i in range(100)], limit=50)
    assert [len(c) for c in ch] == [48, 48, 4]
    assert census._chunks(["A"], []) == [[]]
