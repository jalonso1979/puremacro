"""store_realtime_vintages: row-level parsing survives the vectorised date path.

The per-row loop was replaced by per-distinct-value parsing. A single
vectorised ``pd.to_datetime`` over the column (what several automated
"vectorise iterrows" patches proposed) infers one format from the first
element: rows written another way become NaT and are dropped, and a
``13/02/2020`` first row makes ``01/02/2020`` parse as 1 February instead of
2 January. These tests pin the row-level answers.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from puremacro import _cache_db


def _store(df: pd.DataFrame) -> tuple[int, list[tuple]]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_cache_db._DDL_REALTIME_VINTAGES)
    n = _cache_db.store_realtime_vintages(df, conn=conn)
    rows = conn.execute(
        "SELECT provider, country, series_id, observation_date, vintage_date, value "
        "FROM realtime_vintages ORDER BY observation_date").fetchall()
    return n, rows


def _frame(dates, values, vintage="2021-01-05") -> pd.DataFrame:
    return pd.DataFrame({"provider": "p", "country": "mx", "series_id": "s",
                         "date": dates, "vintage": vintage, "value": values})


def test_mixed_date_formats_are_all_kept():
    n, rows = _store(_frame(["2020-01-01", "02/15/2020", "2020Q2", "2020-03-01 12:00"],
                            [1.0, 2.0, 3.0, 4.0]))
    assert n == 4
    assert [r[3] for r in rows] == ["2020-01-01", "2020-02-15", "2020-03-01", "2020-04-01"]


def test_first_row_does_not_fix_the_format_of_the_others():
    n, rows = _store(_frame(["13/02/2020", "01/02/2020"], [1.0, 2.0]))
    assert n == 2
    assert rows == [("p", "MX", "s", "2020-01-02", "2021-01-05", 2.0),
                    ("p", "MX", "s", "2020-02-13", "2021-01-05", 1.0)]


def test_values_and_dates_that_do_not_parse():
    n, rows = _store(_frame(
        ["2020-01-01", "2020-02-01", "2020-02-30", "not a date", None, "2020-06-01"],
        ["3.5", "abc", 1.0, 2.0, 3.0, "nan"],
        vintage=["2021-01-05"] * 5 + [None]))
    # "abc" is dropped, 2020-02-30 / "not a date" / None dates are dropped,
    # and a missing vintage drops the row even though its value is fine.
    assert n == 1
    assert rows == [("p", "MX", "s", "2020-01-01", "2021-01-05", 3.5)]


def test_missing_value_is_stored_as_null():
    n, rows = _store(_frame(["2020-01-01", "2020-02-01"], [np.nan, "nan"]))
    assert n == 2
    assert [r[5] for r in rows] == [None, None]


def test_datetime64_columns_with_nat():
    obs = pd.to_datetime(["2020-01-01", None, "2020-03-01"])
    vin = pd.to_datetime(["2021-01-05", "2021-01-05", "2021-02-05"]).tz_localize("UTC")
    n, rows = _store(_frame(obs, [1.0, 2.0, 3.0], vintage=vin))
    assert n == 2
    assert rows == [("p", "MX", "s", "2020-01-01", "2021-01-05", 1.0),
                    ("p", "MX", "s", "2020-03-01", "2021-02-05", 3.0)]


def test_later_duplicate_replaces_earlier():
    n, rows = _store(_frame(["2020-01-01", pd.Timestamp("2020-01-01")], [1.0, 2.0]))
    assert n == 2
    assert rows == [("p", "MX", "s", "2020-01-01", "2021-01-05", 2.0)]
