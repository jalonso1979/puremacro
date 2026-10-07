"""Observed ENIGH totals and the units/denominators used by the application."""
import numpy as np
from numpy.testing import assert_allclose

from puremacro.datasets.enigh import load_enigh2024_deciles


def test_official_totals_and_all_household_denominator():
    d = load_enigh2024_deciles()
    assert list(d.index) == ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    assert d.households.sum() == 38_830_230
    # Literal targets from INEGI Cuadro 4.2, GASTO rows; thousands -> MXN.
    assert_allclose(d.households @ d.food, 698246718.7915 * 1000, rtol=1e-11)
    assert_allclose(d.households @ d.monetary_expenditure, 1851206621.94559 * 1000, rtol=1e-11)
    assert_allclose(d.loc["I", "food"], 33659095.74951 * 1000 / 3883023, rtol=1e-11)
    assert_allclose(d[list(d.attrs["categories"])].sum(axis=1) + d.outward_transfers,
                    d.monetary_expenditure, rtol=1e-11)
    assert not d.attrs["is_synthetic"]
    assert d.attrs["source_sha256"] == "7af7850495255fb1e6a9cb139cf531e5a62c7fd2a9a1de5ba0503f03aa82490b"


def test_income_is_an_exposure_not_consumption_or_in_kind_income():
    d = load_enigh2024_deciles()
    cash_total_thousands = sum([1465771586.0952, 13649764.4933, 40773837.11385,
                                80243008.52449, 7616530.03866, 34882837.30935])
    assert_allclose(d.households @ d.cash_wages, cash_total_thousands * 1000, rtol=1e-11)
    assert (d.cash_wages > d[list(d.attrs["categories"])].sum(axis=1)).any()
    assert d.attrs["population_unit"] == "households"
    assert d.attrs["period"] == "quarter"
    assert d.attrs["monetary_unit"] == "MXN"


def test_fresh_load_does_not_share_mutable_data_or_metadata():
    first = load_enigh2024_deciles()
    expected = first.iloc[0, 1]
    first.iloc[0, 1] = -100
    first.attrs["categories"]["food"] = "changed"
    second = load_enigh2024_deciles()
    assert second.iloc[0, 1] == expected
    assert second.attrs["categories"]["food"] != "changed"
    assert np.isfinite(second.to_numpy()).all()
