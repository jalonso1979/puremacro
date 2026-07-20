"""Tests for the EU NMS 2023 (Ciżkowicz-Ledóchowski-Rzońca) action-
based fiscal-consolidation loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


_FIXTURE = (
    Path(__file__).parent / "fixtures" / "narrative" / "synthetic_eu_nms_2023.csv"
)


def _read_fixture() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


def test_eu_nms_loader_uses_impact_t_columns_for_profile():
    """POL Personal income tax 2010 with impact_t=2.0, impact_t+1=1.5,
    impact_t+2=1.0 → 12-quarter profile (3 years × 4 quarters)."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture())
    pol_tax = [e for e in events if e.country == "POL" and e.subtarget == "tax"]
    assert len(pol_tax) == 1
    e = pol_tax[0]
    assert e.date == pd.Timestamp("2010-01-01")
    assert len(e.implementation_profile) == 12
    weights = [w for _, w in e.implementation_profile]
    assert sum(weights) == pytest.approx(1.0)


def test_eu_nms_loader_excludes_endogenous_by_default():
    """2 endogenous rows in fixture → output excludes them (only 2 events:
    POL tax + POL expenditure, both exogenous)."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture())
    assert len(events) == 2
    assert all(e.country == "POL" for e in events)
    assert all(
        e.metadata.get("endogeneity", "").lower() == "exogenous"
        for e in events
    )


def test_eu_nms_loader_include_endogenous_param():
    """include_endogenous=True → all 4 rows in fixture become events."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture(), include_endogenous=True)
    countries = {e.country for e in events}
    assert countries == {"POL", "HUN", "CZE"}


def test_eu_nms_loader_category_to_subtarget_mapping():
    """Personal income tax → tax; Public investment → expenditure;
    'Other widget' (unrecognized) → general."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture(), include_endogenous=True)
    pol_tax = next(e for e in events if e.country == "POL" and e.subtarget == "tax")
    pol_exp = next(e for e in events if e.country == "POL" and e.subtarget == "expenditure")
    hun_unknown = next(e for e in events if e.country == "HUN")
    assert pol_tax.subtarget == "tax"
    assert pol_exp.subtarget == "expenditure"
    assert hun_unknown.subtarget == "general"


def test_eu_nms_loader_normalizes_to_pct_gdp():
    """impact totals are converted to % of GDP via the gdp column."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture())
    pol_tax = next(e for e in events if e.country == "POL" and e.subtarget == "tax")
    # POL Personal income tax: total = 2.0+1.5+1.0 = 4.5 LCU bn,
    # gdp=400 → magnitude = 4.5/400×100 = 1.125% of GDP.
    assert pol_tax.magnitude == pytest.approx(1.125)
    assert pol_tax.magnitude_unit == "pct_gdp"


def test_eu_nms_loader_skips_zero_total():
    """Row with all impact_t+k zero → no event emitted."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    df = pd.DataFrame([
        {"country": "POL", "year": 2010, "category": "VAT",
         "impact_t": 0.0, "impact_t+1": 0.0, "impact_t+2": 0.0,
         "impact_t+3": 0.0, "impact_t+4": 0.0, "impact_t+5": 0.0,
         "gdp": 400, "endogeneity": "exogenous"},
    ])
    events = eu_nms_csv_to_events(df)
    assert events == []


def test_eu_nms_loader_sign_convention_is_consolidation_negative():
    """flip_sign=True (default) → consolidations end with sign=-1."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import (
        eu_nms_csv_to_events,
    )
    events = eu_nms_csv_to_events(_read_fixture())
    assert all(e.sign == -1 for e in events)


def test_eu_nms_loader_load_with_csv_path():
    """`load(csv_path=, countries=['POL'])` returns POL-only dict."""
    from puremacro.narrative.replication.eu_nms_2023_consolidations import load
    out = load(csv_path=str(_FIXTURE), countries=["POL"])
    assert set(out.keys()) == {"POL"}
