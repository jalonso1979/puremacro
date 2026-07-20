"""Within-year quarterization helper for replication-loader profiles.

Loaders that carry annual implementation tranches (DGLP, Devries,
Guajardo) need to project an annual weight onto four (or four-shifted)
quarter starts. Three rules:

- ``uniform``: weight / 4 in each calendar quarter Q1..Q4 of ``year``.
  IMF cross-country fiscal-flow convention; default.
- ``front_loaded``: 40% / 30% / 20% / 10% in Q1..Q4. Reflects the
  empirical pattern that consolidations bite earliest (January VAT,
  Q1 wage freezes). Country-agnostic.
- ``fiscal_year``: weight / 4 in each quarter starting at the country's
  fiscal-year start. Requires a ``country`` ISO3.
"""
from __future__ import annotations

import pandas as pd

# Fiscal-year start month per country. Sourced from each country's
# central-budget calendar; conservative subset — extend as new
# countries enter the panel. Countries not in the dict cannot use
# ``rule="fiscal_year"``.
FY_START_MONTH: dict[str, int] = {
    # Calendar-year economies (Jan start).
    "AUT": 1, "BEL": 1, "CHE": 1, "CZE": 1, "DEU": 1, "DNK": 1,
    "ESP": 1, "EST": 1, "FIN": 1, "FRA": 1, "GRC": 1, "HUN": 1,
    "IRL": 1, "ISL": 1, "ITA": 1, "KOR": 1, "LTU": 1, "LUX": 1,
    "LVA": 1, "MEX": 1, "NLD": 1, "NOR": 1, "POL": 1, "PRT": 1,
    "SVK": 1, "SVN": 1, "SWE": 1, "TUR": 1,
    # Australia, New Zealand: July-June.
    "AUS": 7, "NZL": 7,
    # Canada: April-March.
    "CAN": 4,
    # United Kingdom: April-March.
    "GBR": 4,
    # Japan: April-March.
    "JPN": 4,
    # United States federal: October-September.
    "USA": 10,
}


def annualize_to_quarters(
    year: int,
    *,
    weight: float = 1.0,
    rule: str = "uniform",
    country: str | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    """Project an annual ``weight`` onto four quarter starts.

    Parameters
    ----------
    year : calendar year (or fiscal-year start year for ``rule="fiscal_year"``).
    weight : annual weight to distribute. Output sums to this value.
    rule : ``"uniform"`` | ``"front_loaded"`` | ``"fiscal_year"``.
    country : ISO3, required when ``rule="fiscal_year"``.

    Returns
    -------
    list of (quarter_start_timestamp, weight) tuples. Length 4. Sum
    equals input ``weight`` (within float tolerance).
    """
    if rule == "uniform":
        qs = pd.date_range(f"{year}-01-01", periods=4, freq="QS")
        share = weight / 4.0
        return [(q, share) for q in qs]
    if rule == "front_loaded":
        qs = pd.date_range(f"{year}-01-01", periods=4, freq="QS")
        ratios = [0.40, 0.30, 0.20, 0.10]
        return [(q, weight * r) for q, r in zip(qs, ratios)]
    if rule == "fiscal_year":
        if country is None:
            raise ValueError("rule='fiscal_year' requires a country code.")
        if country not in FY_START_MONTH:
            raise KeyError(
                f"Unknown country code {country!r} for fiscal-year rule. "
                f"Add it to FY_START_MONTH or pass rule='uniform' instead."
            )
        start_month = FY_START_MONTH[country]
        first = pd.Timestamp(year, start_month, 1)
        qs = [
            pd.Period(first + pd.DateOffset(months=3 * k), freq="Q").to_timestamp()
            for k in range(4)
        ]
        share = weight / 4.0
        return [(q, share) for q in qs]
    raise ValueError(
        f"unknown within_year_rule {rule!r}; "
        f"expected 'uniform' | 'front_loaded' | 'fiscal_year'"
    )


__all__ = ["annualize_to_quarters", "FY_START_MONTH"]
