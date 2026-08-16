"""Canonical Macroeconomic Research Datasets.

Provides built-in, offline loaders for empirical macroeconomic benchmark data:
- Galí (1999, AER): Technology shocks and hours worked.
- Mertens & Ravn (2013, AER): Narrative tax liability shocks.
- US Macroeconomic Quarterly Panel (GDP, Investment, Inflation, Fed Funds).
- US Macroeconomic Monthly Panel (CPI, Core CPI, Unemployment, FFR, NFCI).
- DICE-2016 Climate-Macro Parameters (Nordhaus 2018).
"""
from puremacro.datasets.loaders import (
    load_gali1999,
    load_narrative_tax_shocks,
    load_macro_quarterly,
    load_macro_monthly,
    load_banxico_stance,
    load_dice_parameters,
    list_datasets,
)

__all__ = [
    "load_gali1999",
    "load_narrative_tax_shocks",
    "load_macro_quarterly",
    "load_macro_monthly",
    "load_banxico_stance",
    "load_dice_parameters",
    "list_datasets",
]
