"""LWUI-wage variant — labor x uncertainty x wage triple co-occurrence."""
from __future__ import annotations

import pandas as pd

from puremacro.narrative import lwui, lwui_wage
from puremacro.narrative.indices._lexicons import LEXICONS


def _records(corpus):
    for d, txt in corpus:
        yield (pd.Timestamp(d), txt, "https://t/" + d,
               {"language": "en", "doctype": "press"})


# ---------------------------------------------------------------------------
# Lexicon shape
# ---------------------------------------------------------------------------
def test_lexicon_has_lwui_wage_entry_for_all_languages():
    assert "lwui_wage" in LEXICONS
    for lang in ("en", "es", "pt", "de", "fr", "it", "ja", "zh"):
        d = LEXICONS["lwui_wage"][lang]
        assert {"labor_domain", "uncertainty_tone", "wage_domain"}.issubset(d)
        assert len(d["wage_domain"]) >= 10, f"{lang}: wage_domain too small"


def test_wage_and_war_domains_are_disjoint_en():
    war = LEXICONS["lwui"]["en"]["war_domain"]
    wage = LEXICONS["lwui_wage"]["en"]["wage_domain"]
    common = war & wage
    assert common == frozenset(), \
        f"war / wage domains overlap in EN: {sorted(common)}"


# ---------------------------------------------------------------------------
# Function behaviour — synthetic corpora
# ---------------------------------------------------------------------------

# A wage-heavy + uncertainty-heavy + labor-heavy corpus.
_WAGE_CORPUS = [
    ("2020-01-15", "Workers and employers face mounting uncertainty about "
                   "wage growth and the next round of collective bargaining. "
                   "Pay raises were paused as labor market conditions "
                   "weakened."),
    ("2020-04-15", "Wage settlements remain uncertain; labour unions warn "
                   "of pay cuts and a wage freeze for hourly employees as "
                   "uncertainty about employment continues."),
    ("2020-07-15", "Hiring uncertainty persists. Real wages stagnated and "
                   "unit labor cost growth is unclear for workers entering "
                   "wage negotiations."),
    ("2020-10-15", "Labour markets remain uncertain. Minimum-wage increases "
                   "are debated while pay raises and cost-of-living "
                   "adjustments are deferred amid uncertainty."),
]

# A war-heavy + uncertainty-heavy + labor-heavy corpus.
_WAR_CORPUS = [
    ("2020-01-15", "Geopolitical war risk weighs on the labour market as "
                   "uncertainty about sanctions and military mobilisation "
                   "drives layoffs in defence-exposed sectors."),
    ("2020-04-15", "Armed conflict and trade embargoes raise uncertainty "
                   "about workers in occupied territories; refugee flows "
                   "and conscription disrupt employment."),
    ("2020-07-15", "Wartime uncertainty over invasion and missile attacks "
                   "displaces labour markets across the region; hiring "
                   "remains uncertain near the frontlines."),
    ("2020-10-15", "Sanctions and military deployment heighten labour-market "
                   "uncertainty; supply-chain disruption and defence "
                   "spending complicate hiring decisions."),
]


def test_lwui_wage_runs_and_returns_non_empty_index():
    rec = list(_records(_WAGE_CORPUS))
    ri = lwui_wage(rec, country="USA", normalize="raw")
    d = ri.diagnostics()
    assert d["n_quarters"] >= 1
    # At least one quarter should have a positive raw score
    assert ri.series.dropna().max() > 0


def test_lwui_wage_scores_higher_on_wage_corpus_than_war_corpus():
    rec_wage = list(_records(_WAGE_CORPUS))
    rec_war = list(_records(_WAR_CORPUS))
    s_wage_on_wage = lwui_wage(rec_wage, country="USA", normalize="raw")
    s_wage_on_war = lwui_wage(rec_war, country="USA", normalize="raw")
    # lwui_wage should fire more strongly on wage-heavy text than on
    # war-heavy text (since the third domain is wage, not war).
    mean_wage = s_wage_on_wage.series.dropna().mean()
    mean_war = s_wage_on_war.series.dropna().mean()
    assert mean_wage > mean_war, (
        f"lwui_wage should fire more on wage corpus: "
        f"mean_wage_corpus={mean_wage}, mean_war_corpus={mean_war}"
    )


def test_lwui_war_scores_higher_on_war_corpus_than_wage_corpus():
    """Sanity check the other direction: lwui (war flavour) should fire
    more on the war corpus than on the wage corpus."""
    rec_wage = list(_records(_WAGE_CORPUS))
    rec_war = list(_records(_WAR_CORPUS))
    s_war_on_war = lwui(rec_war, country="USA", normalize="raw")
    s_war_on_wage = lwui(rec_wage, country="USA", normalize="raw")
    mean_war = s_war_on_war.series.dropna().mean()
    mean_wage = s_war_on_wage.series.dropna().mean()
    assert mean_war > mean_wage, (
        f"lwui should fire more on war corpus: "
        f"mean_war_corpus={mean_war}, mean_wage_corpus={mean_wage}"
    )


def test_lwui_wage_is_distinct_from_lwui():
    """The two indices must NOT be aliased — running both on the same
    corpus should produce different series."""
    rec = list(_records(_WAGE_CORPUS + _WAR_CORPUS))
    s_wage = lwui_wage(rec, country="USA", normalize="raw")
    s_war = lwui(rec, country="USA", normalize="raw")
    assert s_wage.name != s_war.name
    # On a mixed corpus, the two series differ at at least one quarter
    diff = (s_wage.series - s_war.series).abs().dropna()
    assert (diff > 1e-9).any(), \
        "lwui_wage and lwui returned identical series on mixed corpus"


def test_lwui_wage_metadata_has_correct_index_name():
    rec = list(_records(_WAGE_CORPUS))
    ri = lwui_wage(rec, country="USA", normalize="raw")
    assert ri.metadata["index"] == "lwui_wage"
    assert ri.name == "lwui_wage_usa"
