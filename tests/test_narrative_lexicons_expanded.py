"""alpha.2 stability lexicons must be (a) supersets of their base and
(b) at most 1.20 * base_size.

For the 5 in-scope indices (lui, ltui, ltui_up, ltui_down, lwui) the
LEXICONS entry is a dict-of-components keyed by language, so we iterate
into each component's frozenset and check the cap on a per-component
basis.
"""
from puremacro.narrative.indices._lexicons import LEXICONS
from puremacro.narrative.indices._lexicons_expanded import LEXICONS_EXPANDED


IN_SCOPE = {"lui", "ltui", "ltui_up", "ltui_down", "lwui"}


def _iter_cells(by_lang):
    """Yield (lang, component_name, frozenset) tuples regardless of whether
    the per-language value is a flat frozenset or a dict-of-components."""
    for lang, val in by_lang.items():
        if isinstance(val, dict):
            for component, fs in val.items():
                yield lang, component, fs
        else:
            yield lang, "_", val


def test_expanded_is_superset_and_capped():
    for idx_name in IN_SCOPE:
        base_by_lang = LEXICONS[idx_name]
        exp_by_lang = LEXICONS_EXPANDED[idx_name]
        # Walk base + expanded in lockstep on (lang, component) pairs.
        for lang, component, base in _iter_cells(base_by_lang):
            exp_val = exp_by_lang[lang]
            expanded = exp_val[component] if isinstance(exp_val, dict) else exp_val
            assert expanded >= base, (
                f"{idx_name}/{lang}/{component} expanded missing base terms"
            )
            # Cap: floor(1.20 * base) + 1 to allow tiny-base edge cases.
            cap = int(len(base) * 1.20) + 1
            assert len(expanded) <= cap, (
                f"{idx_name}/{lang}/{component}: expansion "
                f"{len(expanded)} > 1.2 x {len(base)} (+1) = {cap}"
            )


def test_expanded_covers_same_index_names():
    assert set(LEXICONS_EXPANDED.keys()) == IN_SCOPE


def test_expanded_covers_same_languages_per_index():
    for idx_name in IN_SCOPE:
        assert set(LEXICONS_EXPANDED[idx_name].keys()) >= set(
            LEXICONS[idx_name].keys()
        ), f"{idx_name} expanded missing languages"


def test_expanded_components_match_base_components():
    for idx_name in IN_SCOPE:
        for lang, base_val in LEXICONS[idx_name].items():
            exp_val = LEXICONS_EXPANDED[idx_name][lang]
            if isinstance(base_val, dict):
                assert isinstance(exp_val, dict), (
                    f"{idx_name}/{lang}: base is dict-of-components but "
                    f"expanded is not"
                )
                assert set(exp_val.keys()) == set(base_val.keys()), (
                    f"{idx_name}/{lang}: component keys differ "
                    f"(base={sorted(base_val.keys())}, "
                    f"expanded={sorted(exp_val.keys())})"
                )
