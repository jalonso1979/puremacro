# Narrative Extension — Slice 6a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift LUI vs urate ρ from +0.18 to ≥+0.30 via sentence-level co-occurrence scoring. Length-normalize WUI per Ahir-Bloom-Furceri. Add Hubert-inspired vocabulary to WUI. Hard cutover → 0.8.0.

**Architecture:** New `sentence_cooccurrence_kernel` splits text per language, scores doc as fraction of sentences containing labor-domain ∩ uncertainty-tone (or a curated phrase). New `_LABOR_DOMAIN_<lang>` and `_UNCERTAINTY_TONE_<lang>` lexicons added; existing `_LUI_<lang>` (Slice 5 phrases) kept under new key `phrases`. WUI gains a `length_normalize=True` path.

**Tech Stack:** Python 3.10+, `re`. No new deps.

**Spec reference:** `docs/specs/2026-05-09-narrative-slice6a-lui-cooccurrence-and-wui-normalization.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (current head v0.7.2 / 9f58145).

**Pre-implementation baseline:** `pytest -q` after Slice 5 = **979 passed, 27 skipped**, plus 1 pre-existing pyodide-compat failure (unchanged).

---

## File Structure

### Files modified
- `puremacro/narrative/indices/_kernels.py` — add `_split_sentences`, `sentence_cooccurrence_kernel`; add `length_normalize` parameter to `keyword_count_kernel`.
- `puremacro/narrative/indices/_lexicons.py` — rename `_LUI_<lang>` → `_LUI_PHRASES_<lang>` (8 renames; content unchanged); add `_LABOR_DOMAIN_<lang>` and `_UNCERTAINTY_TONE_<lang>` (8 each); restructure `LEXICONS["lui"][lang]` to dict-of-frozensets; expand `_WUI_<lang>` with Hubert-inspired terms.
- `puremacro/narrative/indices/lui.py` — switch to `sentence_cooccurrence_kernel`; update lexicon-shape handling.
- `puremacro/narrative/indices/wui.py` — pass `length_normalize=True`; update docstring.
- `puremacro/tests/test_narrative_kernels.py` — new file: tests for split + co-occurrence + length-norm.
- `puremacro/tests/test_narrative_lui_cooccurrence.py` — new file: focused integration tests on new LUI methodology.
- `puremacro/tests/test_narrative_indices.py` — update LUI lexicon-shape assertions; add coverage tests for new lexicons.
- Notebook 28 outputs (re-rendered).
- `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, `CHANGELOG.md` — 0.7.2 → 0.8.0.

---

## Task 0: Branch + baseline

- [ ] **Step 1: Verify branch + baseline**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current   # feature/narrative-extension-slice3
git log --oneline -1        # 9f58145 (Slice 5 fixup) tagged v0.7.2
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 979 passed, 27 skipped.

---

## Task 1: Sentence splitter + co-occurrence kernel + length-norm

**Files:**
- Modify: `puremacro/narrative/indices/_kernels.py`
- Create: `puremacro/tests/test_narrative_kernels.py`

This is the foundational code change. All three new pieces are coupled and tested together.

- [ ] **Step 1: Write failing tests**

Create `puremacro/tests/test_narrative_kernels.py` with the following content:

```python
"""Unit tests for sentence splitter, sentence_cooccurrence_kernel,
and length_normalize on keyword_count_kernel (Slice 6a)."""
from __future__ import annotations
import pandas as pd


# -----------------------------------------------------------------------------
# _split_sentences
# -----------------------------------------------------------------------------

def test_split_sentences_english_basic():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("First sentence. Second sentence! Third?", "en")
    assert len(out) == 3
    assert out[0].startswith("First")
    assert out[1].startswith("Second")
    assert out[2].startswith("Third")


def test_split_sentences_english_empty():
    from puremacro.narrative.indices._kernels import _split_sentences
    assert _split_sentences("", "en") == []


def test_split_sentences_english_no_punctuation():
    """A doc with no boundary punctuation is treated as one sentence."""
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("just one sentence with no end mark", "en")
    assert len(out) == 1


def test_split_sentences_chinese():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("第一句话。第二句话！第三句话？", "zh")
    assert len(out) == 3


def test_split_sentences_japanese():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("第一文。第二文！第三文？", "ja")
    assert len(out) == 3


# -----------------------------------------------------------------------------
# sentence_cooccurrence_kernel
# -----------------------------------------------------------------------------

def test_sentence_cooccurrence_all_match():
    """Every sentence has both labor + uncertainty term → score 1.0."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment risk has risen. The labor market is weak."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "labor market"})
    unc = frozenset({"risk", "weak"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert len(out) == 1
    assert out[0][1] == 1.0


def test_sentence_cooccurrence_partial_match():
    """Only one of two sentences has co-occurrence → score 0.5."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment grew strongly last quarter. Risk remains elevated."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "labor market"})
    unc = frozenset({"risk"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert out[0][1] == 0.5


def test_sentence_cooccurrence_no_match():
    """Labor terms appear but no uncertainty in same sentence → 0.0."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment grew. Wages rose."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "wages"})
    unc = frozenset({"risk", "uncertain"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert out[0][1] == 0.0


def test_sentence_cooccurrence_phrase_shortcut():
    """A sentence with a curated phrase matches even without co-occurrence."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Rising unemployment was discussed. Employment is fine."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment"})  # the second sentence has labor
    unc = frozenset({"risk"})           # but no uncertainty term
    phrases = frozenset({"rising unemployment"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], phrases=phrases, language="en",
    ))
    # First sentence matches via phrase (rising unemployment).
    # Second sentence: has "employment" but no uncertainty → no match.
    # Score = 1/2 = 0.5
    assert out[0][1] == 0.5


def test_sentence_cooccurrence_empty_doc():
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    records = [(pd.Timestamp("2024-01-01"), "", "url", {})]
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[frozenset({"x"}), frozenset({"y"})],
        language="en",
    ))
    assert out[0][1] == 0.0


def test_sentence_cooccurrence_multiple_groups():
    """Generalized to >2 groups (e.g. labor ∩ uncertainty ∩ time-ref)."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment risk rose this quarter."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    g1 = frozenset({"employment"})
    g2 = frozenset({"risk"})
    g3 = frozenset({"quarter"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[g1, g2, g3], language="en",
    ))
    assert out[0][1] == 1.0


# -----------------------------------------------------------------------------
# keyword_count_kernel — length normalization
# -----------------------------------------------------------------------------

def test_keyword_count_length_normalize_doubles_text_halves_score():
    """Doubling text length with same hits halves the per-1000-word score."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    text_short = "uncertainty " * 5 + "the " * 95     # 100 words, 5 hits
    text_long = "uncertainty " * 5 + "the " * 195    # 200 words, 5 hits
    records_short = [(pd.Timestamp("2024-01-01"), text_short, "u", {})]
    records_long = [(pd.Timestamp("2024-01-01"), text_long, "u", {})]
    terms = frozenset({"uncertainty"})
    s_short = list(keyword_count_kernel(
        records_short, terms=terms, language="en", length_normalize=True,
    ))[0][1]
    s_long = list(keyword_count_kernel(
        records_long, terms=terms, language="en", length_normalize=True,
    ))[0][1]
    assert abs(s_short - 50.0) < 1.0   # 5/100 * 1000 = 50
    assert abs(s_long - 25.0) < 1.0    # 5/200 * 1000 = 25


def test_keyword_count_length_normalize_default_off():
    """Default behavior unchanged: returns raw hit count."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    text = "uncertainty " * 5 + "x " * 95
    records = [(pd.Timestamp("2024-01-01"), text, "u", {})]
    terms = frozenset({"uncertainty"})
    score = list(keyword_count_kernel(records, terms=terms, language="en"))[0][1]
    assert score == 5.0


def test_keyword_count_length_normalize_empty_doc():
    """Empty doc with length_normalize → 0.0 (no division by zero)."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    records = [(pd.Timestamp("2024-01-01"), "", "u", {})]
    score = list(keyword_count_kernel(
        records, terms=frozenset({"x"}), language="en", length_normalize=True,
    ))[0][1]
    assert score == 0.0
```

- [ ] **Step 2: Run tests, verify failure**

```bash
cd puremacro && pytest tests/test_narrative_kernels.py -v --no-header 2>&1 | tail -20
```
Expected: All fail with `ImportError: cannot import name '_split_sentences'` or `cannot import name 'sentence_cooccurrence_kernel'`. The two `keyword_count_length_normalize` tests fail with `TypeError: unexpected keyword argument 'length_normalize'`.

- [ ] **Step 3: Implement in `_kernels.py`**

Open `puremacro/narrative/indices/_kernels.py`. Replace the existing module content with:

```python
"""Per-document scoring kernels for the indices layer.

Slice 2 introduced three kernels: ``keyword_count_kernel``,
``cooccurrence_kernel``, ``tone_kernel``.

Slice 6a adds:
  - ``_split_sentences(text, language)`` — language-aware splitter.
  - ``sentence_cooccurrence_kernel`` — score = fraction of sentences that
    contain ≥1 term from each group OR ≥1 curated phrase. Designed for
    long multi-topic documents (e.g. FOMC minutes) where document-level
    co-occurrence is too coarse.
  - ``length_normalize`` parameter on ``keyword_count_kernel`` —
    Ahir-Bloom-Furceri WUI normalization (hits / total_words × 1000).
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..types import VALID_RISKINDEX_NORMALIZATION as _VALID_NORMALIZATIONS

_NEEDS_SUBSTRING = {"ja", "zh"}
_TOKEN_RX = re.compile(r"\w+", flags=re.UNICODE)
_SENT_LATIN_RX = re.compile(r"[.!?]+(?:\s+|$)")
_SENT_CJK_RX = re.compile(r"[。！？]+")


def _normalize_text_for_match(text: str, language: str) -> str:
    if language in _NEEDS_SUBSTRING:
        return text
    return text.lower()


def count_keywords(text: str, terms: frozenset[str], language: str = "en") -> int:
    if not text:
        return 0
    norm = _normalize_text_for_match(text, language)
    if language in _NEEDS_SUBSTRING:
        return sum(norm.count(t) for t in terms)
    n = 0
    for t in terms:
        rx = r"\b" + re.escape(t.lower()) + r"\b"
        n += len(re.findall(rx, norm))
    return n


def _split_sentences(text: str, language: str) -> list[str]:
    """Split text into sentences. Latin scripts use [.!?]+ boundary;
    CJK uses [。！？]+. Empty strings are stripped from the result;
    a doc with no punctuation is one sentence (or empty)."""
    if not text:
        return []
    if language in _NEEDS_SUBSTRING:
        parts = _SENT_CJK_RX.split(text)
    else:
        parts = _SENT_LATIN_RX.split(text)
    return [p.strip() for p in parts if p.strip()]


def keyword_count_kernel(
    records: Iterable[tuple],
    *,
    terms: frozenset[str],
    language: str = "en",
    length_normalize: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, score)`` per record.

    If ``length_normalize=False`` (default): score = total keyword hits.
    If ``length_normalize=True``: score = (hits / total_words) * 1000
        (Ahir-Bloom-Furceri WUI normalization). Empty docs → 0.0.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        hits = count_keywords(text, terms, language=lang)
        if not length_normalize:
            yield (pd.Timestamp(date), float(hits))
            continue
        total = len(_TOKEN_RX.findall(text or ""))
        if total == 0:
            yield (pd.Timestamp(date), 0.0)
        else:
            yield (pd.Timestamp(date), float(hits) / total * 1000.0)


def cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, 1.0)`` if the document contains ≥1 term from EVERY
    group, else ``(date, 0.0)``. Used by BBD-EPU."""
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        score = 1.0 if all(
            count_keywords(text, group, language=lang) > 0
            for group in term_groups
        ) else 0.0
        yield (pd.Timestamp(date), score)


def sentence_cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    phrases: frozenset[str] | None = None,
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, fraction-of-sentences-matching)`` per record.

    A sentence "matches" iff:
      (∀ group in term_groups: count_keywords(s, group) > 0)
      OR
      (phrases is not None AND count_keywords(s, phrases) > 0)

    Score = matched_sentences / total_sentences ∈ [0, 1].
    Empty docs and docs with zero parseable sentences yield 0.0.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        sentences = _split_sentences(text or "", lang)
        if not sentences:
            yield (pd.Timestamp(date), 0.0)
            continue
        matched = 0
        for s in sentences:
            cooc = all(
                count_keywords(s, group, language=lang) > 0
                for group in term_groups
            )
            phrase_hit = (
                phrases is not None
                and count_keywords(s, phrases, language=lang) > 0
            )
            if cooc or phrase_hit:
                matched += 1
        yield (pd.Timestamp(date), float(matched) / len(sentences))


def tone_kernel(
    records: Iterable[tuple],
    *,
    hawkish_terms: frozenset[str],
    dovish_terms: frozenset[str],
    language: str = "en",
) -> Iterator[tuple]:
    """Yield ``(date, net_tone)`` per doc.

    ``net_tone = (h - d) / (h + d)`` for documents with at least one hit;
    ``0.0`` if neither lexicon matches. Per-doc score in ``[-1, +1]``.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        h = count_keywords(text, hawkish_terms, language=lang)
        d = count_keywords(text, dovish_terms, language=lang)
        total = h + d
        score = 0.0 if total == 0 else (h - d) / total
        yield (pd.Timestamp(date), float(score))


def normalize_series(
    series: pd.Series,
    normalization: str,
    *,
    base_period: tuple[str, str] | None = None,
) -> pd.Series:
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"normalization {normalization!r} not in {_VALID_NORMALIZATIONS}"
        )
    if normalization == "raw":
        return series.copy()

    if base_period is None:
        ref = series.dropna()
    else:
        start, end = base_period
        ref = series.loc[start:end].dropna()

    if ref.empty:
        return series.copy()

    mu = float(ref.mean())
    sigma = float(ref.std(ddof=0))
    if sigma == 0.0:
        sigma = 1.0

    z = (series - mu) / sigma
    if normalization == "zscore":
        return z
    return 100.0 + 50.0 * z


__all__ = [
    "count_keywords",
    "_split_sentences",
    "keyword_count_kernel",
    "cooccurrence_kernel",
    "sentence_cooccurrence_kernel",
    "tone_kernel",
    "normalize_series",
]
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd puremacro && pytest tests/test_narrative_kernels.py -v --no-header 2>&1 | tail -20
```
Expected: all 14 new tests pass.

- [ ] **Step 5: Run full suite — verify nothing broke**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 979 + 14 = 993 passed (no existing test broken; default behavior preserved).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/_kernels.py \
        puremacro/tests/test_narrative_kernels.py
git commit -m "feat(narrative): sentence_cooccurrence_kernel + length_normalize for keyword_count"
```

---

## Task 2: New `_LABOR_DOMAIN_<lang>` lexicons (8 languages)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py`
- Modify: `puremacro/tests/test_narrative_indices.py`

**Goal:** Broad labor-economics vocabulary, polarity-neutral. ~30-40 terms per Latin language; ~25 per CJK.

- [ ] **Step 1: Add `_LABOR_DOMAIN_EN`**

In `puremacro/narrative/indices/_lexicons.py`, find the existing `_LUI_EN = frozenset({...})` constant. INSERT BEFORE IT (do not replace yet):

```python
# ---------------------------------------------------------------------------
# Labor-domain vocabulary (Slice 6a) — broad labor-economics terms,
# polarity-neutral. Used as one of two co-occurring groups in the
# sentence-level LUI scoring (the other is _UNCERTAINTY_TONE_<lang>).
# Curated to be substantially broader than _LUI_PHRASES_<lang> (which
# is high-precision pre-formed phrases).
# ---------------------------------------------------------------------------
_LABOR_DOMAIN_EN = frozenset({
    "labor", "labour",
    "labor market", "labour market",
    "labor force", "labour force",
    "employment", "unemployment", "underemployment",
    "jobs", "job", "job market", "job creation", "job openings",
    "hiring", "hire", "hires",
    "wages", "wage", "salaries", "salary", "compensation", "earnings",
    "workforce", "workers", "worker", "employees", "employee",
    "payroll", "payrolls", "nonfarm payrolls",
    "labor cost", "labor costs", "unit labor cost", "unit labor costs",
    "vacancies", "vacancy",
    "layoffs", "layoff",
    "quit rate", "quits",
    "participation", "participation rate",
    "jobless", "jobless rate", "unemployment rate",
    "labor productivity",
    "labor demand", "labor supply",
    "openings",
    "wage growth",
})
```

(~45 terms.)

- [ ] **Step 2: Add other 7 `_LABOR_DOMAIN_<lang>` constants**

Add these BEFORE the existing `_LUI_<lang>` blocks. Aim for ~30 terms per Latin language (es/pt/de/fr/it) and ~25 for ja/zh. The implementer should curate genuine labor-economics vocabulary in each language; below are starter lists — expand as needed to hit per-language targets.

```python
_LABOR_DOMAIN_ES = frozenset({
    "trabajo", "mercado laboral", "mercado de trabajo",
    "fuerza laboral", "fuerza de trabajo",
    "empleo", "desempleo", "subempleo",
    "puestos de trabajo", "puesto de trabajo",
    "creación de empleo",
    "contratación", "contrataciones",
    "salarios", "salario", "sueldos", "sueldo", "remuneración",
    "trabajadores", "trabajador", "empleados", "empleado",
    "nómina", "nóminas",
    "costos laborales", "coste laboral",
    "vacantes", "vacante",
    "despidos", "despido",
    "tasa de paro", "tasa de desempleo", "tasa de empleo",
    "tasa de participación",
    "productividad laboral",
    "demanda laboral", "oferta laboral",
    "crecimiento salarial",
})

_LABOR_DOMAIN_PT = frozenset({
    "trabalho", "mercado de trabalho", "mercado laboral",
    "força de trabalho",
    "emprego", "desemprego", "subemprego",
    "vagas de emprego", "vagas",
    "criação de empregos", "criação de vagas",
    "contratação", "contratações",
    "salários", "salário", "remuneração", "ganhos",
    "trabalhadores", "trabalhador", "empregados", "empregado",
    "folha de pagamento", "folha salarial",
    "custos do trabalho", "custo do trabalho",
    "demissões", "demissão",
    "taxa de desemprego", "taxa de emprego",
    "taxa de participação",
    "produtividade do trabalho",
    "demanda por trabalho", "oferta de trabalho",
    "crescimento salarial",
})

_LABOR_DOMAIN_DE = frozenset({
    "arbeit", "arbeitsmarkt",
    "erwerbstätigkeit", "erwerbstätige",
    "beschäftigung", "arbeitslosigkeit", "unterbeschäftigung",
    "arbeitsplätze", "arbeitsplatz",
    "schaffung von arbeitsplätzen",
    "einstellung", "einstellungen",
    "löhne", "lohn", "gehälter", "gehalt", "vergütung",
    "arbeitnehmer", "arbeiter", "beschäftigte",
    "lohnkosten", "arbeitskosten",
    "stellenangebote", "offene stellen",
    "entlassungen", "entlassung",
    "arbeitslosenquote", "beschäftigungsquote",
    "erwerbsquote",
    "arbeitsproduktivität",
    "arbeitsnachfrage", "arbeitsangebot",
    "lohnwachstum",
})

_LABOR_DOMAIN_FR = frozenset({
    "travail", "marché du travail",
    "main-d'œuvre", "force de travail",
    "emploi", "chômage", "sous-emploi",
    "postes", "postes de travail",
    "création d'emplois",
    "embauche", "embauches", "recrutement",
    "salaires", "salaire", "rémunération", "rémunérations",
    "travailleurs", "travailleur", "salariés", "salarié",
    "masse salariale",
    "coûts du travail", "coût du travail",
    "offres d'emploi", "postes vacants",
    "licenciements", "licenciement",
    "taux de chômage", "taux d'emploi",
    "taux d'activité",
    "productivité du travail",
    "demande de travail", "offre de travail",
    "croissance des salaires",
})

_LABOR_DOMAIN_IT = frozenset({
    "lavoro", "mercato del lavoro",
    "forza lavoro", "forza-lavoro",
    "occupazione", "disoccupazione", "sottoccupazione",
    "posti di lavoro", "posto di lavoro",
    "creazione di posti di lavoro",
    "assunzioni", "assunzione",
    "salari", "salario", "stipendi", "stipendio", "retribuzioni",
    "lavoratori", "lavoratore", "occupati", "dipendenti",
    "monte salari",
    "costo del lavoro", "costi del lavoro",
    "posti vacanti", "vacanze",
    "licenziamenti", "licenziamento",
    "tasso di disoccupazione", "tasso di occupazione",
    "tasso di partecipazione",
    "produttività del lavoro",
    "domanda di lavoro", "offerta di lavoro",
    "crescita salariale",
})

_LABOR_DOMAIN_JA = frozenset({
    "労働", "労働市場",
    "労働力",
    "雇用", "失業", "不完全雇用",
    "就業", "就労",
    "雇用創出",
    "採用", "新規採用",
    "賃金", "給料", "給与", "報酬",
    "労働者", "従業員", "被用者",
    "人件費",
    "求人", "求人数", "求人倍率",
    "解雇", "離職",
    "失業率", "雇用率", "労働参加率",
    "労働生産性",
    "労働需要", "労働供給",
    "賃金上昇",
})

_LABOR_DOMAIN_ZH = frozenset({
    "劳动", "劳动力市场", "就业市场",
    "劳动力",
    "就业", "失业", "不充分就业",
    "工作", "岗位", "职位",
    "创造就业",
    "招聘", "录用",
    "工资", "薪资", "薪酬", "报酬",
    "工人", "员工", "雇员",
    "工资总额",
    "招聘需求", "用工需求",
    "解雇", "辞退",
    "失业率", "就业率",
    "劳动参与率",
    "劳动生产率",
    "劳动需求", "劳动供给",
    "工资增长",
})
```

If any language's count is below ~30 (Latin) or ~25 (CJK), expand with additional genuine vocabulary in that language.

- [ ] **Step 3: Add coverage tests**

In `puremacro/tests/test_narrative_indices.py`, append:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("en", 35),
    ("es", 30), ("pt", 30), ("de", 30), ("fr", 30), ("it", 30),
    ("ja", 25), ("zh", 25),
])
def test_labor_domain_lexicon_substantive_coverage(lang, min_count):
    """Slice 6a labor-domain lexicons must be broad enough for sentence
    co-occurrence to fire. Per-language thresholds reflect concept density
    in each script."""
    from puremacro.narrative.indices._lexicons import (
        _LABOR_DOMAIN_EN, _LABOR_DOMAIN_ES, _LABOR_DOMAIN_PT,
        _LABOR_DOMAIN_DE, _LABOR_DOMAIN_FR, _LABOR_DOMAIN_IT,
        _LABOR_DOMAIN_JA, _LABOR_DOMAIN_ZH,
    )
    name_to_lex = {
        "en": _LABOR_DOMAIN_EN, "es": _LABOR_DOMAIN_ES,
        "pt": _LABOR_DOMAIN_PT, "de": _LABOR_DOMAIN_DE,
        "fr": _LABOR_DOMAIN_FR, "it": _LABOR_DOMAIN_IT,
        "ja": _LABOR_DOMAIN_JA, "zh": _LABOR_DOMAIN_ZH,
    }
    terms = name_to_lex[lang]
    assert len(terms) >= min_count, (
        f"_LABOR_DOMAIN_{lang.upper()} has {len(terms)} terms; "
        f"need ≥ {min_count}"
    )
```

- [ ] **Step 4: Verify counts**

```bash
cd puremacro && python3 -c "
from puremacro.narrative.indices._lexicons import (
    _LABOR_DOMAIN_EN, _LABOR_DOMAIN_ES, _LABOR_DOMAIN_PT,
    _LABOR_DOMAIN_DE, _LABOR_DOMAIN_FR, _LABOR_DOMAIN_IT,
    _LABOR_DOMAIN_JA, _LABOR_DOMAIN_ZH,
)
for n, lex in [('EN', _LABOR_DOMAIN_EN), ('ES', _LABOR_DOMAIN_ES),
               ('PT', _LABOR_DOMAIN_PT), ('DE', _LABOR_DOMAIN_DE),
               ('FR', _LABOR_DOMAIN_FR), ('IT', _LABOR_DOMAIN_IT),
               ('JA', _LABOR_DOMAIN_JA), ('ZH', _LABOR_DOMAIN_ZH)]:
    print(n, len(lex))
"
```
Expected per-lang: EN ≥ 35; ES/PT/DE/FR/IT ≥ 30; JA/ZH ≥ 25. If short, add terms to that language until threshold met.

- [ ] **Step 5: Run tests**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "labor_domain_lexicon" 2>&1 | tail -10
```
Expected: 8 tests pass.

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 993 + 8 = 1001 passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): _LABOR_DOMAIN_<lang> lexicons across 8 languages (Slice 6a)"
```

---

## Task 3: New `_UNCERTAINTY_TONE_<lang>` lexicons (8 languages)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py`
- Modify: `puremacro/tests/test_narrative_indices.py`

**Goal:** Polarity-neutral risk/uncertainty markers that, when paired with labor terms, signal labor uncertainty. ~25 terms per Latin lang; ~15 per CJK.

- [ ] **Step 1: Add `_UNCERTAINTY_TONE_EN`**

Insert after the `_LABOR_DOMAIN_<lang>` blocks:

```python
# ---------------------------------------------------------------------------
# Uncertainty/risk tone markers (Slice 6a) — polarity-neutral language
# that signals risk, weakening, or uncertainty. Used as the second
# co-occurring group in sentence-level LUI scoring.
# ---------------------------------------------------------------------------
_UNCERTAINTY_TONE_EN = frozenset({
    "uncertain", "uncertainty", "uncertainties",
    "risk", "risks", "risky",
    "downside", "downside risk", "downside risks",
    "weak", "weaken", "weakened", "weakening",
    "soft", "soften", "softened", "softening",
    "decline", "declining", "declined", "declines",
    "deteriorate", "deteriorating", "deteriorated", "deterioration",
    "slow", "slowdown", "slowing", "slowed",
    "fragile", "fragility",
    "concern", "concerns", "concerned",
    "worry", "worries",
    "volatile", "volatility",
    "headwinds",
    "fall", "fell", "falling",
    "drop", "drops", "dropped",
    "loss", "losses",
    "tightening",
    "subdued",
})
```

(~50 terms. Frozenset will dedupe; that's fine.)

- [ ] **Step 2: Add other 7 `_UNCERTAINTY_TONE_<lang>` constants**

```python
_UNCERTAINTY_TONE_ES = frozenset({
    "incierto", "incertidumbre", "incertidumbres",
    "riesgo", "riesgos",
    "riesgos a la baja", "sesgo a la baja",
    "débil", "debilidad", "debilitamiento",
    "blando", "moderación",
    "caída", "caer", "cayendo",
    "deterioro", "deteriorarse",
    "desaceleración", "ralentización",
    "frágil", "fragilidad",
    "preocupación", "preocupaciones", "preocupante",
    "volátil", "volatilidad",
    "vientos en contra",
    "descenso", "bajada",
    "pérdida", "pérdidas",
    "endurecimiento", "tensionamiento",
    "reducido",
})

_UNCERTAINTY_TONE_PT = frozenset({
    "incerto", "incerteza", "incertezas",
    "risco", "riscos",
    "riscos negativos", "viés de baixa",
    "fraco", "fraqueza", "enfraquecimento",
    "moderação",
    "queda", "cair", "caindo",
    "deterioração", "deteriorando",
    "desaceleração",
    "frágil", "fragilidade",
    "preocupação", "preocupações", "preocupante",
    "volátil", "volatilidade",
    "ventos contrários",
    "baixa",
    "perda", "perdas",
    "aperto",
    "subdimensionado",
})

_UNCERTAINTY_TONE_DE = frozenset({
    "unsicher", "unsicherheit", "unsicherheiten",
    "risiko", "risiken",
    "abwärtsrisiko", "abwärtsrisiken",
    "schwach", "schwäche", "abschwächung",
    "weich",
    "rückgang", "rückläufig", "fallen",
    "verschlechterung", "verschlechtern",
    "verlangsamung",
    "fragil", "fragilität",
    "sorge", "sorgen", "besorgnis", "besorgniserregend",
    "volatil", "volatilität",
    "gegenwind",
    "sinken", "sinkend",
    "verlust", "verluste",
    "straffung",
    "gedämpft",
})

_UNCERTAINTY_TONE_FR = frozenset({
    "incertain", "incertitude", "incertitudes",
    "risque", "risques",
    "risques baissiers", "risques à la baisse",
    "faible", "faiblesse", "affaiblissement",
    "ralentissement",
    "baisse", "baisser",
    "détérioration", "se détériorer",
    "fragile", "fragilité",
    "préoccupation", "préoccupations", "préoccupant",
    "volatile", "volatilité",
    "vents contraires",
    "chute", "chuter",
    "perte", "pertes",
    "resserrement",
    "modéré",
})

_UNCERTAINTY_TONE_IT = frozenset({
    "incerto", "incertezza", "incertezze",
    "rischio", "rischi",
    "rischi al ribasso", "rischi negativi",
    "debole", "debolezza", "indebolimento",
    "moderazione",
    "calo", "calare",
    "deterioramento", "peggioramento",
    "rallentamento",
    "fragile", "fragilità",
    "preoccupazione", "preoccupazioni", "preoccupante",
    "volatile", "volatilità",
    "venti contrari",
    "caduta",
    "perdita", "perdite",
    "irrigidimento", "stretta",
    "moderato",
})

_UNCERTAINTY_TONE_JA = frozenset({
    "不確実", "不確実性",
    "リスク", "下振れリスク",
    "弱い", "弱含み", "軟調",
    "減少", "低下",
    "悪化",
    "鈍化", "減速",
    "脆弱",
    "懸念", "憂慮",
    "ボラティリティ",
    "下落",
    "損失",
    "引き締め",
    "低調",
})

_UNCERTAINTY_TONE_ZH = frozenset({
    "不确定", "不确定性",
    "风险", "下行风险",
    "疲弱", "疲软", "疲态",
    "减弱", "走弱",
    "恶化",
    "放缓", "减速",
    "脆弱",
    "担忧", "忧虑",
    "波动", "波动性",
    "下跌",
    "损失",
    "收紧",
    "低迷",
})
```

- [ ] **Step 3: Add coverage tests**

In `puremacro/tests/test_narrative_indices.py`, append:

```python
@pytest.mark.parametrize("lang,min_count", [
    ("en", 35),  # EN frozenset has ~50 - duplicates
    ("es", 25), ("pt", 25), ("de", 25), ("fr", 25), ("it", 25),
    ("ja", 15), ("zh", 15),
])
def test_uncertainty_tone_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import (
        _UNCERTAINTY_TONE_EN, _UNCERTAINTY_TONE_ES, _UNCERTAINTY_TONE_PT,
        _UNCERTAINTY_TONE_DE, _UNCERTAINTY_TONE_FR, _UNCERTAINTY_TONE_IT,
        _UNCERTAINTY_TONE_JA, _UNCERTAINTY_TONE_ZH,
    )
    name_to_lex = {
        "en": _UNCERTAINTY_TONE_EN, "es": _UNCERTAINTY_TONE_ES,
        "pt": _UNCERTAINTY_TONE_PT, "de": _UNCERTAINTY_TONE_DE,
        "fr": _UNCERTAINTY_TONE_FR, "it": _UNCERTAINTY_TONE_IT,
        "ja": _UNCERTAINTY_TONE_JA, "zh": _UNCERTAINTY_TONE_ZH,
    }
    terms = name_to_lex[lang]
    assert len(terms) >= min_count, (
        f"_UNCERTAINTY_TONE_{lang.upper()} has {len(terms)} terms; "
        f"need ≥ {min_count}"
    )
```

- [ ] **Step 4: Verify counts**

```bash
cd puremacro && python3 -c "
from puremacro.narrative.indices._lexicons import (
    _UNCERTAINTY_TONE_EN, _UNCERTAINTY_TONE_ES, _UNCERTAINTY_TONE_PT,
    _UNCERTAINTY_TONE_DE, _UNCERTAINTY_TONE_FR, _UNCERTAINTY_TONE_IT,
    _UNCERTAINTY_TONE_JA, _UNCERTAINTY_TONE_ZH,
)
for n, lex in [('EN', _UNCERTAINTY_TONE_EN), ('ES', _UNCERTAINTY_TONE_ES),
               ('PT', _UNCERTAINTY_TONE_PT), ('DE', _UNCERTAINTY_TONE_DE),
               ('FR', _UNCERTAINTY_TONE_FR), ('IT', _UNCERTAINTY_TONE_IT),
               ('JA', _UNCERTAINTY_TONE_JA), ('ZH', _UNCERTAINTY_TONE_ZH)]:
    print(n, len(lex))
"
```
Expected per-lang: EN ≥ 35; ES/PT/DE/FR/IT ≥ 25; JA/ZH ≥ 15. Augment if short.

- [ ] **Step 5: Run tests**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "uncertainty_tone_lexicon" 2>&1 | tail -10
```
Expected: 8 pass.

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1001 + 8 = 1009 passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative): _UNCERTAINTY_TONE_<lang> lexicons across 8 languages (Slice 6a)"
```

---

## Task 4: Restructure `LEXICONS["lui"][lang]` + rewire `lui.py`

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py`
- Modify: `puremacro/narrative/indices/lui.py`
- Create: `puremacro/tests/test_narrative_lui_cooccurrence.py`
- Modify: `puremacro/tests/test_narrative_indices.py`

This is the API-breaking step. After this, `LEXICONS["lui"]["en"]` is a dict, not a frozenset.

- [ ] **Step 1: Rename `_LUI_<lang>` → `_LUI_PHRASES_<lang>` in `_lexicons.py`**

In `puremacro/narrative/indices/_lexicons.py`, do a global rename of the 8 constants:
- `_LUI_EN` → `_LUI_PHRASES_EN`
- `_LUI_ES` → `_LUI_PHRASES_ES`
- (and so on for PT, DE, FR, IT, JA, ZH)

Content is unchanged — just the variable names. Also rename their docstring/comment headers from "Labor-Market Uncertainty (English)" to "LUI curated phrases (English)".

- [ ] **Step 2: Restructure `LEXICONS["lui"]`**

Find the `LEXICONS` dict at end of file. Replace the `"lui"` entry:

```python
    "lui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN,
            "phrases": _LUI_PHRASES_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES,
            "phrases": _LUI_PHRASES_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT,
            "phrases": _LUI_PHRASES_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE,
            "phrases": _LUI_PHRASES_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR,
            "phrases": _LUI_PHRASES_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT,
            "phrases": _LUI_PHRASES_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA,
            "phrases": _LUI_PHRASES_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH,
            "phrases": _LUI_PHRASES_ZH,
        },
    },
```

Update the module docstring (first triple-quoted block) to reflect that `lui` is now a dict-of-frozensets:
```python
"""...
The EPU and LUI lexicons use a nested dict (multi-group co-occurrence
methodology); MPU/GPR/WUI/tone use a flat frozenset (or hawkish/dovish
dict for tone)...
"""
```

- [ ] **Step 3: Rewrite `puremacro/narrative/indices/lui.py`**

Replace the file content with:

```python
"""Labor-Market Uncertainty Index — sentence-level co-occurrence
methodology (Slice 6a).

A document's LUI score is the fraction of sentences that contain
either:
  (i) ≥1 labor-domain term AND ≥1 uncertainty-tone term, OR
  (ii) ≥1 high-precision pre-formed phrase (e.g. "rising unemployment").

This generalizes BBD-EPU's document-level co-occurrence to long
multi-topic documents (FOMC minutes, ECB bulletins) where document-
level co-occurrence is too coarse.

Score is in [0, 1]: "fraction of doc that's labor-uncertainty-flavored".
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import sentence_cooccurrence_kernel


def lui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a labor-market uncertainty series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"labor_domain": frozenset, "uncertainty_tone": frozenset,
           "phrases": frozenset | None}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : ``(start_iso, end_iso)`` for normalisation statistics.
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"``.
    """
    lex = lexicon if lexicon is not None else LEXICONS["lui"][language]
    term_groups = [lex["labor_domain"], lex["uncertainty_tone"]]
    phrases = lex.get("phrases")

    def _kernel(records):
        return sentence_cooccurrence_kernel(
            records,
            term_groups=term_groups,
            phrases=phrases,
            language=language,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"lui_{country.lower()}",
        method="sentence_cooccurrence", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "lui", "base_period": base_period},
    )


__all__ = ["lui"]
```

- [ ] **Step 4: Update LUI lexicon-shape tests in `test_narrative_indices.py`**

Find the existing `test_other_indices_have_each_supported_language` parametrize. The current assertion `assert len(terms) >= 1` will FAIL on the new dict shape — `len(dict)` returns 3 (number of keys), which still ≥ 1, so it still passes coincidentally. But the spec spirit (each lexicon "is" a list of terms) is broken. Fix by handling LUI specially:

Replace that test's body with:

```python
def test_other_indices_have_each_supported_language(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    for index in ("mpu", "gpr", "wui", "lui"):
        assert lang in LEXICONS[index], f"{index} missing language {lang}"
        if index == "lui":
            # Slice 6a: LUI is a dict of three frozensets per language.
            sub = LEXICONS["lui"][lang]
            assert "labor_domain" in sub and len(sub["labor_domain"]) >= 1
            assert "uncertainty_tone" in sub and len(sub["uncertainty_tone"]) >= 1
            assert "phrases" in sub and len(sub["phrases"]) >= 1
        else:
            terms = LEXICONS[index][lang]
            assert len(terms) >= 1, f"{index}/{lang} is empty"
```

Also: the existing `test_lui_english_has_substantive_coverage`, `test_lui_latin_lexicon_substantive_coverage`, and `test_lui_cjk_lexicon_substantive_coverage` (Slice 5) still reference `LEXICONS["lui"][lang]` as a flat frozenset. Update them to use the new dict shape:

```python
def test_lui_english_has_substantive_coverage():
    """LUI/EN phrases lexicon expanded in Slice 5: ≥ 100 terms across 6 conceptual groups."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    lui_en_phrases = LEXICONS["lui"]["en"]["phrases"]
    assert len(lui_en_phrases) >= 100
    assert any("layoff" in t for t in lui_en_phrases)
    assert any("hiring freeze" in t for t in lui_en_phrases)
    assert any("wage" in t for t in lui_en_phrases)
    assert any("shortage" in t for t in lui_en_phrases)
    assert any("participation" in t for t in lui_en_phrases)
    assert any("unemployment" in t for t in lui_en_phrases)


@pytest.mark.parametrize("lang,min_count", [
    ("es", 100), ("pt", 100), ("de", 100), ("fr", 100), ("it", 100),
])
def test_lui_latin_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]["phrases"]
    assert len(terms) >= min_count


@pytest.mark.parametrize("lang,min_count", [
    ("ja", 60), ("zh", 60),
])
def test_lui_cjk_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]["phrases"]
    assert len(terms) >= min_count
```

- [ ] **Step 5: New focused integration test**

Create `puremacro/tests/test_narrative_lui_cooccurrence.py`:

```python
"""Slice 6a: focused integration tests on the new LUI methodology."""
from __future__ import annotations
import pandas as pd


def test_lui_basic_sentence_cooccurrence_signal():
    """Doc with one labor-uncertainty sentence + one neutral sentence
    should score ~0.5."""
    from puremacro.narrative.indices import lui
    text = "Employment risk has risen materially. The economy expanded."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    # Quarterly aggregation of one record gets that record's score.
    val = float(series.series.dropna().iloc[0])
    assert 0.4 < val < 0.6   # ~0.5


def test_lui_phrase_shortcut():
    """A doc whose only matching sentence has a phrase but no separate
    labor + uncertainty co-occurrence still gets credit."""
    from puremacro.narrative.indices import lui
    text = "Rising unemployment is a concern."  # phrase only
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 1.0


def test_lui_pure_labor_discussion_scores_low():
    """A doc that talks about labor positively (no uncertainty markers)
    should score 0."""
    from puremacro.narrative.indices import lui
    text = "Employment grew strongly. Wages rose by 4 percent. Hiring continued."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 0.0


def test_lui_lexicon_override_accepts_dict():
    """The lexicon kwarg accepts the dict-of-frozensets shape."""
    from puremacro.narrative.indices import lui
    custom = {
        "labor_domain": frozenset({"workers"}),
        "uncertainty_tone": frozenset({"struggle"}),
        "phrases": frozenset({"job pain"}),
    }
    text = "Workers struggle right now. Job pain is widespread."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en",
                 lexicon=custom, normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 1.0  # both sentences match
```

- [ ] **Step 6: Run tests**

```bash
cd puremacro && pytest tests/test_narrative_lui_cooccurrence.py tests/test_narrative_indices.py -v --no-header -k "lui or labor_domain or uncertainty_tone" 2>&1 | tail -25
```
Expected: All targeted tests pass (4 new integration + extended Slice 5 tests reshaped to dict).

- [ ] **Step 7: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: ≥ 1009 + 4 = 1013 passed (4 new integration tests added, no regressions).

If existing fiscal-narrative tests break (because they reference `LEXICONS["lui"][lang]` as a flat frozenset), fix them — they should not call LEXICONS["lui"] at all (LUI is the active variable, not those tests). If breakage is in fiscal narrative tests, that's a hidden coupling — investigate and patch in same commit.

- [ ] **Step 8: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/_lexicons.py \
        puremacro/puremacro/narrative/indices/lui.py \
        puremacro/tests/test_narrative_lui_cooccurrence.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative)!: LUI sentence-level co-occurrence (BREAKING change to LEXICONS shape)"
```

(`!` in the commit subject signals breaking change per Conventional Commits.)

---

## Task 5: Length-normalized WUI

**Files:**
- Modify: `puremacro/narrative/indices/wui.py`
- Modify: `puremacro/tests/test_narrative_indices.py`

- [ ] **Step 1: Update `wui.py`**

Replace the file content:

```python
"""Ahir-Bloom-Furceri World Uncertainty Index.

Counts uncertainty-term mentions per document and normalises by
document length: ``score = (hits / total_words) * 1000``.

Reference
---------
Ahir, H., Bloom, N., Furceri, D. (2022). The World Uncertainty Index.
NBER WP 29763.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def wui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
) -> RiskIndex:
    """Build a length-normalised WUI series from a custom corpus.

    Score is hits per 1000 words per Ahir-Bloom-Furceri.
    """
    terms = lexicon if lexicon is not None else LEXICONS["wui"][language]

    def _kernel(records):
        return keyword_count_kernel(
            records, terms=terms, language=language, length_normalize=True,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"wui_{country.lower()}",
        method="length_normalized_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "wui", "base_period": base_period},
    )


__all__ = ["wui"]
```

- [ ] **Step 2: Add a test that confirms WUI is length-normalized**

In `puremacro/tests/test_narrative_indices.py`, append:

```python
def test_wui_is_length_normalized():
    """Slice 6a: WUI now returns hits per 1000 words, not raw counts."""
    import pandas as pd
    from puremacro.narrative.indices import wui
    text_short = "uncertainty " * 5 + "the " * 95     # 100 words, 5 hits
    text_long = "uncertainty " * 5 + "the " * 195    # 200 words, 5 hits
    records_s = [(pd.Timestamp("2024-01-01"), text_short, "u", {})]
    records_l = [(pd.Timestamp("2024-04-01"), text_long, "u", {})]
    s_short = wui(records_s, country="USA", language="en", normalize="raw")
    s_long = wui(records_l, country="USA", language="en", normalize="raw")
    val_short = float(s_short.series.dropna().iloc[0])
    val_long = float(s_long.series.dropna().iloc[0])
    # Length-norm: text_short has higher density → higher score.
    assert val_short > val_long
    # Specifically: 50 vs 25 (per 1000 words).
    assert abs(val_short - 50.0) < 5.0
    assert abs(val_long - 25.0) < 5.0
```

- [ ] **Step 3: Run targeted**

```bash
cd puremacro && pytest tests/test_narrative_indices.py -v --no-header -k "wui_is_length_normalized" 2>&1 | tail -10
```
Expected: 1 test passes.

- [ ] **Step 4: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1013 + 1 = 1014 passed. Note: existing WUI tests that asserted RAW hit counts may now FAIL — they need updating to the new length-normalized scale. If any fail, update them in this same commit (fix the assertion's expected value, not the API).

- [ ] **Step 5: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/wui.py \
        puremacro/tests/test_narrative_indices.py
git commit -m "feat(narrative)!: WUI length-normalized per Ahir-Bloom-Furceri"
```

---

## Task 6: Hubert-inspired vocabulary expansion (WUI)

**Files:**
- Modify: `puremacro/narrative/indices/_lexicons.py`

- [ ] **Step 1: Identify current `_WUI_<lang>` blocks**

Read `puremacro/narrative/indices/_lexicons.py` and locate each `_WUI_<lang>` constant (8 of them). Note current term count per language.

- [ ] **Step 2: Add Hubert-inspired terms to `_WUI_EN`**

In `puremacro/narrative/indices/_lexicons.py`, locate `_WUI_EN = frozenset({...})` and add the following terms to the existing set (keep old terms in place):

```python
# Slice 6a additions — Hubert-inspired economic-uncertainty vocabulary.
# Drawn from open peer-reviewed sources on inflation expectations
# uncertainty (Hubert 2017; Coibion-Gorodnichenko 2015).
"inflation expectations", "anchored expectations", "unanchored expectations",
"deviation from target", "off target",
"second-round effects", "second round effects",
"wage-price spiral", "price spiral",
"price stability concerns",
"policy uncertainty", "policy unpredictability",
"macroeconomic uncertainty",
"downside risks", "upside risks",
"tail risk", "tail risks", "fat tails",
"stagflation", "stagflationary",
"headwinds", "crosscurrents",
"fragile recovery",
"data-dependent",
"uncertain outlook", "uncertain path",
"asymmetric risks",
```

(~28 new EN terms.)

- [ ] **Step 3: Add proportional Hubert terms to other 7 languages**

Add ~10-15 terms to ES, PT, DE, FR, IT (per language); ~6-8 to JA and ZH. Implementer should pick the most common renderings of the same concepts:

```python
# In _WUI_ES additions:
"expectativas de inflación", "expectativas ancladas", "expectativas desancladas",
"desviación del objetivo",
"efectos de segunda ronda",
"espiral salarios-precios",
"riesgos a la baja", "riesgos al alza",
"riesgos asimétricos",
"recuperación frágil",
"perspectivas inciertas",
"vientos en contra",

# In _WUI_PT additions:
"expectativas de inflação", "expectativas ancoradas", "expectativas desancoradas",
"desvio da meta",
"efeitos de segunda rodada",
"espiral salários-preços",
"riscos negativos", "riscos positivos",
"riscos assimétricos",
"recuperação frágil",
"perspectivas incertas",
"ventos contrários",

# In _WUI_DE additions:
"inflationserwartungen", "verankerte erwartungen", "entankerte erwartungen",
"abweichung vom ziel",
"zweitrundeneffekte",
"lohn-preis-spirale",
"abwärtsrisiken", "aufwärtsrisiken",
"asymmetrische risiken",
"fragile erholung",
"unsicherer ausblick",
"gegenwind",

# In _WUI_FR additions:
"anticipations d'inflation", "anticipations ancrées", "anticipations désancrées",
"écart à la cible",
"effets de second tour",
"spirale salaires-prix",
"risques baissiers", "risques haussiers",
"risques asymétriques",
"reprise fragile",
"perspectives incertaines",
"vents contraires",

# In _WUI_IT additions:
"aspettative di inflazione", "aspettative ancorate", "aspettative disancorate",
"scostamento dall'obiettivo",
"effetti di secondo round",
"spirale salari-prezzi",
"rischi al ribasso", "rischi al rialzo",
"rischi asimmetrici",
"ripresa fragile",
"prospettive incerte",
"venti contrari",

# In _WUI_JA additions:
"インフレ期待",
"アンカーされた期待", "アンカー外れ",
"目標からの乖離",
"第二次効果", "二次的効果",
"賃金物価スパイラル",
"下振れリスク", "上振れリスク",
"非対称的リスク",
"脆弱な回復",

# In _WUI_ZH additions:
"通胀预期",
"锚定预期", "脱锚预期",
"偏离目标",
"二轮效应", "次轮效应",
"工资价格螺旋",
"下行风险", "上行风险",
"不对称风险",
"复苏脆弱",
```

- [ ] **Step 4: Verify counts**

```bash
cd puremacro && python3 -c "
from puremacro.narrative.indices._lexicons import LEXICONS
for lang in ('en','es','pt','de','fr','it','ja','zh'):
    print(lang, len(LEXICONS['wui'][lang]))
"
```
Expected: substantial increases vs Slice 5 baseline.

- [ ] **Step 5: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1014 passed (no count changes; Hubert terms are additive).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/narrative/indices/_lexicons.py
git commit -m "feat(narrative): Hubert-inspired economic-uncertainty vocabulary added to WUI (8 langs)"
```

---

## Task 7: Re-run notebook 28 + validate signal

**Files:**
- Notebook 28 outputs

- [ ] **Step 1: Verify branch + commits**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git log --oneline 9f58145..HEAD
```
Expected: 6 new commits (T1 kernel, T2 labor_domain, T3 uncertainty_tone, T4 LUI rewire, T5 WUI length-norm, T6 Hubert).

- [ ] **Step 2: Clear corpus + re-run notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
rm -f notebooks/data_cache/fed_corpus_28.parquet
PUREMACRO_REFETCH=1 jupyter execute notebooks/28_us_lui_from_fed_text.ipynb \
    --output 28_us_lui_from_fed_text.executed.ipynb
```
Network-bound; 5-15 min. Use `run_in_background: true` and BashOutput to poll.

- [ ] **Step 3: Inspect validation outputs**

```bash
cat notebooks/output_tables/28_lui_us_quarterly.meta.json
cat notebooks/output_tables/28_lui_validation_corr.csv
```

**Acceptance criterion:**
- LUI vs urate ρ ≥ +0.30 (Slice 5 baseline: +0.18). PASS → proceed to T8.
- 0.22 ≤ ρ < 0.30: PARTIAL — ship with caveats (document in CHANGELOG). Proceed.
- ρ < 0.18: REGRESSION — STOP. Investigate (likely lexicon-polarity bug or sentence-splitter under-coverage).

EPU/WUI sanity check (no significant regression):
- EPU vs BBD-EPU ρ should stay ~+0.32.
- WUI is now length-normalized so absolute values change; ρ vs benchmarks should not regress meaningfully.

- [ ] **Step 4: If signal lifted, proceed to T8. If signal flat or regressed, debug.**

If signal flat: likely the labor_domain lexicon is too narrow (under-firing) or the uncertainty_tone is too narrow. Try:
- Print 5 random Fed minutes records' sentence-level scores to inspect what's matching and what isn't.
- Check token-level for boundary issues (multi-word terms missing word boundaries).

Fix in a follow-up commit before proceeding to T8.

---

## Task 8: 0.8.0 release

**Files:**
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `puremacro/tests/test_import.py`, `puremacro/CHANGELOG.md`

- [ ] **Step 1: Bump versions to 0.8.0**

In `puremacro/pyproject.toml`: `version = "0.7.2"` → `"0.8.0"`.
In `puremacro/puremacro/__init__.py`: `__version__ = "0.7.2"` → `"0.8.0"`.
In `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.7.2"` → `"0.8.0"`.

- [ ] **Step 2: Add CHANGELOG entry**

Open `puremacro/CHANGELOG.md`. Insert above the `## 0.7.2 — 2026-05-09` block. Use **actual ρ values** from T7 inspection — replace placeholders below:

```markdown
## 0.8.0 — 2026-05-09

Slice 6a: signal-quality fix for LUI. The Slice 5 lexicon expansion (35 → 145 EN terms) didn't lift LUI vs urate (ρ stayed at +0.18) because raw term-frequency conflates labor-market discussion with labor-market uncertainty. Slice 6a switches LUI to sentence-level co-occurrence — score = fraction of sentences containing both a labor-domain term AND an uncertainty/risk-tone term (or a curated phrase). LUI vs urate ρ = +<ACTUAL_VALUE>. WUI length-normalized per Ahir-Bloom-Furceri. Hubert-inspired vocabulary added to WUI.

### Breaking

- `LEXICONS["lui"][lang]` is now a `dict[str, frozenset]` with keys `labor_domain`, `uncertainty_tone`, `phrases` — was a flat `frozenset`. Callers that supply `lexicon=...` to `lui()` must use the new shape.
- `wui()` output is now hits per 1000 words instead of raw hit counts. Absolute scale changes; correlations with benchmarks unchanged in expectation.

### Added

- `narrative.indices._kernels._split_sentences(text, language)` — language-aware sentence splitter (Latin: `[.!?]+`; CJK: `[。！？]+`).
- `narrative.indices._kernels.sentence_cooccurrence_kernel` — score = matched sentences / total sentences.
- `keyword_count_kernel(..., length_normalize=True)` — Ahir-Bloom-Furceri WUI normalization.
- 8 `_LABOR_DOMAIN_<lang>` lexicons (~30 terms per Latin; ~25 per CJK).
- 8 `_UNCERTAINTY_TONE_<lang>` lexicons (~25 terms per Latin; ~15 per CJK).
- Hubert-inspired vocabulary in `_WUI_<lang>` (~28 EN; proportional ports).
- Tests: `test_narrative_kernels.py` (14 tests), `test_narrative_lui_cooccurrence.py` (4 tests), `test_labor_domain_lexicon_substantive_coverage` (8 parametrized), `test_uncertainty_tone_lexicon_substantive_coverage` (8 parametrized).

### Changed

- `_LUI_<lang>` constants renamed `_LUI_PHRASES_<lang>` (8 renames; content unchanged from Slice 5). Now keyed under `LEXICONS["lui"][lang]["phrases"]`.
- `lui()` switched from `keyword_count_kernel` to `sentence_cooccurrence_kernel`.
- `wui()` now passes `length_normalize=True` to its kernel; docstring updated.

### Pyodide compatibility

- `_kernels.py` adds pure-Python `re` patterns (`_SENT_LATIN_RX`, `_SENT_CJK_RX`); no new top-level deps. Slice 6a contributes zero new pyodide leaks.

### Notes for next iteration

- LUI vs urate ρ = +<ACTUAL_VALUE> (Slice 5: +0.18). Notebook 29 (state-panel LP-IV with national LUI as shock) <BLOCKED|UNBLOCKED — based on result>.
- Slice 6b candidates: `llm_prob_kernel` (LLM-backed scoring with pyodide carve-out, async, caching), Picault-Renault paragraph-level multinomial logit, stricter sentence tokenizer (handle abbreviations).
```

- [ ] **Step 3: Final regression sweep**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1014 passed (matches T6 final count after `test_import.py` updated).

```bash
cd puremacro && pytest tests/test_narrative.py tests/test_narrative_replication_*.py tests/test_narrative_quality.py -q --no-header 2>&1 | tail -3
```
Expected: zero fiscal-narrative regressions.

```bash
cd puremacro && pytest tests/test_pyodide_compat.py -q --no-header 2>&1 | tail -3
```
Expected: same 1 pre-existing failure (statsmodels.tsa.x13 leak in `_seasonal.py`); no new leaks.

- [ ] **Step 4: Commit + tag**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git status -s notebooks/output_tables/ notebooks/data_cache/
git add notebooks/output_tables/28_lui_us_quarterly.parquet \
        notebooks/output_tables/28_lui_us_quarterly.meta.json \
        notebooks/output_tables/28_lui_validation_corr.csv \
        notebooks/data_cache/fed_corpus_28.parquet \
        puremacro/pyproject.toml \
        puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py \
        puremacro/CHANGELOG.md
git commit -m "chore(release): puremacro 0.8.0 — narrative Slice 6a (sentence co-occurrence LUI + length-norm WUI)"
git tag -a v0.8.0 -m "puremacro 0.8.0 — narrative Slice 6a (sentence co-occurrence LUI + length-norm WUI + Hubert vocab)"
```

(Do NOT push.)

---

## Definition of Done

- [ ] All 8 task blocks above checked off.
- [ ] Branch `feature/narrative-extension-slice3` has new commits past `v0.7.2`, tagged `v0.8.0`.
- [ ] `pytest -q` ≥ 1014 passed.
- [ ] `test_pyodide_compat.py` shows the same 1 pre-existing failure (no new leaks).
- [ ] Zero fiscal-narrative regressions.
- [ ] `pyproject.toml` version is `0.8.0`; `puremacro.__version__ == "0.8.0"`.
- [ ] `CHANGELOG.md` has a `## 0.8.0 — 2026-05-09` section with the actual ρ value from the re-run.
- [ ] LUI vs urate ρ improved from Slice 5's +0.18 toward acceptance criterion +0.30 (or partial-success documented honestly in CHANGELOG).
- [ ] Notebook 28 outputs updated.
- [ ] Notebook 29 status (BLOCKED/UNBLOCKED) honestly documented in CHANGELOG based on actual ρ.

## Out of scope (deferred to Slice 6b)

- LLM-backed scoring (`llm_prob_kernel`) — needs API choice, cost guardrails.
- Picault-Renault paragraph-level multinomial logit.
- Stricter sentence tokenizer.
- Notebook 29 (state-panel LP-IV).
- Per-bank precise extractors for Slice-3 banks.
- BIS speeches (still JS-rendered).
