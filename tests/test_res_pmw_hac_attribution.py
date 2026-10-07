"""RES-PMW: the LP HAC truncation lag h+1 is a convention, not Plagborg-Moller & Wolf (2021).

``lp_hac`` (and ``teaching.lp_sm.lp_ols_hac``) use Newey-West standard errors
with a Bartlett kernel and a fixed truncation lag ``h + 1`` at horizon ``h``.
Plagborg-Moller and Wolf (2021, Econometrica 89(2)) prove that LPs and VARs
estimate the same impulse responses; the paper makes no HAC or bandwidth
recommendation. The notebooks and the technical report used to credit the
lag to that paper, and the technical report also said the bandwidth was
selected automatically and that lag-augmented LPs rest on the PMW theorem
(they are Montiel Olea and Plagborg-Moller 2021, Econometrica 89(4)).

These tests pin the corrected prose in the files that carried the claim and
check the prose against the code: the standard error ``lp_hac`` reports is
the fixed-lag ``h + 1`` Bartlett estimator, whatever the data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_STEMS = (
    "notebooks/07_local_projections",
    "notebooks/07_local_projections_es",
    "notebooks/course/06_lp_narrativa_es",
    "notebooks/course/10_b1_capital_ajuste_es",
)
TECHREPORT = ROOT / "techreport" / "puremacro_technical_report.md"

# A sentence credits the lag to PMW when it names the HAC lag h+1 and PMW
# and says nothing that negates or qualifies the credit.
_H_PLUS_1 = re.compile(r"(?<![\w\\])h\s*\+\s*1(?!\d)")
_HAC_WORD = re.compile(
    r"HAC|Newey|bandwidth|ancho de banda|truncation|truncamiento|Bartlett", re.I)
_PMW = re.compile(r"Wolf|PMW")
_NEGATION = re.compile(
    r"\b(?:not|no|nor|ningún|ninguna)\b|rule[ -]of[ -]thumb|regla práctica|convention",
    re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def _py_text(path: Path) -> str:
    """Percent-format source with comment markers stripped (markdown and comments)."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# %%"):
            lines.append("")                         # cell boundary = paragraph break
        elif stripped.startswith("#"):
            lines.append(stripped[1:].strip())
        else:
            lines.append(line)
    return "\n".join(lines)


def _ipynb_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join("".join(c["source"]) for c in nb["cells"])


def _sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        flat = " ".join(para.split())
        out.extend(s for s in _SENTENCE_END.split(flat) if s)
    return out


def _pmw_hac_credits(text: str) -> list[str]:
    return [s for s in _sentences(text)
            if _H_PLUS_1.search(s) and _HAC_WORD.search(s) and _PMW.search(s)
            and not _NEGATION.search(s)]


def _owned_texts():
    for stem in NOTEBOOK_STEMS:
        yield f"{stem}.py", _py_text(ROOT / f"{stem}.py")
        yield f"{stem}.ipynb", _ipynb_text(ROOT / f"{stem}.ipynb")
    yield "techreport/puremacro_technical_report.md", TECHREPORT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The detector itself: it flags each phrasing the files used to carry.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old", [
    # notebooks/07_local_projections.py (EN, before)
    "HAC (Newey–West) standard errors with a\nBartlett kernel and bandwidth $h+1$ "
    "(Plagborg-Møller–Wolf 2021), and report bands\n$\\beta_h \\pm z$.",
    # notebooks/07_local_projections_es.py (before)
    "errores estándar HAC (Newey–West) con un núcleo de Bartlett y ancho de banda $h+1$\n"
    "(Plagborg-Møller–Wolf 2021), y reportamos bandas.",
    # notebooks/course/06_lp_narrativa_es.py:73 (before)
    "En `puremacro`: `lp.lp_hac` usa **ancho de banda HAC $= h+1$** (recomendación de PMW 2021).",
    # notebooks/course/10_b1_capital_ajuste_es.py:206 (before)
    "Estimamos con proyecciones locales de Jordà (2005), con errores estándar HAC\n"
    "(Newey–West, ancho de banda $h+1$ según Plagborg-Møller–Wolf 2021), la respuesta\n"
    "acumulada del (log) de la inversión ante un choque de $\\Delta\\log q$:",
    # puremacro/lp/jorda.py docstring (before)
    "HAC bandwidth = h + 1 (Plagborg-Møller-Wolf 2021 recommendation).",
])
def test_detector_flags_the_old_credit(old):
    assert _pmw_hac_credits(old), old


def test_detector_accepts_the_corrected_wording():
    ok = ("En `puremacro`: `lp.lp_hac` usa errores estándar HAC de Newey-West con **rezago de\n"
          "truncamiento $h+1$**. Es una regla práctica, no una recomendación de PMW (2021), cuyo\n"
          "artículo no propone ningún ancho de banda.")
    assert _pmw_hac_credits(ok) == []


# ---------------------------------------------------------------------------
# The owned files: no credit left, and the convention is stated instead.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [pytest.param(n, t, id=n) for n, t in _owned_texts()])
def test_no_pmw_credit_for_hac_lag(name, text):
    assert _pmw_hac_credits(text) == [], name


@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
@pytest.mark.parametrize("suffix", [".py", ".ipynb"])
def test_notebook_states_h_plus_1_as_a_rule_of_thumb(stem, suffix):
    path = ROOT / f"{stem}{suffix}"
    text = _py_text(path) if suffix == ".py" else _ipynb_text(path)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    stated = [p for p in paras
              if _H_PLUS_1.search(p) and re.search(r"truncation lag|rezago de truncamiento", p)
              and re.search(r"rule of thumb|regla práctica", p)]
    assert stated, f"{path.name}: the truncation lag h+1 is not described as a rule of thumb"


@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
def test_committed_notebook_matches_its_source(stem):
    """The .ipynb was rebuilt from the corrected .py (cell sources identical)."""
    jupytext = pytest.importorskip("jupytext")
    src = jupytext.read(ROOT / f"{stem}.py")
    nb = json.loads((ROOT / f"{stem}.ipynb").read_text(encoding="utf-8"))
    assert [(c.cell_type, c.source) for c in src.cells] == [
        (c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    errors = [o for c in nb["cells"] if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []


# ---------------------------------------------------------------------------
# Technical report, section 3.2
# ---------------------------------------------------------------------------

def _lp_inference_paragraph() -> str:
    text = TECHREPORT.read_text(encoding="utf-8")
    paras = [p for p in text.split("\n") if "(LP-HAC)" in p]
    assert len(paras) == 1
    return paras[0]


def test_techreport_describes_the_fixed_lag_not_automatic_selection():
    para = _lp_inference_paragraph()
    assert not re.search(r"automatic\w*\s+bandwidth", para, re.I)
    assert "fixed truncation lag of $h+1$" in para
    assert "no data-driven bandwidth selection" in para
    assert "rule-of-thumb convention" in para
    assert "$1 - \\ell/(h+2)$" in para


def test_techreport_credits_la_lp_to_montiel_olea_plagborg_moller():
    para = _lp_inference_paragraph()
    for s in _sentences(para):
        if re.search(r"LA-LP|lag-augmented", s, re.I):
            assert "Wolf" not in s, s
            assert "Montiel Olea and Plagborg-Møller (2021)" in s, s
    text = TECHREPORT.read_text(encoding="utf-8")
    refs = text[text.index("### References"):]
    assert ("- Montiel Olea, J. L., & Plagborg-Møller, M. (2021). Local Projection Inference "
            "Is Simpler and More Robust Than You Think. *Econometrica*, 89(4), 1789–1823.") in refs
    # PMW stays cited, for what the paper proves.
    assert "Plagborg-Møller, M., & Wolf, C. K. (2021)" in refs
    assert re.search(r"Plagborg-Møller and Wolf \(2021\)[^.]*estimate the same impulse responses",
                     para)


# ---------------------------------------------------------------------------
# The prose matches the code: fixed lag h+1, Bartlett weights 1 - l/(h+2),
# no small-sample correction, whatever the persistence of the data.
# ---------------------------------------------------------------------------

def _newey_west_se(h: int, df: pd.DataFrame, n_lags: int, L: int) -> float:
    """Independent Newey-West SE of the x_t slope in lp_hac's regression."""
    y, x = df["y"], df["x"]
    cols = {"dy": y.shift(-h) - y.shift(1), "const": 1.0, "x": x}
    for lag in range(1, n_lags + 1):
        cols[f"x{lag}"] = x.shift(lag)
        cols[f"y{lag}"] = y.shift(lag)
    d = pd.DataFrame(cols).dropna()
    X = d.drop(columns="dy").to_numpy(float)
    yy = d["dy"].to_numpy(float)
    bread = np.linalg.inv(X.T @ X)
    u = yy - X @ (bread @ X.T @ yy)
    g = X * u[:, None]
    S = g.T @ g
    for ell in range(1, L + 1):
        G = g[ell:].T @ g[:-ell]
        S += (1.0 - ell / (L + 1.0)) * (G + G.T)
    return float(np.sqrt((bread @ S @ bread)[1, 1]))


@pytest.mark.parametrize("phi", [0.0, 0.95])
def test_lp_hac_uses_fixed_truncation_lag_h_plus_1(phi):
    from puremacro.lp import lp_hac

    rng = np.random.default_rng(20260930)
    T = 300
    x = rng.standard_normal(T)
    e = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + 0.5 * x[t - 1] + 0.2 * x[t] + e[t]
    df = pd.DataFrame({"y": y, "x": x})
    res = lp_hac(df, y="y", x="x", horizons=range(0, 9), n_lags=2)
    for h, se in zip(res["h"], res["se"]):
        h = int(h)
        assert se == pytest.approx(_newey_west_se(h, df, 2, h + 1), rel=1e-10)
        # A different lag gives a different SE, so the check has power.
        assert abs(se - _newey_west_se(h, df, 2, h + 3)) > 1e-8 * se
