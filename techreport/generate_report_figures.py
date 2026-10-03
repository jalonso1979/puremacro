"""Draw the figures of the puremacro technical report in grayscale print style.

Run from the repository root:

    python techreport/generate_report_figures.py

Every figure is drawn at its printed size (6.5 in wide, the PDF text width) and
saved at 300 dpi without ``bbox_inches="tight"``, so a font set to 7.25 pt
prints at 7.25 pt. Text is wrapped to each card's inner width as measured by
the renderer, cards are sized to their content, and after drawing each figure
:func:`check_layout` raises if any text leaves its card or overlaps other text
or an arrow. The palette is the grayscale style of
:mod:`puremacro.plotting.bw_style`; nothing is distinguished by colour.

The statements on the cards describe release 4.5.0 and are sourced from the
technical report, CHANGELOG.md, docs/ADVISORY.md,
docs/STRUCTURAL_VALIDATION_STATUS.md and the code. Figure 7 is computed from
``puremacro.validation.scorecard()`` each time the script runs.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from puremacro.plotting.bw_style import apply_bw_style, bw_colors, bw_hatches  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Print geometry and type
# ---------------------------------------------------------------------------
WIDTH_IN = 6.5          # PDF text width
DPI = 300               # 6.5 in x 300 dpi = 1950 px
BODY_H = (4.0, 4.6)     # allowed height range of body figures, inches
ABSTRACT_H = 8.0        # graphical abstract, full-page portrait

TITLE_PT = 8.5
BODY_PT = 7.25
SUB_PT = 7.25
BAND_PT = 9.0
MIN_PT = 7.0

# Grayscale tokens (0 = black, 1 = white).
INK = "0.05"            # body text
INK_SOFT = "0.22"       # subtitles
ARROW = "0.25"
EDGE_CARD = "0.45"
EDGE_BAND = "0.30"
FILL_WHITE = "1.0"
FILL_LIGHT = "0.97"
FILL_BAND = "0.95"
FILL_HEAD = "0.94"

PAD = 0.075             # card inner padding, inches
INDENT = 0.10           # bullet hanging indent
GAP_TITLE = 0.035       # title -> subtitle
GAP_HEAD = 0.05         # header block -> first bullet
GAP_BULLET = 0.028      # between bullets
MAX_SPREAD = 0.10       # max extra space added to one gap when a card is stretched
MARGIN = 0.06           # figure margin
ROUND = 0.035           # corner radius


def _style() -> None:
    apply_bw_style(dpi=DPI, serif=True)
    matplotlib.rcParams.update({
        # Times-like STIX ships with matplotlib: compact, with Greek and math glyphs.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Exact paper size: apply_bw_style sets savefig.bbox="tight".
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "savefig.format": "png",
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------
class Canvas:
    """A figure whose single axes is measured in inches from the top-left corner."""

    def __init__(self, height: float):
        self.fig = plt.figure(figsize=(WIDTH_IN, height), dpi=DPI)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, WIDTH_IN)
        self.ax.set_ylim(height, 0)        # y grows downward
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.cards: list[tuple[tuple[float, float, float, float], list[Text]]] = []
        self.arrows: list[FancyArrowPatch] = []
        # Invisible text measures as a unit box, so the probe stays visible
        # and is emptied after every measurement.
        self._probe = self.ax.text(0, 0, "")

    def set_height(self, height: float) -> None:
        self.fig.set_size_inches(WIDTH_IN, height)
        self.ax.set_ylim(height, 0)
        self.renderer = self.fig.canvas.get_renderer()

    # -- measurement -------------------------------------------------------
    def width_in(self, s: str, **font) -> float:
        self._probe.update({"fontsize": BODY_PT, "fontweight": "normal",
                            "fontstyle": "normal", **font})
        self._probe.set_text(s)
        w = self._probe.get_window_extent(self.renderer).width / DPI
        self._probe.set_text("")
        return w

    def wrap(self, s: str, width: float, **font) -> str:
        """Greedy word wrap to ``width`` inches; $...$ math stays on one line."""
        tokens = re.findall(r"\$[^$]*\$\S*|\S+", s)
        lines, cur = [], ""
        for tok in tokens:
            cand = f"{cur} {tok}" if cur else tok
            if not cur or self.width_in(cand, **font) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = tok
        lines.append(cur)
        return "\n".join(lines)

    def text(self, s: str, width: float | None = None, **font) -> tuple[Text, float]:
        font = {"va": "top", "ha": "left", "color": INK, "linespacing": 1.18, **font}
        if width is not None:
            s = self.wrap(s, width, **{k: v for k, v in font.items()
                                       if k in ("fontsize", "fontweight", "fontstyle")})
        t = self.ax.text(0, 0, s, **font)
        h = t.get_window_extent(self.renderer).height / DPI
        return t, h

    # -- drawing -------------------------------------------------------------
    def box(self, x, y, w, h, fill, edge, lw, z=1, hatch=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={ROUND}",
                           facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
                           hatch=hatch)
        self.ax.add_patch(p)
        return p

    def arrow(self, x0, y0, x1, y1, lw=1.1):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                            color=ARROW, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5)
        self.ax.add_patch(a)
        self.arrows.append(a)
        return a

    def save(self, name: str) -> Path:
        check_layout(self)
        path = OUTPUT_DIR / name
        self.fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(self.fig)
        check_png(path, self.fig.get_size_inches()[1])
        print(f"wrote {path.relative_to(OUTPUT_DIR.parent.parent)}")
        return path


class Card:
    """A titled box of bullets. ``build`` wraps and measures, ``place`` positions."""

    def __init__(self, title, bullets=(), subtitle=None, *, fill=FILL_WHITE,
                 edge=EDGE_CARD, lw=0.6, title_pt=TITLE_PT, body_pt=BODY_PT,
                 plain=False):
        self.title, self.subtitle, self.bullets = title, subtitle, list(bullets)
        self.fill, self.edge, self.lw = fill, edge, lw
        self.title_pt, self.body_pt, self.plain = title_pt, body_pt, plain
        self.rect = None

    def build(self, cv: Canvas, w: float) -> float:
        inner = w - 2 * PAD
        self.cv, self.items = cv, []
        self.title_t, self.title_h = cv.text(self.title, inner, fontsize=self.title_pt,
                                             fontweight="bold")
        h = PAD + self.title_h
        self.sub_t = None
        if self.subtitle:
            self.sub_t, self.sub_h = cv.text(self.subtitle, inner, fontsize=SUB_PT,
                                             fontstyle="italic", color=INK_SOFT)
            h += GAP_TITLE + self.sub_h
        for b in self.bullets:
            if self.plain:
                body, bh = cv.text(b, inner, fontsize=self.body_pt)
                dot = None
            else:
                dot, _ = cv.text("•", fontsize=self.body_pt)
                body, bh = cv.text(b, inner - INDENT, fontsize=self.body_pt)
            self.items.append((dot, body, bh))
        if self.items:
            h += GAP_HEAD + sum(bh for *_, bh in self.items)
            h += GAP_BULLET * (len(self.items) - 1)
        self.natural = h + PAD
        return self.natural

    def place(self, x: float, y: float, w: float, h: float) -> None:
        cv = self.cv
        self.rect = (x, y, w, h)
        cv.box(x, y, w, h, self.fill, self.edge, self.lw)
        extra = max(0.0, h - self.natural)
        n_gaps = len(self.items) + 1 if self.items else 1
        spread = min(MAX_SPREAD, extra / n_gaps)
        yy = y + PAD
        self.title_t.set_position((x + PAD, yy))
        yy += self.title_h
        texts = [self.title_t]
        if self.sub_t is not None:
            yy += GAP_TITLE
            self.sub_t.set_position((x + PAD, yy))
            texts.append(self.sub_t)
            yy += self.sub_h
        yy += GAP_HEAD + spread
        for dot, body, bh in self.items:
            if dot is not None:
                dot.set_position((x + PAD, yy))
                body.set_position((x + PAD + INDENT, yy))
                texts += [dot, body]
            else:
                body.set_position((x + PAD, yy))
                texts.append(body)
            yy += bh + GAP_BULLET + spread
        cv.cards.append(((x, y, w, h), texts))


class Row:
    """Children side by side with equal heights. ``weights`` split the width."""

    def __init__(self, children, gap=0.12, weights=None):
        self.children, self.gap = children, gap
        self.weights = weights or [1] * len(children)
        self.rect = None

    def _widths(self, w):
        avail = w - self.gap * (len(self.children) - 1)
        tot = sum(self.weights)
        return [avail * k / tot for k in self.weights]

    def build(self, cv, w):
        self.natural = max(c.build(cv, cw) for c, cw in zip(self.children, self._widths(w)))
        return self.natural

    def place(self, x, y, w, h):
        self.rect = (x, y, w, h)
        for c, cw in zip(self.children, self._widths(w)):
            c.place(x, y, cw, h)
            x += cw + self.gap


class Column:
    """Children stacked vertically; spare height is shared equally."""

    def __init__(self, children, gap=0.12):
        self.children, self.gap = children, gap
        self.rect = None

    def build(self, cv, w):
        self.nat = [c.build(cv, w) for c in self.children]
        self.natural = sum(self.nat) + self.gap * (len(self.children) - 1)
        return self.natural

    def place(self, x, y, w, h):
        self.rect = (x, y, w, h)
        extra = max(0.0, h - self.natural) / len(self.children)
        for c, nh in zip(self.children, self.nat):
            c.place(x, y, w, nh + extra)
            y += nh + extra + self.gap


class Band:
    """A labelled container (light fill, darker border) around one child."""

    def __init__(self, title, child, *, fill=FILL_BAND, edge=EDGE_BAND, lw=0.8, pad=0.08):
        self.title, self.child = title, child
        self.fill, self.edge, self.lw, self.pad = fill, edge, lw, pad
        self.rect = None

    def build(self, cv, w):
        self.cv = cv
        self.title_t, self.title_h = cv.text(self.title, w - 2 * self.pad,
                                             fontsize=BAND_PT, fontweight="bold")
        self.natural = (2 * self.pad + self.title_h + 0.06
                        + self.child.build(cv, w - 2 * self.pad))
        return self.natural

    def place(self, x, y, w, h):
        self.rect = (x, y, w, h)
        self.cv.box(x, y, w, h, self.fill, self.edge, self.lw, z=0.5)
        self.title_t.set_position((x + self.pad, y + self.pad))
        self.cv.cards.append(((x, y, w, h), [self.title_t]))
        top = y + self.pad + self.title_h + 0.06
        self.child.place(x + self.pad, top, w - 2 * self.pad, y + h - self.pad - top)


def layout(root, height_range, name_for_errors) -> Canvas:
    """Build ``root`` on a canvas of the right height and place it."""
    lo, hi = height_range
    cv = Canvas(hi)
    natural = root.build(cv, WIDTH_IN - 2 * MARGIN)
    need = natural + 2 * MARGIN
    if need > hi + 1e-9:
        raise RuntimeError(f"{name_for_errors}: content needs {need:.2f} in, more than {hi} in")
    height = math.ceil(max(lo, need) * DPI - 1e-6) / DPI    # whole pixels
    cv.set_height(height)
    root.place(MARGIN, MARGIN, WIDTH_IN - 2 * MARGIN, height - 2 * MARGIN)
    return cv


# ---------------------------------------------------------------------------
# Automatic checks
# ---------------------------------------------------------------------------
def _shrink(bb, px=0.6):
    return bb.from_extents(bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px)


def check_layout(cv: Canvas) -> None:
    """Raise if text leaves its card, overlaps text or an arrow, or is too small."""
    cv.fig.canvas.draw()            # legends and ticks are positioned at draw time
    fig, r = cv.fig, cv.fig.canvas.get_renderer()
    fig_bb = fig.bbox
    owner = {}
    for rect, texts in cv.cards:
        x, y, w, h = rect
        (x0, y0), (x1, y1) = cv.ax.transData.transform([(x, y + h), (x + w, y)])
        for t in texts:
            owner[id(t)] = (x0, y0, x1, y1)
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    boxes = []
    problems = []
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-9:
            problems.append(f"font {t.get_fontsize()} pt < {MIN_PT}: {t.get_text()[:40]!r}")
        bb = t.get_window_extent(r)
        boxes.append((t, bb))
        if (bb.x0 < fig_bb.x0 - 0.5 or bb.y0 < fig_bb.y0 - 0.5
                or bb.x1 > fig_bb.x1 + 0.5 or bb.y1 > fig_bb.y1 + 0.5):
            problems.append(f"outside figure: {t.get_text()[:40]!r}")
        card = owner.get(id(t))
        if card is not None:
            x0, y0, x1, y1 = card
            inset = 1.5   # stay clear of the border line
            if (bb.x0 < x0 + inset or bb.x1 > x1 - inset
                    or bb.y0 < y0 + inset or bb.y1 > y1 - inset):
                problems.append(f"leaves its card: {t.get_text()[:50]!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _shrink(boxes[i][1]).overlaps(_shrink(boxes[j][1])):
                problems.append("text overlap: "
                                f"{boxes[i][0].get_text()[:30]!r} / {boxes[j][0].get_text()[:30]!r}")
    for a in cv.arrows:
        abb = a.get_window_extent(r)
        for t, bb in boxes:
            if _shrink(abb, 1.0).overlaps(_shrink(bb)):
                problems.append(f"arrow over text: {t.get_text()[:40]!r}")
    if problems:
        raise RuntimeError("layout check failed:\n  " + "\n  ".join(problems))


def check_png(path: Path, height_in: float) -> None:
    """Raise unless the PNG is 1950 px wide, at the stated height and grayscale."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        rgb = np.asarray(im.convert("RGB")).astype(int)
    if w != round(WIDTH_IN * DPI) or h != round(height_in * DPI):
        raise RuntimeError(f"{path.name}: {w}x{h} px, expected {WIDTH_IN * DPI:.0f} wide")
    dev = max(np.abs(rgb[..., 0] - rgb[..., 1]).max(), np.abs(rgb[..., 1] - rgb[..., 2]).max())
    if dev > 2:
        raise RuntimeError(f"{path.name}: not grayscale (max channel difference {dev})")


# ---------------------------------------------------------------------------
# Figure 1: graphical abstract
# ---------------------------------------------------------------------------
def generate_graphical_abstract():
    header = Card(
        "puremacro 4.5.0: quantitative macroeconomics and macroeconometrics in Python",
        ["One package on the scientific-Python stack for estimation, DSGE, heterogeneous "
         "agents, dynamic programming and trade, with no compiled extension of its own "
         "and case-specific numerical validation."],
        fill=FILL_HEAD, edge=EDGE_BAND, lw=0.8, title_pt=10.5, plain=True)
    band1 = Band("1   Architecture and invariants", Row([
        Card("Import contract", [
            "Module-scope imports: NumPy, SciPy, pandas, Matplotlib",
            "requests is also mandatory, for the data layer",
            "No C, C++, Cython or Rust extension of its own",
            "Optional Numba and GPU backends sit outside the contract",
            "Import sweeps run in every CI job",
        ]),
        Card("Runtimes", [
            "CPython 3.11–3.13 on Linux, macOS and Windows (nine CI targets)",
            "Pyodide and JupyterLite: shippable modules are import-checked",
            "Realistic browser and GPU workloads: not yet verified",
            "Colab offload: notebook out, .pmz result back",
        ]),
        Card("Result objects", [
            "Typed result dataclasses (311 frozen, 56 plain)",
            ".summary(), .plot(), .to_latex(), .to_typst()",
            "Convergence flags refer to the residual that was checked",
            "Since 4.5, calls that cannot be answered honestly raise",
        ]),
    ]))
    band2 = Band("2   Four computational pillars", Column([
        Row([
            Card("Macroeconometrics and causal inference", [
                "SVAR: Cholesky, Blanchard–Quah, sign and zero, narrative, proxy",
                "Local projections: HAC, lag-augmented, panel, state-dependent",
                "Minnesota BVAR, Markov-switching VAR, GARCH family, GARCH-MIDAS",
                "Staggered DiD, synthetic control, synthetic DiD, double ML",
                "Factor-model nowcasting; structural moment targets (4.4)",
            ]),
            Card("DSGE models", [
                "Parser for a subset of Dynare .mod syntax",
                "QZ first order; pruned second and third order",
                "Exact pruned moments at orders two and three (4.4)",
                "NUTS with an analytic first-order likelihood score",
                "Stacked-time Newton–Krylov for perfect foresight (4.4)",
            ]),
        ]),
        Row([
            Card("Heterogeneous agents and dynamic programming", [
                "VFI, EGM and DC-EGM; histogram distributions (Young, 2010)",
                "Sequence-space Jacobians by the fake-news algorithm",
                "Continuous-time HJB–KFE with an implicit upwind scheme",
                "Chebyshev collocation, Smolyak grids, finite elements",
            ]),
            Card("Spatial, trade and climate", [
                "Allen–Arkolakis spatial equilibrium",
                "Caliendo–Parro hat algebra; IO-based CGE with consistent accounting",
                "Seven input-output engines, including a dynamic MRIO (4.4)",
                "Hicksian EV/CV; ENIGH 2024 household incidence",
                "DICE-2016R forward simulator with a social cost of carbon",
            ]),
        ]),
    ], gap=0.10))
    band3 = Band("3   Verification and use", Row([
        Card("Validation gallery", [
            "110 cases in 15 subsystems, all passing",
            "20 external: 13 package, 5 SciPy, 2 published",
            "31 analytical, 59 internal consistency",
            "Reference libraries are not imported at run time",
            "Tolerances stated per case; no trade cases",
        ]),
        Card("Benchmarks and replications", [
            "Research benchmarks: 7 of 8 pass",
            "Romer–Romer t statistic misses its printed rounding by "
            r"$3.9\times10^{-6}$, reported as a failure",
            "14 replication cases, all passing",
            "Five models checked in Dynare 7.0 at orders two and three",
        ]),
        Card("Open science and teaching", [
            "806 modules, 281,431 lines, 19,090 tests",
            "70 bilingual notebook pairs; 22 ITAM course lessons",
            "JupyterLite site with an in-browser Pyodide kernel",
            "Public correctness advisories with affected versions",
            "MIT licence; PyPI release only after nine-target CI",
        ]),
    ]))
    gap = 0.20
    root = Column([header, band1, band2, band3], gap=gap)
    cv = layout(root, (ABSTRACT_H, ABSTRACT_H), "graphical abstract")
    for upper, lower in ((band1, band2), (band2, band3)):
        xc = upper.rect[0] + upper.rect[2] / 2
        y0 = upper.rect[1] + upper.rect[3] + 0.025
        cv.arrow(xc, y0, xc, lower.rect[1] - 0.025, lw=1.3)
    cv.save("graphical_abstract.png")


# ---------------------------------------------------------------------------
# Figure 2: architecture and the oracle loop
# ---------------------------------------------------------------------------
def generate_architecture_oracle_figure():
    left = Band("Offline: reference generation", Column([
        Card("Reference libraries (never imported by the package)", [
            "statsmodels 0.14.6: VAR impulse responses, HAC errors, state-space smoother",
            "arch 8.0.0: GARCH(1,1)",
            "linearmodels 7.0: IV2SLS, PanelOLS",
            "esda 2.10 and libpysal 4.15: Moran's I, Geary's C",
        ]),
        Card("Generator scripts", [
            "tools/gen_validation_goldens_*.py",
            "Seeded simulated data from the gallery's own fixtures",
            "Each file records the generator, library version and data seed",
        ]),
        Card("Frozen goldens", [
            "JSON files in puremacro/validation/goldens/ (156 kB)",
            "Shipped as package data inside the wheel",
            "Dynare, Stata and Romer–Romer references are frozen the same way",
        ]),
    ], gap=0.10), fill=FILL_BAND)
    right = Band("Shipped package: user runtime", Column([
        Card("Import contract", [
            "Shippable modules import only NumPy, SciPy, pandas and Matplotlib at module scope",
            "A sweep fails if statsmodels, linearmodels, arch, bs4, pdfplumber or pypdf is loaded",
            "A blocked-import subprocess test runs in every CI job",
        ]),
        Card("Where it runs", [
            "pip install puremacro: CPython 3.11–3.13; CI on Linux, macOS, Windows",
            "Pyodide and JupyterLite: import-checked; workloads not verified",
            "Optional Colab offload for long computations",
        ]),
        Card("puremacro.validation.scorecard()", [
            "110 cases in 15 subsystems, all passing",
            "20 external (13 package, 5 SciPy, 2 published), 31 analytical, 59 internal",
            "Each case states its reference mechanism and tolerance",
        ]),
    ], gap=0.10), fill=FILL_BAND)
    root = Row([left, right], gap=0.42)
    cv = layout(root, BODY_H, "architecture")
    # The goldens flow from the offline column into the package.
    golden = left.child.children[2].rect
    score = right.child.children[2].rect
    y = (max(golden[1], score[1]) + min(golden[1] + golden[3], score[1] + score[3])) / 2
    x0 = left.rect[0] + left.rect[2] + 0.03
    x1 = right.rect[0] - 0.03
    cv.arrow(x0, y, x1, y, lw=1.4)
    t, _ = cv.text("ships", fontsize=BODY_PT, fontstyle="italic", ha="center", va="bottom")
    t.set_position(((x0 + x1) / 2, y - 0.04))
    # Downward flow inside the offline column.
    col = left.child.children
    for a, b in zip(col[:-1], col[1:]):
        xc = a.rect[0] + a.rect[2] / 2
        cv.arrow(xc, a.rect[1] + a.rect[3] + 0.012, xc, b.rect[1] - 0.012, lw=0.9)
    cv.save("fig_architecture_oracle.png")


# ---------------------------------------------------------------------------
# Figure 3: SVAR identification
# ---------------------------------------------------------------------------
def generate_svar_identification_figure():
    head = Card("Reduced-form VAR and the structural impact matrix", [
        r"$Y_t = c + A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + u_t,\quad "
        r"\mathrm{E}[u_t u_t^{\top}] = \Sigma$",
        r"$u_t = B_0^{-1}\varepsilon_t,\quad \mathrm{E}[\varepsilon_t \varepsilon_t^{\top}] = I_K,"
        r"\quad B_0^{-1}(B_0^{-1})^{\top} = \Sigma$;  "
        r"$K(K-1)/2$ further restrictions are needed",
    ], fill=FILL_HEAD, edge=EDGE_BAND, lw=0.8, plain=True)
    row1 = Row([
        Card("1  Recursive (Cholesky)", [
            r"$B_0^{-1} = P$, lower triangular, $PP^{\top} = \Sigma$",
            r"Variable $j$ has no impact effect on variable $i$ if $j > i$",
            "Eigenvalue floor keeps the factor defined when Σ is near-singular",
            "Results depend on the ordering",
        ], "Sims (1980)"),
        Card("2  Long-run (Blanchard–Quah)", [
            r"Long-run matrix $A(1)^{-1}B_0^{-1}$ lower triangular, "
            r"$A(1) = I - A_1 - \cdots - A_p$",
            r"Cholesky factor of $A(1)^{-1}\,\Sigma\,[A(1)^{-1}]^{\top}$",
            "bq_svar(cumulate=...) sets which responses are cumulated (4.4)",
            "The same choice is applied to every bootstrap draw",
        ], "Blanchard and Quah (1989)"),
        Card("3  Sign and zero restrictions", [
            r"$B_0^{-1} = PQ$ with $Q$ drawn from the Haar measure",
            "Sign checks on impulse responses at chosen horizons",
            "Impact zeros by the Rubio-Ramírez et al. algorithm",
            "Set identification: bands over accepted draws",
        ], "Uhlig (2005); Rubio-Ramírez et al. (2010)"),
    ], gap=0.10)
    row2 = Row([
        Card("4  Narrative sign restrictions", [
            "Shock-sign restrictions in dated historical episodes",
            "Historical-contribution restrictions: one shock is the main driver",
            "Accepted draws reweighted by importance weights",
            "Kish effective sample size reported; a warning when weights concentrate",
        ], "Antólin-Díaz and Rubio-Ramírez (2018)"),
        Card("5  External instruments and local projections", [
            "Proxy SVAR with the Montiel Olea–Pflueger effective F",
            "LP-IV with Anderson–Rubin weak-IV-robust intervals",
            r"LP-HAC: Newey–West, Bartlett kernel, lag $h+1$",
            "Lag-augmented LP with HC0 errors (Montiel Olea and Plagborg-Møller, 2021)",
            "Panel LP with Driscoll–Kraay errors; smooth-transition state-dependent LP",
        ], "Mertens and Ravn (2013); Stock and Watson (2018); Jordà (2005)"),
    ], gap=0.10)
    root = Column([head, row1, row2], gap=0.17)
    cv = layout(root, BODY_H, "svar")
    hx, hy, hw, hh = head.rect
    for c in row1.children:
        xc = c.rect[0] + c.rect[2] / 2
        cv.arrow(xc, hy + hh + 0.02, xc, c.rect[1] - 0.02)
    cv.save("fig_svar_identification.png")


# ---------------------------------------------------------------------------
# Figure 4: DSGE and sequence space
# ---------------------------------------------------------------------------
def generate_dsge_ssj_figure():
    c1 = Card("1  Parse and differentiate", [
        "var, varexo, parameters, model, initval and shocks blocks; leads and lags",
        "Model-local # variables substituted symbolically",
        r"Complex-step Jacobian ($h = 10^{-20}$), cross-checked by finite differences",
        "steady() rejects structurally singular steady states (4.4)",
    ], "A subset of Dynare .mod syntax")
    c2 = Card("2  Perturbation", [
        "First order by QZ with the Blanchard–Kahn check",
        r"Pruned second and third order; risk correction $g_{\sigma\sigma}$",
        "Exact pruned moments at orders two and three (4.4)",
        r"Five models agree with Dynare 7.0 to $1.2\times10^{-13}$: "
        "a benchmark, not general parity",
    ], "Klein (2000); Kim et al. (2008)")
    c3 = Card("3  Bayesian estimation", [
        "Kalman likelihood; NUTS with dual-averaging step size",
        "Analytic score for first-order solutions (Sylvester and Lyapunov derivatives)",
        r"Score checked against finite differences at relative tolerance $10^{-5}$",
        "Split R-hat, Geyer ESS and Geweke z corrected in 4.4",
    ], "Hoffman and Gelman (2014)")
    c4 = Card("4  Perfect-foresight transitions", [
        "Opt-in stacked-time inexact Newton–Krylov (4.4)",
        "LGMRES with a block-tridiagonal steady-state preconditioner",
        "A path is accepted only if residual and next step are small; failures raise",
        r"Matches solve_perfect_foresight to $3.5\times10^{-11}$ on 4,000 unknowns",
    ], "puremacro.dsge.stacked_newton")
    c5 = Card("5  Sequence-space Jacobians", [
        "Household block: EGM and histogram distribution (Young, 2010)",
        r"Fake-news algorithm: $T\times T$ Jacobians from one backward and one forward pass",
        r"Two-asset Jacobians match brute-force perturbation to $10^{-7}$ (4.4)",
    ], "Auclert et al. (2021)")
    c6 = Card("6  General-equilibrium coupling", [
        "Household Jacobians joined to aggregate equations (Phillips curve, policy rule)",
        "Linear transition paths solved in sequence space",
        "Optional nonlinear path by Broyden updates",
        "solve_hank_bridge(source, shock=...)",
    ], "hetagent_block in .mod files (puremacro extension)")
    row1 = Row([c1, c2, c3], gap=0.20)
    row2 = Row([c4, c5, c6], gap=0.20)
    root = Column([row1, row2], gap=0.22)
    cv = layout(root, BODY_H, "dsge")
    for a, b in ((c1, c2), (c2, c3), (c5, c6)):
        y = a.rect[1] + a.rect[3] / 2
        cv.arrow(a.rect[0] + a.rect[2] + 0.02, y, b.rect[0] - 0.02, y)
    for a, b in ((c1, c4), (c2, c5)):
        xc = a.rect[0] + a.rect[2] / 2
        cv.arrow(xc, a.rect[1] + a.rect[3] + 0.02, xc, b.rect[1] - 0.02)
    cv.save("fig_dsge_ssj_workflow.png")


# ---------------------------------------------------------------------------
# Figure 5: projection methods and dynamic programming
# ---------------------------------------------------------------------------
def generate_vfi_projections_figure():
    root = Column([
        Row([
            Card("1  Chebyshev collocation", [
                r"Policy $c(k) = \Sigma_j\, \gamma_j T_j(\phi(k))$ on Chebyshev nodes",
                "Euler residuals solved by scipy.optimize.root (hybr, then lm)",
                "converged means residuals within tolerance at the fitting nodes; "
                "off-grid accuracy is not certified",
                "CRRA curvature read as sigma, alias gamma (4.4; splines 4.5)",
                "Euler and Bellman variants (method=\"euler\" or \"bellman\")",
            ], "Judd (1992)"),
            Card("2  Smolyak sparse grids", [
                "Nested Clenshaw–Curtis nodes",
                r"$d = 5$, $\mu = 2$: 61 nodes against 3,125 for the tensor grid "
                "(5 per dimension)",
                r"$d = 5$, $\mu = 3$: 241 nodes against 59,049 (9 per dimension)",
                "method=\"euler\" is the general route",
                "method=\"bellman\" raises outside log utility with full depreciation (4.5)",
            ], "Smolyak (1963); Krueger and Kubler (2004)"),
        ], gap=0.12),
        Row([
            Card("3  Finite elements", [
                "Piecewise-linear hat functions; uniform, clustered or kink-aligned meshes",
                "Galerkin residuals by Gauss–Legendre quadrature",
                "Fischer–Burmeister function encodes the borrowing constraint and its "
                "multiplier as one equation",
                "How well the kink is located depends on the mesh",
            ], "McGrattan (1996)"),
            Card("4  Continuous time: HJB and KFE", [
                "Implicit upwind finite differences for the HJB equation",
                "Monotone sparse (M-matrix) system allows large time steps",
                r"Stationary distribution from $A^{\top} g = 0$ by sparse LU",
                "Accuracy depends on the grid; only residuals are checked",
                "solve_aiyagari_continuous_hjb: stationary general equilibrium",
            ], "Achdou et al. (2022)"),
        ], gap=0.12),
    ], gap=0.12)
    cv = layout(root, BODY_H, "vfi")
    cv.save("fig_vfi_projections.png")


# ---------------------------------------------------------------------------
# Figure 6: trade, spatial and climate
# ---------------------------------------------------------------------------
def generate_trade_spatial_figure():
    trade = Column([
        Card("Caliendo–Parro exact hat algebra", [
            "Many sectors and countries linked by input-output shares",
            r"Counterfactuals in changes $\hat{x} = x'/x$; sector elasticities $\theta_j$",
            "Damped fixed-point iteration on wages and deficits",
            "The published NAFTA results have not been re-replicated",
        ], "Caliendo and Parro (2015)"),
        Card("IO-based CGE with consistent accounting", [
            "Producer and purchaser prices; duties paid once and rebated",
            r"Independent two-country check: price error below $5\times10^{-15}$",
            "Hicksian EV and CV from the expenditure function",
            "Full-size native OECD tables rejected (negative investment cells)",
        ], "solve_trade_equilibrium(accounting=\"consistent\")"),
        Card("Flexible CGE settings", [
            "Nested CES, Stone–Geary demand, Atkeson–Burstein markups",
            "Solved by the quasi-condensed Newton route (4.4)",
            "Reports the residual its convergence flag was judged on (4.5)",
            "Welfare decomposition is a historical proxy, not validated",
        ], "puremacro.trade.flexible"),
    ], gap=0.10)
    other = Column([
        Card("Input-output engines (4.4)", [
            "Native OECD ICIO 2023, FIGARO and EXIOBASE readers; the corrupted 2020 export is refused",
            "Condensed Leontief tariff model with an independent certificate",
            "Nested-CES block Newton, household demand, continuation, stability",
            "Dynamic MRIO: indeterminate on the 77×11 table, so no welfare number there",
        ], "seven engines ported from the research workspace"),
        Card("Allen–Arkolakis spatial equilibrium", [
            r"$N$ locations, iceberg costs $\tau_{ij} \geq 1$, free mobility equalizes utility",
            "is_unique checks the parameter condition of their Theorem 2, "
            "not a computed equilibrium",
            "Counterfactual trade costs by fixed-point iteration",
        ], "Allen and Arkolakis (2014)"),
        Card("Climate: DICE-2016R", [
            "Forward simulation under a given carbon-tax path; nothing is optimized",
            "Three-reservoir carbon cycle, two-layer temperature, quadratic damages",
            "Social cost of carbon from a marginal emissions perturbation",
            "Outside the validation gallery",
        ], "Nordhaus (2017) calibration"),
    ], gap=0.10)
    root = Row([
        Band("Trade general equilibrium", trade),
        Band("Input-output data, space and climate", other),
    ], gap=0.12)
    cv = layout(root, BODY_H, "trade")
    cv.save("fig_trade_spatial_cge.png")


# ---------------------------------------------------------------------------
# Figure 7: validation scorecard
# ---------------------------------------------------------------------------
KIND = {"package": "External reference", "scipy": "External reference",
        "published": "External reference", "analytical": "Analytical result",
        "internal": "Internal consistency"}
KINDS = ["External reference", "Analytical result", "Internal consistency"]
SUBSYSTEM = {
    "var": "VAR / SVAR", "spatial": "Spatial econometrics",
    "did": "Difference-in-differences", "inference": "HAC, fixed-b and weak-IV inference",
    "narrative": "Text-based narrative indices", "garch": "GARCH volatility",
    "dynpanel": "Dynamic panels (GMM)", "state_space": "State space / Kalman",
    "spectral": "Spectral analysis", "lp": "Local projections",
    "forecast": "Forecast evaluation", "dsge": "Linear RE / DSGE",
    "vfi": "Dynamic programming (VFI)", "cointegration": "Cointegration",
    "unit_root": "Unit roots",
}


def generate_scorecard_figure():
    from puremacro.validation import scorecard

    df = scorecard()
    unknown = set(df["mechanism"]) - set(KIND)
    if unknown:
        raise RuntimeError(f"unmapped mechanism labels: {sorted(unknown)}")
    df = df.assign(kind=df["mechanism"].map(KIND))
    tab = (df.pivot_table(index="subsystem", columns="kind", values="id",
                          aggfunc="count", fill_value=0)
             .reindex(columns=KINDS, fill_value=0))
    rows = sorted(tab.index, key=lambda s: (int(tab.loc[s].sum()), s))
    tab = tab.loc[rows]
    ext = df[df["kind"] == "External reference"]["mechanism"].value_counts()

    fills = {KINDS[0]: bw_colors(1)[0], KINDS[1]: "0.62", KINDS[2]: "0.93"}
    hatches = {KINDS[0]: bw_hatches(1)[0], KINDS[1]: bw_hatches(2)[1], KINDS[2]: ""}

    fig = plt.figure(figsize=(WIDTH_IN, 4.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.30, 0.115, 0.66, 0.80))
    left = [0] * len(tab)
    for kind in KINDS:
        vals = [int(v) for v in tab[kind]]
        n = sum(vals)
        if kind == KINDS[0]:
            label = (f"{kind} ({n}: {ext.get('package', 0)} package, "
                     f"{ext.get('scipy', 0)} SciPy, {ext.get('published', 0)} published)")
        else:
            label = f"{kind} ({n})"
        ax.barh(range(len(tab)), vals, left=left, height=0.64, color=fills[kind],
                hatch=hatches[kind], edgecolor="0.15", linewidth=0.6, label=label)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + 0.25, i, str(total), va="center", ha="left",
                fontsize=BODY_PT, color=INK)
    ax.set_yticks(range(len(tab)), [SUBSYSTEM.get(s, s) for s in tab.index])
    ax.tick_params(axis="y", length=0, labelsize=BODY_PT, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=BODY_PT, labelcolor=INK, direction="out")
    ax.set_xlim(0, max(left) * 1.12)
    ax.set_xticks(range(0, max(left) + 1, 3))
    ax.set_ylim(-0.6, len(tab) - 0.4)
    ax.set_xlabel(f"Validation checks ({int(df['passed'].sum())} of {len(df)} pass, "
                  f"{df['subsystem'].nunique()} subsystems)", fontsize=BODY_PT, color=INK)
    ax.xaxis.grid(True, color="0.82", linewidth=0.5, linestyle="-")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.35")
    ax.spines["bottom"].set_linewidth(0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=BODY_PT, handlelength=1.6,
               handleheight=1.0, columnspacing=1.4, borderaxespad=0.2)

    cv = Canvas.__new__(Canvas)        # reuse the checks without the card layout
    cv.fig, cv.ax, cv.cards, cv.arrows = fig, ax, [], []
    check_layout(cv)
    path = OUTPUT_DIR / "fig_scorecard.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    check_png(path, 4.0)
    print(f"wrote {path.relative_to(OUTPUT_DIR.parent.parent)} "
          f"({len(df)} cases, {dict((k, int(tab[k].sum())) for k in KINDS)})")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    with warnings.catch_warnings():
        # A missing glyph would print as a box: treat it as an error.
        warnings.filterwarnings("error", message=".*missing from.*font.*")
        warnings.filterwarnings("error", message=".*Glyph.*missing.*")
        generate_graphical_abstract()
        generate_architecture_oracle_figure()
        generate_svar_identification_figure()
        generate_dsge_ssj_figure()
        generate_vfi_projections_figure()
        generate_trade_spatial_figure()
        generate_scorecard_figure()
    print("All figures generated in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
