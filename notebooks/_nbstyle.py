"""Course notebook plot style: a black-and-white card that reads the same in
Juno's light and dark modes.

The previous style drew on a transparent background, so its inks had to be
legible on black *and* on white.  The only inks that manage both are
mid-grays (#767676 is 4.6:1 against each), and they look dim on either.  This
style gives every figure its own opaque background -- the card -- so the
notebook's theme no longer reaches the figure, and the ink can be as bright as
the card calls for.

Every neutral on the card is an *ink fraction* ``t``: 0 is the card, 1 the
ink, interpolated in perceptual lightness (OKLab L, which for a neutral gray is
the cube root of its linear luminance).  The two themes share those
fractions, so a figure keeps its hierarchy when the theme flips:

* ``grafito`` (default) -- graphite card #161616, ink #EDEDED.  On a black page
  it reads as a raised card, on a white page as a dark tile.
* ``papel`` -- white card, ink #141414: the printed page.

Series are told apart by lightness *and* dash pattern, never by hue.  Choose
the theme with ``apply_style(theme="papel")`` or the ``MAV_NB_TEMA``
environment variable; the notebooks themselves never name one.

Tokens (read them as ``_nbstyle.NAME`` at plot time, not ``from _nbstyle
import NAME``: a theme switch rebinds them):

    FONDO    the card                 TINTA    titles, main text, hero series
    TEXTO    axis labels, legends     NOTA     tick labels, notes, arrows
    SPINE    axes, zero/45-degree     REJILLA  grid hairlines
    S1..S6   series (color, dash, width) in emphasis order
    BANDA    confidence band fill     RECESION shaded episode fill
    tono(t)  the gray at ink fraction t
    gris(c)  the gray that plays, on this card, the role c plays on white paper

Helpers: ``figura(nrows, ncols)`` (subplots at the course size, constrained
layout), ``etiquetar(ax)`` (the series' names next to the lines instead of a
legend), ``etiquetar_barras(ax)``, ``TAMANO``/``LAMINA`` (notebook and slide
sizes), ``apply_style(perfil="lamina")`` (slide type sizes) and
``exportar(fig, nombre)`` (vector PDF for the decks; a no-op unless
``$MAV_EXPORTAR_FIGURAS`` names a folder).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
import warnings

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ``FT2Font`` takes its flags as ``enum.Flag``/``enum.Enum`` members, and
# pybind11 checks them by class identity.  Juno on the iPad ends up with two
# live copies of ``matplotlib.ft2font``, so the members matplotlib passes are
# not the ones the compiled font accepts -- and a plain ``Flag`` has no
# ``__int__``, so the integer overload does not rescue it either.  Every text
# measurement then dies: tick label, title, the "lp" probe in
# ``Text._get_layout``.  It surfaces as
# ``TypeError: set_text(): incompatible function arguments``, and it takes
# ``tight_layout()``, the PNG and the cell with it.
_FT2FONT_ENUMS = ("LoadFlags", "Kerning", "RenderMode")

# Modules that pass those enums into compiled ft2font calls.  Imported eagerly
# so a lazily loaded one cannot keep a stale copy the repair never saw.
_FT2FONT_USERS = ("_mathtext", "_text_helpers", "mathtext", "textpath",
                  "backends.backend_agg")


def _matplotlib_modules():
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("matplotlib"):
            yield module


def _enum_named(module, name):
    """The attribute *name* of *module*, if it is an enum class of that name."""
    value = getattr(module, name, None)
    if isinstance(value, type) and getattr(value, "__name__", None) == name:
        return value
    return None


def _live_font():
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    return RendererAgg(4, 4, 72)._prepare_font(FontProperties())


def _text_measures() -> bool:
    """True if matplotlib can measure a string with the Agg renderer."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    try:
        RendererAgg(4, 4, 72).get_text_width_height_descent(
            "lp", FontProperties(), False)
        return True
    except Exception:
        return False


def _text_renders() -> bool:
    """True if matplotlib can rasterize a string, not merely measure it."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(1, 1))
    FigureCanvasAgg(figure)
    figure.text(0.5, 0.5, "lp")
    try:
        figure.canvas.draw()
        return True
    except Exception:
        return False


def _swap(name, replacement):
    """Point every matplotlib module holding enum *name* at *replacement*.

    Returns the undo list: (module, name, previous) triples.
    """
    undone = []
    for module in _matplotlib_modules():
        previous = _enum_named(module, name)
        if previous is not None and previous is not replacement:
            setattr(module, name, replacement)
            undone.append((module, name, previous))
    return undone


def _rebind_to_accepted_enums():
    """Adopt the copy of the enums that the compiled font actually accepts."""
    font = _live_font()
    for module in _matplotlib_modules():
        flags = _enum_named(module, "LoadFlags")
        if flags is None:
            continue
        try:
            font.set_text("lp", 0.0, flags=flags.DEFAULT)
        except Exception:
            continue
        undone = []
        for name in _FT2FONT_ENUMS:                 # all three from one module,
            accepted = getattr(module, name, None)  # so they cannot disagree
            if accepted is not None:
                undone += _swap(name, accepted)
        return undone
    return []


def _rebind_enums_as_ints():
    """Fallback: hand the call sites plain integers.

    ``set_text`` and ``get_kerning`` still take them; the stricter entry points
    do not, which is why this runs only after the enum swap has failed and why
    the caller verifies before keeping it.
    """
    undone = []
    for name in ("LoadFlags", "Kerning"):
        source = next(filter(None, (_enum_named(m, name)
                                    for m in _matplotlib_modules())), None)
        if source is None:
            continue
        # __members__, not iteration: iterating a Flag skips its zero-valued
        # member, and LoadFlags.DEFAULT is exactly that.
        undone += _swap(name, types.SimpleNamespace(
            **{n: m.value for n, m in source.__members__.items()}))
    return undone


def repair_ft2font_enums() -> bool:
    """Repair matplotlib when its ft2font enums are not the compiled copy.

    Returns True only if a repair was needed *and* held -- text both measures
    and rasterizes afterwards.  A strategy that does not hold is undone rather
    than left half-applied, so the original error stays legible.  On any install
    where the Python and compiled halves agree (every desktop one) this costs a
    single text measurement and changes nothing.
    """
    if _text_measures():
        return False

    for name in _FT2FONT_USERS:
        try:
            importlib.import_module(f"matplotlib.{name}")
        except ImportError:
            pass

    for strategy in (_rebind_to_accepted_enums, _rebind_enums_as_ints):
        undone = strategy()
        if undone and _text_measures() and _text_renders():
            if strategy is _rebind_enums_as_ints:
                # Integer flags are deprecated since 3.10; here they are the
                # only ones that work, so keep the notice out of the output.
                warnings.filterwarnings(
                    "ignore", message=".*flags parameter as int.*",
                    category=mpl.MatplotlibDeprecationWarning)
            return True
        for module, name, previous in reversed(undone):
            setattr(module, name, previous)
    return False


def enable_inline_backend() -> bool:
    """Safely activate '%matplotlib inline' if running within an interactive IPython kernel.

    Returns True if inline backend magic was executed, False otherwise.
    Safe across standard Python scripts, pytest test runners, and sandboxed iOS kernels.
    """
    try:
        ip = None
        if "IPython" in sys.modules:
            import IPython
            ip = IPython.get_ipython()
        elif "get_ipython" in globals() or (isinstance(__builtins__, dict) and "get_ipython" in __builtins__) or hasattr(__builtins__, "get_ipython"):
            get_ip = (
                globals().get("get_ipython")
                or (__builtins__.get("get_ipython") if isinstance(__builtins__, dict) else getattr(__builtins__, "get_ipython", None))
            )
            if callable(get_ip):
                ip = get_ip()
        else:
            try:
                import IPython
                ip = IPython.get_ipython()
            except ImportError:
                ip = None

        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
            return True
    except Exception:
        pass
    return False



# ---------------------------------------------------------------------------
# The card and its grays
# ---------------------------------------------------------------------------
TEMAS = {
    "grafito": {"fondo": "#161616", "tinta": "#EDEDED"},
    "papel": {"fondo": "#FFFFFF", "tinta": "#141414"},
}
TEMA_POR_OMISION = "grafito"

# Ink fractions of the chrome.  Text at >= 0.60 clears 4.5:1 on either card,
# and SPINE sits just above the 3:1 floor for marks that carry meaning (zero
# lines, 45-degree lines).
_F_TINTA, _F_TEXTO, _F_NOTA, _F_SPINE, _F_REJILLA = 1.00, 0.80, 0.60, 0.42, 0.10

# Series in emphasis order: (ink fraction, dash, width).  Lightness says which
# series matters most; the dash is what tells two of them apart, so no two
# share one.  The five distinct lightness steps (1.00 .. 0.44) pass the
# ordinal ramp checks -- monotone, >= 0.06 apart in OKLab L, the faintest at
# 3.3:1 -- on both cards.
_SERIES = (
    (1.00, "-", 2.0),
    (0.70, (0, (5, 2.2)), 2.0),
    (0.44, (0, (6, 1.8, 1.2, 1.8)), 2.0),
    (0.86, (0, (1, 1.7)), 2.3),
    (0.56, (0, (10, 3)), 2.0),
    (0.70, (0, (3, 1.5, 1, 1.5, 1, 1.5)), 1.7),
)
LINESTYLES = [dash for _, dash, _ in _SERIES]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Fills: bars read as blocks, so they stop short of the ink.  Neighbouring
# segments are split by a card-colored gap, not by an outline.
_F_BARRAS = (0.86, 0.60, 0.38, 0.22)
BARRA_HATCH = ("", "//", "..", "xx")
EPISODIO_MARCA = ("o", "s", "^", "D", "v")
BANDA_ALPHA = 0.16      # confidence bands: ink over the card
RECESION_ALPHA = 0.10   # shaded episodes: fainter than a band, heavier than the grid


def _lineal(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _codificar(y):
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


def _claridad(color) -> float:
    """OKLab lightness of *color* seen as a gray: cube root of its luminance."""
    r, g, b = mpl.colors.to_rgb(color)
    return float((0.2126 * _lineal(r) + 0.7152 * _lineal(g) + 0.0722 * _lineal(b)) ** (1 / 3))


def _hex(lightness) -> str:
    v = int(round(float(_codificar(float(lightness) ** 3)) * 255))
    return f"#{v:02X}{v:02X}{v:02X}"


def tono(t) -> str:
    """The gray at ink fraction *t* (0 = the card, 1 = the ink) on this card."""
    t = min(max(float(t), 0.0), 1.0)
    return _hex(_L_FONDO + t * (_L_TINTA - _L_FONDO))


def gris(color) -> str:
    """The gray that plays, on this card, the role *color* plays on white paper.

    ``gris("0.85")`` is as faint here as 0.85 is on paper, ``gris("black")`` is
    the ink and ``gris("white")`` the card, so a literal written for a white
    page keeps its weight on either theme.  Colors are taken by luminance.
    """
    if isinstance(color, (int, float)) and not isinstance(color, bool):
        color = str(float(color))
    return tono(1.0 - _claridad(color))


def _tema_por_omision() -> str:
    tema = os.environ.get("MAV_NB_TEMA", "").strip().lower()
    if tema in TEMAS:
        return tema
    # The Colab copies in colab_es/ still carry colors written for a white
    # page (black lines, white gaps) that a graphite card would swallow.
    if "google.colab" in sys.modules:
        return "papel"
    return TEMA_POR_OMISION


def _usar_tema(nombre: str) -> None:
    """Rebind every module-level token to theme *nombre*."""
    global TEMA, FONDO, _L_FONDO, _L_TINTA
    if nombre not in TEMAS:
        raise ValueError(f"unknown theme {nombre!r}; choose from {sorted(TEMAS)}")
    TEMA = nombre
    FONDO = TEMAS[nombre]["fondo"]
    _L_FONDO = _claridad(FONDO)
    _L_TINTA = _claridad(TEMAS[nombre]["tinta"])

    series = [tono(t) for t, _, _ in _SERIES]
    tokens = dict(
        TINTA=tono(_F_TINTA), TEXTO=tono(_F_TEXTO), NOTA=tono(_F_NOTA),
        SPINE=tono(_F_SPINE), REJILLA=tono(_F_REJILLA),
        UNIVERSAL_COLORS=series,
        GRAYS=[tono(t) for t in np.linspace(1.0, 0.42, 8)],   # 0.42: >= 3:1 on either card
        BARRA_COLORES=tuple(tono(t) for t in _F_BARRAS),
        EPISODIO_COLORES=tuple(tono(t) for t in (1.00, 0.70, 0.44, 0.86, 0.56)),
    )
    tokens["BARRA_GRIS"] = tokens["BARRA_COLORES"]
    tokens["EPISODIO_GRIS"] = tokens["EPISODIO_COLORES"]
    for i, (t, dash, width) in enumerate(_SERIES, start=1):
        tokens[f"S{i}"] = dict(color=series[i - 1], linestyle=dash, linewidth=width)
    tokens["ESCALERA"] = tuple(tokens[f"S{i}"] for i in range(1, 7))

    ink = mpl.colors.to_rgb(tokens["TINTA"])
    tokens.update(
        BANDA_HEX=tokens["TINTA"], BANDA=(*ink, BANDA_ALPHA), BANDA_BORDE=tokens["NOTA"],
        RECESION_HEX=tokens["TINTA"], RECESION=(*ink, RECESION_ALPHA),
        FLECHA=dict(arrowstyle="->", color=tokens["NOTA"], lw=0.9),
        COLOR_CYCLER=cycler(color=series),
        UNIVERSAL_CYCLER=(cycler(color=series) + cycler(linestyle=LINESTYLES)
                          + cycler(marker=MARKERS)),
    )
    tokens["SEMANTIC_COLORS"] = {
        "card": FONDO, "text": tokens["TINTA"], "label": tokens["TEXTO"],
        "note": tokens["NOTA"], "spine": tokens["SPINE"], "grid": tokens["REJILLA"],
        "grid_alpha": 1.0, "grid_lw": 0.6, "recession": tokens["RECESION_HEX"],
        "recession_alpha": RECESION_ALPHA, "band": tokens["BANDA_HEX"],
        "band_alpha": BANDA_ALPHA, "arrow": tokens["NOTA"],
    }
    globals().update(tokens)
    _registrar_mapas()


def _registrar_mapas() -> None:
    """Register ``mav_seq`` (card -> ink) and ``mav_seq_r`` for this theme.

    Low values fade into the card and high ones reach the ink, which is the
    right reading on either theme -- unlike ``Greys``, whose top is black and
    so vanishes into a graphite card.  For a signed quantity use it with
    ``matplotlib.colors.TwoSlopeNorm(vcenter=0)`` and label the colorbar.
    """
    global CMAP_SEQ, CMAP_SEQ_R
    anchors = [tono(t) for t in np.linspace(0.06, 1.0, 11)]
    CMAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("mav_seq", anchors, N=256)
    CMAP_SEQ_R = CMAP_SEQ.reversed()
    for cmap in (CMAP_SEQ, CMAP_SEQ_R):
        with warnings.catch_warnings():   # re-registering on a theme switch is the point
            warnings.simplefilter("ignore", UserWarning)
            mpl.colormaps.register(cmap, name=cmap.name, force=True)


# ---------------------------------------------------------------------------
# Sizes: one table for the notebook, one for the slide slots
# ---------------------------------------------------------------------------
# Notebook figures, in inches at figure.dpi 150.  A notebook column on an iPad
# is ~1000 px wide, so a 6.4-in figure shows at full size and a 9.6-in one is
# scaled to ~2/3 -- which is why the two-panel widths stay under 10 in.
TAMANO = {
    "1": (6.4, 4.0),          # one panel
    "1_alto": (6.4, 5.0),     # one panel, taller (phase diagrams, maps)
    "1_bajo": (6.4, 3.0),     # one panel, a strip (a single time series)
    "ancho": (9.6, 3.6),      # one wide panel (long time series, many bars)
    "2": (9.6, 4.0),          # two panels side by side
    "2_alto": (9.6, 5.0),
    "3": (11.4, 3.8),         # three panels side by side
    "2x2": (9.6, 7.2),
    "2x3": (11.4, 7.2),
    "3x2": (9.6, 9.6),
    "3x3": (11.4, 10.2),
}

# Slide slots, in inches at the deck's geometry (16:9, 9 pt): the text block
# is 5.51 x 3.49 in, and under a frame title about 2.9 in remain.  A figure
# drawn at the slot's size and included at width=\linewidth keeps its fonts
# at the size they were drawn (see the "lamina" profile), so the figure's
# type matches the slide's.
LAMINA = {
    "ancha": (5.5, 2.5),      # full width, one row of panels
    "ancha_baja": (5.5, 1.9), # full width, a strip
    "completa": (5.5, 2.9),   # full width, the whole body (no text)
    "media": (2.65, 2.3),     # a 0.48\textwidth column
    "media_alta": (2.65, 2.8),
    "media_baja": (2.65, 1.8),
    "tercio": (1.75, 1.9),    # a third of the width
    "dos_tercios": (3.6, 2.5),
}

PERFILES = {
    # (font.size, axes.titlesize, tick labelsize, legend/notes, lines.linewidth,
    #  figure.figsize, savefig.pad_inches, axes.titlepad, axes.labelpad)
    "cuaderno": dict(font=12, titulo=12, tick=11, nota=11, lw=2.0,
                     figsize=TAMANO["1"], pad=0.18, titlepad=10, labelpad=6),
    "lamina": dict(font=8.5, titulo=9, tick=8, nota=8, lw=1.5,
                   figsize=LAMINA["media"], pad=0.04, titlepad=5, labelpad=3),
}
PERFIL_POR_OMISION = "cuaderno"


def _perfil_por_omision() -> str:
    perfil = os.environ.get("MAV_NB_PERFIL", "").strip().lower()
    return perfil if perfil in PERFILES else PERFIL_POR_OMISION


def tamano(nrows: int = 1, ncols: int = 1, *, alto: bool = False) -> tuple[float, float]:
    """The notebook size for an *nrows* x *ncols* grid (from ``TAMANO``)."""
    clave = {(1, 1): "1", (1, 2): "2", (1, 3): "3", (2, 2): "2x2", (2, 3): "2x3",
             (3, 2): "3x2", (3, 3): "3x3"}.get((nrows, ncols))
    if clave is None:
        w = min(6.4 * ncols, 11.4)
        return (w, min(3.6 * nrows + 0.4, 10.2))
    if alto and clave in ("1", "2"):
        clave += "_alto"
    return TAMANO[clave]


def figura(nrows: int = 1, ncols: int = 1, *, figsize=None, alto: bool | float = False,
           layout: str | None = "constrained", ancho: float | None = None, **kwargs):
    """``plt.subplots`` with the course size for the grid and constrained layout.

    ``figsize`` can be a key of ``TAMANO`` (or of ``LAMINA``) or a pair of
    inches.  A figure made here needs no ``tight_layout()``.
    """
    if isinstance(figsize, str):
        figsize = TAMANO.get(figsize) or LAMINA.get(figsize) or TAMANO["ancho"]
    if ancho is not None or (isinstance(alto, (int, float)) and not isinstance(alto, bool)):
        base_w, base_h = tamano(nrows, ncols, alto=bool(alto) if isinstance(alto, bool) else False)
        w = float(ancho) if ancho is not None else (figsize[0] if figsize else base_w)
        h = float(alto) if (isinstance(alto, (int, float)) and not isinstance(alto, bool)) else (figsize[1] if figsize else base_h)
        figsize = (w, h)
    elif figsize is None:
        figsize = tamano(nrows, ncols, alto=bool(alto))
    if layout is not None:
        kwargs.setdefault("layout", layout)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def _rc(transparent: bool, perfil: str = PERFIL_POR_OMISION) -> dict:
    fondo = "none" if transparent else FONDO
    P = PERFILES[perfil]
    return {
        "figure.facecolor": fondo, "figure.edgecolor": fondo,
        "axes.facecolor": fondo,
        "savefig.facecolor": fondo, "savefig.edgecolor": fondo,
        "savefig.transparent": bool(transparent),
        # In card mode the legend sits on a card-colored plate with no border, so a
        # line running under it passes behind the text instead of through it.
        "legend.frameon": not transparent, "legend.facecolor": fondo,
        "legend.edgecolor": fondo, "legend.framealpha": 0.85,

        "text.color": TINTA, "axes.titlecolor": TINTA, "axes.labelcolor": TEXTO,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.8,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "xtick.labelcolor": NOTA, "ytick.labelcolor": NOTA,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,

        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": REJILLA, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,

        "font.size": P["font"], "axes.titlesize": P["titulo"], "axes.labelsize": P["font"],
        "xtick.labelsize": P["tick"], "ytick.labelsize": P["tick"],
        "legend.fontsize": P["nota"], "legend.title_fontsize": P["nota"],
        "figure.titlesize": P["titulo"] + 1, "figure.titleweight": "bold",
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": P["titlepad"], "axes.labelpad": P["labelpad"],
        # Pin DejaVu Sans, which ships inside matplotlib itself: Juno would
        # otherwise draw in Noto Sans, which lacks arrows and >=, <= (they come out
        # as boxes) and has other widths than the desktop the layout was checked on.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",

        "legend.labelcolor": TEXTO,
        "legend.handlelength": 2.8, "legend.borderaxespad": 0.6,

        "axes.prop_cycle": cycler(color=UNIVERSAL_COLORS) + cycler(linestyle=LINESTYLES),
        "lines.linewidth": P["lw"], "lines.markersize": 5.5 * P["lw"] / 2.0,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
        "scatter.edgecolors": FONDO,          # a ring of card around each marker

        # Arrows and unfilled patches take their edge from here; matplotlib's
        # default is black, which a graphite card would swallow.
        "patch.edgecolor": NOTA, "patch.facecolor": NOTA,
        "hatch.color": FONDO, "hatch.linewidth": 0.8,   # engraved: card-colored lines read on every fill

        "boxplot.boxprops.color": TEXTO, "boxplot.whiskerprops.color": NOTA,
        "boxplot.capprops.color": NOTA, "boxplot.medianprops.color": TINTA,
        "boxplot.flierprops.color": NOTA, "boxplot.flierprops.markeredgecolor": NOTA,
        "boxplot.flierprops.markerfacecolor": "none",
        "boxplot.meanprops.color": TINTA, "boxplot.meanprops.markerfacecolor": TINTA,
        "boxplot.meanprops.markeredgecolor": TINTA,

        "image.cmap": "mav_seq",
        "errorbar.capsize": 0,

        "figure.dpi": 150, "savefig.dpi": 200 if perfil == "cuaderno" else 300,
        "savefig.bbox": "tight", "savefig.pad_inches": P["pad"],
        "figure.figsize": P["figsize"],
        # Reproducible PDFs: no creation date, so a regenerated figure that did
        # not change keeps its hash (graphs/ is checked by sha256 like data_curso/).
        "pdf.compression": 6,
    }


def apply_style(theme: str | None = None, *, transparent: bool = False,
                perfil: str | None = None) -> None:
    """Set the course rcParams: the ``grafito`` card unless told otherwise.

    Parameters
    ----------
    theme : {"grafito", "papel"}, optional
        Defaults to ``$MAV_NB_TEMA``, else ``grafito``.
    transparent : bool, default False
        Leave the card out (figure and axes backgrounds ``none``).  Only for a
        caller that composites the figure onto a surface it controls; in a
        notebook it brings back the problem the card solves.
    perfil : {"cuaderno", "lamina"}, optional
        Type sizes and default figure size.  ``cuaderno`` (default, or
        ``$MAV_NB_PERFIL``) is the notebook; ``lamina`` draws at slide-slot
        size with 8.5 pt type, for figures that go into the decks.
    """
    global PERFIL
    if isinstance(theme, bool):          # the old signature: apply_style(transparent)
        theme, transparent = None, theme
    try:
        repair_ft2font_enums()
    except Exception:      # a repair that misfires must not cost the style
        pass

    enable_inline_backend()
    _usar_tema(theme or _tema_por_omision())
    PERFIL = perfil or _perfil_por_omision()
    if PERFIL not in PERFILES:
        raise ValueError(f"unknown profile {PERFIL!r}; choose from {sorted(PERFILES)}")
    rc = _rc(transparent, PERFIL)
    plt.rcParams.update({k: v for k, v in rc.items() if k in plt.rcParams})


apply_course_style = apply_style


def palette(n: int, universal: bool = True) -> list[str]:
    """*n* grays in emphasis order: the most prominent first, evenly spaced.

    Up to five stay at or above 3:1 against the card; beyond that the ramp
    reaches down to ink fraction 0.30, so pair it with dashes or labels.
    ``universal=False`` is the old name for the fixed ramp: the first *n* of
    ``GRAYS`` (an evenly spaced ramp once *n* passes eight).
    """
    if n <= 0:
        return []
    if not universal:
        return list(GRAYS[:n]) if n <= len(GRAYS) else [tono(t) for t in np.linspace(1.0, 0.30, n)]
    if n == 1:
        return [tono(1.0)]
    floor = 0.44 if n <= 5 else 0.30
    return [tono(t) for t in np.linspace(1.0, floor, n)]


def styles(n: int) -> list:
    """*n* dash patterns, cycling the six series dashes if needed."""
    if n <= 0:
        return []
    return (LINESTYLES * (n // len(LINESTYLES) + 1))[:n]


def get_color_cycle() -> list[str]:
    """The six series grays, in emphasis order."""
    return list(UNIVERSAL_COLORS)


def get_series_cycler(with_markers: bool = False):
    """Color + dash cycler (plus markers if *with_markers*)."""
    if with_markers:
        return UNIVERSAL_CYCLER
    return cycler(color=UNIVERSAL_COLORS) + cycler(linestyle=LINESTYLES)


def sombrear_recesion(ax=None, x0=None, x1=None, *, facecolor=None, alpha=None,
                      edgecolor="none", lw=0.0, linestyle="-", zorder=0, **kwargs):
    """Shade the span [x0, x1] as an episode: ink at ``RECESION_ALPHA``."""
    if ax is None:
        return None
    if x0 is None and "xmin" in kwargs:
        x0 = kwargs.pop("xmin")
    if x1 is None and "xmax" in kwargs:
        x1 = kwargs.pop("xmax")
    if x0 is None or x1 is None:
        return None
    return ax.axvspan(x0, x1, facecolor=RECESION_HEX if facecolor is None else facecolor,
                      alpha=RECESION_ALPHA if alpha is None else alpha,
                      edgecolor=edgecolor, linewidth=lw, linestyle=linestyle,
                      zorder=zorder, **kwargs)


# ---------------------------------------------------------------------------
# Direct labels: the series' name next to the series, never a legend
# ---------------------------------------------------------------------------
def _lineas_etiquetadas(ax, artistas=None):
    from matplotlib.lines import Line2D
    if artistas is None:
        artistas = ax.get_lines()
    out = []
    for a in artistas:
        if not isinstance(a, Line2D) or not a.get_visible():
            continue
        lab = a.get_label()
        if not lab or str(lab).startswith("_"):
            continue
        x = a.get_xdata()
        y = np.asarray(a.get_ydata(), dtype=float)
        if x.size == 0:
            continue
        out.append((a, x, y))
    return out


def _a_num(ax, x):
    """x data as floats in the axis' units (dates, categories included)."""
    try:
        return np.asarray(ax.xaxis.convert_units(x), dtype=float)
    except Exception:
        x = np.asarray(x)
        if x.dtype.kind in "fiu":
            return x.astype(float)
        return np.arange(len(x), dtype=float)


def _color_legible(color) -> str:
    """The series' gray, but never fainter than NOTA, so the label reads."""
    try:
        t = (_claridad(color) - _L_FONDO) / (_L_TINTA - _L_FONDO)
    except Exception:
        return TEXTO
    return tono(max(float(t), _F_NOTA))


def etiquetar(ax=None, *, artistas=None, donde="auto", dx: float = 5.0,
              fontsize=None, halo: bool = True, sep: float = 1.15,
              nombres: dict | None = None, quitar_leyenda: bool = True,
              fuera: bool | None = None):
    """Label each line next to itself and drop the legend.

    ``donde``: ``"fin"`` puts the label after the last point (outside the axes
    when *fuera*, which defaults to True for the rightmost panel of a row);
    ``"auto"`` chooses, over the right 60 % of each line, the point farthest
    from every other line and writes the label there, above or below the
    line where there is room -- the reading for IRFs that all return to zero.
    A number is an x position in data units.  ``nombres`` renames labels
    (``{"old": "new"}``); ``sep`` is the minimum vertical gap between labels
    in line heights.  Call it last, after the limits are set.
    """
    from matplotlib import patheffects
    ax = ax or plt.gca()
    lineas = _lineas_etiquetadas(ax, artistas)
    if not lineas:
        return []
    if quitar_leyenda and ax.get_legend() is not None:
        ax.get_legend().remove()
    fig = ax.figure
    fig.canvas.draw()                        # limits and transforms are final now
    if fontsize is None:
        fontsize = plt.rcParams["legend.fontsize"]
    fontsize = mpl.font_manager.FontProperties(size=fontsize).get_size_in_points()
    px = fontsize * fig.dpi / 72.0           # one line height in display pixels
    if fuera is None:
        try:
            fuera = ax.get_subplotspec().is_last_col()
        except Exception:
            fuera = True
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tr = ax.transData

    # candidate positions per line
    puestos = []                             # (line, x_data, y_data, ha, va, dx_pts)
    for a, x, y in lineas:
        xn = _a_num(ax, x)
        ok = np.isfinite(xn) & np.isfinite(y)
        if not ok.any():
            continue
        xn, yv = xn[ok], y[ok]
        if donde == "fin" or (donde == "auto" and len(lineas) == 1):
            i = int(np.argmax(xn)) if not fuera else int(np.argmax(xn))
            xd, yd = xn[i], yv[i]
            if fuera:
                puestos.append((a, xd, yd, "left", "center", dx))
            else:
                puestos.append((a, xd, yd, "right", "bottom", 0.0))
        elif donde == "auto":
            lo = x0 + 0.40 * (x1 - x0)
            cand = np.linspace(max(lo, xn.min()), min(x1, xn.max()), 25)
            yc = np.interp(cand, xn, yv)
            best, best_score, best_side = None, -np.inf, 1
            otras = []
            for b, xb, yb in lineas:
                if b is a:
                    continue
                xbn = _a_num(ax, xb); okb = np.isfinite(xbn) & np.isfinite(yb)
                if okb.any():
                    otras.append((xbn[okb], yb[okb]))
            for k, xc in enumerate(cand):
                _, yc_px = tr.transform((xc, yc[k]))
                dist_up, dist_dn = np.inf, np.inf
                for xbn, ybn in otras:
                    if xc < xbn.min() or xc > xbn.max():
                        continue
                    yo = np.interp(xc, xbn, ybn)
                    _, yo_px = tr.transform((xc, yo))
                    d = yo_px - yc_px
                    if d >= 0:
                        dist_up = min(dist_up, d)
                    else:
                        dist_dn = min(dist_dn, -d)
                # room to the edges of the axes too
                _, top_px = tr.transform((xc, y1)); _, bot_px = tr.transform((xc, y0))
                dist_up = min(dist_up, top_px - yc_px)
                dist_dn = min(dist_dn, yc_px - bot_px)
                side = 1 if dist_up >= dist_dn else -1
                score = max(dist_up, dist_dn) + 0.15 * px * k / len(cand)
                if score > best_score:
                    best, best_score, best_side = k, score, side
            xd, yd = cand[best], yc[best]
            puestos.append((a, xd, yd, "center", "bottom" if best_side > 0 else "top",
                            0.0))
        else:                                # a number: x position in data units
            xd = float(donde)
            yd = float(np.interp(xd, xn, yv))
            puestos.append((a, xd, yd, "center", "bottom", 0.0))

    # push overlapping labels apart (in display pixels), keeping the order
    pts = [tr.transform((xd, yd)) for _, xd, yd, _, _, _ in puestos]
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    orden = np.argsort(ys)
    minsep = sep * px
    for _ in range(50):
        moved = False
        for i, j in zip(orden[:-1], orden[1:]):
            same_col = abs(xs[i] - xs[j]) < 6.0 * px   # only neighbours in x collide
            if same_col and ys[j] - ys[i] < minsep:
                d = (minsep - (ys[j] - ys[i])) / 2.0
                ys[i] -= d; ys[j] += d; moved = True
        if not moved:
            break
    inv = tr.inverted()
    textos = []
    efectos = [patheffects.withStroke(linewidth=3.0, foreground=FONDO)] if halo else None
    for k, (a, xd, yd, ha, va, ddx) in enumerate(puestos):
        xk, yk = inv.transform((xs[k], ys[k]))
        lab = str(a.get_label())
        if nombres and lab in nombres:
            lab = nombres[lab]
        dy = 0.0
        if va == "bottom":
            dy = 2.0
        elif va == "top":
            dy = -2.0
        t = ax.annotate(lab, xy=(xk, yk), xytext=(ddx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=fontsize, color=_color_legible(a.get_color()),
                        annotation_clip=False, zorder=a.get_zorder() + 0.5)
        if efectos and not (fuera and ha == "left"):
            t.set_path_effects(efectos)
        textos.append(t)
    return textos


def etiquetar_barras(ax=None, *, fmt: str = "{:.1f}", fontsize=None, dentro: bool = False,
                     contenedores=None, color=None, pad: float = 2.0):
    """Write each bar's value at its end (or inside it, in card color)."""
    ax = ax or plt.gca()
    if fontsize is None:
        fontsize = plt.rcParams["xtick.labelsize"]
    out = []
    for cont in (contenedores or ax.containers):
        try:
            labels = [fmt.format(v) if np.isfinite(v) else "" for v in cont.datavalues]
        except AttributeError:
            continue
        col = FONDO if dentro else (color or TEXTO)
        out += ax.bar_label(cont, labels=labels, fontsize=fontsize, color=col,
                            label_type="center" if dentro else "edge", padding=pad)
    return out


# ---------------------------------------------------------------------------
# Export: the same figure, at slide-slot size, as a vector PDF
# ---------------------------------------------------------------------------
def _carpeta_exportacion(carpeta=None):
    carpeta = carpeta or os.environ.get("MAV_EXPORTAR_FIGURAS", "").strip()
    return Path(carpeta) if carpeta else None


def exportar(fig, nombre: str, *, tamano=None, carpeta=None, titulo: bool = True,
             png: bool = True) -> Path | None:
    """Save *fig* as ``<carpeta>/<nombre>.pdf`` (and a PNG preview).

    Does nothing unless *carpeta* or ``$MAV_EXPORTAR_FIGURAS`` names a folder,
    so a notebook can carry the call without ever writing files when a student
    runs it.  ``tamano`` is a ``LAMINA`` key or a pair of inches: the figure is
    resized before saving (its type stays at the profile's point size, so on
    the slide it reads at that size).  ``titulo=False`` drops the axes titles
    and suptitle, for a slot whose frame already says what the figure shows.
    """
    carpeta = _carpeta_exportacion(carpeta)
    if carpeta is None:
        return None
    carpeta.mkdir(parents=True, exist_ok=True)
    if tamano is not None:
        if isinstance(tamano, str):
            tamano = LAMINA.get(tamano) or TAMANO[tamano]
        actual = tuple(fig.get_size_inches())
        # Only resize when the figure was not already built at that size:
        # resizing after the fact leaves annotations where the old axes were.
        if max(abs(actual[0] - tamano[0]), abs(actual[1] - tamano[1])) > 0.05:
            fig.set_size_inches(*tamano, forward=True)
    if not titulo:
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        for ax in fig.axes:
            ax.set_title("")
    nombre = nombre[:-4] if nombre.endswith(".pdf") else nombre
    out = carpeta / f"{nombre}.pdf"
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    if png:   # same bbox as the PDF, so the preview is what the deck will show
        fig.savefig(carpeta / f"{nombre}.png", bbox_inches="tight",
                    pad_inches=plt.rcParams["savefig.pad_inches"], dpi=200)
    return out


def exportar_abiertas(nombre: str, **kwargs) -> list:
    """Export every open figure under *nombre* (``_a``, ``_b`` ... if several)."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    out = []
    for k, fig in enumerate(figs):
        sufijo = "" if len(figs) == 1 else "_" + "abcdefghij"[k]
        out.append(exportar(fig, nombre + sufijo, **kwargs))
    return out


# ---------------------------------------------------------------------------
# WCAG arithmetic, kept for the tests and for anyone checking a new token
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_code: str) -> np.ndarray:
    """'#RRGGBB' (or '#RGB') -> RGB array in [0, 1]."""
    h = hex_code.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def srgb_to_linear(rgb) -> np.ndarray:
    """sRGB components in [0, 1] -> linear RGB."""
    return _lineal(rgb)


def relative_luminance(color) -> float:
    """WCAG 2 relative luminance of a hex string, RGB triple or a luminance."""
    if isinstance(color, (int, float)) and not isinstance(color, bool):
        return float(color)
    if isinstance(color, str):
        rgb = mpl.colors.to_rgb(color)
    else:
        arr = np.asarray(color, dtype=float)
        rgb = (arr / 255.0 if np.any(arr > 1.0) else arr)[:3]
    r, g, b = _lineal(rgb)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(c1, c2) -> float:
    """WCAG 2 contrast ratio, always >= 1."""
    l1, l2 = relative_luminance(c1), relative_luminance(c2)
    return float((max(l1, l2) + 0.05) / (min(l1, l2) + 0.05))


def is_dual_compliant(color, min_cr: float = 3.0) -> bool:
    """True if *color* reaches *min_cr* against both #000000 and #FFFFFF."""
    return (contrast_ratio(color, "#000000") >= min_cr
            and contrast_ratio(color, "#FFFFFF") >= min_cr)


universal_palette = palette
universal_styles = styles
COLOR_NAMES = ["ink", "light gray", "mid gray", "pale (dotted)", "gray (long dash)",
               "light gray (dash-dot-dot)"]

try:                                   # the simulator lives with the slide style
    from mav_plot_style import CVDSimulator
except Exception:
    try:
        _root = str(Path(__file__).resolve().parent.parent)
        if _root not in sys.path:
            sys.path.append(_root)
        from mav_plot_style import CVDSimulator
    except Exception:
        class CVDSimulator:
            """Placeholder: the simulator needs ``mav_plot_style.py``."""

            def __init__(self, *args, **kwargs):
                raise NotImplementedError("CVDSimulator requires mav_plot_style.py")


# Tokens exist from import on, so a cell that reads _nbstyle.TINTA before
# apply_style() still gets a color.
PERFIL = _perfil_por_omision()
_usar_tema(_tema_por_omision())
