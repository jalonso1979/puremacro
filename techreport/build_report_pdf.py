"""Build puremacro_technical_report.pdf from puremacro_technical_report.md.

The Markdown file is the source of truth. This script converts its body with
pandoc, wraps it in the report's LaTeX house style (navy headings, boxed
call-outs, running heads) with a hand-set title page, and compiles with
LuaLaTeX so the few symbols outside Latin Modern (★ ● ◐ Σ, superscripts) fall
back to DejaVu Serif. Relative links are pointed at the tagged release on
GitHub so they work in the PDF.

Usage (from the repository root):

    python techreport/build_report_pdf.py [--tag v4.5.0]

Requires pandoc >= 3 and a TeX distribution with lualatex.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "puremacro_technical_report.md"
TEX = HERE / "puremacro_technical_report.tex"
REPO = "https://github.com/jalonso1979/puremacro"

PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{fontspec}
\directlua{luaotfload.add_fallback("pmfallback", {"DejaVuSerif.ttf:mode=node;", "DejaVuSans.ttf:mode=node;"})}
\setmainfont{Latin Modern Roman}[RawFeature={fallback=pmfallback}]
\setsansfont{Latin Modern Sans}[RawFeature={fallback=pmfallback}]
\setmonofont{Latin Modern Mono}[RawFeature={fallback=pmfallback}, Scale=0.95]
\usepackage{booktabs,longtable,array,calc,multirow}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage[most]{tcolorbox}
\usepackage{xurl}
\usepackage[bookmarks=true,bookmarksnumbered=true,colorlinks=true,linkcolor=navy,
            citecolor=navy,urlcolor=forest,pdfauthor={Jorge Alonso Ortiz, Claude, Codex, Antigravity},
            pdftitle={puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for
            Quantitative Macroeconomics and Macroeconometrics}]{hyperref}
\definecolor{navy}{HTML}{1E3D59}
\definecolor{forest}{HTML}{17B978}
\definecolor{slate}{HTML}{334155}
\definecolor{lightslate}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\titleformat{\section}{\Large\bfseries\color{navy}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\large\bfseries\color{navy}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{navy}}{\thesubsubsection}{1em}{}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyhead[L]{\small\scshape puremacro: Technical Report v4.5}
\fancyhead[R]{\small\scshape Alonso Ortiz, Claude, Codex \& Antigravity}
\fancyfoot[C]{\thepage}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\captionsetup{font=small,labelfont={bf,color=navy}}
% Block quotes in the source are the report's call-out boxes.
\renewenvironment{quote}{\begin{tcolorbox}[colback=lightslate,colframe=navy,boxrule=0.8pt,
  arc=3mm,left=10pt,right=10pt,top=6pt,bottom=6pt,breakable]}{\end{tcolorbox}}
% Helpers that pandoc's LaTeX writer expects.
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\newcounter{none} % pandoc uses it for tables without a caption
% Long code identifiers may break after an underscore.
\let\pmunderscore\_
\renewcommand{\_}{\pmunderscore\allowbreak}
\makeatletter
\newsavebox\pandoc@box
\newcommand*\pandocbounded[1]{%
  \sbox\pandoc@box{#1}%
  \Gscale@div\@tempa{\textheight}{\dimexpr\ht\pandoc@box+\dp\pandoc@box\relax}%
  \Gscale@div\@tempb{\linewidth}{\wd\pandoc@box}%
  \ifdim\@tempb\p@<\@tempa\p@\let\@tempa\@tempb\fi
  \ifdim\@tempa\p@<\p@\scalebox{\@tempa}{\usebox\pandoc@box}%
  \else\usebox{\pandoc@box}\fi}
\def\fps@figure{htbp}
\makeatother
"""

TITLEPAGE = r"""\begin{titlepage}
\begin{center}
\vspace*{-0.6cm}
{\huge\bfseries\color{navy} puremacro: A Unified, Dependency-Minimal Scientific-Python Engine for Quantitative Macroeconomics and Macroeconometrics\par}
\vspace{0.35cm}
{\large\scshape Architecture, Algorithmic Foundations, Numerical Validation, and Browser-Native Reproducibility\par}
\vspace{0.6cm}
{\Large Jorge Alonso Ortiz$^{1,\star}$ \quad Claude$^{2}$ \quad Codex$^{3}$ \quad Antigravity$^{4}$\par}
\vspace{0.4cm}
{\small
$^{1}$Department of Economics, Instituto Tecnológico Autónomo de México (ITAM), Mexico City\\
\texttt{jorge.alonso@itam.mx} $\cdot$ ORCID \href{https://orcid.org/0000-0002-5941-9928}{0000-0002-5941-9928}\\[2pt]
$^{2}$AI coding agent, Anthropic (Claude Opus 5, Opus 5.5, Fable 5 and Fable 5.1)\\
$^{3}$AI coding agent, OpenAI \quad $^{4}$AI coding agent, Google\\[2pt]
$^{\star}$Corresponding author; the human author takes sole responsibility for the content.\par}
\vspace{0.45cm}
{\large Technical Report \& Working Paper v4.5 $\cdot$ 3 October 2026\par}
\vspace{0.25cm}
{\small \url{REPO} $\cdot$ \url{https://jalonso1979.github.io/puremacro/}\par}
\end{center}
\vspace{0.3cm}
\begin{tcolorbox}[colback=lightslate,colframe=navy,boxrule=0.8pt,arc=3mm,title={\bfseries Abstract},
  fonttitle=\color{white}\bfseries,colbacktitle=navy,fontupper=\small]
ABSTRACT
\end{tcolorbox}
\end{titlepage}
"""

NUMBERED = re.compile(r"^(#{2,3}) (\d+(?:\.\d+)?)\.?\s+(.*)$")


def preprocess(md: str, tag: str) -> tuple[str, str]:
    """Return (abstract_markdown, body_markdown) ready for pandoc."""
    abstract = re.search(r"^### Abstract\n(.+?)\n\n", md, flags=re.S | re.M).group(1).strip()
    body = md[md.index("> **Validation scope"):]
    body = re.sub(r"^### Abstract\n.+?\n\n", "", body, count=1, flags=re.S | re.M)
    # Thematic breaks separate sections in the Markdown; headings do it in LaTeX.
    body = re.sub(r"^---\s*$", "", body, flags=re.M)
    # Repository-relative links resolve against the tagged release.
    blob = f"{REPO}/blob/{tag}/"
    body = re.sub(r"\]\(\.\./", "](" + blob, body)
    abstract = re.sub(r"\]\(\.\./", "](" + blob, abstract)
    # "![alt](img)\n\n*Figure N: caption*" becomes one captioned figure.
    body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)\s*\n\s*\n\*Figure \d+:\s*(.+?)\*[ \t]*\n",
                  lambda m: f"![{m.group(2)}]({m.group(1)})\n", body, flags=re.S)
    lines = []
    for line in body.splitlines():
        m = NUMBERED.match(line)
        if m:  # "## 3. Title" -> numbered section, "### 3.1 Title" -> subsection
            level = "#" if len(m.group(1)) == 2 else "##"
            lines.append(f"{level} {m.group(3)}")
        elif line.startswith("### "):  # unnumbered back matter and front matter
            lines.append(f"# {line[4:].strip()} {{.unnumbered}}")
        elif line.startswith("## "):
            lines.append(f"# {line[3:].strip()}")
        else:
            lines.append(line)
    return abstract, "\n".join(lines) + "\n"


def pandoc(markdown: str) -> str:
    return subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_dollars+gfm_auto_identifiers",
         "-t", "latex", "--wrap=preserve"],
        input=markdown, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v4.5.0", help="release tag that relative links point to")
    args = parser.parse_args()
    for tool in ("pandoc", "lualatex"):
        if shutil.which(tool) is None:
            print(f"{tool} not found on PATH", file=sys.stderr)
            return 1
    md = SOURCE.read_text(encoding="utf-8")
    abstract_md, body_md = preprocess(md, args.tag)
    title = TITLEPAGE.replace("REPO", REPO).replace("ABSTRACT", pandoc(abstract_md).strip())
    body = pandoc(body_md)
    # The validation-scope and what-changed boxes open the document after the
    # title page; the graphical abstract and the contents follow.
    marker = "\\section*{Graphical Abstract}"
    if marker in body:
        head, tail = body.split(marker, 1)
        body = head + "\\clearpage\n" + marker + tail
    body = body.replace("\\section*{Graphical Abstract}",
                        "\\section*{Graphical Abstract}\\addcontentsline{toc}{section}{Graphical Abstract}", 1)
    first = body.index("\\section{")
    body = body[:first] + "\\clearpage\n\\tableofcontents\n\\clearpage\n" + body[first:]
    tex = PREAMBLE + "\\begin{document}\n" + title + body + "\\end{document}\n"
    TEX.write_text(tex, encoding="utf-8")
    for _ in range(2):  # second pass resolves the table of contents
        run = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", TEX.name],
                             cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            print(run.stdout[-3000:], file=sys.stderr)
            return run.returncode
    print(f"wrote {TEX.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
