"""
Script to generate publication-grade figures for the puremacro technical report / working paper.
Outputs high-resolution (300 DPI) schematics using matplotlib with professional styling.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color Palette: Academic / Modern Slate
C_PRIMARY = "#1e3d59"       # Deep Navy
C_SECONDARY = "#17b978"     # Emerald Green
C_ACCENT = "#ff6e40"        # Coral / Amber
C_HIGHLIGHT = "#438a5e"     # Forest / Jade
C_BG_CARD = "#f8f9fa"       # Off-white card background
C_BORDER = "#d1d5db"        # Slate light border
C_TEXT_DARK = "#111827"     # Dark Slate
C_TEXT_MUTED = "#4b5563"    # Medium Gray
C_BLUE_LIGHT = "#e8f0fe"    # Pale Blue
C_GREEN_LIGHT = "#e6f4ea"   # Pale Green
C_AMBER_LIGHT = "#fef7e0"   # Pale Amber
C_PURPLE_LIGHT = "#f3e8fd"  # Pale Purple
C_PURPLE = "#7b1fa2"        # Purple


def draw_card(ax, x, y, w, h, title, subtitle="", bg_color=C_BG_CARD, border_color=C_BORDER, title_color=C_PRIMARY):
    """Draws a rounded card container with title and optional subtitle."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + 0.02 * w, y + h - 0.06 * h, title,
                fontsize=11, fontweight="bold", color=title_color, zorder=3, va="top")
    if subtitle:
        ax.text(x + 0.02 * w, y + h - 0.12 * h, subtitle,
                fontsize=8.5, fontstyle="italic", color=C_TEXT_MUTED, zorder=3, va="top")


def generate_graphical_abstract():
    """Generates the Graphical Abstract required by scientific-writing guidelines."""
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Header Banner
    banner = patches.Rectangle((0, 0.90), 1, 0.10, facecolor=C_PRIMARY, zorder=1)
    ax.add_patch(banner)
    ax.text(0.5, 0.955, "puremacro: Universal Quantitative Macroeconomics & Econometrics",
            ha="center", va="center", color="#ffffff", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.92, "Pure Scientific-Python Stack  •  Zero C-Extensions  •  Workstation to WebAssembly Portability  •  Self-Verifying Oracles",
            ha="center", va="center", color="#e0e7ff", fontsize=9.5)

    # Three Main Architectural Columns:
    # Col 1: Foundations & Architecture (0.03 to 0.28)
    # Col 2: Unified Methodology Engines (0.31 to 0.69)
    # Col 3: Verification & Zero-Friction Delivery (0.72 to 0.97)

    # --- COLUMN 1: ARCHITECTURE & RUNTIME ---
    draw_card(ax, 0.02, 0.04, 0.27, 0.83, "1. Architecture & Invariants", "Pure Scientific-Python Contract",
              bg_color="#f9fafb", border_color="#cbd5e1", title_color=C_PRIMARY)

    # Subcards in Col 1
    # 1.1 Strict Dependency Invariant
    draw_card(ax, 0.035, 0.58, 0.24, 0.22, "Strict Four-Package Core", "", bg_color=C_BLUE_LIGHT, border_color="#bfdbfe")
    ax.text(0.045, 0.74, "• Core: NumPy, SciPy, pandas, Matplotlib\n• Requests only in data ingestion layer\n• ZERO custom C/C++/Rust extensions\n• Strictly no Numba or Cython in core\n• Run-time sandboxes prevent leakages",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # 1.2 Dual Execution Runtime
    draw_card(ax, 0.035, 0.32, 0.24, 0.23, "Dual Multi-Target Runtime", "", bg_color=C_GREEN_LIGHT, border_color="#bbf7d0")
    ax.text(0.045, 0.48, "• CPython Workstations (Linux, macOS, Win)\n• WebAssembly / Pyodide (In-Browser)\n• iPad & Tablet Native via Juno.sh\n• JupyterLite Client-Side Zero-Install\n• Transparent Google Colab RPC Offload",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # 1.3 Immutable Results
    draw_card(ax, 0.035, 0.06, 0.24, 0.23, "Immutable Unified Results", "", bg_color=C_AMBER_LIGHT, border_color="#fde68a")
    ax.text(0.045, 0.22, "• Frozen Dataclass result containers\n• Standardized API: lags, horizon, ci\n• Direct .to_latex() & .to_typst() export\n• Integrated publication-grade .plot()\n• Structured economic diagnostics",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # --- COLUMN 2: METHODOLOGICAL SUITE ---
    draw_card(ax, 0.31, 0.04, 0.38, 0.83, "2. Unified Quantitative Engine", "Four Computational Pillars",
              bg_color="#f8fafc", border_color="#cbd5e1", title_color=C_PRIMARY)

    # Pillar A: Macroeconometrics & Causal Inference
    draw_card(ax, 0.325, 0.65, 0.35, 0.17, "Macroeconometrics & Time Series", "", bg_color="#ffffff", border_color="#93c5fd")
    ax.text(0.335, 0.77, "SVAR: Cholesky, Blanchard-Quah, Uhlig Sign, Rubio-Ramirez Sign/Zero\nNarrative SVAR: Antolin-Diaz & Rubio-Ramirez historical restrictions\nProxy SVAR & LP-IV: Mertens-Ravn, Stock-Watson, Olea-Pflueger\nLocal Projections: Jordà HAC, Lag-Augmented (PMW), Driscoll-Kraay Panel\nModern Causal: Staggered DiD (CS, SA), Synthetic Control, SDID, DML",
            fontsize=7.8, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Pillar B: Dynamic General Equilibrium (DSGE)
    draw_card(ax, 0.325, 0.45, 0.35, 0.17, "DSGE & Exact Analytical Gradients", "", bg_color="#ffffff", border_color="#86efac")
    ax.text(0.335, 0.57, "Dynare .mod Parser: Zero MATLAB/Octave dependency\nFirst-Order Perturbation: Klein (2000) QZ generalized Schur solver\nSecond-Order Perturbation: Kim et al. pruning, cross-terms, oo_.dr parity\nBayesian NUTS Estimation: Hamiltonian MC via No-U-Turn Sampler\nExact Analytical Gradients: Implicit Sylvester derivatives of Kalman score",
            fontsize=7.8, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Pillar C: Heterogeneous-Agent & Dynamic Programming
    draw_card(ax, 0.325, 0.25, 0.35, 0.17, "Heterogeneous Agents & Projections", "", bg_color="#ffffff", border_color="#fed7aa")
    ax.text(0.335, 0.37, "Sequence-Space Jacobian (SSJ): Auclert et al. bridge in Dynare .mod\nHousehold Solvers: Value Function Iteration, EGM, DCEGM, Schumaker\nContinuous-Time Macro: Implicit upwind HJB + adjoint KFE (Achdou et al.)\nOrthogonal Projections: Chebyshev collocation & Smolyak sparse grids\nFinite Element Galerkin: Fischer-Burmeister NCP borrowing constraints",
            fontsize=7.8, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Pillar D: Spatial, Trade & Integrated Climate
    draw_card(ax, 0.325, 0.06, 0.35, 0.16, "Spatial, Trade & Climate General Equilibrium", "", bg_color="#ffffff", border_color="#e9d5ff")
    ax.text(0.335, 0.17, "Quantitative Spatial GE: Allen-Arkolakis gravity topography equilibrium\nQuantitative Trade: Caliendo-Parro exact hat algebra with multi-sector IO\nFlexible CGE Engine: Nested CES technology & Atkeson-Burstein markups\nClimate Economics: Nordhaus DICE & Golosov et al. optimal taxation",
            fontsize=7.8, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # --- COLUMN 3: VERIFICATION & ADOPTION ---
    draw_card(ax, 0.71, 0.04, 0.27, 0.83, "3. Verification & Impact", "Self-Certifying Evidence",
              bg_color="#f9fafb", border_color="#cbd5e1", title_color=C_PRIMARY)

    # 3.1 Oracle Hardening
    draw_card(ax, 0.725, 0.58, 0.24, 0.22, "The Oracle Validation Gallery", "", bg_color=C_BLUE_LIGHT, border_color="#bfdbfe")
    ax.text(0.735, 0.74, "• 107 Checks across 15 Subsystems (100% Pass)\n• Oracles: statsmodels, arch, linearmodels\n• Offline serialization of oracle references\n• ZERO oracle run-time import footprint\n• Numerical accuracy verified to 10^-15",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # 3.2 Empirical Landmark Replications
    draw_card(ax, 0.725, 0.32, 0.24, 0.23, "Empirical Replications", "", bg_color=C_GREEN_LIGHT, border_color="#bbf7d0")
    ax.text(0.735, 0.48, "• Card (1995) Returns to Schooling (IV)\n• Mroz (1987) Female Labor Participation\n• Romer & Romer (2010) Tax Shocks (Narrative)\n• Huggett (1993) & Aiyagari (1994) Precautionary\n  saving interest rate bounds verified",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # 3.3 Classroom & Open Science
    draw_card(ax, 0.725, 0.06, 0.24, 0.23, "Classroom & Web Deployment", "", bg_color=C_AMBER_LIGHT, border_color="#fde68a")
    ax.text(0.735, 0.22, "• Adopted in ITAM Macroeconomía Avanzada\n• 60 Bilingual (EN/ES) Notebook Pairs\n• Interactive JupyterLite WebAssembly site\n• 755 modules, 240k LOC, 14.3k tests\n• Open Source (MIT) on PyPI",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.4)

    # Arrows between columns
    ax.annotate("", xy=(0.31, 0.45), xytext=(0.28, 0.45),
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.5))
    ax.annotate("", xy=(0.71, 0.45), xytext=(0.68, 0.45),
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.5))

    fig.savefig(OUTPUT_DIR / "graphical_abstract.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated graphical_abstract.png")


def generate_architecture_oracle_figure():
    """Generates the Architecture & Oracle Verification schematic."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(0.5, 0.96, "The puremacro Architecture & The Oracle Hardening Loop",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(0.5, 0.92, "Decoupling Verification Rigor from Run-Time Dependency Graph",
            ha="center", va="center", fontsize=10, fontstyle="italic", color=C_TEXT_MUTED)

    # Left Container: Development / Test Environment (The Oracle Factory)
    draw_card(ax, 0.03, 0.06, 0.42, 0.81, "Offline Development & Oracle Generation",
              "Developer Machine / Nightly Drift CI (Heavy Dependencies Permitted)",
              bg_color="#f8fafc", border_color="#94a3b8", title_color=C_PRIMARY)

    draw_card(ax, 0.05, 0.61, 0.38, 0.18, "External Reference Implementations (Oracles)", "", bg_color="#fee2e2", border_color="#fca5a5")
    ax.text(0.065, 0.73, "• statsmodels: OLS, VAR, VECM, State Space, ARMA\n• arch: ARCH, GARCH, EGARCH, GJR-GARCH, HAR-RV\n• linearmodels: Panel OLS, 2SLS/IV, Dynamic Panel GMM\n• esda / PySAL: Spatial weights, Moran's I, spatial lag/error",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    draw_card(ax, 0.05, 0.36, 0.38, 0.19, "Oracle Extraction Harness (Offline Scripts)", "", bg_color="#fef3c7", border_color="#fde047")
    ax.text(0.065, 0.49, "1. Evaluate standardized benchmark datasets (e.g. macro panels)\n2. Extract parameter estimates, covariance matrices, residuals, IRFs\n3. Serialize ground-truth outputs into compact package fixtures\n4. Continuous drift tests check if upstream package versions change",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    draw_card(ax, 0.05, 0.10, 0.38, 0.20, "Frozen Oracle Test Fixtures", "", bg_color="#e0e7ff", border_color="#a5b4fc")
    ax.text(0.065, 0.24, "• High-precision JSON & NumPy binary fixtures (.npz)\n• Stored in puremacro/validation/fixtures/\n• Shipped directly inside the pure-Python wheel (< 2 MB total)\n• Zero external dependency overhead for downstream users",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Connecting Flow Arrow
    ax.annotate("", xy=(0.24, 0.58), xytext=(0.24, 0.61),
                arrowprops=dict(arrowstyle="->", color=C_TEXT_MUTED, lw=1.5))
    ax.annotate("", xy=(0.24, 0.33), xytext=(0.24, 0.36),
                arrowprops=dict(arrowstyle="->", color=C_TEXT_MUTED, lw=1.5))

    # Center Flow Arrow: Shipping into the Pure Library
    ax.annotate("", xy=(0.52, 0.50), xytext=(0.46, 0.50),
                arrowprops=dict(arrowstyle="->", color=C_SECONDARY, lw=3.0))
    ax.text(0.49, 0.53, "Ships Fixtures\nInto Wheel", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=C_SECONDARY)

    # Right Container: Production / Client Environment (The Pure Invariant)
    draw_card(ax, 0.52, 0.06, 0.45, 0.81, "Client Runtime & Continuous Verification",
              "Zero-Compiled Dependencies  •  Universal Portability",
              bg_color="#f0fdf4", border_color="#86efac", title_color=C_HIGHLIGHT)

    draw_card(ax, 0.54, 0.64, 0.41, 0.16, "Strict Import Invariant Boundary", "", bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.555, 0.75, "Enforced by automated CI subprocess sweeps:\n• statsmodels, arch, linearmodels, numba FORBIDDEN in sys.modules\n• Pure NumPy, SciPy, pandas, Matplotlib ONLY\n• Sandboxed execution guarantees pure Python execution",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    draw_card(ax, 0.54, 0.37, 0.41, 0.21, "puremacro.validation.scorecard()", "", bg_color=C_BLUE_LIGHT, border_color="#93c5fd")
    ax.text(0.555, 0.52, "• 107 Validation checks across 15 economic subsystems\n• 19 External reference checks (vs stored fixtures, diff < 10^-14)\n• 29 Analytical checks (closed-form identities, Euler residuals)\n• 59 Internal consistency checks (cross-algorithm & simulated recovery)\n• Executes in < 45 seconds on standard laptop or browser",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    draw_card(ax, 0.54, 0.10, 0.41, 0.21, "Deployment Targets (Zero Installation Friction)", "", bg_color="#ffffff", border_color="#bbf7d0")
    ax.text(0.555, 0.25, "• Local Workstations: pip install puremacro (Linux, macOS, Win)\n• Web Browsers: JupyterLite & Pyodide (Wasm) zero-install\n• Mobile / Tablets: Full offline capability on Juno.sh for iPad\n• Compute Offload: One-click export to Google Colab for large MCMC",
            fontsize=8.5, color=C_TEXT_DARK, va="top", linespacing=1.35)

    fig.savefig(OUTPUT_DIR / "fig_architecture_oracle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_architecture_oracle.png")


def generate_svar_identification_figure():
    """Generates the Structural Macroeconometrics & Identification schematic."""
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(0.5, 0.96, "Structural Macroeconometric Identification Spectrum in puremacro",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(0.5, 0.92, "From Canonical Restrictions to Narrative Priors, External Proxies, and Local Projections",
            ha="center", va="center", fontsize=10, fontstyle="italic", color=C_TEXT_MUTED)

    # Core Reduced Form Box at Top
    draw_card(ax, 0.25, 0.77, 0.50, 0.12, "Reduced-Form System:  Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + u_t,   E[u_t u_t'] = Σ",
              "Structural Shocks: u_t = B_0^{-1} ε_t,   E[ε_t ε_t'] = I_K,   B_0^{-1} (B_0^{-1})' = Σ",
              bg_color="#e0e7ff", border_color="#818cf8", title_color=C_PRIMARY)

    # Five Structural Identification Cards
    # Card 1: Cholesky Recursive
    draw_card(ax, 0.03, 0.40, 0.28, 0.32, "1. Recursive (Cholesky)", "Sims (1980)",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.045, 0.65, "• Identification: Lower-triangular B_0^{-1} = P\n  where Σ = P P' (Cholesky factor)\n• Restriction: Exact zero causal ordering\n  Variable j cannot affect variable i on\n  impact if j > i.\n• Pure-Python: safe_cholesky with\n  ridge regularization if ill-conditioned\n• Limitations: Strict ordering rarely\n  supported by modern theory.",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 2: Blanchard-Quah Long-Run
    draw_card(ax, 0.36, 0.40, 0.28, 0.32, "2. Long-Run Neutrality", "Blanchard & Quah (1989)",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.375, 0.65, "• Identification: Long-run multiplier matrix\n  C(1) = (I - A_1 - ... - A_p)^{-1} B_0^{-1}\n  is constrained to be lower triangular.\n• Economic restriction: Demand shocks have\n  zero long-run impact on output level.\n• Algorithm: Spectral factorization of\n  long-run covariance C(1) Σ C(1)'\n• Fully verified against original AER data.",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 3: Sign & Zero Restrictions
    draw_card(ax, 0.69, 0.40, 0.28, 0.32, "3. Sign & Sign-Zero Bounds", "Uhlig (2005), Rubio-Ramirez et al. (2010)",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.705, 0.65, "• Identification: Set-identification via\n  orthogonal rotations Q ∈ O(K), B_0^{-1} = P Q\n• Sign constraints: IRF_h(k, j) ≥ 0 or ≤ 0\n• Sign-zero: Exact impact zeros via QR\n  null-space projection (Rubio-Ramirez 2010)\n• Bayesian Haar prior over O(K)\n• Median, HPDI, and IRF fan charts.",
            fontsize=8.2, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 4: Narrative Restrictions (Antolin-Diaz & Rubio-Ramirez 2018)
    draw_card(ax, 0.03, 0.04, 0.45, 0.32, "4. Narrative Sign Restrictions", "Antolin-Diaz & Rubio-Ramirez (AER 2018)",
              bg_color="#f0fdf4", border_color="#86efac")
    ax.text(0.045, 0.29, "• Conditions posterior draws on historical events and episode narratives:\n  (a) Shock sign restriction: Structural shock ε_{j, t*} > 0 during known historical episode.\n  (b) Historical contribution restriction: Shock j is the overwhelming driver of variable i at t*.\n• Algorithmic implementation: Importance sampling algorithm updating Haar prior weights.\n• Tested on oil shocks (1973, 1979, 1990) and Volcker disinflation (1979:10-1982:10).",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 5: Proxy SVAR & Local Projections
    draw_card(ax, 0.52, 0.04, 0.45, 0.32, "5. External Instruments & Local Projections", "Mertens-Ravn (2013), Stock-Watson (2018), Jorda (2005)",
              bg_color="#fef7ee", border_color="#fed7aa")
    ax.text(0.535, 0.29, "• Proxy SVAR (SVAR-IV): External instrument z_t correlated with shock ε_{1,t}, orthogonal to ε_{2:K,t}.\n• Weak Instrument Robustness: Anderson-Rubin (1949) & Montiel Olea-Pflueger (2013) F-tests.\n• Local Projections (LP): Direct multi-step regression y_{t+h} = β_h x_t + controls + e_{t+h}.\n• Unified LP Suite: Single-country LP-HAC, Lag-Augmented (PMW 2021), Panel LP with\n  Driscoll-Kraay SE, and State-Dependent / Smooth Transition Local Projections.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Connection lines from top to cards
    ax.annotate("", xy=(0.17, 0.72), xytext=(0.35, 0.77), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.5))
    ax.annotate("", xy=(0.50, 0.72), xytext=(0.50, 0.77), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.5))
    ax.annotate("", xy=(0.83, 0.72), xytext=(0.65, 0.77), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.5))

    fig.savefig(OUTPUT_DIR / "fig_svar_identification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_svar_identification.png")


def generate_dsge_ssj_figure():
    """Generates the DSGE, Bayesian NUTS, and Sequence-Space Jacobian schematic."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(0.5, 0.96, "The Dynamic General Equilibrium (DSGE) & Sequence-Space Workflow",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(0.5, 0.92, "From Native Dynare .mod Parsing to Higher-Order Pruning, NUTS Estimation, and HANK Coupling",
            ha="center", va="center", fontsize=10, fontstyle="italic", color=C_TEXT_MUTED)

    # Stage 1: Input & Parsing
    draw_card(ax, 0.03, 0.52, 0.28, 0.36, "1. Native Dynare .mod Engine", "Zero External Runtimes",
              bg_color="#f8fafc", border_color="#94a3b8")
    ax.text(0.045, 0.81, "• Pure-Python lexer & recursive parser\n• Parses var, varexo, parameters,\n  model, initval, shocks blocks\n• Supports arbitrary forward/backward leads\n  (e.g. c(+1), k(-1))\n• Complex-step numerical differentiation:\n  Machine-precision Jacobians without\n  symbolic toolboxes or AD overhead\n• Automatic steady-state finding.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Stage 2: Perturbation Solvers
    draw_card(ax, 0.36, 0.52, 0.28, 0.36, "2. Perturbation & Decision Rules", "Dynare oo_.dr Exact Parity",
              bg_color="#ffffff", border_color="#93c5fd")
    ax.text(0.375, 0.81, "• 1st-Order: Klein (2000) QZ solver\n  Checks Blanchard-Kahn conditions\n  (determinate, indeterminate, explosive)\n• 2nd-Order: State-space Hessian expansion\n• Pruned Perturbation (Kim et al. 2008):\n  Eliminates spurious explosive dynamics\n• Cross-derivatives (g_xu, g_uu) & risk\n  corrections (g_σσ) computed analytically\n• Exact parity with Dynare oo_.dr structures.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Stage 3: Bayesian NUTS Estimation
    draw_card(ax, 0.69, 0.52, 0.28, 0.36, "3. Bayesian NUTS Estimation", "Exact Analytical Gradients",
              bg_color="#ffffff", border_color="#86efac")
    ax.text(0.705, 0.81, "• Kalman Filter Likelihood Evaluation\n• Exact Analytical Score Gradient (∇_θ ln L):\n  Implicit differentiation of continuous-time\n  Sylvester and Lyapunov equations\n  (zero finite-difference truncation error!)\n• No-U-Turn Sampler (NUTS):\n  Hamiltonian MC with dual-averaging step\n• Fast convergence on complex macro posteriors.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Bottom Stage: Sequence-Space Jacobian (SSJ) HANK Bridge
    draw_card(ax, 0.03, 0.06, 0.94, 0.40, "4. The Heterogeneous-Agent Sequence-Space Jacobian (SSJ) Bridge",
              "Unifying Micro Incomplete Markets with Macro General Equilibrium (Auclert et al. 2021)",
              bg_color="#f0fdf4", border_color="#4ade80", title_color=C_PRIMARY)

    # Subcolumns inside Stage 4
    # Sub 4.1: Household Block
    draw_card(ax, 0.05, 0.10, 0.28, 0.28, "Micro Household Block", "hetagent_block in .mod",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.065, 0.32, "• Incomplete markets: Idiosyncratic\n  labor risk & borrowing constraints\n• Solved via Endogenous Grid Method (EGM)\n• Non-stochastic density iteration (Young 2010)\n• Steady-state ergodic wealth distribution\n• Fast fake-news algorithm for Jacobians.",
            fontsize=8.0, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Sub 4.2: Sequence-Space Jacobians
    draw_card(ax, 0.36, 0.10, 0.28, 0.28, "Sequence-Space Jacobians", "J_C,r,  J_C,Y,  J_C,T",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.375, 0.32, "• Column-by-column impulse responses\n  in sequence space over horizon T (e.g. T=300)\n• Exact linear mapping from input shock paths\n  to aggregate consumption & labor supply\n• Sparsity & decay properties exploited in\n  pure vectorized NumPy matrix operations.",
            fontsize=8.0, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Sub 4.3: General Equilibrium Solver
    draw_card(ax, 0.67, 0.10, 0.28, 0.28, "General Equilibrium Coupling", "Broyden Sequence Solver",
              bg_color="#ffffff", border_color="#cbd5e1")
    ax.text(0.685, 0.32, "• Combines household Jacobians with\n  aggregate DAG equations (Taylor rule, NKPC)\n• Formulates sequence residual H(U, Z) = 0\n• Non-linear MIT shock transitions solved\n  via Broyden method in seconds\n• One-line Python call: solve_hank_bridge().",
            fontsize=8.0, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Flow arrows
    ax.annotate("", xy=(0.36, 0.70), xytext=(0.31, 0.70), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.0))
    ax.annotate("", xy=(0.69, 0.70), xytext=(0.64, 0.70), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.0))
    ax.annotate("", xy=(0.17, 0.46), xytext=(0.17, 0.52), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.0))
    ax.annotate("", xy=(0.50, 0.46), xytext=(0.50, 0.52), arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2.0))

    fig.savefig(OUTPUT_DIR / "fig_dsge_ssj_workflow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_dsge_ssj_workflow.png")


def generate_vfi_projections_figure():
    """Generates the Continuous State-Space Projection & Dynamic Programming schematic."""
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(0.5, 0.96, "Numerical Dynamic Programming & Continuous Projection Engines",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(0.5, 0.92, "From Orthogonal Collocation to Smolyak Sparse Grids, Finite Elements, and Continuous-Time PDEs",
            ha="center", va="center", fontsize=10, fontstyle="italic", color=C_TEXT_MUTED)

    # 4 Cards for 4 Numerical Paradigms
    # Card 1: Orthogonal Chebyshev Collocation
    draw_card(ax, 0.03, 0.48, 0.45, 0.39, "1. Orthogonal Chebyshev Collocation", "Spectral Precision for Smooth Problems (Judd 1992)",
              bg_color="#ffffff", border_color="#93c5fd")
    ax.text(0.045, 0.80, "• Basis: Orthogonal Chebyshev polynomials T_n(x) = cos(n arccos x)\n• Grid: Roots of Chebyshev polynomials (optimal node clustering at boundaries)\n• Solves Euler equation residuals R(k; θ) = 0 via Newton-Krylov rootfinding\n• Exponential convergence rate for analytic policy functions\n• Pure-NumPy Clenshaw recurrence and fast cosine transforms\n• Verified to machine precision on neoclassical stochastic growth models.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 2: Smolyak Sparse Grids
    draw_card(ax, 0.52, 0.48, 0.45, 0.39, "2. Smolyak Sparse Grids", "Mitigating the Curse of Dimensionality (Smolyak 1963, Krueger-Kubler 2004)",
              bg_color="#ffffff", border_color="#86efac")
    ax.text(0.535, 0.80, "• Tensor product grids grow exponentially: N = m^d (infeasible for d ≥ 4)\n• Smolyak combination technique selects multi-indices with |i| ≤ d + μ\n• Reduces node count by orders of magnitude (e.g. d=5, level=2: 243 vs. 59,049 nodes)\n• Nested Clenshaw-Curtis nodes enable hierarchical interpolation\n• Seamlessly solves multi-asset and OLG models with intermediate dimensions d ∈ [2, 6]\n• Implemented in pure NumPy with zero compiled dependencies.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 3: Finite Element Galerkin + Fischer-Burmeister
    draw_card(ax, 0.03, 0.05, 0.45, 0.39, "3. Finite Element Galerkin Projection", "Handling Kinks & Borrowing Constraints (McGrattan 1996)",
              bg_color="#ffffff", border_color="#fed7aa")
    ax.text(0.045, 0.37, "• Basis: Locally supported piecewise linear 'hat' / 'tent' functions\n• Galerkin projection enforces orthogonality of residual to trial basis\n• Fischer-Burmeister NCP formulation for borrowing constraints:\n  Ψ(a, b) = a + b - sqrt(a^2 + b^2) = 0   <=>   a ≥ 0, b ≥ 0, a · b = 0\n• Reformulates inequality constraints into smooth non-linear system\n• Accurately resolves non-differentiable policy kinks without smoothing errors.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Card 4: Continuous-Time Macroeconomics (HJB-KFE)
    draw_card(ax, 0.52, 0.05, 0.45, 0.39, "4. Continuous-Time Upwind PDE Solvers", "Income and Wealth Distributions (Achdou et al. 2022)",
              bg_color="#ffffff", border_color="#e9d5ff")
    ax.text(0.535, 0.37, "• Hamilton-Jacobi-Bellman (HJB) equation solved by implicit upwind finite difference:\n  ρ v(a, y) = max_c u(c) + v_a(a, y) (r a + w y - c) + λ_y [v(a, y') - v(a, y)]\n• Monotone finite-difference scheme yields M-matrix, guaranteeing convergence\n• Stationary wealth distribution g(a, y) solved via adjoint Kolmogorov Forward Equation:\n  A' g = 0 subject to mass normalization\n• Coupled general equilibrium loops in pure NumPy/SciPy sparse linear algebra.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    fig.savefig(OUTPUT_DIR / "fig_vfi_projections.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_vfi_projections.png")


def generate_trade_spatial_figure():
    """Generates the Quantitative Spatial & Trade General Equilibrium schematic."""
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Title
    ax.text(0.5, 0.96, "Quantitative Spatial & International Trade General Equilibrium",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_PRIMARY)
    ax.text(0.5, 0.92, "Caliendo-Parro Exact Hat Algebra, Allen-Arkolakis Geography, and Flexible CGE Extensions",
            ha="center", va="center", fontsize=10, fontstyle="italic", color=C_TEXT_MUTED)

    # Left Container: Caliendo & Parro (2015) Trade Engine
    draw_card(ax, 0.03, 0.06, 0.45, 0.81, "1. Caliendo & Parro (2015) Exact Hat Algebra",
              "Multi-Sector Multi-Country Trade with Input-Output Linkages",
              bg_color="#f8fafc", border_color="#93c5fd")
    ax.text(0.045, 0.77, "• Evaluates counterfactual tariff and trade shock scenarios:\n  x̂ = x' / x (proportional change relative to baseline)\n• Inter-industry input-output structure: Sector j uses materials from sector k\n• Bilateral trade shares updated via sectoral trade elasticities θ_j\n• General equilibrium condition: Goods market clearing + trade balance\n• Solved via fast fixed-point dampening algorithms in vectorised NumPy\n• Full welfare decomposition into terms-of-trade and volume-of-trade effects.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Sub-card: Flexible CGE Extensions
    draw_card(ax, 0.05, 0.10, 0.41, 0.32, "Flexible CGE Engine Extensions", "puremacro.trade.flexible",
              bg_color="#ffffff", border_color="#60a5fa")
    ax.text(0.065, 0.36, "• Nested CES Technology: Inner nest VA c_va(r, w) with elasticity ρ_va;\n  Outer nest gross output c_y with material elasticity σ_y.\n• Stone-Geary LES Preferences: Linear Expenditure System with\n  subsistence consumption, capturing non-homothetic Engel curves.\n• Atkeson-Burstein Variable Markups: Cournot competition yields endogenous\n  markups depending on market shares s_{ni}^j, bounded in [1.0, 5.0].\n• Zero-profit variety condensation strictly preserving GE state vector.",
            fontsize=8.0, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Right Container: Allen & Arkolakis (2014) Spatial Economics
    draw_card(ax, 0.52, 0.06, 0.45, 0.81, "2. Allen & Arkolakis (2014) Spatial Topography",
              "Economic Geography, Trade Costs, and Labor Mobility",
              bg_color="#f0fdf4", border_color="#86efac")
    ax.text(0.535, 0.77, "• Continuous or discrete spatial geography with N locations / regions\n• Bilateral iceberg trade costs τ_{ij} and labor migration frictions\n• Workers freely mobile: Utility equalization across all inhabited locations:\n  u_i = u_j = U   for all locations\n• Agglomeration economies (productivity spillovers) vs. dispersion forces\n  (congestion and land consumption)\n• Existence and uniqueness of spatial equilibrium guaranteed by spectral radius\n• Counterfactual analysis of transport infrastructure investments.",
            fontsize=8.3, color=C_TEXT_DARK, va="top", linespacing=1.35)

    # Sub-card: Integrated Assessment & Climate Economy
    draw_card(ax, 0.54, 0.10, 0.41, 0.32, "Climate-Economy IAM Integration", "Nordhaus DICE & Golosov et al. (2014)",
              bg_color="#ffffff", border_color="#4ade80")
    ax.text(0.555, 0.36, "• Nordhaus DICE-2018: Geophysical carbon cycle coupled to macro growth\n• Temperature anomaly dynamics, economic damage functions, and abatement\n• Golosov et al. (2014): Micro-founded optimal carbon tax paths with\n  log-utility and linear carbon decay structures\n• Evaluates global emission pathways, social cost of carbon (SCC), and\n  regional spatial climate damage incidence.",
            fontsize=8.0, color=C_TEXT_DARK, va="top", linespacing=1.35)

    fig.savefig(OUTPUT_DIR / "fig_trade_spatial_cge.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated fig_trade_spatial_cge.png")


if __name__ == "__main__":
    generate_graphical_abstract()
    generate_architecture_oracle_figure()
    generate_svar_identification_figure()
    generate_dsge_ssj_figure()
    generate_vfi_projections_figure()
    generate_trade_spatial_figure()
    print("All figures successfully generated in", OUTPUT_DIR)
