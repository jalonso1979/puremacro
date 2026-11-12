"""Generate Publication Figures for the Public JOSS Paper.

Generates:
- paper/figures/graphical_abstract.png
- paper/figures/architecture_schematic.png
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
})

# Academic palette
NAVY = "#1b365d"
BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"
DARK_TEAL = "#0f766e"
TEAL = "#0d9488"
LIGHT_TEAL = "#ccfbf1"
SLATE = "#334155"
LIGHT_GRAY = "#f8fafc"
BORDER_GRAY = "#94a3b8"
CRIMSON = "#991b1b"
AMBER = "#b45309"
LIGHT_AMBER = "#fef3c7"
PURPLE = "#6b21a8"
LIGHT_PURPLE = "#f3e8ff"


def generate_graphical_abstract():
    """Generate high-impact graphical abstract for JOSS."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Header title banner
    ax.add_patch(patches.FancyBboxPatch(
        (0.5, 5.1), 13.0, 0.75,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=NAVY, edgecolor="none"
    ))
    ax.text(7.0, 5.47, "puremacro: Unified Scientific Macroeconomics on Core Python Stack",
            color="white", fontsize=15, fontweight="bold", ha="center", va="center")

    # 3 Primary Pillars
    # Pillar 1: Empirical Econometrics
    ax.add_patch(patches.FancyBboxPatch(
        (0.5, 0.5), 3.9, 4.3,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.8
    ))
    ax.text(2.45, 4.45, "Empirical Macroeconometrics", color=NAVY, fontsize=12, fontweight="bold", ha="center")

    emp_features = [
        "• Structural VARs (Sign, Narrative, IV)",
        "• Local Projections (Linear & Smooth)",
        "• GARCH & GARCH-MIDAS Volatility",
        "• Synthetic Control & DiD Estimators",
        "• Mixed-Frequency Nowcasting (DFMs)",
        "• Survey Microdata & Replicate Weights",
        "• Text Uncertainty Indices (BBD)"
    ]
    for idx, feat in enumerate(emp_features):
        ax.text(0.8, 3.95 - idx * 0.48, feat, color=SLATE, fontsize=9.5, va="center")

    # Pillar 2: Quantitative Structural Modeling
    ax.add_patch(patches.FancyBboxPatch(
        (4.7, 0.5), 4.6, 4.3,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=LIGHT_TEAL, edgecolor=DARK_TEAL, lw=1.8
    ))
    ax.text(7.0, 4.45, "Quantitative Structural Modeling", color=DARK_TEAL, fontsize=12, fontweight="bold", ha="center")

    model_features = [
        "• Declarative dp.Model Domain-Specific Language",
        "• Sequence-Space HANK Jacobians (ABRS 2021)",
        "• Heterogeneous Agents (EGM, VFI, Upwind HJB)",
        "• Projection Methods (Smolyak, Chebyshev, FEM)",
        "• Linear Rational Expectations (Real Schur QZ)",
        "• Quantitative Spatial & Trade Models",
        "• Integrated Climate-Economy DSGE (DICE)"
    ]
    for idx, feat in enumerate(model_features):
        ax.text(5.0, 3.95 - idx * 0.48, feat, color=SLATE, fontsize=9.5, va="center")

    # Pillar 3: Architecture & Verification
    ax.add_patch(patches.FancyBboxPatch(
        (9.6, 0.5), 3.9, 4.3,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=LIGHT_AMBER, edgecolor=AMBER, lw=1.8
    ))
    ax.text(11.55, 4.45, "Portable Numerical Core", color=AMBER, fontsize=12, fontweight="bold", ha="center")

    arch_features = [
        "• Pure Python wheel; requests for data",
        "• NumPy, SciPy, pandas, Matplotlib core",
        "• In-Browser Pyodide / WebAssembly Support",
        "• 116 Scorecard Self-Validation Checks",
        "• External Reference Oracles (SciPy, statsmodels)",
        "• 24,000+ Automated Test Suite",
        "• 77 Bilingual Notebook Pairs"
    ]
    for idx, feat in enumerate(arch_features):
        ax.text(9.9, 3.95 - idx * 0.48, feat, color=SLATE, fontsize=9.5, va="center")

    plt.tight_layout()
    out_file = FIGURES_DIR / "graphical_abstract.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file} ({out_file.stat().st_size} bytes)")


def generate_architecture_schematic():
    """Generate architectural schematic diagram for JOSS."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Main Title
    ax.text(6.5, 7.15, "puremacro Architectural Schematic & Modular Design",
            color=NAVY, fontsize=15, fontweight="bold", ha="center")

    # Layer 1: Frontends and Domain Interfaces
    ax.add_patch(patches.FancyBboxPatch(
        (0.6, 4.9), 11.8, 1.8,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.5
    ))
    ax.text(1.0, 6.35, "Domain Interfaces & Public APIs (puremacro.*)", color=NAVY, fontsize=11, fontweight="bold")

    modules = [
        ("var", "SVARs & Identification"),
        ("lp", "Local Projections"),
        ("dp", "Declarative DP DSL"),
        ("dsge", "Rational Expectations"),
        ("dsge.hank", "Sequence-Space HANK"),
        ("trade", "Quantitative Spatial/Trade"),
        ("micro", "Microdata & Weights"),
        ("climate", "Integrated DICE Models"),
    ]
    for i, (name, desc) in enumerate(modules):
        x = 0.9 + (i % 4) * 2.85
        y = 5.65 if i < 4 else 5.1
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), 2.65, 0.45,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="white", edgecolor=BORDER_GRAY, lw=1.0
        ))
        ax.text(x + 0.1, y + 0.22, f"puremacro.{name}", color=NAVY, fontsize=8.5, fontweight="bold", va="center")
        ax.text(x + 0.1, y + 0.08, desc, color=SLATE, fontsize=7.0, va="center")

    # Connecting Arrow
    ax.annotate("", xy=(6.5, 4.45), xytext=(6.5, 4.85),
                arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))

    # Layer 2: Computational Engines & Mathematical Solvers
    ax.add_patch(patches.FancyBboxPatch(
        (0.6, 2.5), 11.8, 1.9,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=LIGHT_TEAL, edgecolor=DARK_TEAL, lw=1.5
    ))
    ax.text(1.0, 4.05, "Computational Core & Algorithmic Engines", color=DARK_TEAL, fontsize=11, fontweight="bold")

    engines = [
        ("EGM & VFI Solvers", "Endogenous Grid & Howard Policy"),
        ("SSJ Fake-News", "Auclert et al. Sequence Jacobians"),
        ("Real Schur Sylvester", "Generalized Sylvester Solvers"),
        ("State-Space Engines", "Kalman Filter & Tempered Particle Filter"),
        ("NUTS & Hamiltonian MCMC", "Analytical Score Gradients"),
        ("Upwind Continuous Time", "Achdou et al. Sparse HJB"),
        ("Nonlinear Solvers", "Trust-Region, Powell, Levenberg-M."),
        ("Microdata Engine", "BRR, Jackknife, Replicate Weights"),
    ]
    for i, (name, desc) in enumerate(engines):
        x = 0.9 + (i % 4) * 2.85
        y = 3.35 if i < 4 else 2.75
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), 2.65, 0.48,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="white", edgecolor=BORDER_GRAY, lw=1.0
        ))
        ax.text(x + 0.1, y + 0.26, name, color=DARK_TEAL, fontsize=8.5, fontweight="bold", va="center")
        ax.text(x + 0.1, y + 0.10, desc, color=SLATE, fontsize=7.0, va="center")

    # Connecting Arrow
    ax.annotate("", xy=(6.5, 2.05), xytext=(6.5, 2.45),
                arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))

    # Layer 3: Numerical Foundation & Runtime Targets
    ax.add_patch(patches.FancyBboxPatch(
        (0.6, 0.4), 11.8, 1.6,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=LIGHT_AMBER, edgecolor=AMBER, lw=1.5
    ))
    ax.text(1.0, 1.65, "Execution Layer & Portability Targets (Zero C Extensions)", color=AMBER, fontsize=11, fontweight="bold")

    targets = [
        ("Core Stack", "NumPy, SciPy, pandas, Matplotlib", BLUE),
        ("CPython", "Python 3.11, 3.12, 3.13 (macOS, Linux, Win)", NAVY),
        ("WebAssembly (Pyodide)", "Runs directly in client browser", CRIMSON),
        ("Validation Gallery", "116 Scorecard self-contained checks", DARK_TEAL),
    ]
    for i, (title, sub, col) in enumerate(targets):
        x = 0.9 + i * 2.85
        y = 0.65
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), 2.65, 0.75,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="white", edgecolor=BORDER_GRAY, lw=1.0
        ))
        ax.text(x + 0.15, y + 0.48, title, color=col, fontsize=9.0, fontweight="bold", va="center")
        ax.text(x + 0.15, y + 0.20, sub, color=SLATE, fontsize=7.2, va="center")

    plt.tight_layout()
    out_file = FIGURES_DIR / "architecture_schematic.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated: {out_file} ({out_file.stat().st_size} bytes)")


def main():
    print("Generating JOSS publication figures...")
    generate_graphical_abstract()
    generate_architecture_schematic()
    print("JOSS figures generated successfully.")


if __name__ == "__main__":
    main()
