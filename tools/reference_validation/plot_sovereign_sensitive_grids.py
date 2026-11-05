"""Plot completed uniform and curved sovereign spread-volatility refinements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import t as student_t


def plot(uniform_path: Path, curved_path: Path, output: Path, language: str):
    uniform = json.loads(uniform_path.read_text(encoding="utf-8"))
    curved = json.loads(curved_path.read_text(encoding="utf-8"))
    if "status" not in uniform or "status" not in curved:
        raise ValueError("completed, assessed studies are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    spanish = language == "es"
    labels = {"uniform": "Uniforme" if spanish else "Uniform",
              "curved": "No uniforme" if spanish else "Nonuniform"}
    colors = {"uniform": "#17648c", "curved": "#b65e17"}
    with plt.rc_context({"font.size": 10, "svg.hashsalt": "puremacro-sovereign-refinement"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
        titles = ((r"$\theta = 0{,}5$", r"$\rho_y = 0{,}97$") if spanish
                  else (r"$\theta = 0.5$", r"$\rho_y = 0.97$"))
        for ax, scenario, title in zip(axes, ("theta=0.5", "rho_y=0.97"), titles):
            families = {"uniform": uniform["scenarios"][scenario]["cases"]}
            if scenario == "theta=0.5":
                families["curved"] = curved["cases"]
            for family, cases in families.items():
                for income, style, marker in ((641, "--", "o"), (1281, "-", "s")):
                    selected = sorted((c for c in cases.values() if c["n_y"] == income),
                                      key=lambda c: c["n_b"])
                    points, means, halves = [], [], []
                    for case in selected:
                        values = np.asarray(case["simulation"]["per_seed"]["spread_sd_pp"])
                        points.append(case["n_b"])
                        means.append(values.mean())
                        halves.append(student_t.ppf(.975, len(values) - 1)
                                      * values.std(ddof=1) / np.sqrt(len(values)))
                    income_label = f"{income:,}".replace(",", "." if spanish else ",")
                    ax.errorbar(points, means, yerr=halves, color=colors[family],
                                linestyle=style, marker=marker, markersize=5, capsize=3,
                                label=f"{labels[family]}, {income_label}")
            ax.set_xscale("log", base=2)
            ticks = [6401, 12801, 25601]
            ax.set_xticks(ticks, [f"{n:,}".replace(",", "." if spanish else ",") for n in ticks])
            ax.set_title(title)
            ax.set_xlabel("Nodos de deuda" if spanish else "Debt nodes")
            ax.set_ylabel("DE del diferencial (pp)" if spanish else "Spread SD (pp)")
            if spanish:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
            ax.grid(axis="y", color=".88", linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(title="Malla, estados de ingreso" if spanish else "Grid, income states",
                      fontsize=8, title_fontsize=8, loc="best", frameon=False)
        fig.suptitle("Refinamiento de los escenarios sensibles" if spanish
                     else "Refinement of the sensitive scenarios", fontsize=13)
        metadata = ({"Date": None, "Creator": "PureMacro"} if output.suffix == ".svg"
                    else {"CreationDate": None, "Creator": "PureMacro"} if output.suffix == ".pdf"
                    else {"Software": "PureMacro"})
        fig.savefig(output, metadata=metadata)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path,
                        default=Path("tests/fixtures/sovereign_brief/sensitive_grid_convergence.json"))
    parser.add_argument("--curved", type=Path,
                        default=Path("tests/fixtures/sovereign_brief/curved_grid_convergence.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", choices=["en", "es"], default="en")
    args = parser.parse_args()
    plot(args.uniform, args.curved, args.output, args.lang)
