"""Saved DSGE experiments with an independently known AR(1) response.

Run ``python -m puremacro.examples.dsge_workbench`` to print the comparison.
Add ``--out DIRECTORY`` to export the archive, recipe, reports and plots.
"""
from __future__ import annotations

from pathlib import Path

from puremacro.dsge import DSGEWorkbench

SOURCE = """
var output; varexo productivity; parameters persistence; persistence=0.8;
model;
  [name='Output persistence'] output=persistence*output(-1)+productivity;
end;
initval; output=0; end;
"""


def run_example():
    """Return the workbench, baseline and alternative without writing files."""
    bench = DSGEWorkbench.from_mod(SOURCE, name="Persistence experiment")
    baseline = bench.irf("productivity", horizon=20, size=.01, label="Baseline: persistence 0.8")
    alternative = bench.irf("productivity", horizon=20, size=.01,
                            params={"persistence": .5}, label="Alternative: persistence 0.5")
    if baseline.status != "succeeded" or alternative.status != "succeeded":
        raise RuntimeError(f"Example failed: {baseline.summary()}; {alternative.summary()}")
    return bench, baseline, alternative


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Directory for reproducible run and report exports")
    args = parser.parse_args()
    bench, baseline, alternative = run_example()
    comparison = baseline.compare(alternative)
    print(bench.history().to_string(index=False))
    print(comparison.to_frame().to_string())
    print(baseline.accuracy.summary())
    if args.out is not None:
        import matplotlib.pyplot as plt
        args.out.mkdir(parents=True, exist_ok=True)
        baseline.save(args.out / "baseline.json")
        alternative.save(args.out / "alternative.json")
        (args.out / "baseline.py").write_text(baseline.to_python(), encoding="utf-8")
        (args.out / "baseline.html").write_text(baseline.to_html(), encoding="utf-8")
        (args.out / "comparison.html").write_text(comparison.to_html(), encoding="utf-8")
        comparison.plot().savefig(args.out / "comparison.png", dpi=150)
        ax = baseline.accuracy.plot()
        ax.figure.tight_layout()
        ax.figure.savefig(args.out / "accuracy.png", dpi=150)
        widget = bench.interactive_irf(parameters={"persistence": (.1, .95)},
                                       shocks="productivity", horizon=20, size=.01,
                                       title="Persistence experiment")
        widget.set_value("persistence", .65)
        widget.plot().savefig(args.out / "edited.png", dpi=150)
        widget.apply()
        widget.plot().savefig(args.out / "applied.png", dpi=150)
        widget.disconnect()
        plt.close("all")


if __name__ == "__main__":
    main()
