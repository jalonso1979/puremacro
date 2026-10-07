"""Write an offline evidence dossier; a failed comparison exits nonzero.

Run ``python tools/run_research_benchmarks.py --output /tmp/puremacro-benchmarks``
from a checkout, or use ``run_research_benchmarks().write(...)`` from a wheel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Direct script execution puts tools/, rather than the checkout root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puremacro.validation.research import research_benchmarks, run_research_benchmarks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research-benchmarks"))
    parser.add_argument("--case", action="append", dest="cases", help="Case ID; repeat to select several")
    args = parser.parse_args(argv)
    cases = research_benchmarks()
    if args.cases:
        unknown = set(args.cases)-{case.id for case in cases}
        if unknown:
            parser.error("Unknown case IDs: " + ", ".join(sorted(unknown)))
        cases = tuple(case for case in cases if case.id in args.cases)
    report = run_research_benchmarks(cases)
    paths = report.write(args.output)
    for case in report.results:
        print(f"{'PASS' if case.passed else 'FAIL'} {case.id}" + (f": {case.error}" if case.error else ""))
    print(paths["json"])
    print(paths["markdown"])
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
