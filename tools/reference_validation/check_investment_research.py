"""Execute both investment guides against a specified source or wheel install.

Example (from outside the checkout, after installing a wheel into TARGET):
  PYTHONPATH=TARGET python /checkout/tools/reference_validation/check_investment_research.py \
      --expected-package-root TARGET --wheel /path/puremacro-6.1.0-py3-none-any.whl \
      --output /tmp/investment-research.json

For a source check, set PYTHONPATH=/checkout and --expected-package-root=/checkout.
This records synthetic workflow evidence, not empirical performance validation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import puremacro
    from puremacro import asset_pricing as ap
    import numpy as np
    import pandas as pd
    import scipy

    expected = args.expected_package_root.resolve()
    if not Path(puremacro.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f"wrong import: {puremacro.__file__}; expected under {expected}")
    repo = Path(__file__).resolve().parents[2]
    results = []
    for locale, directory in [("en", ""), ("es", "es")]:
        page = repo / "docs" / directory / "investment_research.md"
        blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)
        namespace: dict = {"__name__": "__investment_guide__"}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            for i, code in enumerate(blocks):
                exec(compile(code, f"{page}:block{i}", "exec"), namespace)
        ledger = namespace["runs"]["minimum_variance"].ledger
        if ledger.accounting_residual.max() >= 2e-14:
            raise AssertionError("guide accounting residual exceeded its tolerance")
        results.append({
            "locale": locale, "blocks": len(blocks),
            "guide_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
            "maximum_accounting_residual": float(ledger.accounting_residual.max()),
            "input_sha256": namespace["replayed"].input_sha256,
            "terminal_wealth": namespace["wealth"].iloc[-1].to_dict(),
            "valuation_per_share": namespace["valuation"].value_per_share,
            "max_test_pvalue": namespace["comparison"].global_pvalue,
        })
    for field in ("input_sha256", "terminal_wealth", "valuation_per_share", "max_test_pvalue"):
        if results[0][field] != results[1][field]:
            raise AssertionError(f"English/Spanish mismatch: {field}")
    package = Path(ap.__file__).parent
    report = {
        "schema": "puremacro.investment_research.validation.v1",
        "claim": "synthetic guide execution and accounting; not empirical performance",
        "python": sys.version, "platform": platform.platform(),
        "versions": {"puremacro": puremacro.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__},
        "import_path": puremacro.__file__,
        "module_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                          for name in ("__init__.py", "_research.py", "backtest.py", "allocation.py",
                                       "technical.py", "fundamentals.py", "beliefs.py")},
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest() if args.wheel else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Both guides passed ({sum(r['blocks'] for r in results)} blocks): {args.output}")


if __name__ == "__main__":
    main()
