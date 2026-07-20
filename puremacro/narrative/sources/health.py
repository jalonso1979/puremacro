"""Connector health-check.

Probes every ``iter_*`` function exported from :mod:`puremacro.narrative.sources`
and reports its current operational status: live record count, date range,
and error (if any). Run via::

    python -m puremacro.narrative.sources.health           # table
    python -m puremacro.narrative.sources.health --json    # machine-readable

The intent is pain-prevention — RSS endpoints retire silently (we saw the
2025 BoE / BoJ / PBoC reorgs), and a connector returning zero records is
indistinguishable from a working one until something downstream breaks.
This module makes that drift visible.

Connectors that need positional/required arguments (e.g. ``iter_google_news``,
``iter_local_csv``, ``iter_oecd_listing``, ``iter_imf_listing``,
``iter_imf_articleiv``) are listed in ``_NEEDS_ARGS`` and reported as
``"needs args"`` without probing.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from datetime import datetime
from typing import Any


# Connectors that don't accept zero-arg probing. They get reported as
# "needs args" (a known limitation, not a failure).
_NEEDS_ARGS: frozenset[str] = frozenset({
    "iter_google_news",         # query is required
    "iter_local_csv",           # path is required
    "iter_oecd_listing",        # category is required
    "iter_imf_listing",
    "iter_imf_articleiv",       # country code is required
    "iter_imf_news",            # query is required
    "iter_gdelt_v2",            # query is required
    "iter_oecd_surveys",        # may need country code
})

# Connectors known to take a long time and that we cap with a per-probe
# wall-clock budget (seconds). Default budget applies otherwise.
_DEFAULT_BUDGET_SEC = 30.0
_LONG_BUDGET_SEC = 90.0
_LONG: frozenset[str] = frozenset({
    "iter_boe_minutes",         # year × month URL enumeration, ~30s
    "iter_boj_speeches",        # multi-year HTML scrape
    "iter_fed_decision",        # 186 records, slow RSS
    "iter_fed_minutes",         # 165 records
    "iter_federal_register",    # large
    "iter_dof_press",           # 200 records
    "iter_tresor_press",        # 50 records
})


@dataclasses.dataclass
class ConnectorStatus:
    name: str
    status: str          # "ok" | "empty" | "skipped" | "error" | "timeout"
    n_records: int
    first_date: str | None
    last_date: str | None
    error: str | None
    elapsed_sec: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _probe(name: str, fn) -> ConnectorStatus:
    """Run a single connector with a wall-clock budget; tolerate any error."""
    if name in _NEEDS_ARGS:
        return ConnectorStatus(name, "skipped", 0, None, None,
                               "needs positional args", 0.0)
    budget = _LONG_BUDGET_SEC if name in _LONG else _DEFAULT_BUDGET_SEC
    t0 = time.monotonic()
    try:
        recs = []
        for rec in fn():
            recs.append(rec)
            if time.monotonic() - t0 > budget:
                return ConnectorStatus(
                    name, "timeout", len(recs), None, None,
                    f"> {budget:.0f}s wall-clock budget", time.monotonic() - t0,
                )
        dt = time.monotonic() - t0
        if not recs:
            return ConnectorStatus(name, "empty", 0, None, None, None, dt)
        dates = sorted(r[0] for r in recs)
        first = dates[0].strftime("%Y-%m-%d") if dates else None
        last = dates[-1].strftime("%Y-%m-%d") if dates else None
        return ConnectorStatus(name, "ok", len(recs), first, last, None, dt)
    except Exception as e:
        return ConnectorStatus(
            name, "error", 0, None, None,
            f"{type(e).__name__}: {str(e)[:120]}",
            time.monotonic() - t0,
        )


def run() -> list[ConnectorStatus]:
    """Probe every ``iter_*`` connector and return per-connector status."""
    mod = importlib.import_module("puremacro.narrative.sources")
    names = sorted(
        n for n in dir(mod)
        if n.startswith("iter_") and callable(getattr(mod, n))
    )
    return [_probe(n, getattr(mod, n)) for n in names]


def format_table(results: list[ConnectorStatus]) -> str:
    """Pretty-print results as a fixed-width text table."""
    headers = ("STATUS", "CONNECTOR", "N", "FIRST", "LAST", "TIME", "ERROR")
    rows = []
    for r in results:
        marker = {"ok": "✓", "empty": "·", "skipped": "—",
                  "error": "✗", "timeout": "⏱"}.get(r.status, "?")
        rows.append((
            f"{marker} {r.status}",
            r.name,
            str(r.n_records),
            r.first_date or "-",
            r.last_date or "-",
            f"{r.elapsed_sec:.1f}s",
            r.error or "",
        ))
    widths = [max(len(h), max(len(row[i]) for row in rows))
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for row in rows:
        lines.append(fmt.format(*row))
    # Counts summary
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    lines.append("")
    lines.append(f"summary: {summary}  (total {len(results)})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.narrative.sources.health",
        description="Probe every iter_* connector and report status.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of a text table.",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Optional substring filter on connector names "
             "(e.g. --filter ecb).",
    )
    args = parser.parse_args(argv)

    results = run()
    if args.filter:
        results = [r for r in results if args.filter.lower() in r.name.lower()]

    if args.json:
        payload = {
            "computed_at": datetime.utcnow().isoformat(),
            "n_connectors": len(results),
            "results": [r.as_dict() for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_table(results))

    # Exit code: 0 if no errors AND at least one "ok"; 1 otherwise.
    has_error = any(r.status == "error" for r in results)
    has_any_ok = any(r.status == "ok" for r in results)
    return 1 if has_error or not has_any_ok else 0


if __name__ == "__main__":
    sys.exit(main())
