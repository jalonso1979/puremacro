"""Public-API freeze test.

Snapshots two things:
  1. ``__all__`` of every shippable subpackage.
  2. Field names of every public ``<MethodName>Result`` dataclass.

If either drifts, this test fails loudly and prints a diff. Regenerate the
snapshot deliberately when intentional API changes happen — never silently.
"""
import dataclasses
import importlib
import json
import pkgutil
from pathlib import Path

import pytest
import puremacro

SNAPSHOT = Path(__file__).parent / "fixtures" / "public_api_snapshot.json"


def _walk_subpackages():
    """Yield (qualname, module) for every shippable subpackage."""
    skip_prefixes = (
        "puremacro.examples",
        "puremacro.tests",
        # Optional-backend kernel modules import numba at module level by
        # design (mirrors test_pyodide_compat._SKIP_PREFIXES). Excluded so
        # the snapshot is invariant to which extras are installed.
        "puremacro.vfi.kernels_numba",
        "puremacro.models.nested_dmp.kernels_numba",
    )
    for finder, name, is_pkg in pkgutil.walk_packages(
        puremacro.__path__, prefix="puremacro."
    ):
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        # Skip experimental network/LLM modules — see ARCHITECTURE.md
        if name.startswith("puremacro.narrative.sources"):
            continue
        if name == "puremacro.narrative.scoring.llm":
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        yield name, mod


def collect_current_api():
    api = {"all": {}, "result_classes": {}}
    for name, mod in _walk_subpackages():
        if hasattr(mod, "__all__"):
            api["all"][name] = sorted(mod.__all__)
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name)
            if dataclasses.is_dataclass(attr) and isinstance(attr, type):
                # only count classes defined in this module (avoid double-counting re-exports)
                if attr.__module__ != name:
                    continue
                api["result_classes"][f"{name}.{attr_name}"] = sorted(
                    f.name for f in dataclasses.fields(attr)
                )
    return api


def test_public_api_matches_snapshot():
    if not SNAPSHOT.exists():
        pytest.fail(
            f"No snapshot at {SNAPSHOT}. Generate with:\n"
            f"  python -c \"from tests.test_public_api import collect_current_api; "
            f"import json; print(json.dumps(collect_current_api(), indent=2))\" "
            f"> {SNAPSHOT}"
        )
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = collect_current_api()
    if current != expected:
        diff_lines = []
        for key in ("all", "result_classes"):
            added = set(current[key]) - set(expected[key])
            removed = set(expected[key]) - set(current[key])
            for k in sorted(added):
                diff_lines.append(f"  + {key}.{k} = {current[key][k]}")
            for k in sorted(removed):
                diff_lines.append(f"  - {key}.{k} (was {expected[key][k]})")
            for k in sorted(set(current[key]) & set(expected[key])):
                if current[key][k] != expected[key][k]:
                    diff_lines.append(
                        f"  ~ {key}.{k}: {expected[key][k]} -> {current[key][k]}"
                    )
        pytest.fail(
            "Public API drift detected:\n"
            + "\n".join(diff_lines)
            + "\n\nRegenerate snapshot only if the drift is intentional."
        )
