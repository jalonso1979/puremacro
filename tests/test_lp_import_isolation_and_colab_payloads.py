"""Subprocess import-isolation, Colab offload payload, and NumPy 1.26 trapezoid shim tests."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _run_fresh_python(code: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    raw = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
    ).strip()
    return json.loads(raw)


def test_import_puremacro_lp_does_not_load_dsge_models_or_trade() -> None:
    res = _run_fresh_python(
        """
import json, sys
from collections import Counter
import puremacro.lp
pm_mods = [m for m in sys.modules if m == "puremacro" or m.startswith("puremacro.")]
counts = Counter(m.split(".")[1] if "." in m else "<root>" for m in pm_mods)
print(json.dumps({
    "total": len(pm_mods),
    "dsge": counts.get("dsge", 0),
    "trade": counts.get("trade", 0),
    "models": counts.get("models", 0),
}))
"""
    )
    assert res["dsge"] == 0, f"import puremacro.lp eagerly loaded {res['dsge']} dsge modules"
    assert res["trade"] == 0, f"import puremacro.lp eagerly loaded {res['trade']} trade modules"
    assert res["models"] == 0, f"import puremacro.lp eagerly loaded {res['models']} models modules"
    assert res["total"] < 80, f"import puremacro.lp loaded unexpectedly many modules ({res['total']})"


def test_lazy_inference_and_models_exports_preserve_public_surface() -> None:
    res = _run_fresh_python(
        """
import json, sys
import puremacro.inference as inf

# Before lazy attribute access on inference, neither dsge nor trade should be loaded
inf_pre_dsge = sum(1 for m in sys.modules if m.startswith("puremacro.dsge"))
inf_pre_trade = sum(1 for m in sys.modules if m.startswith("puremacro.trade"))

import puremacro.models as mdl
# Before lazy TradePolicySimulator access on models, trade should not be loaded
mdl_pre_trade = sum(1 for m in sys.modules if m.startswith("puremacro.trade"))

# dir() still advertises all __all__ symbols
inf_dir_ok = set(inf.__all__).issubset(set(dir(inf)))
mdl_dir_ok = set(mdl.__all__).issubset(set(dir(mdl)))

# Lazy attribute access resolves cleanly
from puremacro.inference import AdaptiveTemperingSchedule, SMCResult, smc_sample, quandt_andrews_supF
from puremacro.models import TradePolicySimulator, TradePolicySimulationResult

print(json.dumps({
    "inf_pre_dsge": inf_pre_dsge,
    "inf_pre_trade": inf_pre_trade,
    "mdl_pre_trade": mdl_pre_trade,
    "inf_dir_ok": inf_dir_ok,
    "mdl_dir_ok": mdl_dir_ok,
    "smc_callable": callable(smc_sample),
    "qa_callable": callable(quandt_andrews_supF),
    "tp_class": isinstance(TradePolicySimulator, type),
}))
"""
    )
    assert res["inf_pre_dsge"] == 0
    assert res["inf_pre_trade"] == 0
    assert res["mdl_pre_trade"] == 0
    assert res["inf_dir_ok"] is True
    assert res["mdl_dir_ok"] is True
    assert res["smc_callable"] is True
    assert res["qa_callable"] is True
    assert res["tp_class"] is True


@pytest.mark.parametrize(
    "doc_rel",
    [
        "README.md",
        "README.es.md",
        "docs/quickstart.md",
        "docs/es/quickstart.md",
        "docs/es/tablet.md",
    ],
)
def test_colab_offload_doc_snippets_import_real_symbols_and_bind_signatures(
    doc_rel: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import puremacro.dsge as dsge_pkg
    from puremacro.dsge import estimate_sw07 as real_estimate_sw07

    text = (REPO / doc_rel).read_text(encoding="utf-8")
    assert "pm.dsge." not in text, f"{doc_rel} still references unexported pm.dsge.*"

    blocks = [
        b
        for b in re.findall(r"```python\n(.*?)```", text, flags=re.S)
        if "generate_colab_notebook(" in b
    ]
    assert len(blocks) >= 1, f"{doc_rel} expected at least one generate_colab_notebook snippet"

    real_sig = inspect.signature(real_estimate_sw07)
    bound_calls: list[dict] = []

    def _checked_stub(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        bound_calls.append(dict(bound.arguments))
        return {"stubbed_sw07": True, "n_draws": bound.arguments.get("n_draws")}

    monkeypatch.setattr(dsge_pkg, "estimate_sw07", _checked_stub)
    monkeypatch.chdir(tmp_path)

    for idx, block in enumerate(blocks):
        ns: dict = {"__name__": "__colab_doc_test__"}
        exec(compile(block, f"{doc_rel}:colab_block_{idx}", "exec"), ns)
        nb = ns.get("nb")
        assert isinstance(nb, dict) and "cells" in nb
        code_cells = [
            "".join(c["source"]) if isinstance(c["source"], list) else str(c["source"])
            for c in nb["cells"]
            if c.get("cell_type") == "code"
        ]
        task_cells = [src for src in code_cells if "estimate_sw07" in src]
        assert len(task_cells) == 1
        task_ns: dict = {"__name__": "__colab_task_cell__"}
        exec(compile(task_cells[0], f"{doc_rel}:task_cell_{idx}", "exec"), task_ns)
        assert task_ns["result"]["stubbed_sw07"] is True

    assert len(bound_calls) == len(blocks)


def test_skewt_integration_uses_trapz_shim_without_np_trapezoid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import puremacro.gar.skewt as skewt_mod
    from puremacro.gar.skewt import SkewTFit

    fit = SkewTFit(
        mu=1.2, sigma=0.9, alpha=-1.4, nu=6.0, target_quantiles={0.05: -0.5}, success=True
    )
    es_ref = fit.expected_shortfall(0.05)
    ent_ref = fit.downside_entropy()

    # Simulate NumPy 1.26 where np.trapezoid does not exist and only trapezoid/trapz shim runs
    def _fallback_trapz(y: np.ndarray, x: np.ndarray) -> float:
        dx = np.diff(x)
        return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))

    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(skewt_mod, "_trapz", _fallback_trapz)

    es_shim = fit.expected_shortfall(0.05)
    ent_shim = fit.downside_entropy()
    assert es_shim == pytest.approx(es_ref, rel=1e-12, abs=1e-12)
    assert ent_shim == pytest.approx(ent_ref, rel=1e-12, abs=1e-12)
