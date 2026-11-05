"""The bilingual macroeconomic showcase notebook executes its assertions and ships current outputs."""
from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from puremacro.dsge.ssj_policy import ssj_ramsey
from puremacro.lp.counterfactual import mckay_wolf_counterfactual
from puremacro.portfolio import optimize_portfolio, backtest_portfolio

ROOT = Path(__file__).resolve().parents[2]
STEMS = ("76_bilingual_macro_showcase", "76_bilingual_macro_showcase_es")
ALL_NOTEBOOKS = STEMS


def _source_cells(path: Path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)\n", text, flags=re.MULTILINE)
    cells = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].strip()
        kind = "markdown" if "[markdown]" in parts[i] else "code"
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else "" if line == "#" else line
                for line in body.splitlines()
            ).strip()
        if body:
            cells.append((kind, body))
    return cells


def _source_code(path: Path):
    return [source for kind, source in _source_cells(path) if kind == "code"]


def _text(source):
    return "".join(source) if isinstance(source, list) else source


def test_bilingual_showcase_computations_and_saved_sources_match():
    sources = [_source_code(ROOT / "notebooks" / (stem + ".py")) for stem in STEMS]
    assert len(sources[0]) >= 7
    # Exact 1-to-1 code block parity between English and Spanish editions
    assert sources[0] == sources[1]

    for stem, expected in zip(STEMS, sources):
        nb_path = ROOT / "notebooks" / (stem + ".ipynb")
        assert nb_path.exists(), f"Missing compiled notebook: {nb_path}"
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        actual = [
            _text(cell["source"]).strip()
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and _text(cell["source"]).strip()
        ]
        assert actual == expected
        for block in actual:
            ast.parse(block)


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_saved_showcase_outputs_are_executed_and_figures_opaque(name):
    nb_path = ROOT / "notebooks" / (name + ".ipynb")
    assert nb_path.exists(), f"Missing notebook file: {nb_path}"
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    images, counts, text = [], [], []

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if _text(cell["source"]).strip():
            counts.append(cell["execution_count"])
        for output in cell["outputs"]:
            assert output["output_type"] != "error", f"Error output in {name}: {output}"
            if "text" in output:
                text.append(_text(output["text"]))
            if "image/png" in output.get("data", {}):
                raw = base64.b64decode(_text(output["data"]["image/png"]))
                with Image.open(io.BytesIO(raw)) as image:
                    rgba = np.array(image.convert("RGBA"))
                assert rgba.shape[0] >= 200 and rgba.shape[1] >= 600
                assert np.all(rgba[:, :, 3] == 255), f"Found transparent canvas in {name}"
                images.append(raw)

    assert len(counts) >= 7 and None not in counts
    assert len(images) == 1
    transcript = "\n".join(text)
    assert "All 14 macroeconomic showcase workflow checks pass." in transcript


def test_showcase_runs_offline_and_exercise_corners_hold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "notebooks"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__notebook_test__"}
    with matplotlib.rc_context():
        try:
            exec(
                compile(
                    (ROOT / "notebooks" / (STEMS[0] + ".py")).read_text(encoding="utf-8"),
                    STEMS[0] + ".py",
                    "exec",
                ),
                namespace,
            )
        finally:
            plt.close("all")

    scorecard = namespace["scorecard"]
    assert len(scorecard) == 14
    assert (scorecard["measured"] <= scorecard["bound"]).all()

    # Corner validation of interactive "Your turn" knobs
    base_target = namespace["base_target"]
    base_inst = namespace["base_inst"]
    Theta_target = namespace["Theta_target"]
    Theta_inst = namespace["Theta_inst"]
    returns_df = namespace["returns_df"]
    rp_res = namespace["rp_res"]

    for weight in (0.5, 2.0, 3.5):
        cf_corner = mckay_wolf_counterfactual(
            baseline_target=base_target,
            baseline_instrument=base_inst,
            policy_target_irfs=Theta_target,
            policy_instrument_irfs=Theta_inst,
            rule_target_weights=float(weight),
        )
        assert cf_corner.rule_rmse < 1e-10
        assert np.allclose(
            cf_corner.counterfactual_instrument,
            float(weight) * cf_corner.counterfactual_target,
            atol=1e-10,
        )

    for freq in ("monthly", "quarterly", "annual"):
        bt_corner = backtest_portfolio(
            returns_df,
            weights=rp_res.weights,
            rebalance=freq,
            tc_bps=10.0,
            periods_per_year=12,
        )
        assert bt_corner.sharpe_ratio is not None
        assert 0.0 <= abs(bt_corner.max_drawdown) <= 0.40

    for target_ret in (0.03, 0.05, 0.08):
        opt_corner = optimize_portfolio(
            returns_df,
            method="target_return",
            target_return=float(target_ret),
        )
        assert np.isclose(opt_corner.expected_return, float(target_ret), atol=1e-4)
