"""Tests for the zero-dependency puremacro MCP stdio server."""
from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import puremacro.mcp_server as mcp_server
from puremacro.mcp_server import (
    MCP_PROTOCOL_VERSION,
    PUREMACRO_MCP_TOOLS,
    UnknownToolError,
    _to_json_safe,
    call_tool,
    list_tools,
    serve_stdio,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _encode_content_length(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _decode_content_length_stream(raw: bytes) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    pos = 0
    while pos < len(raw):
        sep = raw.find(b"\r\n\r\n", pos)
        assert sep != -1, f"Missing header terminator in {raw[pos:]!r}"
        header_block = raw[pos:sep].decode("ascii")
        headers: dict[str, str] = {}
        for line in header_block.split("\r\n"):
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()
        length = int(headers["content-length"])
        body_start = sep + 4
        body_end = body_start + length
        body_bytes = raw[body_start:body_end]
        assert len(body_bytes) == length
        frames.append(json.loads(body_bytes.decode("utf-8")))
        pos = body_end
    return frames


def _sample_panel_rows() -> list[dict[str, object]]:
    return [
        {"country": "US", "date": "2020-01-01", "y": 1.0, "x": 0.2},
        {"country": "US", "date": "2020-04-01", "y": 1.4, "x": 0.5},
        {"country": "US", "date": "2020-07-01", "y": 1.7, "x": None},
        {"country": "US", "date": "2020-10-01", "y": 2.0, "x": 0.1},
        {"country": "EA", "date": "2020-01-01", "y": 0.8, "x": 0.3},
        {"country": "EA", "date": "2020-04-01", "y": 1.1, "x": 0.4},
        {"country": "EA", "date": "2020-07-01", "y": 1.3, "x": 0.6},
        {"country": "EA", "date": "2020-10-01", "y": 1.5, "x": 0.2},
    ]


def _sample_ts_rows(n: int = 60, seed: int = 7) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="QS")
    x = rng.normal(0.0, 1.0, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t] + 0.4 * x[t - 1] + 0.2 * z[t] + rng.normal(0.0, 0.25)
    return [
        {"date": str(d.date()), "y": float(y[i]), "x": float(x[i]), "z": float(z[i])}
        for i, d in enumerate(dates)
    ]


# ---------------------------------------------------------------------------
# Import / Pyodide invariant & registry metadata
# ---------------------------------------------------------------------------


def test_top_level_imports_are_stdlib_only_and_lazy() -> None:
    src_path = Path(mcp_server.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    top_level_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_modules.add(node.module.split(".")[0])

    forbidden_top = {
        "puremacro",
        "statsmodels",
        "linearmodels",
        "arch",
        "sqlite3",
        "ssl",
        "bs4",
        "pdfplumber",
        "pypdf",
        "pyarrow",
        "openpyxl",
    }
    assert top_level_modules.isdisjoint(forbidden_top)
    assert "sqlite3" not in top_level_modules

    probe = (
        "import json, sys, puremacro.mcp_server as m; "
        "tools = [t['name'] for t in m.list_tools()]; "
        "loaded = sorted(sys.modules); "
        "print(json.dumps({'tools': tools, 'loaded': loaded}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    data = json.loads(proc.stdout)
    assert len(data["tools"]) == 8
    loaded = set(data["loaded"])
    lazy_submodules = {
        "statsmodels",
        "linearmodels",
        "arch",
        "sqlite3",
        "ssl",
        "puremacro.data",
        "puremacro.lp",
        "puremacro.reports",
        "puremacro.dsge",
        "puremacro.trade",
        "puremacro.fetch",
        "puremacro.validation",
    }
    assert loaded.isdisjoint(lazy_submodules), f"Eagerly loaded: {sorted(loaded & lazy_submodules)}"


def test_tool_registry_schema_and_deepcopy() -> None:
    expected = {
        "preflight_panel",
        "lp_hac",
        "lp_sensitivity_grid",
        "compare_irfs",
        "score_irf_against_targets",
        "tariff_scenario_batch",
        "connector_freshness_report",
        "validation_scorecard",
    }
    assert set(PUREMACRO_MCP_TOOLS) == expected
    tools = list_tools()
    assert {t["name"] for t in tools} == expected

    # Deep copy protection
    tools[0]["description"] = "mutated"
    assert PUREMACRO_MCP_TOOLS[tools[0]["name"]]["description"] != "mutated"

    for spec in list_tools():
        assert isinstance(spec["description"], str) and len(spec["description"]) > 15
        schema = spec["inputSchema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        annotations = spec["annotations"]
        assert annotations["readOnlyHint"] is True
        assert annotations["destructiveHint"] is False
        encoded = json.dumps(spec, allow_nan=False)
        assert json.loads(encoded) == spec


def test_call_tool_validation_and_unknown_tool() -> None:
    with pytest.raises(UnknownToolError):
        call_tool("nonexistent_tool", {})

    # Unknown property rejected via additionalProperties=False
    with pytest.raises(ValueError, match="unsupported field"):
        call_tool("validation_scorecard", {"list_only": True, "bogus": 1})

    # Missing required parameter
    with pytest.raises(ValueError, match="is required"):
        call_tool("lp_hac", {"rows": _sample_ts_rows(30)})

    # Boolean is not an integer
    with pytest.raises(ValueError, match="must be an integer"):
        call_tool("lp_hac", {"rows": _sample_ts_rows(30), "y": "y", "x": "x", "horizon": True})

    # Non-finite floats rejected cleanly
    with pytest.raises(ValueError, match="finite number"):
        call_tool(
            "lp_hac",
            {"rows": _sample_ts_rows(30), "y": "y", "x": "x", "alpha": float("nan")},
        )


def test_json_safe_serializer_handles_nonfinite_containers() -> None:
    @dataclasses.dataclass
    class _Demo:
        val: float
        when: datetime
        arr: np.ndarray

    df = pd.DataFrame({"h": [0, 1], "beta": [1.25, float("nan")], "margin": [float("inf"), float("-inf")]})
    # LPResult sets index = df['h'], leaving duplicate 'h' index/column names
    df.index = df["h"]

    obj = {
        "demo": _Demo(
            val=float("nan"),
            when=datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc),
            arr=np.array([1.0, float("inf"), float("-inf")]),
        ),
        "table": df,
        "path": Path("a/b.csv"),
    }
    safe = _to_json_safe(obj)
    dumped = json.dumps(safe, allow_nan=False)
    decoded = json.loads(dumped)
    assert decoded["demo"]["val"] is None
    assert decoded["demo"]["arr"] == [1.0, None, None]
    assert decoded["table"] == [
        {"h": 0, "beta": 1.25, "margin": None},
        {"h": 1, "beta": None, "margin": None},
    ]
    assert math.isfinite(decoded["table"][0]["beta"])
    assert decoded["path"] == os.fspath(Path("a/b.csv"))
    assert Path(decoded["path"]) == Path("a/b.csv")


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 protocol, framing & error handling
# ---------------------------------------------------------------------------


def test_stdio_ndjson_handshake_notifications_and_errors() -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": MCP_PROTOCOL_VERSION}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
        {"jsonrpc": "2.0", "id": 0, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "validation_scorecard", "arguments": {"subsystem": "unit_root", "list_only": True}}},
        {"jsonrpc": "2.0", "id": 4, "method": "unknown/method"},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "missing_tool"}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "lp_hac", "arguments": {"y": "y"}}},
        {"jsonrpc": "1.0", "id": 7, "method": "ping"},
    ]
    raw_in = "\n".join(json.dumps(r) for r in requests) + "\n{not-valid-json\n"
    stdin = io.StringIO(raw_in)
    stdout = io.StringIO()

    rc = serve_stdio(stdin=stdin, stdout=stdout)
    assert rc == 0

    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    # Notifications produce zero messages -> 9 responses (1,0,2,3,4,5,6,7 + parse error)
    assert len(lines) == 9

    init_resp, ping_resp, list_resp, call_resp, unk_meth, unk_tool, bad_args, bad_ver, parse_err = lines
    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert init_resp["result"]["serverInfo"]["name"] == "puremacro"
    assert init_resp["result"]["capabilities"]["tools"]["listChanged"] is False

    assert ping_resp == {"jsonrpc": "2.0", "id": 0, "result": {}}
    assert len(list_resp["result"]["tools"]) == 8

    assert call_resp["id"] == 3
    assert call_resp["result"]["isError"] is False
    assert call_resp["result"]["structuredContent"]["n_cases"] == 3
    assert len(call_resp["result"]["content"]) == 1
    assert call_resp["result"]["content"][0]["type"] == "text"

    assert unk_meth["error"]["code"] == -32601
    assert unk_tool["error"]["code"] == -32602
    assert bad_args["result"]["isError"] is True
    assert bad_args["result"]["structuredContent"]["error_type"] == "ValueError"
    assert bad_ver["error"]["code"] == -32600
    assert parse_err["error"]["code"] == -32700


def test_stdio_content_length_binary_and_utf8_text() -> None:
    utf8_rows = [
        {"country": "ES", "date": "2020-01-01", "pib_españa": 1.0},
        {"country": "ES", "date": "2020-04-01", "pib_españa": 1.5},
        {"country": "ES", "date": "2020-07-01", "pib_españa": 1.9},
        {"country": "ES", "date": "2020-10-01", "pib_españa": 2.2},
    ]
    reqs = [
        {"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": "pf",
            "method": "tools/call",
            "params": {
                "name": "preflight_panel",
                "arguments": {
                    "rows": utf8_rows,
                    "unit_col": "country",
                    "time_col": "date",
                    "min_obs": 6,
                    "run_stationarity": False,
                    "run_seasonality": False,
                },
            },
        },
        [{"jsonrpc": "2.0", "id": 9, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/note"}],
    ]

    # Binary Content-Length framing (multi-byte UTF-8)
    raw_bytes = b"".join(_encode_content_length(r) if isinstance(r, dict) else _encode_content_length_raw(r) for r in reqs)
    bin_out = io.BytesIO()
    serve_stdio(stdin=io.BytesIO(raw_bytes), stdout=bin_out)
    decoded = _decode_content_length_stream(bin_out.getvalue())
    assert len(decoded) == 3
    assert decoded[0]["result"]["protocolVersion"] == "2025-03-26"
    assert decoded[1]["id"] == "pf"
    assert decoded[1]["result"]["isError"] is False
    assert isinstance(decoded[2], list) and decoded[2] == [{"jsonrpc": "2.0", "id": 9, "result": {}}]

    # TextIO fallback for Content-Length (byte counting on UTF-8 text stream)
    text_in = io.StringIO(raw_bytes.decode("utf-8"))
    text_out = io.StringIO()
    serve_stdio(stdin=text_in, stdout=text_out)
    text_frames = _decode_content_length_stream(text_out.getvalue().encode("utf-8"))
    assert len(text_frames) == 3
    assert text_frames[1]["result"]["structuredContent"]["panel_shape"][0] == 4


def _encode_content_length_raw(payload: object) -> bytes:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


# ---------------------------------------------------------------------------
# allowed_root path safety
# ---------------------------------------------------------------------------


def test_allowed_root_security_controls(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "secret.csv"
    outside.write_text("date,y,x\n2020-01-01,1.0,0.5\n", encoding="utf-8")

    inside_csv = root / "panel.csv"
    pd.DataFrame(_sample_panel_rows()).to_csv(inside_csv, index=False)

    bad_ext = root / "notes.txt"
    bad_ext.write_text("hello", encoding="utf-8")

    # Disallowed when allowed_root=None
    with pytest.raises(PermissionError, match="allowed_root"):
        call_tool("preflight_panel", {"data_path": str(inside_csv)})

    # Traversal rejected
    with pytest.raises(PermissionError, match="escapes allowed_root"):
        call_tool("preflight_panel", {"data_path": "../secret.csv"}, allowed_root=root)

    # Unsupported extension rejected
    with pytest.raises(ValueError, match="not permitted"):
        call_tool("preflight_panel", {"data_path": "notes.txt"}, allowed_root=root)

    # Valid relative path succeeds
    res = call_tool(
        "preflight_panel",
        {
            "data_path": "panel.csv",
            "unit_col": "country",
            "time_col": "date",
            "run_stationarity": False,
            "run_seasonality": False,
        },
        allowed_root=root,
    )
    assert res["panel_shape"][0] == 8
    assert res["n_units"] == 2


def test_allowed_root_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "secret.csv"
    outside.write_text("date,y,x\n2020-01-01,1.0,0.5\n", encoding="utf-8")
    link_path = root / "escape.csv"
    try:
        os.symlink(outside, link_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on host: {exc}")

    with pytest.raises(PermissionError, match="escapes allowed_root"):
        call_tool("preflight_panel", {"data_path": "escape.csv"}, allowed_root=root)


# ---------------------------------------------------------------------------
# Individual tool execution tests (all 8 tools)
# ---------------------------------------------------------------------------


def test_tool_preflight_panel_columns_and_markdown() -> None:
    rows = _sample_panel_rows()
    columns = {k: [r[k] for r in rows] for k in rows[0]}
    out = call_tool(
        "preflight_panel",
        {
            "columns": columns,
            "unit_col": "country",
            "time_col": "date",
            "min_obs": 6,
            "run_stationarity": False,
            "run_seasonality": False,
            "include_markdown": True,
        },
    )
    assert out["panel_shape"][0] == 8
    assert out["n_units"] == 2
    assert "summary" in out and isinstance(out["summary"], str)
    assert "markdown" in out and "|" in out["markdown"] and "variable" in out["markdown"]
    json.dumps(out, allow_nan=False)


def test_tool_lp_hac_and_sensitivity_grid(tmp_path: Path) -> None:
    rows = _sample_ts_rows(n=64, seed=11)
    csv_path = tmp_path / "ts.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    hac = call_tool(
        "lp_hac",
        {
            "data_path": "ts.csv",
            "y": "y",
            "x": "x",
            "controls": ["z"],
            "time_col": "date",
            "horizons": [0, 1, 2, 3],
            "n_lags": 2,
            "ci": 0.90,
        },
        allowed_root=tmp_path,
    )
    assert hac["horizons"] == [0, 1, 2, 3]
    assert len(hac["table"]) == 4
    assert all(math.isfinite(row["beta"]) and math.isfinite(row["se"]) for row in hac["table"])

    grid_out = call_tool(
        "lp_sensitivity_grid",
        {
            "rows": rows,
            "time_col": "date",
            "y": "y",
            "x": "x",
            "estimator": "lp_hac",
            "grid": {
                "n_lags": [1, 2],
                "estimator": ["lp_hac", "la_lp"],
            },
            "base_kwargs": {"horizons": [0, 1, 2]},
        },
    )
    assert grid_out["horizons"] == [0, 1, 2]
    assert len(grid_out["table"]) >= 4
    assert "summary" in grid_out and grid_out["summary"]
    json.dumps(grid_out, allow_nan=False)


def test_tool_lp_sensitivity_grid_lp_did_and_mix_guard() -> None:
    rng = np.random.default_rng(42)
    panel_rows: list[dict[str, object]] = []
    for u in range(16):
        cohort = 5 if u < 6 else (7 if u < 10 else 999)
        unit_fe = 0.2 * u
        for t in range(12):
            treat = 1 if t >= cohort else 0
            effect = 1.5 * treat * (1 + 0.1 * max(t - cohort, 0))
            panel_rows.append(
                {
                    "unit": f"u{u:02d}",
                    "time": t,
                    "treat": treat,
                    "y": float(unit_fe + 0.1 * t + effect + rng.normal(0.0, 0.1)),
                }
            )

    did_out = call_tool(
        "lp_sensitivity_grid",
        {
            "rows": panel_rows,
            "y": "y",
            "x": "treat",
            "unit_col": "unit",
            "time_col": "time",
            "estimator": "lp_did",
            "grid": {"weights": ["equal", "vw"]},
            "base_kwargs": {"horizons": [0, 1, 2], "pre_window": 2},
        },
    )
    assert set([0, 1, 2]).issubset(set(did_out["horizons"]))
    assert len(did_out["table"]) >= 4
    json.dumps(did_out, allow_nan=False)

    with pytest.raises(ValueError, match="Cannot combine 'lp_did'"):
        call_tool(
            "lp_sensitivity_grid",
            {
                "rows": panel_rows,
                "y": "y",
                "x": "treat",
                "unit_col": "unit",
                "time_col": "time",
                "grid": {"estimator": ["lp_did", "panel_lp"], "weights": ["equal"]},
            },
        )

    with pytest.raises(ValueError, match="requires 'unit_col'"):
        call_tool(
            "lp_sensitivity_grid",
            {
                "rows": panel_rows,
                "y": "y",
                "x": "treat",
                "estimator": "lp_did",
                "grid": {"weights": ["equal", "vw"]},
            },
        )


def test_csv_row_bounds_enforced_across_irf_tools(tmp_path: Path) -> None:
    n_over = 50_002
    cmp_csv = tmp_path / "oversize_compare.csv"
    cmp_lines = ["spec,h,point,se"] + [f"s{i % 2},{i},0.1,0.05" for i in range(n_over)]
    cmp_csv.write_text("\n".join(cmp_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds maximum of 50000 rows"):
        call_tool("compare_irfs", {"data_path": "oversize_compare.csv"}, allowed_root=tmp_path)

    tgt_csv = tmp_path / "oversize_targets.csv"
    tgt_lines = ["variable,shock,h,value,se"] + [f"y,mp,{i},0.1,0.05" for i in range(n_over)]
    tgt_csv.write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")

    small_model = [{"h": 0, "y": 0.1}]
    with pytest.raises(ValueError, match="exceeds maximum of 50000 rows"):
        call_tool(
            "score_irf_against_targets",
            {"model_irf": small_model, "targets_path": "oversize_targets.csv"},
            allowed_root=tmp_path,
        )

    mdl_csv = tmp_path / "oversize_model.csv"
    mdl_lines = ["h,y"] + [f"{i},0.1" for i in range(n_over)]
    mdl_csv.write_text("\n".join(mdl_lines) + "\n", encoding="utf-8")
    small_targets = [{"variable": "y", "shock": "mp", "h": 0, "value": 0.1, "se": 0.05}]
    with pytest.raises(ValueError, match="exceeds maximum of 50000 rows"):
        call_tool(
            "score_irf_against_targets",
            {"model_irf_path": "oversize_model.csv", "targets": small_targets},
            allowed_root=tmp_path,
        )


def test_tool_compare_irfs_and_score_irf_against_targets(tmp_path: Path) -> None:
    irf_a = [
        {"horizon": 0, "irf": -0.5, "se": 0.1},
        {"horizon": 1, "irf": -0.35, "se": 0.1},
        {"horizon": 2, "irf": -0.15, "se": 0.08},
        {"horizon": 3, "irf": -0.05, "se": 0.06},
    ]
    irf_b = [
        {"horizon": 0, "irf": -0.45, "se": 0.12},
        {"horizon": 1, "irf": -0.30, "se": 0.11},
        {"horizon": 2, "irf": -0.10, "se": 0.09},
        {"horizon": 3, "irf": -0.02, "se": 0.07},
    ]
    cmp_out = call_tool(
        "compare_irfs",
        {
            "results": {"baseline_lp": irf_a, "smooth_lp": irf_b},
            "baseline": "baseline_lp",
            "include_markdown": True,
        },
    )
    assert cmp_out["baseline"] == "baseline_lp"
    assert cmp_out["horizons"] == [0, 1, 2, 3]
    assert "markdown" in cmp_out

    # 1) Table-based scoring
    model_rows = [
        {"shock": "mp", "variable": "y", "horizon": h, "irf": val}
        for h, val in [(0, -0.48), (1, -0.32), (2, -0.14), (3, -0.04)]
    ]
    target_rows = [
        {"shock": "mp", "variable": "y", "horizon": r["horizon"], "irf": r["irf"], "se": r["se"]}
        for r in irf_a
    ]
    sc_diag = call_tool(
        "score_irf_against_targets",
        {
            "model_irf": model_rows,
            "targets": target_rows,
            "weight_mode": "diagonal",
            "include_markdown": True,
        },
    )
    assert sc_diag["weight_mode_used"] == "diagonal"
    assert sc_diag["rmse"] < 0.05
    assert "markdown" in sc_diag

    # 2) Covariance-weighted MomentTargets scoring via JSON files under allowed_root
    cov = np.diag([0.01, 0.01, 0.0064, 0.0036]).tolist()
    model_json = tmp_path / "model.json"
    targets_json = tmp_path / "targets.json"
    model_json.write_text(json.dumps(model_rows), encoding="utf-8")
    targets_json.write_text(
        json.dumps(
            {
                "values": [r["irf"] for r in irf_a],
                "labels": [f"y:mp:h={r['horizon']}" for r in irf_a],
                "covariance": cov,
                "metadata": {"shock": "mp"},
            }
        ),
        encoding="utf-8",
    )
    sc_cov = call_tool(
        "score_irf_against_targets",
        {
            "model_irf_path": "model.json",
            "targets_path": "targets.json",
            "weight_mode": "covariance",
        },
        allowed_root=tmp_path,
    )
    assert sc_cov["weight_mode_used"] == "covariance"
    assert sc_cov["chi2_df"] == 4
    assert math.isfinite(sc_cov["chi2_stat"])
    assert math.isfinite(sc_cov["covariance_condition_number"])


def test_tool_tariff_scenario_batch_composite_and_file_calibration(tmp_path: Path) -> None:
    from puremacro.trade.data import synthetic_codes, synthetic_icio_table

    # 1) Synthetic ICIO with overlapping CompositeTariffScenario
    out = call_tool(
        "tariff_scenario_batch",
        {
            "model_source": "synthetic",
            "n_countries": 3,
            "n_sectors": 2,
            "seed": 3,
            "scenarios": [
                {
                    "name": "c01_on_c02_10pct",
                    "importer": "C01",
                    "description": "10% across the board on imports from C02",
                    "tariffs": [
                        {"exporter": "C02", "sector": "*", "rate": 0.10}
                    ],
                },
                {
                    "name": "us_china_composite",
                    "on_conflict": "last_wins",
                    "scenarios": [
                        {
                            "name": "broad",
                            "importer": "C01",
                            "tariffs": {"C02:*": 0.10},
                        },
                        {
                            "name": "s02_override",
                            "importer": "C01",
                            "tariffs": [{"exporter": "C02", "sector": "S02", "rate": 0.25}],
                        },
                    ],
                },
            ],
            "include_markdown": True,
        },
    )
    assert out["model_source"] == "synthetic"
    assert out["resolve_only"] is False
    assert "us_china_composite" in out["composite_conflicts"]
    assert len(out["composite_conflicts"]["us_china_composite"]) == 1
    assert out["composite_conflicts"]["us_china_composite"][0]["resolved_rate"] == pytest.approx(0.25)
    assert len(out["scenario_summary"]) >= 2
    assert "markdown" in out

    # 2) File-based calibration (resolve_only=True)
    tbl = synthetic_icio_table(3, 2, seed=5)
    c_codes, s_codes = synthetic_codes(3, 2)
    cal_path = tmp_path / "icio.json"
    cal_path.write_text(
        json.dumps({"matrix": tbl.tolist(), "country_codes": c_codes, "sector_codes": s_codes}),
        encoding="utf-8",
    )
    res_only = call_tool(
        "tariff_scenario_batch",
        {
            "model_source": "file",
            "data_path": "icio.json",
            "resolve_only": True,
            "scenarios": [
                {"name": "targeted", "importer": "C01", "tariffs": [{"exporter": "C02", "sector": "S01", "rate": 0.15}]}
            ],
        },
        allowed_root=tmp_path,
    )
    assert res_only["resolve_only"] is True
    assert res_only["country_codes"] == c_codes


def test_tool_connector_freshness_report_and_validation_scorecard(tmp_path: Path) -> None:
    from puremacro._cache_db import bootstrap_schema

    rec = call_tool(
        "connector_freshness_report",
        {
            "mode": "recorded",
            "sources": ["fred.csv", "oecd.qna_panel"],
            "lang": "es",
            "include_markdown": True,
        },
    )
    assert rec["mode"] == "recorded"
    assert len(rec["records"]) == 2
    assert "summary" in rec and isinstance(rec["summary"], str)
    assert "markdown" in rec

    db_file = tmp_path / "empty_cache.db"
    with contextlib.closing(sqlite3.connect(db_file)) as conn:
        with conn:
            bootstrap_schema(conn)
    cache_rep = call_tool(
        "connector_freshness_report",
        {"mode": "cache", "sources": ["fred.csv"], "db_path": "empty_cache.db"},
        allowed_root=tmp_path,
    )
    assert cache_rep["mode"] == "cache"
    assert len(cache_rep["records"]) == 1

    val_list = call_tool("validation_scorecard", {"subsystem": "unit_root", "list_only": True})
    assert val_list["subsystem"] == "unit_root"
    assert val_list["n_cases"] == 3

    val_run = call_tool("validation_scorecard", {"subsystem": "unit_root"})
    assert val_run["all_passed"] is True
    assert val_run["n_cases"] == 3
    assert val_run["n_passed"] == 3
    json.dumps(val_run, allow_nan=False)


# ---------------------------------------------------------------------------
# End-to-end subprocess stdio smoke test
# ---------------------------------------------------------------------------


def test_subprocess_cli_smoke_list_and_stdio(tmp_path: Path) -> None:
    list_proc = subprocess.run(
        [sys.executable, "-m", "puremacro.mcp_server", "--list-tools"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    listed = json.loads(list_proc.stdout)
    assert len(listed["tools"]) == 8

    csv_path = tmp_path / "panel.csv"
    pd.DataFrame(_sample_panel_rows()).to_csv(csv_path, index=False)

    ndjson_in = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": MCP_PROTOCOL_VERSION}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "preflight_panel",
                        "arguments": {
                            "data_path": "panel.csv",
                            "unit_col": "country",
                            "time_col": "date",
                            "run_stationarity": False,
                            "run_seasonality": False,
                        },
                    },
                }
            ),
        ]
    ) + "\n"

    stdio_proc = subprocess.run(
        [sys.executable, "-m", "puremacro.mcp_server", "--allowed-root", str(tmp_path)],
        cwd=str(REPO_ROOT),
        input=ndjson_in,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    responses = [json.loads(line) for line in stdio_proc.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in responses] == [1, 2, 3]
    assert responses[0]["result"]["serverInfo"]["name"] == "puremacro"
    assert len(responses[1]["result"]["tools"]) == 8
    assert responses[2]["result"]["structuredContent"]["panel_shape"][0] == 8
