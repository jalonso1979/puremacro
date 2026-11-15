"""Pyodide-contract smoke: the econometrics subsystems used by the browser
playground demos import cleanly and run on synthetic data WITHOUT pulling in any
forbidden (non-Pyodide) module. Marked pyodide_smoke so the in-browser Gate 6
runner (tools/pyodide/runner.js) also executes it.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

_FORBIDDEN = ("statsmodels", "linearmodels", "arch", "bs4", "requests", "numba")


@pytest.mark.pyodide_smoke
def test_playground_econ_subsystems_are_pyodide_clean():
    # Measure the DELTA: a forbidden module imported by an unrelated test earlier
    # in the same process must not be misattributed to these subsystems.
    before = set(sys.modules)
    from puremacro.var.identify.cholesky import cholesky_svar
    from puremacro.lp.jorda import lp_hac
    from puremacro.garch import garch11_fit
    from puremacro.gar import qar
    from puremacro.did import callaway_santanna

    rng = np.random.default_rng(0)

    # SVAR on a tiny synthetic VAR
    Y = rng.standard_normal((120, 3)).cumsum(0)
    svar = cholesky_svar(Y, p=2, horizon=8, n_boot=50, ci=0.9, seed=0)
    assert svar.irf_point.shape == (9, 3, 3)

    # LP-HAC
    df = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(120)})
    lp = lp_hac(df, y="y", x="shock", horizons=range(0, 6), n_lags=2)
    assert {"h", "beta", "lo", "hi"}.issubset(lp.columns)

    # GARCH(1,1)
    g = garch11_fit(pd.Series(rng.standard_normal(300)))
    assert g.persistence < 1.0

    # Quantile-AR
    q = qar(rng.standard_normal(200), quantiles=(0.5,), horizons=(1,), p=1,
            n_boot=20, seed=0)
    assert len(q) >= 1

    # Callaway-Sant'Anna on a tiny staggered panel
    rows = []
    for i in range(40):
        gi = (5, 8, np.nan)[i % 3]
        for t in range(1, 11):
            treated = 1.0 if (not pd.isna(gi)) and t >= gi else 0.0
            rows.append({"unit": i, "time": t,
                         "y": float(i % 5) + 0.1 * t + treated + 0.05 * rng.standard_normal(),
                         "treat_time": gi})
    cs = callaway_santanna(pd.DataFrame(rows), unit="unit", time="time",
                           outcome="y", treat_time="treat_time", n_boot=50, seed=0)
    assert np.isfinite(cs.att_overall)

    bad = [m for m in _FORBIDDEN if m in sys.modules and m not in before]
    assert bad == [], f"forbidden modules imported: {bad}"


@pytest.mark.pyodide_smoke
def test_workbench_hank_fiscal_cce_mg_pyodide_clean():
    """HANK fiscal solver, CCE mean-group LP, and workbench dossier stay Pyodide-clean."""
    before = set(sys.modules)

    from puremacro.lp.cce import cce_mg_panel_lp, cce_panel_lp
    from puremacro.models.hank_fiscal import solve_hank_fiscal
    from puremacro.workbench import run_empirical_structural_workbench

    fiscal_res = solve_hank_fiscal(
        closure="debt_rule",
        tax_type="proportional",
        horizon=40,
        n_a=50,
        a_max=100.0,
        tol=1e-6,
    )
    assert fiscal_res.converged
    assert fiscal_res.solver_diagnostics is not None
    assert fiscal_res.solver_diagnostics.overall_status == "PASS"

    rng = np.random.default_rng(7)
    n_units = 6
    n_periods = 32
    n_horizons = 5
    true_irf = np.asarray(fiscal_res.dY[:n_horizons], dtype=float) / 0.01
    factor_t = rng.normal(0.0, 0.8, size=n_periods)
    rows: list[dict[str, float | int | str]] = []
    for i in range(n_units):
        unit = f"U{i:02d}"
        shocks = rng.normal(0.0, 1.0, size=n_periods)
        x_series = shocks + (0.4 + 0.1 * i) * factor_t
        y_series = (0.6 + 0.1 * i) * factor_t + rng.normal(0.0, 0.2, size=n_periods)
        for h_lag in range(n_horizons):
            shifted = np.roll(shocks, h_lag)
            shifted[:h_lag] = 0.0
            y_series = y_series + true_irf[h_lag] * shifted
        for t in range(n_periods):
            rows.append({"unit": unit, "period": t, "y": float(y_series[t]), "x": float(x_series[t])})

    panel_df = pd.DataFrame(rows)
    cce_res = cce_panel_lp(
        panel_df,
        y="y",
        x="x",
        horizons=range(n_horizons),
        n_lags=1,
        estimator="mean_group",
        unit_col="unit",
        time_col="period",
    )
    assert len(cce_res) == n_horizons
    assert cce_res["beta"].notna().all()

    dossier = run_empirical_structural_workbench(
        panel_df,
        y="y",
        x="x",
        unit_col="unit",
        time_col="period",
        horizons=range(n_horizons),
        lags=1,
        estimator=cce_mg_panel_lp,
        run_sensitivity=False,
        model_irf=pd.DataFrame({"h": np.arange(n_horizons, dtype=int), "point": true_irf}),
        solver_result=fiscal_res,
    )
    assert dossier.passed
    assert dossier.irf_scorecard is not None
    score_df = dossier.irf_scorecard.to_frame()
    assert not score_df.empty
    assert "Structural IRF Scorecard" in dossier.to_markdown()

    bad = [m for m in _FORBIDDEN if m in sys.modules and m not in before]
    assert bad == [], f"forbidden modules imported: {bad}"


@pytest.mark.pyodide_smoke
def test_cli_and_mcp_server_pyodide_clean(tmp_path):
    """CLI offline preflight and MCP JSON-RPC initialize/tools/list stay Pyodide-clean."""
    import io
    import json

    before = set(sys.modules)

    from puremacro import cli, mcp_server

    csv_path = tmp_path / "panel.csv"
    out_json = tmp_path / "preflight.json"
    pd.DataFrame(
        {
            "unit": ["A"] * 16 + ["B"] * 16,
            "period": list(range(16)) * 2,
            "y": np.linspace(0.0, 3.0, 32),
            "x": np.sin(np.arange(32, dtype=float)),
        }
    ).to_csv(csv_path, index=False)

    parser = cli.create_parser()
    parsed = parser.parse_args(["preflight", str(csv_path), "--format", "json"])
    assert parsed.subcommand == "preflight"

    rc = cli.main(
        [
            "preflight",
            str(csv_path),
            "--unit-col",
            "unit",
            "--time-col",
            "period",
            "--format",
            "json",
            "-o",
            str(out_json),
            "--quiet",
        ]
    )
    assert rc == 0
    preflight_payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert preflight_payload["balanced"] is True

    stdin_buf = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": mcp_server.MCP_PROTOCOL_VERSION},
            }
        )
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        + "\n"
    )
    stdout_buf = io.StringIO()
    mcp_rc = mcp_server.serve_stdio(stdin=stdin_buf, stdout=stdout_buf, allowed_root=str(tmp_path))
    assert mcp_rc == 0
    responses = [json.loads(line) for line in stdout_buf.getvalue().strip().splitlines()]
    assert len(responses) == 2
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["serverInfo"]["name"] == "puremacro"
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert {"preflight_panel", "lp_hac", "validation_scorecard"}.issubset(tool_names)

    bad = [m for m in _FORBIDDEN if m in sys.modules and m not in before]
    assert bad == [], f"forbidden modules imported: {bad}"
