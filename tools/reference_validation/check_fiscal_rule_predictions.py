"""Measure the registered phase-2 fiscal-rule predictions on the coarse 401 x 21 grid.

Reads ``tests/fixtures/sovereign_fiscal/rule_predictions_protocol.json``, which
was committed before any fiscal-rule solve, and never changes it: thresholds,
tolerances, design and definitions come from that file. Writes a record with
the outcomes and, for each prediction, whether it held. A failed prediction
is reported as found. Rules are compared only within one grid; the grid is
coarse, so moments differ from converged ones.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from puremacro.sovereign import FiscalRule, solve_fiscal_sovereign_default
from puremacro.sovereign._fiscal_paths import MOMENTS, draw_uniforms, ergodic_moments, simulate_paths
from tools.reference_validation.check_arellano_grids import paired_comparison

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "rule_predictions_protocol.json"
RECORD = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "rule_predictions_record.json"
TOL = 1e-12


def _rule(spec: dict | None) -> FiscalRule | None:
    return None if spec is None else FiscalRule(**spec)


def solve(protocol: dict, name: str, n_b: int, n_y: int):
    cal = {k: v for k, v in protocol["calibration"].items() if k != "note"}
    return solve_fiscal_sovereign_default(**cal, n_b=n_b, n_y=n_y, rule=_rule(protocol["rules"][name]),
                                          policy_search=protocol["policy_search"][name])


def _paths(result, design):
    uniforms = draw_uniforms(design["seeds"], design["periods"])
    return simulate_paths(result, int(result.metadata["zero_index"]), len(result.y_grid) // 2, uniforms)


def _runs(paths, design) -> list[dict]:
    return [dict(seed=s, moments=m) for s, m in zip(design["seeds"], ergodic_moments(paths, design["burn"]))]


def _binding(result, paths, burn: int, cap: float | None, limit: float | None) -> dict:
    sl = slice(burn, None)
    rep = paths.repaid[:, sl]
    b_next = result.b_grid[paths.b_next_index[:, sl]]
    b_now = result.b_grid[paths.b_index[:, sl]]
    violation = (b_next > cap + TOL) if cap is not None else (b_next - b_now > limit + TOL)
    binding = rep & violation
    per_seed = 100.0 * binding.sum(axis=1) / rep.sum(axis=1)
    booms = paths.y[:, sl][binding] > 1.0
    return dict(binding_share_pct=float(100.0 * binding.sum() / rep.sum()), per_seed_pct=per_seed.tolist(),
                binding_periods=int(binding.sum()), repayment_periods=int(rep.sum()),
                binding_in_booms=float(booms.mean()) if booms.size else float("nan"))


def _max_b_recession(result, paths, burn: int, threshold: float) -> float:
    sl = slice(burn, None)
    mask = paths.repaid[:, sl] & (paths.y[:, sl] < threshold)
    return float(result.b_grid[paths.b_next_index[:, sl]][mask].max()) if mask.any() else float("nan")


def run(output: Path) -> dict:
    protocol_text = PROTOCOL.read_text()
    protocol = json.loads(protocol_text)
    design = protocol["design"]
    n_b, n_y = design["grid"]["n_b"], design["grid"]["n_y"]
    tolerances = protocol["tolerances"]
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                  repository_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
                  protocol_sha256=hashlib.sha256(protocol_text.encode()).hexdigest(), protocol=protocol,
                  label=design["grid"]["label"], solves={}, cases={}, measurements={}, predictions={})
    results, paths = {}, {}
    for name in protocol["rules"]:
        res = solve(protocol, name, n_b, n_y)
        results[name], paths[name] = res, _paths(res, design)
        record["solves"][name] = dict(converged=bool(res.converged), iterations=int(res.iterations),
                                      residual=float(res.residual), policy_search=res.metadata["policy_search"],
                                      monotone_certification_gap=res.metadata["monotone_certification_gap"],
                                      rule_infeasible_states=res.metadata["rule_infeasible_states"])
        runs = _runs(paths[name], design)
        record["cases"][name] = dict(runs=runs, moments={k: float(np.mean([r["moments"][k] for r in runs]))
                                                         for k in MOMENTS})
        print(name, record["solves"][name], {k: round(v, 4) for k, v in record["cases"][name]["moments"].items()},
              flush=True)

    rules, burn = protocol["rules"], design["burn"]
    free, free_paths = results["none"], paths["none"]
    m = record["measurements"]
    m["ceiling_binding"] = _binding(free, free_paths, burn, rules["ceiling"]["debt_ceiling"], None)
    m["deficit_binding"] = _binding(free, free_paths, burn, None, rules["deficit"]["deficit_limit"])
    escape = rules["ceiling_escape"]["escape_below"]
    m["max_b_recession"] = {f"{n_b}x{n_y}": _max_b_recession(free, free_paths, burn, escape)}
    for grid in design["escape_cross_check_grids"]:
        key = f"{grid[0]}x{grid[1]}"
        if key not in m["max_b_recession"]:
            res = solve(protocol, "none", grid[0], grid[1])
            m["max_b_recession"][key] = _max_b_recession(res, _paths(res, design), burn, escape)
    soft, soft_paths = results["soft_ceiling"], paths["soft_ceiling"]
    cal = protocol["calibration"]
    rep = soft_paths.repaid[:, burn:]
    g = soft_paths.g[:, burn:][rep]
    m["penalty_output_equivalent"] = float(np.mean(rules["soft_ceiling"]["penalty"] * g ** cal["gamma_g"] / cal["psi"]))
    m["soft_ceiling_mean_g"] = float(g.mean())
    # Informational (added after the first run, not a registered prediction): how often the
    # soft and escape-clause models actually choose debt above the cap.
    cap_value = rules["ceiling"]["debt_ceiling"]
    for name in ("soft_ceiling", "ceiling_escape"):
        res_k, paths_k = results[name], paths[name]
        rep_k = paths_k.repaid[:, burn:]
        above = res_k.b_grid[paths_k.b_next_index[:, burn:]] > cap_value + TOL
        m[f"{name}_above_cap_share_pct"] = float(100.0 * (rep_k & above).sum() / rep_k.sum())
    # Informational, not graded: off the ergodic set the soft rule differs from the hard one.
    hard = results["ceiling"]
    above_states = (soft.policy_b > cap_value + TOL) & (soft.default_prob < 0.5)
    at_or_below = soft.b_grid <= cap_value
    m["soft_ceiling_policy_above_cap"] = dict(
        repayment_states=int(above_states.sum()),
        min_current_debt=float(soft.b_grid[np.nonzero(above_states)[0]].min()) if above_states.any() else float("nan"),
        policies_equal_at_or_below_cap=bool(np.array_equal(soft.policy_index[at_or_below],
                                                           hard.policy_index[at_or_below])),
        note="repayment states (default_prob < 0.5) where the soft-ceiling policy chooses b' above the cap; "
             "informational, not a registered prediction")
    m["escape_vs_ceiling"] = paired_comparison(record["cases"]["ceiling"], record["cases"]["ceiling_escape"],
                                               tolerances=tolerances)

    p = protocol["predictions"]
    lo, hi = p["ceiling_binding_share_pct"]
    dlo, dhi = p["deficit_binding_share_pct"]
    cap = rules["ceiling"]["debt_ceiling"]
    record["predictions"] = {
        "escape_inert": dict(
            held=bool(all(v < cap for v in m["max_b_recession"].values()) and m["escape_vs_ceiling"]["passed"]),
            max_b_recession=m["max_b_recession"], escape_vs_ceiling_passed=m["escape_vs_ceiling"]["passed"]),
        "ceiling_binding_share_pct": dict(held=bool(lo <= m["ceiling_binding"]["binding_share_pct"] <= hi),
                                          predicted=[lo, hi], found=m["ceiling_binding"]["binding_share_pct"]),
        "ceiling_binding_in_booms": dict(held=bool(m["ceiling_binding"]["binding_in_booms"] == p["ceiling_binding_in_booms"]),
                                         predicted=p["ceiling_binding_in_booms"],
                                         found=m["ceiling_binding"]["binding_in_booms"]),
        "deficit_binding_share_pct": dict(held=bool(dlo <= m["deficit_binding"]["binding_share_pct"] <= dhi),
                                          predicted=[dlo, dhi], found=m["deficit_binding"]["binding_share_pct"]),
        "penalty_output_equivalent": dict(held=None, predicted=p["penalty_output_equivalent"],
                                          found=m["penalty_output_equivalent"],
                                          note="registered as 'about 0.002' with no numeric band; reported, not graded"),
    }
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record["predictions"], indent=2), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECORD)
    run(parser.parse_args().output)
