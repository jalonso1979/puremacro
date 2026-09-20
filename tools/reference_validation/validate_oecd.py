"""Validate OECD 2023/2019 native ICIO, then a conserving 3-region/3-sector model.

--source reads the original provider CSV/ZIP and refreshes the small empirical
fixture. Without it, only the frozen aggregation and its model checks are run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from puremacro.trade import solve_trade_equilibrium
from puremacro.trade._oecd_icio import read_native, condense_final_demand, SOURCE_URL, README_URL
from puremacro.trade.data import RawIOData, package_mrio_to_calibration_result, verify_accounting_invariants, load_oecd_icio_granular

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/trade/oecd_2023"
ARCHIVE_URL = "https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip"
TOL = 1e-5  # Absolute residuals in current million USD: ten dollars.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(raw):
    """Partition EVERY economy/activity; preserve rest-of-world transactions."""
    countries, sectors = ["USA", "CHN", "REST"], ["RESOURCES", "MANUFACTURING", "SERVICES"]
    ci = np.array([countries.index(c) if c in countries else 2 for c in raw.countries])
    si = np.array([0 if s[0] in "AB" else 1 if s[0] == "C" else 2 for s in raw.sectors])
    node = (ci[:, None]*3 + si[None, :]).ravel()
    fd = (ci[:, None]*raw.K_F + np.arange(raw.K_F)[None, :]).ravel()
    P = csr_matrix((np.ones(raw.M), (node, np.arange(raw.M))), shape=(9, raw.M))
    Q = csr_matrix((np.ones(raw.C*raw.K_F), (fd, np.arange(raw.C*raw.K_F))), shape=(3*raw.K_F, raw.C*raw.K_F))
    result = replace(raw, countries=countries, sectors=sectors,
                     intermediate_matrix=(P @ (P @ raw.Z).T).T,
                     final_demand_matrix=(Q @ (P @ raw.F).T).T,
                     value_added=P @ raw.VA, taxes_less_subsidies=P @ raw.TLS,
                     gross_output=P @ raw.gross_output,
                     taxes_less_subsidies_fd=Q @ raw.taxes_less_subsidies_fd,
                     metadata={**raw.metadata, "aggregation": {
                         "countries": {c: countries[i] for c, i in zip(raw.countries, ci)},
                         "sectors": {s: sectors[i] for s, i in zip(raw.sectors, si)},
                         "rest_includes_provider_ROW": True,
                         "timing": "aggregate original flows before regularization"}})
    for field in ("Z", "F", "VA", "TLS", "gross_output", "taxes_less_subsidies_fd"):
        np.testing.assert_allclose(np.sum(getattr(result, field)), np.sum(getattr(raw, field)), rtol=2e-14, atol=1e-7)
    return result


def save_fixture(raw, full_result, readme=None):
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "2019_3regions_3sectors.npz"
    np.savez_compressed(path, countries=raw.countries, sectors=raw.sectors, fd_categories=raw.fd_categories,
                        Z=raw.Z, F=raw.F, VA=raw.VA, TLS=raw.TLS, Y=raw.gross_output,
                        TLS_fd=raw.taxes_less_subsidies_fd)
    metadata = dict(raw.metadata)
    metadata.pop("file_path", None)
    manifest = {"source_url": SOURCE_URL, "download_url": ARCHIVE_URL, "member": "2019_SML.csv",
                "readme_url": README_URL, "retrieved_utc_date": "2026-09-20",
                "edition": "2023 regular", "year": 2019, "unit": "current million USD",
                "source_metadata": metadata, "fixture_sha256": sha256(path),
                "full_native_validation": full_result,
                "retrieval_note": "Official ZIP downloaded in Safari; automatically extracted. CSV SHA256 identifies input bytes."}
    if readme:
        manifest["readme_sha256"] = sha256(readme)
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_fixture():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    path = FIXTURES / "2019_3regions_3sectors.npz"
    if sha256(path) != manifest["fixture_sha256"]:
        raise ValueError("OECD fixture checksum mismatch")
    with np.load(path, allow_pickle=False) as r:
        return RawIOData(countries=list(r["countries"]), sectors=list(r["sectors"]),
                         fd_categories=list(r["fd_categories"]), intermediate_matrix=r["Z"],
                         final_demand_matrix=r["F"], value_added=r["VA"], taxes_less_subsidies=r["TLS"],
                         gross_output=r["Y"], taxes_less_subsidies_fd=r["TLS_fd"],
                         year=2019, source="OECD_ICIO_2023_regular_aggregated", unit="M_USD",
                         metadata=manifest["source_metadata"])


def _scaled(raw, scale):
    return replace(raw, intermediate_matrix=raw.Z*scale, final_demand_matrix=raw.F*scale,
                   value_added=raw.VA*scale, taxes_less_subsidies=raw.TLS*scale,
                   gross_output=raw.gross_output*scale, taxes_less_subsidies_fd=raw.taxes_less_subsidies_fd*scale,
                   unit="B_USD" if scale == .001 else raw.unit)


def validate_aggregate(raw, accounting="legacy"):
    raw = condense_final_demand(raw)
    calib = package_mrio_to_calibration_result(raw)
    report = {"accounting": verify_accounting_invariants(calib), "adjustments": calib.metadata["adjustments"],
              "final_demand_mapping": calib.metadata["final_demand_mapping"], "checks": {}}
    checks = report["checks"]

    def check(name, got, expected, *, atol=1e-8, rtol=1e-10):
        got, expected = np.asarray(got), np.asarray(expected)
        checks[name] = {"max_abs_error": float(np.max(np.abs(got-expected))),
                        "passed": bool(np.isfinite(got).all() and np.allclose(got, expected, atol=atol, rtol=rtol))}

    kwargs = dict(tariff_revenue_mode="schedule", accounting=accounting, tol=TOL, max_iter=100)
    base = solve_trade_equilibrium(calib, **kwargs)
    check("baseline_output_recovers_transactions", base.y_sol, calib.ytot)
    check("baseline_prices", base.p_sol, np.ones_like(base.p_sol))
    check("baseline_wages", base.w_sol, np.ones_like(base.w_sol))
    check("baseline_rents", base.r_sol, np.ones_like(base.r_sol))
    check("baseline_tariffs", base.tariffs, np.zeros(calib.nc))
    # Independent linear accounting oracle on the calibrated table.
    Z, F = calib.data_calibra[:9, :9], calib.data_calibra[:9, 9:]
    Y = Z.sum(axis=1) + F.sum(axis=1)
    A = Z / Y[None, :]
    check("leontief_inverse_baseline", np.linalg.solve(np.eye(9)-A, F.sum(axis=1)), Y)
    tau = np.ones((9, 3, 3)); tau[3:, :, 0] = 1.1
    tau_fd = np.ones((9, 3, 3)); tau_fd[3:, :, 0] = 1.1
    counter = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, **kwargs)
    pac = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="keller_pac", max_steps=10, **kwargs)
    for field in ("p_sol", "w_sol", "r_sol", "y_sol", "tariffs"):
        check(f"newton_pac_{field}", getattr(counter, field), getattr(pac, field), atol=1e-5)
    scaled_calib = package_mrio_to_calibration_result(_scaled(raw, .001))
    scaled = solve_trade_equilibrium(scaled_calib, tau=tau, tau_fd=tau_fd,
                                     tariff_revenue_mode="schedule", accounting=accounting, tol=TOL*.001, max_iter=100)
    for field in ("p_sol", "w_sol", "r_sol"):
        check(f"currency_unit_invariance_{field}", getattr(counter, field), getattr(scaled, field))
    for field in ("y_sol", "tariffs"):
        check(f"currency_unit_covariance_{field}", getattr(counter, field)*.001, getattr(scaled, field), atol=1e-7)
    # Independently sum the returned transaction quantities and schedule charges.
    inter = counter.intermediate_flows.transpose(1, 0, 3, 2).reshape(9, 9)
    final = counter.final_demand_flows.transpose(1, 0, 3, 2).reshape(9, 9)
    check("counterfactual_goods_clearing", inter.sum(1)+final.sum(1), counter.y_sol.ravel(order="F"), atol=1e-5)
    price = counter.p_sol.ravel(order="F")
    value_i = inter*price[:, None]
    if accounting == "consistent":
        value_f = final*price[:, None]
    else:
        final_price = np.einsum("i,ikc,ikc->kc", price, calib.afd, tau_fd).ravel(order="F")
        value_f = final*final_price[None, :]
    rev_i = (value_i*(tau-1).reshape(9, 9, order="F")).sum(0).reshape(3, 3).sum(1)
    rev_f = (value_f*(tau_fd-1).reshape(9, 9, order="F")).sum(0).reshape(3, 3).sum(1)
    check("counterfactual_schedule_receipts", rev_i+rev_f, counter.metadata["fiscal_tariffs"], atol=1e-5)
    check("non_importers_receive_no_tariffs", counter.tariffs[1:], np.zeros(2))
    report["solvers"] = {name: {"converged": bool(sol.converged), "residual_norm": float(sol.residual_norm)}
                         for name, sol in (("baseline", base), ("tariff_newton", counter), ("tariff_pac", pac), ("tariff_billion_usd", scaled))}
    report["counterfactual"] = {"description": "USA levies 10% on all foreign intermediate and final purchases",
                                "tariff_revenue_mode": "schedule", "accounting": accounting, "sigma": 0., "unit": "current million USD",
                                "prices": counter.p_sol.reshape(3, 3).tolist(),
                                "output_percent_change": (100*(counter.y_sol/calib.ytot-1)).reshape(3, 3).tolist(),
                                "tariff_revenue": counter.tariffs.tolist()}
    if accounting == "consistent":
        report["economic_accounts"] = counter.metadata["account_residuals"]
    report["passed"] = bool(report["accounting"]["valid"] and all(c["passed"] for c in checks.values())
                            and all(s["converged"] for s in report["solvers"].values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--accounting", choices=("legacy", "consistent"), default="legacy")
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source:
        native = read_native(args.source, 2019)
        full = load_oecd_icio_granular(year=2019, file_path=args.source)
        full_result = {"shape": [native.M, native.M + native.C*native.K_F],
                       "provider_accounting": native.metadata["provider_accounting"],
                       "calibrated_accounting": verify_accounting_invariants(full),
                       "adjustments": full.metadata["adjustments"],
                       "calibrated_final_uses": list(full.metadata["fd_categories"])}
        if not full_result["calibrated_accounting"]["valid"]:
            raise ValueError("Full native calibration fails accounting")
        save_fixture(aggregate(native), full_result, args.readme)
    report = validate_aggregate(load_fixture(), accounting=args.accounting)
    report["native_archive_reloaded_this_run"] = args.source is not None
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
