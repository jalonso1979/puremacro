"""Synthetic three-date sovereign-bank equilibria and separately labelled stress.

Run: python -m puremacro.examples.sovereign_banking --output DIR --lang en|es
The joint state distribution is fixed; refinement only concerns the debt grid.
"""
from __future__ import annotations

from puremacro._numeric_contracts import thaw as _thaw

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from puremacro.sovereign.banking import (
    SovereignBankingEquilibrium, SovereignBankingParameters, SovereignBankingResult,
    solve_sovereign_banking, sovereign_bank_accounts,
)

INCOME = [.7, 1., 1.3]
PRODUCTIVITY = [.6, .9, 1.2]
PROBABILITIES = [.25, .5, .25]
SHARES = (0., .2, .4, .6)
GRIDS = (51, 101, 201)
REFINEMENT_LIMITS = dict(value=.005, debt=.015, price=.05, default_probability=.05, expected_credit=.015)


def save_result(result: SovereignBankingResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    scalar_names = ("debt_index", "mask", "price", "initial_consumption", "initial_reserves", "value")
    record = dict(parameters=asdict(result.parameters), price_selection=result.price_selection,
                  decision_tolerance=result.decision_tolerance, selected_index=result.selected_index,
                  fingerprint=result.fingerprint, diagnostics=_thaw(result.diagnostics),
                  equilibria=[{name: getattr(e, name) for name in scalar_names} for e in result.equilibria])
    np.savez_compressed(directory/"arrays.npz", income=result.income, productivity=result.productivity,
        probabilities=result.probabilities, debt_grid=result.debt_grid, candidates=result.candidates,
        selected_indices=result.selected_indices, accounts=np.stack([e.accounts for e in result.equilibria]),
        default=np.stack([e.default for e in result.equilibria]))
    result.equilibria_frame().to_csv(directory/"equilibria.csv", index=False)
    result.candidates_frame().to_csv(directory/"candidates.csv", index=False)
    result.debt_frame().to_csv(directory/"debt.csv", index=False)
    result.to_frame().to_csv(directory/"selected-accounts.csv")
    result.counterfactual_frame().to_csv(directory/"conditional-default.csv", index=False)
    pd.concat([e.to_frame().assign(equilibrium=i) for i, e in enumerate(result.equilibria)]).to_csv(
        directory/"all-accounts.csv")
    record["data_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(directory.iterdir()) if p.suffix in (".npz", ".csv")}
    (directory/"record.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


def load_result(directory: Path) -> SovereignBankingResult:
    """Reconstruct exact saved arrays; verify fingerprints and file hashes first."""
    record = json.loads((directory/"record.json").read_text())
    for name, digest in record["data_hashes"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"saved data hash mismatch: {name}")
    with np.load(directory/"arrays.npz", allow_pickle=False) as data:
        entries = tuple(SovereignBankingEquilibrium(**entry, accounts=data["accounts"][i],
                                                   default=data["default"][i])
                        for i, entry in enumerate(record["equilibria"]))
        result = SovereignBankingResult(SovereignBankingParameters(**record["parameters"]),
            data["income"], data["productivity"], data["probabilities"], data["debt_grid"],
            record["price_selection"], record["decision_tolerance"], entries, data["candidates"],
            data["selected_indices"], record["selected_index"], record["diagnostics"])
    if result.fingerprint != record["fingerprint"]:
        raise ValueError("saved model fingerprint mismatch")
    return result


def run(output: Path, lang: str = "en") -> dict[str, Any]:
    if lang not in ("en", "es"):
        raise ValueError("lang must be en or es")
    output.mkdir(parents=True, exist_ok=True)
    cases: dict[tuple[float, str, int], SovereignBankingResult] = {}
    summaries = []
    for share in SHARES:
        for selection in ("highest", "lowest"):
            for nodes in GRIDS:
                result = solve_sovereign_banking(income=INCOME, productivity=PRODUCTIVITY,
                    probabilities=PROBABILITIES, debt_grid=np.linspace(0, .6, nodes),
                    price_selection=selection, parameters=SovereignBankingParameters(domestic_share=share))
                cases[share, selection, nodes] = result
                name = f"share-{share:.1f}-{selection}-{nodes}"
                save_result(result, output/name)
                row = result.equilibria_frame().query("selected").iloc[0].to_dict()
                row.update(case=name, share=share, selection=selection, nodes=nodes,
                    multiple_debt_points=result.diagnostics["multiple_equilibrium_debt_points"],
                    missing_debt_points=result.diagnostics["debt_points_without_pure_equilibrium"],
                    boundary=result.diagnostics["selected_upper_debt_boundary"],
                    max_error=result.diagnostics["max_error"])
                summaries.append(row)
    selected = pd.DataFrame(summaries)
    selected.to_csv(output/"selected.csv", index=False)
    refinements = []
    for share in SHARES:
        for selection in ("highest", "lowest"):
            data = selected[(selected.share == share) & (selected.selection == selection)].sort_values("nodes")
            for i in range(1, len(data)):
                earlier, later = data.iloc[i-1], data.iloc[i]
                changes = {key: abs(float(later[key])-float(earlier[key])) for key in REFINEMENT_LIMITS}
                refinements.append(dict(share=share, selection=selection, from_nodes=int(earlier.nodes),
                    to_nodes=int(later.nodes), **changes,
                    passed=all(changes[key] <= limit for key, limit in REFINEMENT_LIMITS.items())))
    refinement = pd.DataFrame(refinements)
    refinement.to_csv(output/"debt-refinement.csv", index=False)
    stress = []
    for share in SHARES:
        p = SovereignBankingParameters(initial_equity=.2, domestic_share=share, leverage=3.)
        for default in (False, True):
            row = sovereign_bank_accounts(p, income=[1.], productivity=[4.], debt=.25, price=.8,
                                           default=[default]).iloc[0].to_dict()
            row.update(share=share, debt=.25, price=.8, scope="fixed-contract stress; no repricing")
            stress.append(row)
    pd.DataFrame(stress).to_csv(output/"fixed-contract-stress.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    spanish = lang == "es"
    for share, color in zip(SHARES, ("#697386", "#007c91", "#d47b19", "#9a416f")):
        result = cases[share, "highest", GRIDS[-1]]
        eq = result.equilibria_frame()
        chosen = eq[eq.selected_at_debt]
        axes[0, 0].plot(chosen.debt, chosen.price, color=color, label=f"s={share:.1f}")
    axes[0, 0].set(title="Precio mayor por deuda" if spanish else "Highest price at each debt",
                   xlabel="Deuda B" if spanish else "Debt B", ylabel="q")
    axes[0, 0].legend()
    eq = cases[.4, "highest", GRIDS[-1]].equilibria_frame()
    colors = axes[0, 1].scatter(eq.debt, eq.price, c=eq.default_probability, s=10, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(colors, ax=axes[0, 1], label="Probabilidad de default" if spanish else "Default probability")
    axes[0, 1].set(title="Todos los precios puros, s=0.4" if spanish else "All pure prices, s=0.4",
                   xlabel="Deuda B" if spanish else "Debt B", ylabel="q")
    contract = pd.DataFrame(stress)
    for default, label in ((False, "Pago" if spanish else "Repayment"), (True, "Default")):
        frame = contract[contract.default == default]
        axes[1, 0].plot(frame.share, frame.credit, marker="o", label=label)
    axes[1, 0].set(title="Crédito: estrés a B=.25, q=.8" if spanish else "Credit: stress at B=.25, q=.8",
                   xlabel="Participación doméstica s" if spanish else "Domestic share s",
                   ylabel="Crédito L" if spanish else "Credit L")
    axes[1, 0].legend()
    for share in SHARES:
        frame = selected[(selected.share == share) & (selected.selection == "highest")]
        axes[1, 1].plot(frame.nodes, frame.debt, marker="o", label=f"s={share:.1f}")
    axes[1, 1].set(title="Deuda elegida: refinamiento" if spanish else "Chosen debt: grid refinement",
                   xlabel="Nodos de deuda" if spanish else "Debt nodes", ylabel="B")
    axes[1, 1].legend()
    axes[1, 1].text(.03, .92, "Las ocho especificaciones coinciden" if spanish else
                    "All eight specifications coincide", transform=axes[1, 1].transAxes, fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Riesgo soberano y bancos: economía de tres fechas" if spanish
                 else "Sovereign risk and banks: a three-date economy", fontsize=15)
    fig.savefig(output/"sovereign-banking.png", dpi=160)
    fig.savefig(output/"sovereign-banking.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    from puremacro.sovereign import banking, _banking_audit
    module_paths = [Path(banking.__file__), Path(_banking_audit.__file__), Path(__file__)]
    final_refinement = refinement[refinement.to_nodes == GRIDS[-1]]
    summary = dict(language=lang, python=platform.python_version(), numpy=np.__version__,
        source_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in module_paths},
        states=dict(income=INCOME, productivity=PRODUCTIVITY, probabilities=PROBABILITIES),
        shares=list(SHARES), debt_grids=list(GRIDS), refinement_limits=REFINEMENT_LIMITS,
        final_refinements_passed=bool(final_refinement.passed.all()),
        maximum_audit_error=float(selected.max_error.max()), cases=len(cases),
        final_selected=selected[selected.nodes == GRIDS[-1]].to_dict("records"),
        limits="synthetic finite-state, finite-debt-grid, three-date model; mandated portfolios; no mixed equilibria")
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", choices=("en", "es"), default="en")
    args = parser.parse_args()
    summary = run(args.output, args.lang)
    print(json.dumps({key: summary[key] for key in ("cases", "maximum_audit_error", "final_refinements_passed")}, indent=2))


if __name__ == "__main__":
    main()
