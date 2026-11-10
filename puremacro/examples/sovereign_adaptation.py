"""Synthetic adaptation/debt choices, joint grid refinement and domain checks.

Run: python -m puremacro.examples.sovereign_adaptation --output DIR --lang en|es
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

from puremacro.sovereign.adaptation import (
    AdaptationDistribution, SovereignAdaptationParameters, SovereignAdaptationResult,
    solve_sovereign_adaptation,
)

ARRAYS = ("income","damage","disaster","transition","debt_grid","adaptation_grid","access_value",
          "repayment_value","excluded_value","default","repayment_debt_policy","repayment_adaptation_policy",
          "excluded_adaptation_policy","prices")
GRIDS = ((21,11),(41,21),(81,41))
CASES = ("baseline","grant","relief","no_adaptation","no_borrowing")
LIMITS = dict(value=.005,initial_investment=.015,total_investment=.03,
              cumulative_default_probability=.03,cumulative_damage=.03)


def save_result(r: SovereignAdaptationResult, d: AdaptationDistribution, directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    arrays = {name:getattr(r,name) for name in ARRAYS}
    arrays.update(access_mass=d.access_mass,excluded_mass=d.excluded_mass,initial_probabilities=d.initial_probabilities)
    np.savez_compressed(directory/"arrays.npz",**arrays)
    r.to_frame().to_csv(directory/"initial-policies.csv",index=False)
    r.to_frame(excluded=True).to_csv(directory/"initial-excluded-policies.csv",index=False)
    r.pricing_frame().to_csv(directory/"initial-prices.csv",index=False)
    d.to_frame().to_csv(directory/"expected-path.csv")
    record = dict(parameters=asdict(r.parameters),horizon=r.horizon,decision_tolerance=r.decision_tolerance,
        fingerprint=r.fingerprint,diagnostics=_thaw(r.diagnostics),distribution_diagnostics=_thaw(d.diagnostics),
        initial_debt_index=d.initial_debt_index,initial_adaptation_index=d.initial_adaptation_index,initial_access=d.initial_access,
        data_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
                     if p.suffix in (".csv",".npz")})
    (directory/"record.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


def load_result(directory: Path) -> tuple[SovereignAdaptationResult,AdaptationDistribution]:
    record = json.loads((directory/"record.json").read_text())
    for name,digest in record["data_hashes"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"saved data hash mismatch: {name}")
    with np.load(directory/"arrays.npz",allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    r = SovereignAdaptationResult(parameters=SovereignAdaptationParameters(**record["parameters"]),
        horizon=record["horizon"],decision_tolerance=record["decision_tolerance"],diagnostics=record["diagnostics"],
        **{name:data[name] for name in ARRAYS})
    if r.fingerprint!=record["fingerprint"]: raise ValueError("saved model fingerprint mismatch")
    d = AdaptationDistribution(r,data["initial_probabilities"],record["initial_debt_index"],
        record["initial_adaptation_index"],record["initial_access"],data["access_mass"],data["excluded_mass"],
        record["distribution_diagnostics"])
    return r,d


def _case(name: str, nb: int, na: int, debt_max: float = .4) -> tuple[SovereignAdaptationResult,AdaptationDistribution]:
    p = SovereignAdaptationParameters(grant_rate=.5 if name=="grant" else 0.,disaster_relief=1. if name=="relief" else 0.)
    r = solve_sovereign_adaptation(income=[1.,1.],damage=[0.,.4],disaster=[False,True],
        transition=[[.9,.1],[.75,.25]],debt_grid=[0.] if name=="no_borrowing" else np.linspace(0,debt_max,nb),
        adaptation_grid=[0.] if name=="no_adaptation" else np.linspace(0,.3,na),horizon=6,parameters=p)
    return r,r.distribution([1.,0.])


def _metrics(r: SovereignAdaptationResult, d: AdaptationDistribution) -> dict[str,Any]:
    f = d.to_frame()
    return dict(value=d.value,initial_investment=float(f.investment.iloc[0]),total_investment=float(f.investment.sum()),
        cumulative_default_probability=float(f.new_default_probability.sum()),cumulative_damage=float(f.disaster_damage.sum()),
        total_grant=float(f.grant.sum()),total_scrapping=float(f.scrapping.sum()),
        max_debt_boundary_probability=float(f.debt_upper_boundary_probability.max()),
        max_stock_boundary_probability=float(f.stock_upper_boundary_probability.max()),
        max_error=r.diagnostics["max_error"],distribution_error=d.diagnostics["max_error"],
        debt_nodes=len(r.debt_grid),stock_nodes=len(r.adaptation_grid),debt_max=float(r.debt_grid[-1]))


def run(output: Path, lang: str = "en") -> dict[str,Any]:
    if lang not in ("en","es"): raise ValueError("lang must be en or es")
    output.mkdir(parents=True,exist_ok=True)
    solutions: dict[tuple[str,int],tuple[SovereignAdaptationResult,AdaptationDistribution]] = {}
    rows = []
    for level,(nb,na) in enumerate(GRIDS):
        for case in CASES:
            r,d = _case(case,nb,na)
            solutions[case,level] = r,d
            name = f"{case}-{nb}-{na}"
            save_result(r,d,output/name)
            rows.append(dict(case=case,stage="joint_grid",level=level,name=name,**_metrics(r,d)))
    for nb,debt_max in ((81,.8),(121,1.2)):
        r,d = _case("relief",nb,21,debt_max)
        name = f"relief-domain-{debt_max:.1f}"
        save_result(r,d,output/name)
        rows.append(dict(case="relief",stage="domain",level=-1,name=name,**_metrics(r,d)))
    for na in (81,161):
        r,d = _case("no_borrowing",1,na)
        name = f"no-borrowing-stock-{na}"
        save_result(r,d,output/name)
        rows.append(dict(case="no_borrowing",stage="stock_followup",level=-1,name=name,**_metrics(r,d)))
    table = pd.DataFrame(rows)
    for i,row in table.iterrows():
        reference_level = int(row.level) if row.stage=="joint_grid" else (1 if row.stage=="domain" else 2)
        ref = solutions["baseline",reference_level][1]
        weight = sum(.96**t for t in range(6))
        table.loc[i,"consumption_equivalent"] = np.expm1((row.value-ref.value)/weight)
    table.to_csv(output/"summary.csv",index=False)
    comparisons = []
    for case in CASES:
        rows_case = table[(table.case==case)&(table.stage=="joint_grid")].sort_values("level")
        for i in (1,2):
            lo,hi = rows_case.iloc[i-1],rows_case.iloc[i]
            changes = {key:abs(float(hi[key])-float(lo[key])) for key in LIMITS}
            comparisons.append(dict(case=case,stage="joint_grid",from_name=lo["name"],to_name=hi["name"],
                **changes,passed=all(changes[key]<=limit for key,limit in LIMITS.items())))
    sequence = table[(table["name"]=="relief-41-21")|(table.stage=="domain")].sort_values("debt_max")
    for i in (1,2):
        lo,hi = sequence.iloc[i-1],sequence.iloc[i]
        changes = {key:abs(float(hi[key])-float(lo[key])) for key in LIMITS}
        comparisons.append(dict(case="relief",stage="domain",from_name=lo["name"],to_name=hi["name"],
            **changes,passed=all(changes[key]<=limit for key,limit in LIMITS.items())))
    stock_sequence = table[(table["name"]=="no_borrowing-81-41")|(table.stage=="stock_followup")].sort_values("stock_nodes")
    for i in (1,2):
        lo,hi = stock_sequence.iloc[i-1],stock_sequence.iloc[i]
        changes = {key:abs(float(hi[key])-float(lo[key])) for key in LIMITS}
        comparisons.append(dict(case="no_borrowing",stage="stock_followup",from_name=lo["name"],to_name=hi["name"],
            **changes,passed=all(changes[key]<=limit for key,limit in LIMITS.items())))
    refinement = pd.DataFrame(comparisons)
    refinement.to_csv(output/"refinement.csv",index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spanish = lang=="es"
    labels = dict(baseline="Base",grant="Subsidio 50%" if spanish else "50% grant",
        relief="Condonación en desastre" if spanish else "Disaster forgiveness",
        no_adaptation="Sin adaptación" if spanish else "No adaptation",
        no_borrowing="Sin deuda nueva" if spanish else "No borrowing")
    fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for case,color in zip(CASES,("#343d49","#007c91","#9a416f","#999999","#d47b19")):
        f = solutions[case,2][1].to_frame()
        axes[0,0].plot(f.index,f.investment,marker="o",label=labels[case],color=color)
        axes[0,1].plot(f.index,f.disaster_damage,marker="o",label=labels[case],color=color)
    axes[0,0].set(title="Inversión esperada" if spanish else "Expected adaptation investment",ylabel="Bienes" if spanish else "Goods")
    axes[0,1].set(title="Daño esperado" if spanish else "Expected disaster damage",ylabel="Bienes" if spanish else "Goods")
    axes[0,0].legend(fontsize=8)
    for ax in axes[0]: ax.set_xlabel("Fecha" if spanish else "Date")
    final = table[(table.stage=="joint_grid")&(table.level==2)]
    axes[1,0].bar(np.arange(len(CASES)),100*final.consumption_equivalent,color=["#343d49","#007c91","#9a416f","#999999","#d47b19"])
    axes[1,0].set_xticks(np.arange(len(CASES)),[labels[c] for c in CASES],rotation=18,ha="right",fontsize=8)
    axes[1,0].set(title="Consumo equivalente vs base" if spanish else "Consumption equivalent vs baseline",ylabel="%")
    axes[1,0].axhline(0,color="black",linewidth=.5)
    axes[1,1].plot(sequence.debt_max,sequence.value,marker="o",color="#9a416f")
    axes[1,1].set(title="Condonación: dominio de deuda" if spanish else "Forgiveness: debt-domain sensitivity",
        xlabel="Límite superior de deuda" if spanish else "Upper debt bound",ylabel="Utilidad inicial" if spanish else "Initial utility")
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle("Adaptación y riesgo soberano: seis fechas" if spanish else "Adaptation and sovereign risk: six dates",fontsize=15)
    fig.savefig(output/"sovereign-adaptation.png",dpi=160)
    fig.savefig(output/"sovereign-adaptation.pdf",metadata={"CreationDate":None})
    plt.close(fig)
    from puremacro.sovereign import adaptation,_adaptation_audit
    paths = [Path(adaptation.__file__),Path(_adaptation_audit.__file__),Path(__file__)]
    summary = dict(language=lang,python=platform.python_version(),numpy=np.__version__,cases=len(table),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        parameters=asdict(SovereignAdaptationParameters()),refinement_limits=LIMITS,
        maximum_audit_error=float(max(table.max_error.max(),table.distribution_error.max())),
        joint_final_refinements_passed=bool(refinement[(refinement.stage=="joint_grid")&refinement.to_name.str.endswith("81-41")].passed.all()),
        domain_last_comparison_passed=bool(refinement[refinement.stage=="domain"].iloc[-1].passed),
        stock_last_comparison_passed=bool(refinement[refinement.stage=="stock_followup"].iloc[-1].passed),
        final_fixed_domain=final.to_dict("records"),domain_sensitivity=sequence.to_dict("records"),
        stock_followup=stock_sequence.to_dict("records"),
        scope="synthetic six-date finite-grid model; exogenous hazard; donor welfare excluded; domain limits retained")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--lang",choices=("en","es"),default="en")
    args=parser.parse_args(); s=run(args.output,args.lang)
    print(json.dumps({k:s[k] for k in ("cases","maximum_audit_error","joint_final_refinements_passed","domain_last_comparison_passed","stock_last_comparison_passed")},indent=2))


if __name__=="__main__": main()
