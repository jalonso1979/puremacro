"""Run and retain finite-tree HANK chain, asset and horizon comparisons.

The deterministic oracle is the separate published-model transition solver.
Stochastic refinement is an internal accuracy study, not an external stochastic
model replication. Every failed comparison remains in the JSON record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from puremacro.models import calibrate_one_asset_hank,solve_one_asset_hank
from puremacro.models.hank_scenario_tree import monetary_scenario_tree,solve_hank_scenario_tree


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    records,solved={},{}
    specifications=[('base',120,5,12),('horizon',120,5,24),('assets',240,5,24),('income',240,7,24)]
    for name,na,ne,T in specifications:
        start=time.perf_counter()
        ss=calibrate_one_asset_hank(n_a=na,n_e=ne,a_max=150.)
        tree=monetary_scenario_tree(horizon=T,innovations=(-.005,.005))
        r=solve_hank_scenario_tree(ss,tree)
        solved[name]=r
        r.to_frame().to_csv(args.output/f'{name}_nodes.csv')
        r.expected_paths().to_csv(args.output/f'{name}_expectations.csv')
        np.savez_compressed(args.output/f'{name}_households.npz',a=ss.asset_grid,e=ss.e_grid,
                            saving=r.policy_a,consumption=r.policy_c,hours=r.policy_n,distribution=r.distributions)
        records[name]={'asset_nodes':na,'income_states':ne,'horizon':T,'elapsed_seconds':time.perf_counter()-start,
            'converged':r.converged,'calibration':ss.calibration,'diagnostics':r.diagnostics,'scope':r.metadata,
            'date0_expected_C':float(r.expected_paths().iloc[0].C)}
        print(name,json.dumps({'converged':r.converged,'elapsed':records[name]['elapsed_seconds'],
                               'residual':r.diagnostics['market_residual_max_abs']}),flush=True)
    ss=solved['base'].steady_state
    chain=solve_hank_scenario_tree(ss,monetary_scenario_tree(horizon=12,innovations=(-.0025,),probabilities=(1.,),branch_date=0))
    oracle=solve_one_asset_hank(ss,horizon=12,magnitude=-.0025,nonlinear=True)
    oracle_gap=float(np.max(np.abs(chain.paths.to_numpy()-oracle.paths[chain.paths.columns].to_numpy())))
    comparisons=[]
    for coarse,fine in [('base','horizon'),('horizon','assets'),('assets','income')]:
        left,right=solved[coarse],solved[fine]
        for variable in ['C','Y','pi']:
            # Compare conditional paths as well as expectations. Same branching
            # histories share labels over the first six dates.
            ids=left.nodes.index[left.nodes.period<6]
            x=left.paths.loc[ids,variable].to_numpy()
            y=right.paths.loc[ids,variable].to_numpy()
            bound=max(1e-6,.01*float(np.max(np.abs(y))))
            difference=float(np.max(np.abs(x-y)))
            comparisons.append({'coarse':coarse,'fine':fine,'variable':variable,'window':6,
                                'maximum_change':difference,'tolerance':bound,'passed':difference<=bound})
    sources=['puremacro/models/hank_scenario_tree.py','puremacro/models/one_asset_hank.py',
             'tools/reference_validation/check_hank_scenario_tree.py']
    record={'schema':1,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        'cases':records,'deterministic_oracle':{'maximum_path_gap':oracle_gap,'tolerance':3e-8,
            'passed':bool(chain.converged and oracle.converged and oracle_gap<=3e-8),
            'scope':'separate deterministic implementation; no independent stochastic author-code reference'},
        'refinements':comparisons,'all_refinements_passed':all(c['passed'] for c in comparisons),
        'protocol':'first six dates, conditional paths C/Y/pi, 1% of finer peak with level floor 1e-6',
        'limitations':['finite tree and fixed terminal marginal values','risk-neutral nominal firm valuation',
                       'no stationary stochastic equilibrium or aggregate stochastic welfare certificate'],
        'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}}
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'deterministic_oracle':record['deterministic_oracle'],
                      'all_refinements_passed':record['all_refinements_passed'],'refinements':comparisons},indent=2))


if __name__=='__main__':
    main()
