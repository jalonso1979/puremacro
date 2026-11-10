"""Record second-order HANK amplitude, quadrature and finite-reference evidence.

Run from a checkout: python tools/reference_validation/benchmark_hank_second_order.py
--output /tmp/hank-second-order --references. A completed report retains failed
cases and does not certify global, continuous-grid, infinite-tail or welfare accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from puremacro.models.one_asset_hank import calibrate_one_asset_hank
from puremacro.models.hank_reduction import solve_hank_reduced_policy
from puremacro.models.hank_second_order import solve_hank_second_order
from puremacro.models._hank_second_order import OneAssetEquations
from puremacro.models.hank_nonlinear_continuation import solve_hank_nonlinear_continuation


def run_benchmark(output: Path, *, references: bool = False) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    sources=['puremacro/models/hank_second_order.py','puremacro/models/_hank_second_order.py',
             'puremacro/models/hank_reduction.py','puremacro/models/one_asset_hank.py',
             'puremacro/models/hank_nonlinear_continuation.py','puremacro/dsge/_sylvester.py',
             'tools/reference_validation/benchmark_hank_second_order.py']
    record=dict(scope='small full-grid one-asset second-order benchmark',welfare_supported=False,
        global_accuracy_certified=False,infinite_tail_certified=False,household_grid_certified=False,
        source_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sources},
        grids=[],amplitudes=[],quadrature=[],references=[],held_out_paths=[])
    arrays={}
    def save():
        (output/'record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    for na in (12,20):
        ss=calibrate_one_asset_hank(n_a=na,n_e=2,a_max=120.)
        p=solve_hank_reduced_policy(ss,distribution_rank=None,marginal_rank=None,shock_std=.001)
        start=time.perf_counter();second=solve_hank_second_order(p);elapsed=time.perf_counter()-start
        equations=OneAssetEquations(p)
        record['grids'].append(dict(n_a=na,seconds=elapsed,diagnostics=dict(second.diagnostics),
            base_equation_error=float(np.max(abs(equations.first_order_residual(second.state()))/equations.row_scale))))
        for name in ('transition_curvature','feedback_curvature','transition_variance','feedback_variance'):
            arrays[f'na{na}_{name}']=getattr(second,name)
        for name,value in second.mean_decomposition().items():arrays[f'na{na}_{name}']=value
        for amplitude in (.001,.0005,.00025,.000125):
            for sign in (-1,0,1):
                x=second.state(monetary_shock=sign*amplitude)
                report=second.audit(x,shock_std=amplitude,tolerance=1e-5)
                first=equations.first_order_residual(x,amplitude)
                record['amplitudes'].append(dict(n_a=na,amplitude=amplitude,sign=sign,accepted=report.accepted,
                    first_error=float(np.max(abs(first)/equations.row_scale)),
                    second_error=report.diagnostics['equation_error'],diagnostics=dict(report.diagnostics)))
        for order in (2,3,5):
            report=second.audit(second.state(monetary_shock=.001),shock_std=.001,quadrature_order=order)
            record['quadrature'].append(dict(n_a=na,order=order,accepted=report.accepted,
                residual=report.residual.tolist(),diagnostics=dict(report.diagnostics)))
        for sigma in (.0001,.000025):
            innovations=np.random.default_rng(9917).normal(0.,sigma,100)
            path=second.simulate(innovations,shock_std=sigma)
            arrays[f'na{na}_sigma{sigma}_states']=path.states
            record['held_out_paths'].append(dict(n_a=na,sigma=sigma,seed=9917,periods=100,
                converged=path.converged,diagnostics=path.diagnostics))
        save()
        if references:
            configurations=([(0.,16,1),(.0005,8,7)] if na==12 else
                [(0.,h,1) for h in (8,16,32)]+[(.0005,h,h-1) for h in (4,6,8,10,12)])
            for sigma,horizon,depth in configurations:
                linear=solve_hank_reduced_policy(ss,distribution_rank=None,marginal_rank=None,shock_std=sigma)
                x=linear.state(monetary_shock=.0005)
                first=linear.aggregate_feedback@x
                candidate=second.controls(x,shock_std=sigma)[-3:]
                start=time.perf_counter()
                ref=solve_hank_nonlinear_continuation(linear,aggregate_state=np.array([0.,.0005]),
                    horizon=horizon,branch_depth=depth,quadrature_order=2,tolerance=1e-9,max_iterations=30)
                truth=ref.aggregate_coordinates[0]
                record['references'].append(dict(n_a=na,sigma=sigma,horizon=horizon,branch_depth=depth,
                    quadrature_order=2,tolerance=1e-9,converged=ref.converged,nodes=ref.diagnostics['nodes'],
                    equation_error=ref.diagnostics['market_residual_max_abs'],
                    first_error=float(np.max(abs(first-truth))),second_error=float(np.max(abs(candidate-truth))),
                    reference=truth.tolist(),second_order=candidate.tolist(),seconds=time.perf_counter()-start,
                    terminal_policy_order=ref.tree.metadata['terminal_policy_order']))
                save()
    np.savez_compressed(output/'coefficients-and-paths.npz',**arrays)
    frame=pd.DataFrame([{k:v for k,v in row.items() if k!='diagnostics'} for row in record['amplitudes']])
    frame.to_csv(output/'amplitudes.csv',index=False)
    pd.DataFrame(record['references']).to_csv(output/'nonlinear-references.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for ax,na in zip(axes,(12,20)):
        data=frame[(frame.n_a==na)&(frame.sign==1)].sort_values('amplitude')
        ax.loglog(data.amplitude,data.first_error,'o-',label='First order')
        ax.loglog(data.amplitude,data.second_error,'s-',label='Second order')
        ax.set(title=f'{na} assets, 2 income states',xlabel='Shock and innovation scale',ylabel='Scaled equation error')
        ax.set_xticks(data.amplitude,labels=[f'{v:g}' for v in data.amplitude])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True,which='both',alpha=.25);ax.legend()
    fig.savefig(output/'equation-errors.png',dpi=180);plt.close(fig)
    record['artifact_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='record.json'}
    save()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--references',action='store_true')
    args=parser.parse_args()
    result=run_benchmark(args.output,references=args.references)
    print(json.dumps(dict(grids=len(result['grids']),amplitude_cases=len(result['amplitudes']),
        reference_cases=len(result['references']),rejected_references=sum(not r['converged'] for r in result['references']),
        held_out_paths=len(result['held_out_paths']),failed_paths=sum(not r['converged'] for r in result['held_out_paths']))))
