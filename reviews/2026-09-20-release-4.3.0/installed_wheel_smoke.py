import json
import numpy as np
import puremacro
from puremacro.trade import (calibrate_trade_model, solve_policy_equilibrium,
    compute_hicksian_welfare, compute_unilateral_optimal_tariff,
    solve_multilateral_nash_tariffs, compute_welfare_payoff_matrix)
assert puremacro.__version__ == '4.3.0'
assert 'site-packages' in puremacro.__file__, puremacro.__file__
flows=np.array([[10.,12.,24.,8.,6.,4.,20.,16.], [8.,14.,4.5,15.,10.5,35.,18.,15.]])
taxes=np.array([4.,6.,1.2,1.,.8,1.8,2.,1.2])
va=flows.sum(1)-flows[:,:2].sum(0)-taxes[:2]
data=np.vstack([flows,taxes,np.r_[2*va/3,np.zeros(6)],np.r_[va/3,np.zeros(6)]])
c=calibrate_trade_model(data,ns=1,nc=2,nfd=3,country_codes=['A','B'])
b=solve_policy_equilibrium(c,sigma=2.,tol=1e-9)
settings=dict(metric='hicksian_ev',sigma=2.,ge_tol=1e-9,base_equilibrium=b,consumption_categories=(0,2))
opt=compute_unilateral_optimal_tariff(c,country_idx='A',tariff_max=.4,num_grid=9,**settings)
assert opt.optimal_tariff_rate==.4
assert abs(opt.welfare_gain_pct-9.102834660094734)<1e-6
w=compute_hicksian_welfare(c,opt.equilibrium,base_result=b,consumption_categories=(0,2))
assert abs(opt.optimal_welfare-w.ev)<1e-10
n=solve_multilateral_nash_tariffs(c,player_countries=['A','B'],tariff_max=.4,
    best_response_grid_size=9,relaxation=.8,tol=1e-5,regret_tol=1e-6,**settings)
assert n.converged and n.metadata['relative_max_regret']<=1e-6
m=compute_welfare_payoff_matrix(c,player_a='A',player_b='B',optimal_a=.1,optimal_b=.2,**settings)
assert np.isfinite(m.payoff_matrix).all() and not m.metadata['continuous_nash_verified']
print(json.dumps({'version':puremacro.__version__,'module_path':puremacro.__file__,
    'unilateral_ev_percent':opt.welfare_gain_pct,'nash_converged':n.converged,
    'nash_relative_regret':n.metadata['relative_max_regret'],
    'matrix_finite':bool(np.isfinite(m.payoff_matrix).all()),'passed':True},indent=2))
