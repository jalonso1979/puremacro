% EXAMPLE_11_DYNARE_VFI_AUTOMATED Dynare-Like Declarative VFI Model Solver
%
% Demonstrates how to specify and solve Dynamic Programming models in 5 lines of code
% using the automated Dynare-like interface `puremacro.vfi.model`.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 11: Dynare-Like Automated VFI Model Solver\n');
fprintf('=================================================================\n\n');

% 1. Create Model Object
m = puremacro.vfi.model('Neoclassical Growth Model');

% 2. Set Parameters
m = m.set_param('beta', 0.96);
m = m.set_param('gamma', 2.0);
m = m.set_param('alpha', 0.36);
m = m.set_param('delta', 0.10);

% 3. Declare States and Shocks
m = m.add_state('k', 0.1, 15.0, 50);          % Capital grid: min=0.1, max=15.0, 50 points
m = m.add_shock('z', 0.90, 0.15, 5, 'tauchen'); % Productivity shock: rho=0.90, sigma=0.15, 5 points

% 4. Declare Utility Function and Resource Constraint
m = m.set_utility(@(c, params) (c^(1 - params.gamma)) / (1 - params.gamma));
m = m.set_budget(@(k, kp, z, params) exp(z) * (k^params.alpha) + (1 - params.delta) * k - kp);

% 5. Solve Model Automatically with Howard Acceleration
res = m.solve('howard_steps', 20);

fprintf('AUTOMATED SOLUTION RESULTS (%s):\n', res.model_name);
fprintf('  Iterations to Convergence: %d\n', res.n_iter);
fprintf('  Computation Time:          %.3f seconds\n', res.elapsed);
fprintf('  Stationary Mean Capital:   %.4f\n', res.sim.mean_assets);
fprintf('  Capital Gini Coefficient:  %.3f\n\n', res.sim.gini);

% 6. Auto-Plot Policy & Value Functions
m.plot_results(res);

fprintf('=================================================================\n');
