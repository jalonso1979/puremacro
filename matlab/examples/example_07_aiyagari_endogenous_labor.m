% EXAMPLE_07_AIYAGARI_ENDOGENOUS_LABOR Aiyagari Model with Endogenous Labor Supply
%
% Solves the general equilibrium Aiyagari (1994) model where households make joint
% consumption, asset saving, and endogenous labor supply decisions n(a, e).

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 7: Aiyagari Model with Endogenous Labor Supply\n');
fprintf('=================================================================\n\n');

% Solve General Equilibrium Aiyagari Model with Endogenous Labor
aiyagari_res = puremacro.vfi.aiyagari_endogenous_labor(...
    'beta', 0.95, ...
    'gamma', 2.0, ...
    'frisch', 0.5, ...
    'chi', 1.2, ...
    'alpha', 0.36, ...
    'delta', 0.08, ...
    'Na', 50, ...
    'Ne', 5);

fprintf('GENERAL EQUILIBRIUM RESULTS:\n');
fprintf('  Equilibrium Interest Rate r*:  %.4f (%.2f%%)\n', aiyagari_res.r_star, aiyagari_res.r_star * 100);
fprintf('  Equilibrium Wage Rate w*:      %.4f\n', aiyagari_res.w_star);
fprintf('  Aggregate Capital Stock K*:    %.4f\n', aiyagari_res.K_star);
fprintf('  Aggregate Labor Supply L*:     %.4f\n', aiyagari_res.L_star);
fprintf('  Capital-Labor Ratio (K/L)*:    %.4f\n', aiyagari_res.K_star / aiyagari_res.L_star);
fprintf('  Wealth Gini Coefficient:       %.3f\n', aiyagari_res.sim_res.gini);
fprintf('  Computation Time:              %.2f seconds\n\n', aiyagari_res.elapsed);

fprintf('=================================================================\n');
