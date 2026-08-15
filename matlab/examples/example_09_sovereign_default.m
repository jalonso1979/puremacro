% EXAMPLE_09_SOVEREIGN_DEFAULT Arellano (2008) Sovereign Default & Bond Spreads
%
% Solves a Small Open Economy sovereign default model with endogenous bond pricing,
% default risk spreads, and credit market exclusion.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 9: Sovereign Default & Endogenous Bond Spread Schedule\n');
fprintf('=================================================================\n\n');

sd_res = puremacro.vfi.sovereign_default('beta', 0.95, 'r_star', 0.017, 'Nb', 40, 'Ny', 9);

fprintf('Sovereign Default Model Results:\n');
fprintf('  Max Bond Price q(B=0, y_mid):   %.4f (Risk-free price = %.4f)\n', ...
    sd_res.q_bond(end, ceil(end/2)), 1 / (1 + 0.017));
fprintf('  Lowest Bond Price q(B_max, y):  %.4f (High default risk)\n', ...
    min(sd_res.q_bond(:)));
fprintf('  Max Default Probability:        %.1f%%\n', max(sd_res.default_prob(:)) * 100);
fprintf('  Iterations to Convergence:      %d (%.3f s)\n\n', sd_res.n_iter, sd_res.elapsed);

fprintf('=================================================================\n');
