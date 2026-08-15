% EXAMPLE_10_PORTFOLIO_HOUSING Multi-Asset Portfolio & Housing Choice
%
% Solves joint portfolio choice across liquid bonds, housing stock with adjustment costs,
% and risky income shocks.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 10: Multi-Asset Portfolio Choice with Housing Adjustment Costs\n');
fprintf('=================================================================\n\n');

ph_res = puremacro.vfi.portfolio_housing('Nb', 15, 'Nh', 8, 'Ny', 5);

fprintf('Multi-Asset Portfolio & Housing Results:\n');
fprintf('  Mean Optimal Bond Holding:    %.2f\n', mean(ph_res.policy_b(:)));
fprintf('  Mean Optimal Housing Stock:   %.2f\n', mean(ph_res.policy_h(:)));
fprintf('  Iterations to Convergence:   %d (%.3f s)\n\n', ph_res.n_iter, ph_res.elapsed);

fprintf('=================================================================\n');
