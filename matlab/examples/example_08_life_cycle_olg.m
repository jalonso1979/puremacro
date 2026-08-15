% EXAMPLE_08_LIFE_CYCLE_OLG Life-Cycle OLG Model with Retirement & Bequests
%
% Solves a 50-period adult life cycle OLG model with hump-shaped age productivity,
% social security pension, asset accumulation, and bequest motives.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 8: Life-Cycle OLG Model with Retirement & Bequests\n');
fprintf('=================================================================\n\n');

lc_res = puremacro.vfi.life_cycle_olg('J', 50, 'J_ret', 35, 'beta', 0.96);

fprintf('Life-Cycle OLG Model Results (50 Adult Years):\n');
fprintf('  Peak Asset Age:           Age %d (Mean Assets = %.2f)\n', ...
    find(lc_res.mean_assets_age == max(lc_res.mean_assets_age)), max(lc_res.mean_assets_age));
fprintf('  Retirement Age Index:     Age %d (Pension = %.2f)\n', lc_res.J_ret + 1, lc_res.mean_inc_age(lc_res.J_ret + 1));
fprintf('  Terminal Asset Bequest:   %.2f\n', lc_res.mean_assets_age(end));
fprintf('  Computation Time:         %.3f seconds\n\n', lc_res.elapsed);

fprintf('=================================================================\n');
