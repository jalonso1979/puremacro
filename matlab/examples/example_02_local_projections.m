% EXAMPLE_02_LOCAL_PROJECTIONS Standard & State-Dependent Local Projections
%
% Demonstrates:
%   1. Jordà (2005) Linear Local Projections with Newey-West HAC Standard Errors
%   2. Auerbach & Gorodnichenko (2013) / Ramey & Zubairy (2018) State-Dependent LPs

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 2: Standard & State-Dependent Local Projections (LP)\n');
fprintf('=================================================================\n\n');

rng(123);
T = 200;
y = cumsum(randn(T, 1) * 0.05);
x = randn(T, 1);           % Exogenous monetary shock / fiscal policy shock
z_state = randn(T, 1);     % State indicator (e.g. output gap / unemployment)

% 1. Estimate Jordà (2005) Linear Local Projections
res_lp = puremacro.lp.estimate(y, x, 0:10, 2, [], 0.10);
fprintf('Jordà Linear LP estimated across 11 horizons.\n');
fprintf('Horizon 0 Beta: %+.4f (SE: %.4f, 90%% CI: [%+.4f, %+.4f])\n', ...
    res_lp.beta(1), res_lp.se(1), res_lp.lo(1), res_lp.hi(1));
fprintf('Horizon 5 Beta: %+.4f (SE: %.4f, 90%% CI: [%+.4f, %+.4f])\n\n', ...
    res_lp.beta(6), res_lp.se(6), res_lp.lo(6), res_lp.hi(6));

% 2. Estimate Auerbach-Gorodnichenko Smooth Transition State-Dependent LP
res_sd = puremacro.lp.state_dependent(y, x, z_state, 0:10, 2, 'logistic', 3.0, 0.0, 0.10);
fprintf('State-Dependent LP estimated across 11 horizons:\n');
fprintf('Horizon 0 High-Regime Beta: %+.4f (SE: %.4f)\n', res_sd.beta_high(1), res_sd.se_high(1));
fprintf('Horizon 0 Low-Regime  Beta: %+.4f (SE: %.4f)\n', res_sd.beta_low(1), res_sd.se_low(1));
fprintf('Horizon 5 High-Regime Beta: %+.4f (SE: %.4f)\n', res_sd.beta_high(6), res_sd.se_high(6));
fprintf('Horizon 5 Low-Regime  Beta: %+.4f (SE: %.4f)\n', res_sd.beta_low(6), res_sd.se_low(6));
fprintf('=================================================================\n\n');
