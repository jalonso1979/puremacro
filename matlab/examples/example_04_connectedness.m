% EXAMPLE_04_CONNECTEDNESS Diebold-Yilmaz & Baruník-Křehlík Spillover Indices
%
% Demonstrates:
%   1. Diebold-Yilmaz (2012) Time-Domain Total & Directional Spillover Indices
%   2. Baruník & Křehlík (2018) Frequency-Domain Connectedness (Short vs Long-Run)

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 4: Diebold-Yilmaz & Frequency-Domain Connectedness\n');
fprintf('=================================================================\n\n');

rng(55);
T = 300;
N_assets = 4;
% Simulated asset returns / volatility panel
returns = randn(T, N_assets);

% 1. Diebold-Yilmaz Time-Domain Connectedness
dy_res = puremacro.connectedness.diebold_yilmaz(returns, 2, 12, 'gfevd');

fprintf('Diebold-Yilmaz Total Spillover Index: %.2f%%\n\n', dy_res.total);

fprintf('Directional TO Spillovers (%%):\n');
disp(dy_res.directional_to');

fprintf('Directional FROM Spillovers (%%):\n');
disp(dy_res.directional_from');

fprintf('Net Spillovers (TO - FROM, %%):\n');
disp(dy_res.net');

% 2. Baruník & Křehlík Frequency-Domain Connectedness
% High-frequency band: [pi/4, pi] (short-term spillovers)
bk_short = puremacro.connectedness.barunik_krehlik(returns, 2, 100, [pi/4, pi]);
% Low-frequency band: [0, pi/4] (long-term spillovers)
bk_long = puremacro.connectedness.barunik_krehlik(returns, 2, 100, [0, pi/4]);

fprintf('Baruník-Křehlík Frequency Decomposition:\n');
fprintf('  Short-Term Frequency Spillover (w in [pi/4, pi]): %.2f%%\n', bk_short.total_spillover);
fprintf('  Long-Term  Frequency Spillover (w in [0, pi/4]):   %.2f%%\n', bk_long.total_spillover);
fprintf('=================================================================\n\n');
