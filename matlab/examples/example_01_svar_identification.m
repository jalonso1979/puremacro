% EXAMPLE_01_SVAR_IDENTIFICATION Structural VAR Identification & Counterfactual Analysis
%
% Demonstrates:
%   1. VAR(2) Estimation & Stationarity Check
%   2. Cholesky SVAR
%   3. Blanchard-Quah (1989) Long-Run Restriction SVAR
%   4. Rubio-Ramírez et al. (2010) Sign-Restricted SVAR
%   5. Barsky-Sims (2011) FEVD Max-Share News Shock
%   6. Mertens-Ravn (2013) Proxy-IV External Instrument SVAR
%   7. Impulse Response Functions (IRF), FEVD, and Historical Decomposition (HD)

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 1: Structural VAR (SVAR) Identification & Analysis\n');
fprintf('=================================================================\n\n');

rng(42);
T = 250;
% Simulated macroeconomic series: [GDP Growth, Inflation, Interest Rate]
Y = cumsum(randn(T, 3) * 0.02) + randn(T, 3) * 0.1;

% 1. Estimate reduced-form VAR(2)
p = 2;
res_var = puremacro.var.estimate(Y, p);
fprintf('Estimated VAR(%d) on %d observations (%d variables).\n', p, res_var.T_eff, res_var.n);
fprintf('VAR Companion Matrix Stable: %d\n\n', res_var.is_stable);

% 2. Cholesky Identification
B0_chol = puremacro.var.cholesky(res_var.Sigma);
irf_chol = puremacro.var.irf(res_var.A_list, B0_chol, 12);
fprintf('Cholesky B0 Impact Matrix:\n');
disp(B0_chol);

% 3. Blanchard-Quah (1989) Long-Run Restriction SVAR
B0_bq = puremacro.var.bq(res_var.A_list, res_var.Sigma, 1);
fprintf('Blanchard-Quah Long-Run Impact B0 Matrix:\n');
disp(B0_bq);

% 4. FEVD Max-Share SVAR (Barsky-Sims 2011 News Shock at H=20)
[B0_ms, fev_share] = puremacro.var.maxshare(res_var.A_list, res_var.Sigma, 1, 20);
fprintf('FEVD Max-Share Shock Explains %.1f%% of Variable 1 Variance at H=20.\n\n', fev_share * 100);

% 5. Proxy-IV SVAR (External Instrument)
z_proxy = res_var.resid(:, 1) + randn(res_var.T_eff, 1) * 0.3;
[B0_proxy, f_stat] = puremacro.var.proxy(res_var.Sigma, res_var.resid, z_proxy, 1);
fprintf('Proxy-IV Identification First-Stage F-Statistic: %.1f\n\n', f_stat);

% 6. Forecast Error Variance Decomposition (FEVD)
fevd_table = puremacro.var.fevd(res_var.A_list, B0_chol, 10);
fprintf('Cholesky FEVD Shares at Horizon 10 for Variable 1:\n');
disp(squeeze(fevd_table(11, 1, :))');

% 7. Historical Decomposition
hd_res = puremacro.var.hd(res_var.A_list, B0_chol, res_var.resid);
fprintf('Historical Decomposition computed for %d periods.\n', size(hd_res.shocks, 1));
fprintf('=================================================================\n\n');
