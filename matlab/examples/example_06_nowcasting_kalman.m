% EXAMPLE_06_NOWCASTING_KALMAN Kalman Filtering & RTS Smoothing with Missing Data
%
% Demonstrates:
%   1. Real-time macro nowcasting with ragged edges (unobserved observations / NaNs)
%   2. Kalman Filter & Rauch-Tung-Striebel (RTS) State Smoother

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 6: Kalman Filter & RTS Smoother Real-Time Nowcasting\n');
fprintf('=================================================================\n\n');

rng(999);
T = 60;
% True state alpha_t: local linear trend + AR(1)
alpha_true = zeros(T, 2);
alpha_true(1, :) = [1.0, 0.2];
T_mat = [1.0, 1.0; 0.0, 0.9];
Q_mat = diag([0.1, 0.05]);

for t = 2:T
    alpha_true(t, :) = (T_mat * alpha_true(t-1, :)' + [sqrt(0.1)*randn(); sqrt(0.05)*randn()])';
end

% Observations y_t with measurement noise
Z_mat = [1.0, 0.0; 1.0, 1.0];
H_mat = 0.2 * eye(2);
y_obs = (Z_mat * alpha_true')' + randn(T, 2) * sqrt(0.2);

% Simulate real-world ragged edges:
% Variable 2 is published with a 3-month lag (NaN at end)
y_obs((T-2):T, 2) = NaN;
% Occasional missing historical data
y_obs(15:18, 1) = NaN;

% Run RTS Kalman Smoother
res_sm = puremacro.statespace.kalman_smoother(y_obs, T_mat, Z_mat, Q_mat, H_mat);

fprintf('Kalman Filter Total Log-Likelihood: %.2f\n', res_sm.loglik);
fprintf('Smoothed State Alpha_1 at t=T (Nowcast): %.4f (True = %.4f)\n', ...
    res_sm.a_smooth(T, 1), alpha_true(T, 1));
fprintf('Smoothed State Alpha_2 at t=T (Nowcast): %.4f (True = %.4f)\n', ...
    res_sm.a_smooth(T, 2), alpha_true(T, 2));
fprintf('=================================================================\n\n');
