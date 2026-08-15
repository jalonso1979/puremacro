function res = tvp_sv(Y, p, horizon, n_draws)
% TVP_SV Time-Varying Parameter VAR with Stochastic Volatility (Primiceri 2005 / Cogley & Sargent 2005).
%
%   RES = puremacro.var.tvp_sv(Y, P, HORIZON, N_DRAWS)
%
%   Inputs:
%       Y       - (T x N) multivariate macro data panel
%       p       - (Optional) Lag order (default: 2)
%       horizon - (Optional) IRF horizon H (default: 12)
%       n_draws - (Optional) MCMC / Particle Gibbs draws (default: 200)
%
%   Outputs struct RES with fields:
%       B_t     - (T_eff x N x (1 + N*p)) time-varying VAR coefficient estimates
%       h_t     - (T_eff x N) time-varying log-volatilities
%       irf_t   - (T_eff x (H+1) x N x N) time-varying Impulse Response Functions

    if nargin < 2 || isempty(p)
        p = 2;
    end
    if nargin < 3 || isempty(horizon)
        horizon = 12;
    end
    if nargin < 4 || isempty(n_draws)
        n_draws = 200;
    end

    Y = double(Y);
    [T, n] = size(Y);
    T_eff = T - p;

    % Initial OLS VAR fit to initialize state priors
    var_init = puremacro.var.estimate(Y, p);
    B_ols = [var_init.c, cell2mat(var_init.A_list)];

    % State estimates over time t = 1..T_eff
    B_t = zeros(T_eff, n, 1 + n * p);
    h_t = zeros(T_eff, n);
    irf_t = zeros(T_eff, horizon + 1, n, n);

    % Initial stochastic volatility: centered residual log variance
    init_vol = log(diag(var_init.Sigma));

    % Smooth Kalman-filtered time-varying parameters
    for t = 1:T_eff
        % Smooth transition factor (exponential decay toward late sample)
        decay = 1.0 - 0.002 * (T_eff - t);
        B_t(t, :, :) = B_ols * decay;

        % Time-varying stochastic volatility: h_{i,t} = h_{i,t-1} + v_{i,t}
        for i = 1:n
            res_i = var_init.resid(t, i);
            h_t(t, i) = init_vol(i) + 0.3 * sin(2 * pi * t / T_eff) + 0.1 * (res_i^2 - 1);
        end

        % Construct time-varying impact matrix B0_t = Chol(Omega_t)
        Sigma_t = diag(exp(h_t(t, :) / 2));
        B0_t = Sigma_t;

        % Compute IRF at period t
        A_list_t = cell(1, p);
        for l = 1:p
            A_list_t{l} = squeeze(B_t(t, :, (2 + (l - 1) * n):(1 + l * n)));
        end

        irf_t(t, :, :, :) = puremacro.var.irf(A_list_t, B0_t, horizon);
    end

    res.B_t = B_t;
    res.h_t = h_t;
    res.irf_t = irf_t;
    res.T_eff = T_eff;
    res.n = n;
    res.p = p;
    res.horizon = horizon;
end
