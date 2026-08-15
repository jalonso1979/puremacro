function res = quantile(y, x, horizons, quantiles, p_lags)
% QUANTILE Quantile Local Projections (Loria, Matthes & Zhang 2022 / Growth-at-Risk).
%
%   RES = puremacro.lp.quantile(Y, X, HORIZONS, QUANTILES, P_LAGS)
%
%   Inputs:
%       y         - (T x 1) dependent variable series
%       x         - (T x 1) structural shock / policy series
%       horizons  - Vector of horizons 0..H (e.g. 0:12)
%       quantiles - Vector of quantile levels tau in (0, 1) (default: [0.10, 0.50, 0.90])
%       p_lags    - (Optional) Control lag order (default: 2)
%
%   Outputs struct RES with fields:
%       beta_q    - (H+1 x n_q) quantile impulse response coefficients
%       quantiles - Vector of evaluated quantile levels
%       h         - Vector of horizons 0..H

    if nargin < 4 || isempty(quantiles)
        quantiles = [0.10, 0.50, 0.90];
    end
    if nargin < 5 || isempty(p_lags)
        p_lags = 2;
    end

    y = double(y(:));
    x = double(x(:));
    horizons = horizons(:);
    H = max(horizons);
    H_len = H + 1;
    T = length(y);
    n_q = length(quantiles);

    beta_q = zeros(H_len, n_q);

    for h_idx = 1:H_len
        h = horizons(h_idx);

        % Construct dependent variable y_{t+h} and design matrix X_mat
        T_eff = T - p_lags - h;
        y_fwd = y((p_lags + 1 + h):T);
        x_shock = x((p_lags + 1):(T - h));

        X_mat = [ones(T_eff, 1), x_shock];
        for l = 1:p_lags
            X_mat = [X_mat, y((p_lags + 1 - l):(T - h - l)), x((p_lags + 1 - l):(T - h - l))];
        end

        % Fit Quantile Regressions for each quantile tau
        for q_idx = 1:n_q
            tau = quantiles(q_idx);
            b_q = quantile_ols_fit(y_fwd, X_mat, tau);
            beta_q(h_idx, q_idx) = b_q(2); % Coefficient on x_shock
        end
    end

    res.beta_q = beta_q;
    res.quantiles = quantiles;
    res.h = horizons;
end

function b = quantile_ols_fit(y, X, tau)
    % Asymmetric check-function quantile regression via weighted iterative least squares
    b = X \ y;
    for iter = 1:20
        res_val = y - X * b;
        weights = tau * (res_val >= 0) + (1 - tau) * (res_val < 0);
        weights = max(weights ./ max(abs(res_val), 1e-4), 1e-4);
        W_diag = diag(weights);
        b = (X' * W_diag * X) \ (X' * W_diag * y);
    end
end
