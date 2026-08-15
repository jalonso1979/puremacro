function [se, cov] = driscoll_kraay(X, resid, time_keys, lags)
% DRISCOLL_KRAAY Panel Driscoll-Kraay robust standard errors.
%
%   [SE, COV] = puremacro.inference.driscoll_kraay(X, RESID, TIME_KEYS, LAGS)
%
%   Inputs:
%       X         - (N_panel x K) regressor matrix
%       resid     - (N_panel x 1) vector of residuals
%       time_keys - (N_panel x 1) vector of time period indicators
%       lags      - Bartlett kernel bandwidth for time aggregation
%
%   Outputs:
%       se        - (K x 1) Driscoll-Kraay standard errors
%       cov       - (K x K) Driscoll-Kraay covariance matrix

    [n_panel, k] = size(X);
    if nargin < 4 || isempty(lags)
        lags = 0;
    end

    score = X .* resid(:);
    unique_t = unique(time_keys);
    T = length(unique_t);

    by_t = zeros(T, k);
    for t_idx = 1:T
        mask = (time_keys == unique_t(t_idx));
        by_t(t_idx, :) = sum(score(mask, :), 1);
    end

    S = by_t' * by_t;
    for ell = 1:lags
        if ell >= T
            break;
        end
        w = 1.0 - ell / (lags + 1.0);
        Gamma = by_t((ell + 1):end, :)' * by_t(1:(end - ell), :);
        S = S + w * (Gamma + Gamma');
    end

    XtX_inv = (X' * X) \ eye(k);
    cov = XtX_inv * S * XtX_inv;
    se = sqrt(diag(cov));
end
