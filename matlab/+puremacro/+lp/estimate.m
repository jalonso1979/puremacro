function res = estimate(y, x, horizons, n_lags, controls, alpha)
% ESTIMATE Jordà (2005) Local Projections horizon-by-horizon.
%
%   RES = puremacro.lp.estimate(Y, X_SHOCK, HORIZONS, N_LAGS, CONTROLS, ALPHA)
%
%   Estimates β_h from:
%       y_{t+h} - y_{t-1} = α_h + β_h x_t + Σ γ_l z_{t-l} + ε_{t,h}

    if nargin < 3 || isempty(horizons)
        horizons = 0:20;
    end
    if nargin < 4 || isempty(n_lags)
        n_lags = 2;
    end
    if nargin < 5
        controls = [];
    end
    if nargin < 6 || isempty(alpha)
        alpha = 0.10;
    end

    y = y(:);
    x = x(:);
    T = length(y);
    H = length(horizons);

    beta = nan(H, 1);
    se = nan(H, 1);
    lo = nan(H, 1);
    hi = nan(H, 1);

    % Native inverse normal CDF without requiring Statistics Toolbox
    z_crit = sqrt(2) * erfinv(2 * (1 - alpha / 2) - 1);

    for idx = 1:H
        h = horizons(idx);

        if (T - h) < (n_lags + 1)
            continue;
        end

        t_start = n_lags + 2;
        t_end = T - h;

        if t_start > t_end
            continue;
        end

        Y_dep = y((t_start + h):(t_end + h)) - y((t_start - 1):(t_end - 1));
        N_eff = length(Y_dep);

        X_mat = zeros(N_eff, 2 + 2 * n_lags + size(controls, 2) * (n_lags + 1));
        X_mat(:, 1) = 1.0;
        X_mat(:, 2) = x(t_start:t_end);

        col = 3;
        for lag = 1:n_lags
            X_mat(:, col) = x((t_start - lag):(t_end - lag));
            col = col + 1;
            X_mat(:, col) = y((t_start - lag):(t_end - lag));
            col = col + 1;
        end

        if ~isempty(controls)
            for c_idx = 1:size(controls, 2)
                c_vec = controls(:, c_idx);
                X_mat(:, col) = c_vec(t_start:t_end);
                col = col + 1;
                for lag = 1:n_lags
                    X_mat(:, col) = c_vec((t_start - lag):(t_end - lag));
                    col = col + 1;
                end
            end
        end

        X_mat = X_mat(:, 1:(col - 1));

        b = X_mat \ Y_dep;
        u = Y_dep - X_mat * b;

        [se_h, ~] = puremacro.inference.hac(X_mat, u, h + 1);

        beta(idx) = b(2);
        se(idx) = se_h(2);
        lo(idx) = b(2) - z_crit * se_h(2);
        hi(idx) = b(2) + z_crit * se_h(2);
    end

    res.h = horizons(:);
    res.beta = beta;
    res.se = se;
    res.lo = lo;
    res.hi = hi;
end
