function res = state_dependent(y, x, state_var, horizons, n_lags, transition_type, gamma, threshold, alpha)
% STATE_DEPENDENT State-Dependent Local Projections (Auerbach & Gorodnichenko 2013 / Ramey & Zubairy 2018).
%
%   RES = puremacro.lp.state_dependent(Y, X_SHOCK, STATE_VAR, HORIZONS, N_LAGS, TRANSITION_TYPE, GAMMA, THRESHOLD, ALPHA)
%
%   Estimates:
%       y_{t+h} - y_{t-1} = \alpha_h + \beta_h^H * F(z_t) * x_t + \beta_h^L * (1 - F(z_t)) * x_t + \text{controls} + \varepsilon_{t,h}
%
%   Inputs:
%       y               - Outcome variable time series (T x 1)
%       x               - Shock / treatment variable time series (T x 1)
%       state_var       - Transition state variable time series (T x 1)
%       horizons        - (Optional) Horizons vector (default: 0:20)
%       n_lags          - (Optional) Number of control lags (default: 2)
%       transition_type - (Optional) 'logistic' (default) or 'threshold'
%       gamma           - (Optional) Transition speed steepness parameter (default: 3.0)
%       threshold       - (Optional) Transition threshold (default: 0.0)
%       alpha           - (Optional) Significance level (default: 0.10)
%
%   Outputs struct RES with fields:
%       h         - Vector of horizons
%       beta_high - High-regime impulse responses (H x 1)
%       se_high   - Newey-West standard errors for high regime (H x 1)
%       lo_high   - Lower bounds for high regime
%       hi_high   - Upper bounds for high regime
%       beta_low  - Low-regime impulse responses (H x 1)
%       se_low    - Newey-West standard errors for low regime (H x 1)
%       lo_low    - Lower bounds for low regime
%       hi_low    - Upper bounds for low regime

    if nargin < 4 || isempty(horizons)
        horizons = 0:20;
    end
    if nargin < 5 || isempty(n_lags)
        n_lags = 2;
    end
    if nargin < 6 || isempty(transition_type)
        transition_type = 'logistic';
    end
    if nargin < 7 || isempty(gamma)
        gamma = 3.0;
    end
    if nargin < 8 || isempty(threshold)
        threshold = 0.0;
    end
    if nargin < 9 || isempty(alpha)
        alpha = 0.10;
    end

    y = y(:);
    x = x(:);
    state_var = state_var(:);
    T = length(y);
    H = length(horizons);

    % Standardize transition variable
    s_std = std(state_var);
    if s_std <= 0
        s_std = 1.0;
    end
    z_score = (state_var - mean(state_var)) / s_std;

    if strcmp(transition_type, 'logistic')
        F_val = 1.0 ./ (1.0 + exp(-gamma * (z_score - threshold)));
    elseif strcmp(transition_type, 'threshold')
        F_val = double(z_score > threshold);
    else
        error('state_dependent:unknownTransition', 'Unknown transition_type %s.', transition_type);
    end

    beta_high = nan(H, 1); se_high = nan(H, 1); lo_high = nan(H, 1); hi_high = nan(H, 1);
    beta_low  = nan(H, 1); se_low  = nan(H, 1); lo_low  = nan(H, 1); hi_low  = nan(H, 1);

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

        F_sub = F_val(t_start:t_end);
        x_sub = x(t_start:t_end);

        % Regressors: [const, F * x, (1 - F) * x, x_lags, y_lags]
        X_mat = zeros(N_eff, 3 + 2 * n_lags);
        X_mat(:, 1) = 1.0;
        X_mat(:, 2) = F_sub .* x_sub;
        X_mat(:, 3) = (1.0 - F_sub) .* x_sub;

        col = 4;
        for lag = 1:n_lags
            X_mat(:, col) = x((t_start - lag):(t_end - lag));
            col = col + 1;
            X_mat(:, col) = y((t_start - lag):(t_end - lag));
            col = col + 1;
        end

        b = X_mat \ Y_dep;
        u = Y_dep - X_mat * b;

        [se_h, ~] = puremacro.inference.hac(X_mat, u, h + 1);

        beta_high(idx) = b(2);
        se_high(idx)   = se_h(2);
        lo_high(idx)   = b(2) - z_crit * se_h(2);
        hi_high(idx)   = b(2) + z_crit * se_h(2);

        beta_low(idx)  = b(3);
        se_low(idx)    = se_h(3);
        lo_low(idx)    = b(3) - z_crit * se_h(3);
        hi_low(idx)    = b(3) + z_crit * se_h(3);
    end

    res.h = horizons(:);
    res.beta_high = beta_high;
    res.se_high = se_high;
    res.lo_high = lo_high;
    res.hi_high = hi_high;
    res.beta_low = beta_low;
    res.se_low = se_low;
    res.lo_low = lo_low;
    res.hi_low = hi_low;
end
