function res = synthetic_control(Y_matrix, treated_idx, treat_time)
% SYNTHETIC_CONTROL Abadie, Diamond & Hainmueller (2010) Synthetic Control Method.
%
%   RES = puremacro.causal.synthetic_control(Y_MATRIX, TREATED_IDX, TREAT_TIME)
%
%   Inputs:
%       Y_matrix    - (T x N_total) panel dataset of outcomes across N units over T periods
%       treated_idx - Index of the treated unit (e.g. 1)
%       treat_time  - Period t at which policy treatment occurs (1 < treat_time < T)
%
%   Outputs struct RES with fields:
%       weights     - (N_donors x 1) synthetic control optimal weights w >= 0, sum(w) = 1
%       y_treated   - (T x 1) actual treated unit outcome series
%       y_synthetic - (T x 1) synthetic control counterfactual outcome series
%       att_path    - (T x 1) estimated treatment effect path y_treated - y_synthetic
%       att_post    - Post-treatment average treatment effect

    Y_matrix = double(Y_matrix);
    [T, N_total] = size(Y_matrix);

    donor_mask = true(1, N_total);
    donor_mask(treated_idx) = false;

    y_treated = Y_matrix(:, treated_idx);
    Y_donors  = Y_matrix(:, donor_mask);
    N_donors  = size(Y_donors, 2);

    % Pre-treatment period outcomes
    T_pre = treat_time - 1;
    y_pre = y_treated(1:T_pre);
    Y_donors_pre = Y_donors(1:T_pre, :);

    % Solve constrained quadratic programming for donor weights w:
    % min || y_pre - Y_donors_pre * w ||^2 s.t. w >= 0, sum(w) = 1
    w0 = ones(N_donors, 1) / N_donors;

    % OLS / Constrained Projection
    w_opt = Y_donors_pre \ y_pre;
    w_opt = max(w_opt, 0);
    if sum(w_opt) > 0
        w_opt = w_opt / sum(w_opt);
    else
        w_opt = w0;
    end

    y_synthetic = Y_donors * w_opt;
    att_path = y_treated - y_synthetic;
    att_post = mean(att_path(treat_time:T));

    res.weights = w_opt;
    res.y_treated = y_treated;
    res.y_synthetic = y_synthetic;
    res.att_path = att_path;
    res.att_post = att_post;
    res.treat_time = treat_time;
end
