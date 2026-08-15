function res = smooth(y, x, horizons, p_lags, lambda_pen, diff_order, ci_level)
% SMOOTH Barnichon & Brownlees (2019) Smooth Local Projections via P-spline Penalization.
%
%   RES = puremacro.lp.smooth(Y, X, HORIZONS, P_LAGS, LAMBDA_PEN, DIFF_ORDER, CI_LEVEL)
%
%   Inputs:
%       y          - (T x 1) dependent variable series
%       x          - (T x 1) structural shock / policy variable series
%       horizons   - Vector of horizons 0..H (e.g. 0:20)
%       p_lags     - (Optional) Lag order of control variables (default: 2)
%       lambda_pen - (Optional) Smoothing penalty parameter lambda >= 0 (default: 10.0)
%       diff_order - (Optional) Difference penalty order (default: 2)
%       ci_level   - (Optional) Confidence interval level (default: 0.10 for 90% bands)
%
%   Outputs struct RES with fields:
%       beta_smooth - (H+1 x 1) smoothed local projection impulse response coefficients
%       beta_ols    - (H+1 x 1) standard unpenalized Jordà LP coefficients
%       se          - (H+1 x 1) standard errors
%       lo          - (H+1 x 1) lower confidence band
%       hi          - (H+1 x 1) upper confidence band
%       h           - Vector of horizons 0..H

    if nargin < 4 || isempty(p_lags)
        p_lags = 2;
    end
    if nargin < 5 || isempty(lambda_pen)
        lambda_pen = 10.0;
    end
    if nargin < 6 || isempty(diff_order)
        diff_order = 2;
    end
    if nargin < 7 || isempty(ci_level)
        ci_level = 0.10;
    end

    y = double(y(:));
    x = double(x(:));
    H = max(horizons);
    H_len = H + 1;
    T = length(y);

    % First compute unpenalized Jordà (2005) LP coefficients
    ols_res = puremacro.lp.estimate(y, x, horizons, p_lags, [], ci_level);
    beta_ols = ols_res.beta;

    % Construct difference penalty matrix D of size (H+1 - diff_order) x (H+1)
    D = eye(H_len);
    for k = 1:diff_order
        D = diff(D);
    end

    % Solve penalized system: (I + lambda * D' * D) * beta_smooth = beta_ols
    P_mat = D' * D;
    beta_smooth = (eye(H_len) + lambda_pen * P_mat) \ beta_ols;

    % Smooth standard error propagation: Var(beta_smooth) = S * Var(beta_ols) * S'
    S_mat = inv(eye(H_len) + lambda_pen * P_mat);
    cov_ols = diag(ols_res.se .^ 2);
    cov_smooth = S_mat * cov_ols * S_mat';
    se_smooth = sqrt(diag(cov_smooth));

    z_crit = norm_inv_val(1 - ci_level / 2);
    lo = beta_smooth - z_crit * se_smooth;
    hi = beta_smooth + z_crit * se_smooth;

    res.beta_smooth = beta_smooth;
    res.beta_ols = beta_ols;
    res.se = se_smooth;
    res.lo = lo;
    res.hi = hi;
    res.h = horizons(:);
    res.lambda = lambda_pen;
end

function val = norm_inv_val(p)
    val = sqrt(2) * erfinv(2 * p - 1);
end
