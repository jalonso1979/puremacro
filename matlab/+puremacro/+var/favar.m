function res = favar(X_panel, Y_macro, n_factors, p_lags, horizon)
% FAVAR Bernanke, Boivin & Eliasz (2005) Factor-Augmented VAR.
%
%   RES = puremacro.var.favar(X_PANEL, Y_MACRO, N_FACTORS, P_LAGS, HORIZON)

    if nargin < 3 || isempty(n_factors)
        n_factors = 3;
    end
    if nargin < 4 || isempty(p_lags)
        p_lags = 2;
    end
    if nargin < 5 || isempty(horizon)
        horizon = 15;
    end

    X_panel = double(X_panel);
    Y_macro = double(Y_macro);
    [T, M] = size(X_panel);

    % Standardize large panel X_panel
    X_std = (X_panel - mean(X_panel, 1)) ./ std(X_panel, 0, 1);

    % Extract Principal Component Factors
    [U_svd, S_svd, ~] = svd(X_std, 'econ');
    F_factors = U_svd(:, 1:n_factors) * S_svd(1:n_factors, 1:n_factors) / sqrt(T);
    loadings = X_std' * F_factors / T;

    % Combine Factors F and Macro Variables Y: Z = [F, Y]
    Z_data = [F_factors, Y_macro];

    % Fit Joint VAR(p) on Z_data
    var_res = puremacro.var.estimate(Z_data, p_lags);
    B0_chol = puremacro.var.cholesky(var_res.Sigma);

    % Compute IRF of joint system [F, Y]
    irf_joint = puremacro.var.irf(var_res.A_list, B0_chol, horizon);

    % Map factor IRFs back to all M panel variables: IRF_X = L * IRF_F
    irf_panel = zeros(M, horizon + 1);
    for h = 0:horizon
        irf_F_h = squeeze(irf_joint(h + 1, 1:n_factors, 1)); % (n_factors x 1)
        irf_panel(:, h + 1) = loadings * irf_F_h(:);
    end

    res.factors = F_factors;
    res.loadings = loadings;
    res.B0 = B0_chol;
    res.irf_joint = irf_joint;
    res.irf_panel = irf_panel;
    res.horizon = horizon;
end
