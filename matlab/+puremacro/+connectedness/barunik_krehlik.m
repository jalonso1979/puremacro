function res = barunik_krehlik(returns_panel, var_lags, fevd_horizon, freq_bounds)
% BARUNIK_KREHLIK Baruník & Křehlík (2018) Frequency-Domain Connectedness Index.
%
%   RES = puremacro.connectedness.barunik_krehlik(RETURNS_PANEL, VAR_LAGS, FEVD_HORIZON, FREQ_BOUNDS)
%
%   Decomposes Diebold-Yilmaz connectedness into frequency bands (e.g., short-run vs long-run).
%
%   Inputs:
%       returns_panel - (T x N) asset returns / macro panel
%       var_lags      - (Optional) VAR lag order (default: 4)
%       fevd_horizon  - (Optional) FEVD horizon (default: 100)
%       freq_bounds   - (Optional) Frequency boundaries [w_lower, w_upper] in [0, pi] (default: [pi/4, pi])
%
%   Outputs struct RES with fields:
%       total_spillover - Total frequency spillover index (%)
%       band_matrix     - (N x N) frequency-decomposed FEVD spillover matrix

    if nargin < 2 || isempty(var_lags)
        var_lags = 4;
    end
    if nargin < 3 || isempty(fevd_horizon)
        fevd_horizon = 100;
    end
    if nargin < 4 || isempty(freq_bounds)
        freq_bounds = [pi/4, pi];
    end

    var_res = puremacro.var.estimate(returns_panel, var_lags);
    A_list = var_res.A_list;
    Sigma = var_res.Sigma;

    n = size(Sigma, 1);
    sigma_jj = diag(Sigma);

    % Numerical integration over frequency band [w1, w2]
    n_freq = 500;
    w_grid = linspace(freq_bounds(1), freq_bounds(2), n_freq);
    dw = w_grid(2) - w_grid(1);

    num_int = zeros(n, n);
    den_int = zeros(n, 1);

    for idx = 1:n_freq
        w = w_grid(idx);

        % Compute Psi(w) = (I - sum(A_l * exp(-i * w * l)))^{-1}
        A_w = zeros(n, n);
        for l = 1:var_lags
            A_w = A_w + A_list{l} * exp(-1i * w * l);
        end
        Psi_w = inv(eye(n) - A_w);
        PS_w = Psi_w * Sigma;

        for i = 1:n
            for j = 1:n
                num_int(i, j) = num_int(i, j) + (abs(PS_w(i, j))^2 / max(sigma_jj(j), 1e-12)) * dw;
            end
            den_int(i) = den_int(i) + real((Psi_w(i, :) * Sigma * Psi_w(i, :)')) * dw;
        end
    end

    theta_w = num_int ./ max(den_int, 1e-12);
    row_sums = sum(theta_w, 2);
    row_sums(row_sums == 0) = 1.0;
    theta_w = theta_w ./ row_sums;

    off_diag_sum = sum(theta_w(:)) - sum(diag(theta_w));
    total_spillover = 100.0 * off_diag_sum / n;

    res.total_spillover = total_spillover;
    res.band_matrix = theta_w;
end
