function [B0, f_stat] = proxy(Sigma, resid, instrument, target_idx)
% PROXY External Instrument / Proxy-SVAR identification (Mertens & Ravn 2013 / Stock & Watson 2018).
%
%   [B0, F_STAT] = puremacro.var.proxy(SIGMA, RESID, INSTRUMENT, TARGET_IDX)
%
%   Inputs:
%       Sigma      - (N x N) reduced-form residual covariance matrix
%       resid      - (T_eff x N) VAR residual matrix
%       instrument - (T_eff x 1) external instrument proxy series
%       target_idx - (Optional) Target variable index (default: 1)
%
%   Outputs:
%       B0         - (N x N) identified structural impact matrix (column 1 holds target shock)
%       f_stat     - First-stage Olea-Pflueger / Stock-Yogo F-statistic on proxy strength

    if nargin < 4 || isempty(target_idx)
        target_idx = 1;
    end

    [T_eff, n] = size(resid);
    z = instrument(:);
    if length(z) > T_eff
        z = z((end - T_eff + 1):end);
    end
    z = z - mean(z);

    % First-stage regression of target residual on proxy z
    u_target = resid(:, target_idx);
    b_first = (z' * u_target) / (z' * z);
    e_first = u_target - z * b_first;
    f_stat = (b_first^2 * (z' * z)) / (sum(e_first.^2) / (T_eff - 1));

    % OLS covariance of residuals with instrument
    Pi = (resid' * z) / (z' * z); % (N x 1)
    norm_factor = sqrt(Pi' * Sigma * Pi);

    if norm_factor < 1e-10
        error('proxy:weakInstrument', 'Degenerate instrument in proxy-SVAR (norm factor near zero).');
    end

    b_col1 = (Sigma * Pi) / norm_factor;

    % Complete B0 matrix via SVD of remaining covariance
    B0 = zeros(n, n);
    B0(:, 1) = b_col1;

    residual_cov = Sigma - (b_col1 * b_col1');
    [U_svd, S_svd, ~] = svd((residual_cov + residual_cov') / 2);
    s_diag = diag(S_svd);

    idx = 2;
    for k = 1:n
        if s_diag(k) > 1e-8 && idx <= n
            B0(:, idx) = U_svd(:, k) * sqrt(s_diag(k));
            idx = idx + 1;
        end
    end
end
