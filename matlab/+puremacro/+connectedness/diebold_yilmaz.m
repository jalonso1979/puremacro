function res = diebold_yilmaz(returns_panel, var_lags, fevd_horizon, identification)
% DIEBOLD_YILMAZ Diebold-Yilmaz (2009, 2012) spillover index via FEVD.
%
%   RES = puremacro.connectedness.diebold_yilmaz(RETURNS_PANEL, VAR_LAGS, FEVD_HORIZON, IDENTIFICATION)
%
%   Inputs:
%       returns_panel  - (T x N) matrix of asset returns / macro series
%       var_lags       - (Optional) Number of VAR lags (default: 4)
%       fevd_horizon   - (Optional) FEVD horizon (default: 12)
%       identification - (Optional) 'gfevd' (default, order-invariant) or 'cholesky'
%
%   Outputs struct RES with fields:
%       total            - Total spillover index (%)
%       directional_to   - (N x 1) to-spillover for each variable (%)
%       directional_from - (N x 1) from-spillover for each variable (%)
%       net              - (N x 1) net spillover (to - from) (%)
%       pairwise_matrix  - (N x N) normalized FEVD matrix

    if nargin < 2 || isempty(var_lags)
        var_lags = 4;
    end
    if nargin < 3 || isempty(fevd_horizon)
        fevd_horizon = 12;
    end
    if nargin < 4 || isempty(identification)
        identification = 'gfevd';
    end

    var_res = puremacro.var.estimate(returns_panel, var_lags);
    A_list = var_res.A_list;
    Sigma = var_res.Sigma;

    if strcmp(identification, 'cholesky')
        B0 = puremacro.var.cholesky(Sigma);
        fev_all = puremacro.var.fevd(A_list, B0, fevd_horizon);
        theta = squeeze(fev_all(fevd_horizon + 1, :, :));
    elseif strcmp(identification, 'gfevd')
        % Generalized FEVD (Pesaran-Shin 1998 / Diebold-Yilmaz 2012)
        n = size(Sigma, 1);
        Phi = cell(1, fevd_horizon + 1);
        Phi{1} = eye(n);
        for h = 1:fevd_horizon
            ph = zeros(n, n);
            for j = 1:min(h, var_lags)
                ph = ph + Phi{h - j + 1} * A_list{j};
            end
            Phi{h + 1} = ph;
        end

        sigma_jj = diag(Sigma);
        num_cum = zeros(n, n);
        den_cum = zeros(n, 1);

        for h = 0:fevd_horizon
            Ph = Phi{h + 1};
            PS = Ph * Sigma;
            num_cum = num_cum + (PS.^2) ./ max(sigma_jj', 1e-12);
            var_i = diag(Ph * Sigma * Ph');
            den_cum = den_cum + max(var_i, 0.0);
        end

        theta = num_cum ./ max(den_cum, 1e-12);
        row_sums = sum(theta, 2);
        row_sums(row_sums == 0) = 1.0;
        theta = theta ./ row_sums;
    else
        error('diebold_yilmaz:unknownIdentification', 'Unknown identification method %s.', identification);
    end

    n = size(theta, 1);
    off_diag_sum = sum(theta(:)) - sum(diag(theta));

    total = 100.0 * off_diag_sum / n;
    directional_to = 100.0 * (sum(theta, 1)' - diag(theta)) / n;
    directional_from = 100.0 * (sum(theta, 2) - diag(theta)) / n;
    net = directional_to - directional_from;

    res.total = total;
    res.directional_to = directional_to;
    res.directional_from = directional_from;
    res.net = net;
    res.pairwise_matrix = theta;
end
