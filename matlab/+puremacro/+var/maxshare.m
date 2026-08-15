function [B0, fev_share] = maxshare(A_list, Sigma, target_var, horizon)
% MAXSHARE Faust-Uhlig (2003) / Barsky-Sims (2011) FEVD Max-Share SVAR identification.
%
%   [B0, FEV_SHARE] = puremacro.var.maxshare(A_LIST, SIGMA, TARGET_VAR, HORIZON)
%
%   Identifies the structural shock that maximizes the forecast error variance
%   share of TARGET_VAR over horizons 0..HORIZON (e.g. news shocks).
%
%   Inputs:
%       A_list     - Cell array of (N x N) VAR lag matrices
%       Sigma      - (N x N) reduced-form residual covariance matrix
%       target_var - Target variable index (1..N)
%       horizon    - (Optional) Horizon H over which FEV share is accumulated (default: 40)
%
%   Outputs:
%       B0        - (N x N) structural impact matrix (column 1 holds max-share shock)
%       fev_share - Share of forecast error variance explained by the identified shock at HORIZON

    if nargin < 3 || isempty(target_var)
        target_var = 1;
    end
    if nargin < 4 || isempty(horizon)
        horizon = 40;
    end

    n = size(Sigma, 1);
    p = length(A_list);
    P = chol(Sigma, 'lower');

    % Compute MA coefficient matrices Phi_h for h = 0..H
    Phi = cell(1, horizon + 1);
    Phi{1} = eye(n);
    for h = 1:horizon
        ph = zeros(n, n);
        for j = 1:min(h, p)
            ph = ph + Phi{h - j + 1} * A_list{j};
        end
        Phi{h + 1} = ph;
    end

    % Construct symmetric PSD matrix M = sum_{h=0}^H (e_target' * Phi_h * P)' * (e_target' * Phi_h * P)
    e_target = zeros(1, n);
    e_target(target_var) = 1.0;
    M = zeros(n, n);

    for h = 0:horizon
        v = e_target * Phi{h + 1} * P; % (1 x N)
        M = M + v' * v;
    end

    % Top eigenvector of M is optimal rotation vector q1
    [V_eig, D_eig] = eig((M + M') / 2);
    [~, max_idx] = max(diag(D_eig));
    q1 = V_eig(:, max_idx);
    q1 = q1 / norm(q1);

    b_col1 = P * q1;

    % Complete B0 matrix via Gram-Schmidt
    B0 = zeros(n, n);
    B0(:, 1) = b_col1;

    candidates = eye(n);
    idx = 2;
    for j = 1:n
        v = candidates(:, j);
        for k = 1:(idx - 1)
            v = v - (v' * B0(:, k)) / (B0(:, k)' * B0(:, k)) * B0(:, k);
        end
        v_norm = norm(v);
        if v_norm > 1e-10 && idx <= n
            B0(:, idx) = v / v_norm;
            idx = idx + 1;
        end
    end

    % Compute FEV share explained by shock 1 at horizon H
    irfs = puremacro.var.irf(A_list, B0, horizon);
    fev_shares = puremacro.var.fevd(A_list, B0, horizon);
    fev_share = fev_shares(horizon + 1, target_var, 1);
end
