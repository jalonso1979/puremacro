function res = hd(A_list, B0, residuals, init_y, intercept)
% HD Historical decomposition of time series y_t into structural shock contributions.
%
%   RES = puremacro.var.hd(A_LIST, B0, RESIDUALS, INIT_Y, INTERCEPT)
%
%   Outputs struct RES with fields:
%       shocks        - (T_eff x N x N) shock contributions
%       deterministic - (T_eff x N) deterministic component

    [T_eff, n] = size(residuals);
    p = length(A_list);

    if nargin < 4 || isempty(init_y)
        init_y = zeros(p, n);
    end
    if nargin < 5 || isempty(intercept)
        intercept = zeros(n, 1);
    else
        intercept = intercept(:);
    end

    eps = residuals * inv(B0)'; % Structural shocks (T_eff x N)

    Phi = cell(1, T_eff);
    Phi{1} = eye(n);
    for h = 1:(T_eff - 1)
        ph = zeros(n, n);
        for j = 1:min(h, p)
            ph = ph + Phi{h - j + 1} * A_list{j};
        end
        Phi{h + 1} = ph;
    end

    shocks = zeros(T_eff, n, n);
    for j = 1:n
        b_j = B0(:, j);
        for t = 1:T_eff
            for s = 1:t
                contrib = Phi{t - s + 1} * b_j * eps(s, j);
                shocks(t, :, j) = shocks(t, :, j) + contrib';
            end
        end
    end

    Y_det = zeros(p + T_eff, n);
    Y_det(1:p, :) = init_y;
    for t = (p + 1):(p + T_eff)
        y_t = intercept;
        for k = 1:p
            y_t = y_t + A_list{k} * Y_det(t - k, :)';
        end
        Y_det(t, :) = y_t';
    end

    res.shocks = shocks;
    res.deterministic = Y_det((p + 1):end, :);
end
