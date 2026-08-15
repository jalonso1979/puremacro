function res = estimate(Y, p)
% ESTIMATE Reduced-form VAR(p) estimation via OLS.
%
%   RES = puremacro.var.estimate(Y, P)
%
%   Inputs:
%       Y - (T x N) data matrix of endogenous variables
%       p - Lag order (integer >= 1)
%
%   Outputs struct RES with fields:
%       A_list - Cell array of (N x N) coefficient matrices {A1, A2, ..., Ap}
%       c      - (N x 1) intercept vector
%       Sigma  - (N x N) residual covariance matrix
%       resid  - ((T-p) x N) reduced-form residual matrix
%       X      - Regressor design matrix ((T-p) x (1 + N*p))
%       is_stable - Boolean flag indicating if companion eigenvalues lie inside unit circle

    Y = double(Y);
    [T, n] = size(Y);

    if p < 1 || T <= p
        error('estimate:invalidLags', 'Lag order p must be positive and T > p.');
    end

    % Construct X: [const, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    T_eff = T - p;
    X = zeros(T_eff, 1 + n * p);
    X(:, 1) = 1.0;

    for l = 1:p
        X(:, (2 + (l - 1) * n):(1 + l * n)) = Y((p - l + 1):(T - l), :);
    end

    Yd = Y((p + 1):T, :);

    % OLS regression: B = (X' * X) \ (X' * Yd)
    B = X \ Yd;

    c = B(1, :)';
    A_list = cell(1, p);
    for l = 1:p
        A_list{l} = B((2 + (l - 1) * n):(1 + l * n), :)';
    end

    resid = Yd - X * B;
    df = T_eff - (1 + n * p);
    Sigma = (resid' * resid) / df;

    % Companion matrix & stability check
    C = zeros(n * p, n * p);
    for l = 1:p
        C(1:n, ((l - 1) * n + 1):(l * n)) = A_list{l};
    end
    if p > 1
        C((n + 1):end, 1:(n * (p - 1))) = eye(n * (p - 1));
    end

    eigvals = abs(eig(C));
    is_stable = all(eigvals < 1.0 - 1e-8);

    res.A_list = A_list;
    res.c = c;
    res.Sigma = Sigma;
    res.resid = resid;
    res.X = X;
    res.companion = C;
    res.is_stable = is_stable;
    res.n = n;
    res.p = p;
    res.T_eff = T_eff;
end
