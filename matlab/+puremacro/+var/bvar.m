function res = bvar(Y, p, lambda, mu)
% BVAR Analytical Conjugate Minnesota Bayesian VAR with Dummy Observations.
%
%   RES = puremacro.var.bvar(Y, P, LAMBDA, MU)
%
%   Inputs:
%       Y      - (T x N) data matrix
%       p      - (Optional) Lag order (default: 2)
%       lambda - (Optional) Overall tightness parameter (default: 0.2)
%       mu     - (Optional) Sum of coefficients tightness (default: 1.0)
%
%   Outputs struct RES with fields:
%       A_list - Cell array of posterior mean coefficient matrices
%       c      - (N x 1) posterior mean intercept vector
%       Sigma  - (N x N) posterior scale residual covariance
%       B_post - Full regressor coefficient matrix ((1 + N*p) x N)

    if nargin < 2 || isempty(p)
        p = 2;
    end
    if nargin < 3 || isempty(lambda)
        lambda = 0.2;
    end
    if nargin < 4 || isempty(mu)
        mu = 1.0;
    end

    Y = double(Y);
    [T, n] = size(Y);

    % AR(1) residual standard deviations for scaling Minnesota prior
    sigma_ar = zeros(n, 1);
    for i = 1:n
        y_i = Y(:, i);
        dy = y_i(2:end) - y_i(1:(end - 1));
        sigma_ar(i) = std(dy);
        if sigma_ar(i) == 0, sigma_ar(i) = 1.0; end
    end

    % Standard VAR design matrix X
    T_eff = T - p;
    X = zeros(T_eff, 1 + n * p);
    X(:, 1) = 1.0;
    for l = 1:p
        X(:, (2 + (l - 1) * n):(1 + l * n)) = Y((p - l + 1):(T - l), :);
    end
    Yd = Y((p + 1):T, :);

    % Construct Minnesota dummy observations Y_dumm, X_dumm
    % 1. Minnesota lag prior dummies
    Y_d1 = zeros(n * p, n);
    X_d1 = zeros(n * p, 1 + n * p);
    row = 1;
    for l = 1:p
        for j = 1:n
            if l == 1
                Y_d1(row, j) = sigma_ar(j) / lambda;
            end
            col = 1 + (l - 1) * n + j;
            X_d1(row, col) = (l * sigma_ar(j)) / lambda;
            row = row + 1;
        end
    end

    % 2. Sum-of-coefficients dummy observations
    Y_d2 = zeros(n, n);
    X_d2 = zeros(n, 1 + n * p);
    y_bar = mean(Y(1:p, :), 1);
    for j = 1:n
        Y_d2(j, j) = y_bar(j) / mu;
        for l = 1:p
            X_d2(j, 1 + (l - 1) * n + j) = y_bar(j) / mu;
        end
    end

    % Combine actual and dummy observations
    X_star = [X; X_d1; X_d2];
    Y_star = [Yd; Y_d1; Y_d2];

    % Solve augmented OLS system
    B_post = X_star \ Y_star;

    c = B_post(1, :)';
    A_list = cell(1, p);
    for l = 1:p
        A_list{l} = B_post((2 + (l - 1) * n):(1 + l * n), :)';
    end

    resid = Yd - X * B_post;
    Sigma = (resid' * resid) / (T_eff - (1 + n * p));

    res.A_list = A_list;
    res.c = c;
    res.Sigma = Sigma;
    res.B_post = B_post;
    res.n = n;
    res.p = p;
end
