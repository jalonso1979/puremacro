function B0 = bq(A_list, Sigma, permanent_var_idx)
% BQ Blanchard-Quah (1989) long-run SVAR identification.
%
%   B0 = puremacro.var.bq(A_LIST, SIGMA, PERMANENT_VAR_IDX)
%
%   Inputs:
%       A_list            - Cell array of (N x N) VAR lag matrices
%       Sigma             - (N x N) reduced-form residual covariance matrix
%       permanent_var_idx - (Optional) Index of variable with permanent shock (default: 1)
%
%   Outputs:
%       B0 - (N x N) structural impact matrix such that the long-run impact
%            matrix C = (I - sum(A_l))^{-1} * B0 is lower triangular (with permanent
%            variable ordered first).

    if nargin < 3 || isempty(permanent_var_idx)
        permanent_var_idx = 1;
    end

    n = size(Sigma, 1);
    p = length(A_list);

    A_sum = zeros(n, n);
    for l = 1:p
        A_sum = A_sum + A_list{l};
    end

    Psi_1 = inv(eye(n) - A_sum);

    % Permutation vector so target variable comes first
    perm = [permanent_var_idx, setdiff(1:n, permanent_var_idx)];
    P = eye(n);
    P = P(perm, :);

    Psi_1_p = P * Psi_1 * P';
    Sigma_p = P * Sigma * P';

    Omega = Psi_1_p * Sigma_p * Psi_1_p';
    L = chol((Omega + Omega') / 2, 'lower');

    B_p = Psi_1_p \ L;
    B0 = P' * B_p * P;
end
