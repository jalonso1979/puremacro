function B0 = cholesky(Sigma)
% CHOLESKY Lower-triangular Cholesky SVAR identification matrix B0 such that B0 * B0' = Sigma.
%
%   B0 = puremacro.var.cholesky(SIGMA)
%
%   Inputs:
%       Sigma - (N x N) reduced-form residual covariance matrix
%
%   Outputs:
%       B0    - (N x N) lower triangular matrix (structural impact matrix)

    B0 = chol(Sigma, 'lower');
end
