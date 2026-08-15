function [z_grid, P] = rouwenhorst(N, rho, sigma)
% ROUWENHORST Rouwenhorst (1995) method for discretizing highly persistent AR(1) process.
%
%   [Z_GRID, P] = puremacro.vfi.rouwenhorst(N, RHO, SIGMA)
%
%   Inputs:
%       N     - Number of grid points (integer >= 2)
%       rho   - AR(1) persistence coefficient (|rho| < 1)
%       sigma - Standard deviation of innovation e_t
%
%   Outputs:
%       z_grid - (N x 1) discrete state grid points
%       P      - (N x N) Markov transition matrix

    p_param = (1 + rho) / 2;
    q_param = p_param;
    psi = sigma * sqrt((N - 1) / (1 - rho^2));

    z_grid = linspace(-psi, psi, N)';

    if N == 2
        P = [p_param, 1 - p_param; 1 - q_param, q_param];
        return;
    end

    % Recursive construction of transition matrix P_N
    P_prev = [p_param, 1 - p_param; 1 - q_param, q_param];
    for n_curr = 3:N
        P1 = zeros(n_curr, n_curr);
        P2 = zeros(n_curr, n_curr);
        P3 = zeros(n_curr, n_curr);
        P4 = zeros(n_curr, n_curr);

        P1(1:(n_curr-1), 1:(n_curr-1)) = P_prev;
        P2(1:(n_curr-1), 2:n_curr)     = P_prev;
        P3(2:n_curr, 1:(n_curr-1))     = P_prev;
        P4(2:n_curr, 2:n_curr)         = P_prev;

        P_curr = p_param * P1 + (1 - p_param) * P2 + (1 - q_param) * P3 + q_param * P4;
        P_curr(2:(n_curr-1), :) = P_curr(2:(n_curr-1), :) / 2;
        P_prev = P_curr;
    end
    P = P_prev;
end
