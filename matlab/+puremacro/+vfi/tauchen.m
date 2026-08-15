function [z_grid, P] = tauchen(N, rho, sigma, m)
% TAUCHEN Tauchen (1986) method for discretizing AR(1) process z_t = rho * z_{t-1} + e_t.
%
%   [Z_GRID, P] = puremacro.vfi.tauchen(N, RHO, SIGMA, M)
%
%   Inputs:
%       N     - Number of grid points (integer >= 2)
%       rho   - AR(1) persistence coefficient (|rho| < 1)
%       sigma - Standard deviation of innovation e_t
%       m     - (Optional) Width of grid in std dev units (default: 3.0)
%
%   Outputs:
%       z_grid - (N x 1) discrete state grid points
%       P      - (N x N) Markov transition matrix where P(i,j) = Pr(z_{t+1} = z_j | z_t = z_i)

    if nargin < 4 || isempty(m)
        m = 3.0;
    end

    y_std = sigma / sqrt(1 - rho^2);
    z_max = m * y_std;
    z_min = -z_max;

    z_grid = linspace(z_min, z_max, N)';
    step = z_grid(2) - z_grid(1);

    P = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if j == 1
                P(i, j) = norm_cdf((z_grid(1) - rho * z_grid(i) + step / 2) / sigma);
            elseif j == N
                P(i, j) = 1.0 - norm_cdf((z_grid(N) - rho * z_grid(i) - step / 2) / sigma);
            else
                P(i, j) = norm_cdf((z_grid(j) - rho * z_grid(i) + step / 2) / sigma) - ...
                          norm_cdf((z_grid(j) - rho * z_grid(i) - step / 2) / sigma);
            end
        end
    end
end

function p = norm_cdf(x)
    p = 0.5 * (1 + erf(x / sqrt(2)));
end
