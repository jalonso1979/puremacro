function [lower, upper, c, scale, center] = supt_bands(theta, Sigma, varargin)
% SUPT_BANDS Montiel Olea & Plagborg-Møller (2019) simultaneous sup-t confidence bands.
%
%   [LOWER, UPPER, C, SCALE, CENTER] = puremacro.inference.supt_bands(THETA, SIGMA, 'alpha', 0.10, 'method', 'plugin')
%   [LOWER, UPPER, C, SCALE, CENTER] = puremacro.inference.supt_bands([], [], 'draws', DRAWS, 'method', 'bootstrap')
%
%   Inputs:
%       theta  - Point estimates vector (H x 1)
%       Sigma  - Covariance matrix of theta (H x H)
%       Name-Value Options:
%           'alpha'   - Significance level (default: 0.10 for 90% band)
%           'method'  - 'plugin' (default), 'bootstrap', or 'bayes'
%           'draws'   - Bootstrap/posterior draws matrix (n_draws x H)
%           'n_mc'    - Monte Carlo draws for plugin method (default: 100,000)
%
%   Reference:
%       Montiel Olea, J.L. and Plagborg-Møller, M. (2019). "Simultaneous confidence
%       bands: Theory, implementation, and an application to SVARs". JAE 34(1), 1-17.

    p = inputParser;
    addParameter(p, 'alpha', 0.10);
    addParameter(p, 'method', 'plugin');
    addParameter(p, 'draws', []);
    addParameter(p, 'n_mc', 100000);
    parse(p, varargin{:});

    alpha = p.Results.alpha;
    method = p.Results.method;
    draws = p.Results.draws;
    n_mc = p.Results.n_mc;

    if strcmp(method, 'plugin')
        if isempty(theta) || isempty(Sigma)
            error('supt_bands:missingArgs', 'method=''plugin'' requires theta and Sigma.');
        end
        theta = theta(:);
        H = length(theta);
        sd = sqrt(diag(Sigma));
        R = Sigma ./ (sd * sd');
        R(1:(H+1):end) = 1.0; % Fill diagonal with 1s

        % Draw Z ~ N(0, R) via Cholesky / Eigendecomposition
        [V, D] = eig((R + R') / 2);
        eigvals = max(diag(D), 0);
        factor = V * diag(sqrt(eigvals));

        Z = randn(n_mc, H) * factor';
        max_abs = max(abs(Z), [], 2);
        c = quantile(max_abs, 1 - alpha);

        center = theta;
        scale = sd;
        lower = center - c * scale;
        upper = center + c * scale;

    elseif strcmp(method, 'bootstrap') || strcmp(method, 'bayes')
        if isempty(draws)
            error('supt_bands:missingDraws', 'method=''%s'' requires draws matrix.', method);
        end
        H = size(draws, 2);

        if strcmp(method, 'bayes') || isempty(theta)
            center = mean(draws, 1)';
        else
            center = theta(:);
        end

        scale = std(draws, 0, 1)';
        scale(scale == 0) = 1e-12;

        % Compute max studentized absolute deviation for each draw
        stud_dev = max(abs(draws - center') ./ scale', [], 2);
        c = quantile(stud_dev, 1 - alpha);

        lower = center - c * scale;
        upper = center + c * scale;
    else
        error('supt_bands:unknownMethod', 'Unknown method %s.', method);
    end
end
