function [cycle, trend] = hpfilter(y, lambda)
% HPFILTER Hodrick-Prescott filter for trend-cycle decomposition.
%
%   [CYCLE, TREND] = puremacro.cycles.hpfilter(Y, LAMBDA)
%
%   Inputs:
%       y      - Time series vector (T x 1)
%       lambda - (Optional) Smoothing parameter. Default: 1600 (quarterly data).
%                Common values: 100 (annual), 1600 (quarterly), 14400 (monthly).
%
%   Outputs:
%       cycle  - Cyclical component (T x 1)
%       trend  - Trend component (T x 1)

    if nargin < 2 || isempty(lambda)
        lambda = 1600;
    end

    y_arr = y(:);
    T = length(y_arr);

    if T < 4
        error('hpfilter:inputTooShort', 'HP filter requires at least 4 observations.');
    end

    % Construct second difference operator matrix K of size (T-2) x T
    e = ones(T, 1);
    K = spdiags([e -2*e e], 0:2, T-2, T);

    % Solve linear system: (I + lambda * K' * K) * trend = y
    A = speye(T) + lambda * (K' * K);
    trend = A \ y_arr;
    cycle = y_arr - trend;
end
