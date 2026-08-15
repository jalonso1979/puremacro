function [cycle, trend] = hamilton(y, h, p)
% HAMILTON Hamilton (2018) regression filter for trend-cycle decomposition.
%
%   [CYCLE, TREND] = puremacro.cycles.hamilton(Y, H, P)
%
%   Projects y_{t+h} on a constant and (y_t, y_{t-1}, ..., y_{t-p+1}) via OLS.
%   The fitted value is the trend; the residual is the cycle.
%
%   Inputs:
%       y - Column vector or row vector of time series data (T x 1)
%       h - (Optional) Forecast horizon. Default: 8 (quarterly convention)
%       p - (Optional) Lags on right-hand side. Default: 4 (quarterly convention)
%
%   Outputs:
%       cycle - Cyclical component (T x 1, with first h+p-1 entries NaN)
%       trend - Trend component (T x 1, with first h+p-1 entries NaN)
%
%   Reference:
%       Hamilton, J.D. (2018). Why you should never use the Hodrick-Prescott
%       filter. Review of Economics and Statistics 100(5), 831-843.

    if nargin < 2 || isempty(h)
        h = 8;
    end
    if nargin < 3 || isempty(p)
        p = 4;
    end

    y_arr = y(:);
    T = length(y_arr);
    lags_needed = h + p;

    if T < lags_needed
        error('hamilton:inputTooShort', ...
            'Input length %d is shorter than h + p = %d; need at least %d observations.', ...
            T, lags_needed, lags_needed);
    end

    n_obs = T - h - (p - 1);
    X = zeros(n_obs, p + 1);
    X(:, 1) = 1.0;

    for j = 0:(p - 1)
        % Row i corresponds to t = p + i - 1
        % y_arr index for lag j: (p - j) : (T - h - j)
        X(:, 2 + j) = y_arr((p - j):(T - h - j));
    end

    % Target is y_{t+h} for t = p..T-h -> indices (p + h) : T
    y_target = y_arr((h + p):T);

    % Pseudoinverse OLS fit (matches Moore-Penrose pinv in Python)
    beta = pinv(X) * y_target;
    fitted = X * beta;
    residual = y_target - fitted;

    cycle = nan(T, 1);
    trend = nan(T, 1);

    cycle((h + p):T) = residual;
    trend((h + p):T) = fitted;
end
