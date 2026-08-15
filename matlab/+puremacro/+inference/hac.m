function [se, cov] = hac(X, resid, bw)
% HAC Newey-West Heteroskedasticity and Autocorrelation Consistent covariance & SEs.
%
%   [SE, COV] = puremacro.inference.hac(X, RESID, BW)
%
%   Inputs:
%       X     - (N x K) regressor matrix (including constant if present)
%       resid - (N x 1) vector of OLS residuals
%       bw    - (Optional) Non-negative integer bandwidth (Bartlett kernel).
%               Default: floor(4 * (N/100)^(2/9))
%
%   Outputs:
%       se    - (K x 1) vector of Newey-West standard errors
%       cov   - (K x K) Newey-West covariance matrix

    [n, k] = size(X);
    if nargin < 3 || isempty(bw)
        bw = floor(4 * (n / 100)^(2/9));
    end

    XtX_inv = (X' * X) \ eye(k);
    u = X .* resid(:); % (N x K) score matrix
    S = u' * u;        % lag 0

    for lag = 1:bw
        if lag >= n
            break;
        end
        w = 1 - lag / (bw + 1);
        G = u((lag + 1):end, :)' * u(1:(end - lag), :);
        S = S + w * (G + G');
    end

    cov = XtX_inv * S * XtX_inv;
    se = sqrt(diag(cov));
end
