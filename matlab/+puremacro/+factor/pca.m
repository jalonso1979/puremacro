function res = pca(X, k, demean, standardize)
% PCA High-dimensional principal component factor extraction.
%
%   RES = puremacro.factor.pca(X, K, DEMEAN, STANDARDIZE)
%
%   Inputs:
%       X           - (T x N) data matrix
%       k           - Number of factors to extract (integer >= 1)
%       demean      - (Optional) Boolean flag to demean columns (default: true)
%       standardize - (Optional) Boolean flag to standardize variance (default: true)
%
%   Outputs struct RES with fields:
%       factors          - (T x K) extracted factors
%       loadings         - (N x K) factor loadings matrix
%       eigvals          - (K x 1) factor eigenvalues
%       variance_share   - (K x 1) share of total variance explained
%       cumulative_share - (K x 1) cumulative share of variance explained

    if nargin < 2 || isempty(k)
        k = 1;
    end
    if nargin < 3 || isempty(demean)
        demean = true;
    end
    if nargin < 4 || isempty(standardize)
        standardize = true;
    end

    X = double(X);
    [T, n] = size(X);

    if demean
        X = X - mean(X, 1);
    end
    if standardize
        sd = std(X, 0, 1);
        sd(sd == 0) = 1.0;
        X = X ./ sd;
    end

    [U, S, V] = svd(X, 'econ');
    s_diag = diag(S);

    factors = sqrt(T) * U(:, 1:k);
    loadings = V(:, 1:k) * (diag(s_diag(1:k)) / sqrt(T));

    eigvals = (s_diag(1:k).^2) / T;
    total_var = sum(s_diag.^2 / T);
    if total_var > 0
        var_share = eigvals / total_var;
    else
        var_share = zeros(k, 1);
    end

    res.factors = factors;
    res.loadings = loadings;
    res.eigvals = eigvals;
    res.variance_share = var_share;
    res.cumulative_share = cumsum(var_share);
end
