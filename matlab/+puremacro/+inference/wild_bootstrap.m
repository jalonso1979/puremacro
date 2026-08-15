function draws = wild_bootstrap(residuals, refit_fn, n_boot)
% WILD_BOOTSTRAP Rademacher wild bootstrap for scalar / regression inference.
%
%   DRAWS = puremacro.inference.wild_bootstrap(RESIDUALS, REFIT_FN, N_BOOT)
%
%   Multiplies each residual by an i.i.d. Rademacher weight (+-1), then
%   calls refit_fn(e_boot) for N_BOOT replications.
%
%   Inputs:
%       residuals - (T x 1) or (T x N) fitted residuals matrix
%       refit_fn  - Function handle taking e_boot and returning a vector/matrix
%       n_boot    - (Optional) Number of bootstrap replications. Default: 999
%
%   Outputs:
%       draws     - Matrix/array of bootstrap draws (N_BOOT x ...)

    if nargin < 3 || isempty(n_boot)
        n_boot = 999;
    end

    T = size(residuals, 1);
    first_stat = refit_fn(residuals);
    stat_shape = size(first_stat);

    draws = zeros([n_boot, stat_shape]);

    for b = 1:n_boot
        w = sign(rand(T, 1) - 0.5);
        w(w == 0) = 1;

        if isvector(residuals)
            e_boot = residuals .* w;
        else
            e_boot = residuals .* w;
        end

        stat_b = refit_fn(e_boot);

        if isscalar(first_stat) || isvector(first_stat)
            draws(b, :) = stat_b(:)';
        else
            % For multi-dimensional return
            idx = repmat({':'}, 1, ndims(stat_b));
            draws(b, idx{:}) = stat_b;
        end
    end
end
