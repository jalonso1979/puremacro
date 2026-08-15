function res = estimate(y_lf, x_hf, K, type_name)
% ESTIMATE Mixed-data sampling (MIDAS) regression using native solvers.
%
%   RES = puremacro.midas.estimate(Y_LF, X_HF, K, TYPE_NAME)

    if nargin < 4 || isempty(type_name)
        type_name = 'unrestricted';
    end

    y_lf = y_lf(:);
    x_hf = x_hf(:);
    n_low = length(y_lf);

    if length(x_hf) ~= n_low * K
        error('midas:lengthMismatch', 'x_hf length must be n_low * K = %d.', n_low * K);
    end

    X_hf = zeros(n_low, K);
    for i = 1:n_low
        sub_idx = (i * K):-1:((i - 1) * K + 1);
        X_hf(i, :) = x_hf(sub_idx)';
    end

    if strcmp(type_name, 'unrestricted')
        X_mat = [ones(n_low, 1), X_hf];
        b = X_mat \ y_lf;
        fitted = X_mat * b;
        residuals = y_lf - fitted;
        rss = sum(residuals.^2);
        tss = sum((y_lf - mean(y_lf)).^2);
        R2 = 1.0 - rss / max(tss, 1e-12);

        res.intercept = b(1);
        res.beta = b(2:end);
        res.fitted = fitted;
        res.residuals = residuals;
        res.R2 = R2;

    elseif strcmp(type_name, 'beta')
        % Unconstrained reparameterization for theta1 > 0, theta2 > 0:
        % theta1 = exp(p(1)), theta2 = exp(p(2))
        obj_fn = @(p) beta_loss_unconstrained(p, y_lf, X_hf, K);
        p0 = [log(1.0); log(5.0)];

        % Native fminsearch (no Optimization Toolbox required)
        [p_hat, fval] = fminsearch(obj_fn, p0);

        t1 = exp(p_hat(1));
        t2 = exp(p_hat(2));
        w_hat = get_beta_weights(K, t1, t2);

        z = X_hf * w_hat;
        Z1 = [ones(n_low, 1), z];
        b = Z1 \ y_lf;

        fitted = Z1 * b;
        residuals = y_lf - fitted;
        tss = sum((y_lf - mean(y_lf)).^2);
        R2 = 1.0 - fval / max(tss, 1e-12);

        res.intercept = b(1);
        res.beta = b(2);
        res.theta1 = t1;
        res.theta2 = t2;
        res.weights = w_hat;
        res.fitted = fitted;
        res.residuals = residuals;
        res.R2 = R2;
    else
        error('midas:unknownType', 'Unknown MIDAS type %s.', type_name);
    end
end

function rss = beta_loss_unconstrained(p, y_lf, X_hf, K)
    t1 = exp(p(1));
    t2 = exp(p(2));
    w = get_beta_weights(K, t1, t2);
    z = X_hf * w;
    Z1 = [ones(length(z), 1), z];
    b = Z1 \ y_lf;
    resid = y_lf - Z1 * b;
    rss = sum(resid.^2);
end

function w = get_beta_weights(K, t1, t2)
    grid = ((1:K) - 0.5) / K;
    w = (grid.^(t1 - 1)) .* ((1 - grid).^(t2 - 1));
    w_sum = sum(w);
    if w_sum <= 0
        w = ones(K, 1) / K;
    else
        w = (w / w_sum)';
    end
end
