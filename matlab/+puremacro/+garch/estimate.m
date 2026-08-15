function res = estimate(returns, mean_type)
% ESTIMATE Fit GARCH(1,1) by Gaussian Maximum Likelihood (unconstrained re-parameterization).
%
%   RES = puremacro.garch.estimate(RETURNS, MEAN_TYPE)

    if nargin < 2 || isempty(mean_type)
        mean_type = 'zero';
    end

    r = returns(:);
    T = length(r);

    if strcmp(mean_type, 'constant')
        eps = r - mean(r);
    else
        eps = r;
    end

    var0 = var(eps);
    if var0 <= 0
        error('garch:zeroVariance', 'Returns have zero variance; cannot fit GARCH(1,1).');
    end

    % Initial guess for theta unconstrained parameters
    % theta1 = log(omega), theta2 = logit(alpha / 0.999), theta3 = logit(beta / (0.999 - alpha))
    omega_init = 0.05 * var0;
    alpha_init = 0.10;
    beta_init = 0.80;

    theta0 = [
        log(omega_init);
        log(alpha_init / (0.999 - alpha_init));
        log(beta_init / (0.999 - alpha_init - beta_init))
    ];

    obj_fn = @(theta) unconstrained_neg_loglik(theta, eps, var0);

    % Native fminsearch (available in standard MATLAB & Octave without extra toolboxes)
    [theta_hat, fval, exitflag] = fminsearch(obj_fn, theta0);

    [omega, alpha, beta] = transform_params(theta_hat);

    sigma2 = filter_sigma2(eps, omega, alpha, beta, var0);
    sigma = sqrt(sigma2);

    res.omega = omega;
    res.alpha = alpha;
    res.beta = beta;
    res.sigma = sigma;
    res.sigma2 = sigma2;
    res.loglik = -fval;
    res.persistence = alpha + beta;
    res.converged = (exitflag > 0);
end

function [omega, alpha, beta] = transform_params(theta)
    omega = exp(theta(1));
    alpha = 0.999 / (1 + exp(-theta(2)));
    rem = 0.999 - alpha;
    beta = rem / (1 + exp(-theta(3)));
end

function nll = unconstrained_neg_loglik(theta, eps, var0)
    [omega, alpha, beta] = transform_params(theta);
    sigma2 = filter_sigma2(eps, omega, alpha, beta, var0);
    nll = 0.5 * sum(log(2 * pi) + log(sigma2) + (eps.^2) ./ sigma2);
end

function sigma2 = filter_sigma2(eps, omega, alpha, beta, var0)
    T = length(eps);
    sigma2 = zeros(T, 1);
    sigma2(1) = var0;
    for t = 2:T
        sigma2(t) = omega + alpha * eps(t - 1)^2 + beta * sigma2(t - 1);
    end
    sigma2 = max(sigma2, 1e-10);
end
