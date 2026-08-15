function res = egm(a_grid, z_grid, P, beta, r_rate, gamma_risk, max_iter, tol)
% EGM Endogenous Grid Method (Carroll 2006) for ultra-fast $O(N_a N_z)$ VFI.
%
%   RES = puremacro.vfi.egm(A_GRID, Z_GRID, P, BETA, R_RATE, GAMMA_RISK, MAX_ITER, TOL)
%
%   Solves: u'(c_t) = beta * (1 + r) * E_t [ u'(c_{t+1}) ]
%   Without grid search over a_prime!

    if nargin < 5 || isempty(r_rate)
        r_rate = 0.04;
    end
    if nargin < 6 || isempty(gamma_risk)
        gamma_risk = 2.0;
    end
    if nargin < 7 || isempty(max_iter)
        max_iter = 1000;
    end
    if nargin < 8 || isempty(tol)
        tol = 1e-7;
    end

    a_grid = a_grid(:);
    z_grid = z_grid(:);
    Na = length(a_grid);
    Nz = length(z_grid);

    R_fac = 1.0 + r_rate;

    % Initial policy guess: c(a, z) = r * a + z
    c_policy = zeros(Na, Nz);
    for k = 1:Nz
        c_policy(:, k) = max(r_rate * a_grid + z_grid(k), 1e-4);
    end

    t_start = tic;
    iter = 0;
    dist = 1.0;

    while dist > tol && iter < max_iter
        iter = iter + 1;
        c_old = c_policy;

        % Marginal utility u'(c) = c^(-gamma)
        mu_next = c_old .^ (-gamma_risk);

        % Expected marginal utility E_t [ u'(c_{t+1}) ] = mu_next * P'
        emu = mu_next * P';

        % Endogenous consumption choice c_endog = (beta * R * emu)^(-1 / gamma)
        c_endog = (beta * R_fac * emu) .^ (-1.0 / gamma_risk);

        % Endogenous asset grid: a_endog = (c_endog + a_grid' - z_grid') / R
        % Re-interpolate onto exogenous a_grid
        c_policy_new = zeros(Na, Nz);
        for k = 1:Nz
            a_endog = (c_endog(:, k) + a_grid - z_grid(k)) / R_fac;

            % Monotone interpolation on exogenous a_grid
            for i = 1:Na
                a_target = a_grid(i);
                if a_target <= a_endog(1)
                    % Borrowing constraint binds
                    c_policy_new(i, k) = R_fac * a_target + z_grid(k) - a_grid(1);
                else
                    c_policy_new(i, k) = interp1(a_endog, c_endog(:, k), a_target, 'linear', 'extrap');
                end
            end
        end

        c_policy_new = max(c_policy_new, 1e-6);
        dist = max(abs(c_policy_new(:) - c_old(:)));
        c_policy = c_policy_new;
    end

    elapsed = toc(t_start);

    % Implied asset policy a'(a, z) = R * a + z - c(a, z)
    policy_a = zeros(Na, Nz);
    for k = 1:Nz
        policy_a(:, k) = R_fac * a_grid + z_grid(k) - c_policy(:, k);
        policy_a(:, k) = max(policy_a(:, k), a_grid(1));
    end

    res.c_policy = c_policy;
    res.policy_a = policy_a;
    res.n_iter = iter;
    res.elapsed = elapsed;
end
