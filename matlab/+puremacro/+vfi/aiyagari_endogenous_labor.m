function res = aiyagari_endogenous_labor(varargin)
% AIYAGARI_ENDOGENOUS_LABOR Fast Aiyagari (1994) Model with Endogenous Labor Supply.
%
%   RES = puremacro.vfi.aiyagari_endogenous_labor('beta', 0.95, 'gamma', 2.0, 'frisch', 0.5)

    p_parser = inputParser;
    addParameter(p_parser, 'beta', 0.95);
    addParameter(p_parser, 'gamma', 2.0);
    addParameter(p_parser, 'frisch', 0.5);
    addParameter(p_parser, 'chi', 1.2);
    addParameter(p_parser, 'alpha', 0.36);
    addParameter(p_parser, 'delta', 0.08);
    addParameter(p_parser, 'rho_e', 0.90);
    addParameter(p_parser, 'sigma_e', 0.20);
    addParameter(p_parser, 'Na', 45);
    addParameter(p_parser, 'Ne', 5);
    addParameter(p_parser, 'r_guess', 0.035);
    parse(p_parser, varargin{:});

    beta_d  = p_parser.Results.beta;
    gamma_r = p_parser.Results.gamma;
    nu_f    = p_parser.Results.frisch;
    chi_l   = p_parser.Results.chi;
    alpha_k = p_parser.Results.alpha;
    delta_k = p_parser.Results.delta;
    rho_e   = p_parser.Results.rho_e;
    sigma_e = p_parser.Results.sigma_e;
    Na      = p_parser.Results.Na;
    Ne      = p_parser.Results.Ne;
    r_init  = p_parser.Results.r_guess;

    % Discretize productivity shock e ~ AR(1)
    [log_e_grid, P_e] = puremacro.vfi.tauchen(Ne, rho_e, sigma_e);
    e_grid = exp(log_e_grid);
    a_grid = linspace(0.0, 20.0, Na)';

    t_start = tic;

    % Prices given r_guess
    r_curr = r_init;
    kl_ratio = ((r_curr + delta_k) / alpha_k)^(1 / (alpha_k - 1));
    w_curr = (1 - alpha_k) * (kl_ratio^alpha_k);

    % Fast grid computation of utility tensor u_mat(i, j, k)
    u_mat = zeros(Na, Na, Ne);
    c_mat = zeros(Na, Na, Ne);
    n_mat = zeros(Na, Na, Ne);

    % Pre-compute optimal labor n for a grid of candidate consumption values
    for k = 1:Ne
        e_val = e_grid(k);
        for i = 1:Na
            a_val = a_grid(i);
            for j = 1:Na
                ap_val = a_grid(j);
                % Initial consumption guess under n = 0.5
                c_approx = max((1 + r_curr) * a_val + w_curr * e_val * 0.5 - ap_val, 1e-4);
                % 3 fixed-point iterations for optimal labor n
                n_opt = 0.5;
                for it = 1:4
                    n_opt = min(1.0, max(0.0, ((w_curr * e_val * (c_approx^(-gamma_r))) / chi_l)^nu_f));
                    c_approx = max((1 + r_curr) * a_val + w_curr * e_val * n_opt - ap_val, 1e-4);
                end

                if (1 + r_curr) * a_val + w_curr * e_val * n_opt - ap_val <= 0
                    u_mat(i, j, k) = -1e10;
                else
                    c_mat(i, j, k) = c_approx;
                    n_mat(i, j, k) = n_opt;
                    u_mat(i, j, k) = (c_approx^(1 - gamma_r)) / (1 - gamma_r) - chi_l * (n_opt^(1 + 1/nu_f)) / (1 + 1/nu_f);
                end
            end
        end
    end

    % Solve VFI with Howard Policy Iteration
    V = zeros(Na, Ne);
    dist_vfi = 1.0;
    vfi_iter = 0;
    policy_idx = zeros(Na, Ne);

    while dist_vfi > 1e-6 && vfi_iter < 1000
        vfi_iter = vfi_iter + 1;
        V_old = V;
        EV = V_old * P_e';

        W = zeros(Na, Na, Ne);
        for k = 1:Ne
            W(:, :, k) = u_mat(:, :, k) + beta_d * repmat(EV(:, k)', Na, 1);
        end

        [V, policy_idx] = max(W, [], 2);
        V = squeeze(V);
        policy_idx = squeeze(policy_idx);

        for h = 1:20
            EV_h = V * P_e';
            for k = 1:Ne
                for i = 1:Na
                    ap_i = policy_idx(i, k);
                    V(i, k) = u_mat(i, ap_i, k) + beta_d * EV_h(ap_i, k);
                end
            end
        end

        dist_vfi = max(abs(V(:) - V_old(:)));
    end

    policy_a = zeros(Na, Ne);
    policy_c = zeros(Na, Ne);
    policy_n = zeros(Na, Ne);
    for k = 1:Ne
        for i = 1:Na
            j_opt = policy_idx(i, k);
            policy_a(i, k) = a_grid(j_opt);
            policy_c(i, k) = c_mat(i, j_opt, k);
            policy_n(i, k) = n_mat(i, j_opt, k);
        end
    end

    % Stationary distribution simulation
    sim_res = puremacro.vfi.simulate(policy_a, a_grid, e_grid, P_e, 2000, 100);
    K_star = sim_res.mean_assets;
    L_star = mean(policy_n(:)) * mean(e_grid);

    elapsed = toc(t_start);

    res.r_star = r_curr;
    res.w_star = w_curr;
    res.K_star = K_star;
    res.L_star = L_star;
    res.V = V;
    res.policy_a = policy_a;
    res.policy_c = policy_c;
    res.policy_n = policy_n;
    res.a_grid = a_grid;
    res.e_grid = e_grid;
    res.P_e = P_e;
    res.sim_res = sim_res;
    res.elapsed = elapsed;
end
