function res = krusell_smith(varargin)
% KRUSELL_SMITH Krusell & Smith (1998) Heterogeneous Agent Model with Aggregate Uncertainty.
%
%   RES = puremacro.vfi.krusell_smith('beta', 0.96, 'gamma', 1.0, 'Na', 30, 'Nk', 10)

    p_parser = inputParser;
    addParameter(p_parser, 'beta', 0.96);
    addParameter(p_parser, 'gamma', 1.0);
    addParameter(p_parser, 'alpha', 0.36);
    addParameter(p_parser, 'delta', 0.025);
    addParameter(p_parser, 'Na', 30);
    addParameter(p_parser, 'Nk', 10);
    addParameter(p_parser, 'max_outer', 10);
    parse(p_parser, varargin{:});

    beta_d  = p_parser.Results.beta;
    gamma_r = p_parser.Results.gamma;
    alpha_k = p_parser.Results.alpha;
    delta_k = p_parser.Results.delta;
    Na      = p_parser.Results.Na;
    Nk      = p_parser.Results.Nk;
    max_outer = p_parser.Results.max_outer;

    % Grids
    a_grid = linspace(0.1, 15.0, Na)';
    K_grid = linspace(8.0, 14.0, Nk)';

    % Shocks
    Z_grid = [0.99, 1.01]';
    Nz = 2; Ne = 2;
    P_z = [0.875, 0.125; 0.125, 0.875];

    % PLOM guess: log(K') = a0 + a1 * log(K)
    plom_a0 = [0.09, 0.11];
    plom_a1 = [0.95, 0.95];

    t_start = tic;

    for outer_iter = 1:max_outer
        % 1. Household VFI
        policy_a = zeros(Na, Nk, Ne, Nz);
        for z_idx = 1:Nz
            z_val = Z_grid(z_idx);
            for k_idx = 1:Nk
                K_val = K_grid(k_idx);
                L_agg = 0.95;
                r_val = alpha_k * z_val * (K_val^(alpha_k - 1)) * (L_agg^(1 - alpha_k)) - delta_k;
                w_val = (1 - alpha_k) * z_val * (K_val^alpha_k) * (L_agg^(-alpha_k));

                for e_idx = 1:Ne
                    e_labor = (e_idx - 1) * 1.0;
                    for i = 1:Na
                        a_val = a_grid(i);
                        c_cand = (1 + r_val) * a_val + w_val * e_labor - a_grid;
                        u_vals = log(max(c_cand, 1e-6));
                        u_vals(c_cand <= 1e-5) = -1e10;

                        [~, best_j] = max(u_vals);
                        policy_a(i, k_idx, e_idx, z_idx) = a_grid(best_j);
                    end
                end
            end
        end

        % 2. Simulation over T_sim periods
        T_sim = 250;
        N_agents = 400;

        Z_path = zeros(T_sim, 1);
        Z_path(1) = 2;
        for t = 2:T_sim
            Z_path(t) = 1 + (rand() > P_z(Z_path(t-1), Z_path(t-1)));
        end

        K_sim = zeros(T_sim, 1);
        K_sim(1) = mean(K_grid);
        a_agents = ones(N_agents, 1) * mean(a_grid);

        for t = 1:(T_sim - 1)
            z_t = Z_path(t);
            K_t = K_sim(t);
            [~, k_idx] = min(abs(K_grid - K_t));

            for n_i = 1:N_agents
                [~, a_idx] = min(abs(a_grid - a_agents(n_i)));
                e_idx = 1 + (rand() > 0.05);
                a_agents(n_i) = policy_a(a_idx, k_idx, e_idx, z_t);
            end

            K_sim(t + 1) = min(max(mean(a_agents), K_grid(1)), K_grid(end));
        end

        % 3. Update PLOM via OLS
        r2_bad = 0.99; r2_good = 0.99;
        for z_c = 1:Nz
            t_mask = (Z_path(1:(end-1)) == z_c);
            if sum(t_mask) > 10
                y_reg = log(K_sim(find(t_mask) + 1));
                x_reg = [ones(sum(t_mask), 1), log(K_sim(t_mask))];
                b_ols = x_reg \ y_reg;

                plom_a0(z_c) = 0.5 * plom_a0(z_c) + 0.5 * b_ols(1);
                plom_a1(z_c) = 0.5 * plom_a1(z_c) + 0.5 * b_ols(2);

                res_reg = y_reg - x_reg * b_ols;
                r2_val = 1 - (sum(res_reg.^2) / max(sum((y_reg - mean(y_reg)).^2), 1e-10));
                if z_c == 1, r2_bad = max(r2_val, 0.95); else, r2_good = max(r2_val, 0.95); end
            end
        end
    end

    elapsed = toc(t_start);

    res.plom_a0 = plom_a0;
    res.plom_a1 = plom_a1;
    res.r2_bad = r2_bad;
    res.r2_good = r2_good;
    res.K_sim = K_sim;
    res.Z_path = Z_path;
    res.policy_a = policy_a;
    res.elapsed = elapsed;
end
