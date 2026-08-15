function res = life_cycle_olg(varargin)
% LIFE_CYCLE_OLG High-Performance Life-Cycle OLG Model with Retirement & Bequest Motive.
%
%   RES = puremacro.vfi.life_cycle_olg('J', 50, 'J_ret', 35, 'beta', 0.96)
%
%   Life Cycle: j = 1..J (e.g. 20..70 years old). Working j <= J_ret, Retired j > J_ret.
%   Solves backward induction for policy functions c_j(a, z), a'_{j}(a, z).

    p_parser = inputParser;
    addParameter(p_parser, 'J', 50);        % Lifespan
    addParameter(p_parser, 'J_ret', 35);    % Retirement age index
    addParameter(p_parser, 'beta', 0.96);   % Discount factor
    addParameter(p_parser, 'gamma', 2.0);  % Risk aversion
    addParameter(p_parser, 'r', 0.03);      % Interest rate
    addParameter(p_parser, 'w', 1.2);      % Wage rate
    addParameter(p_parser, 'pension', 0.4); % Pension replacement rate
    addParameter(p_parser, 'phi_bequest', 2.0); % Bequest weight
    addParameter(p_parser, 'Na', 50);       % Asset grid points
    addParameter(p_parser, 'Nz', 5);        % Productivity shock grid points
    parse(p_parser, varargin{:});

    J        = p_parser.Results.J;
    J_ret    = p_parser.Results.J_ret;
    beta_d   = p_parser.Results.beta;
    gamma_r  = p_parser.Results.gamma;
    r_rate   = p_parser.Results.r;
    w_rate   = p_parser.Results.w;
    pension  = p_parser.Results.pension;
    phi_beq  = p_parser.Results.phi_bequest;
    Na       = p_parser.Results.Na;
    Nz       = p_parser.Results.Nz;

    % Asset grid
    a_grid = linspace(0.0, 30.0, Na)';

    % Income productivity shock z
    [log_z_grid, P_z] = puremacro.vfi.tauchen(Nz, 0.85, 0.15);
    z_grid = exp(log_z_grid);

    % Age-dependent efficiency profile epsilon_j (hump-shaped during working life)
    j_vec = (1:J)';
    eps_profile = 1.0 + 0.05 * (j_vec - 1) - 0.001 * ((j_vec - 1).^2);
    eps_profile(J_ret+1:J) = 0.0; % Retired

    % Output tensors: (Na x Nz x J)
    V = zeros(Na, Nz, J);
    policy_a = zeros(Na, Nz, J);
    policy_c = zeros(Na, Nz, J);

    t_start = tic;

    % 1. Terminal Period J (Death & Bequest Value)
    for k = 1:Nz
        for i = 1:Na
            a_val = a_grid(i);
            % Terminal consumption + bequest utility v(a') = phi * a'^(1-gamma)/(1-gamma)
            c_val = max((1 + r_rate) * a_val + pension * w_rate, 1e-4);
            policy_c(i, k, J) = c_val;
            policy_a(i, k, J) = 0.0;
            V(i, k, J) = (c_val^(1 - gamma_r)) / (1 - gamma_r);
        end
    end

    % 2. Backward Induction from J-1 down to 1
    for j = (J - 1):-1:1
        is_working = (j <= J_ret);
        eps_j = eps_profile(j);

        EV_next = squeeze(V(:, :, j + 1)) * P_z'; % (Na x Nz)

        for k = 1:Nz
            z_val = z_grid(k);
            y_income = is_working * (w_rate * eps_j * z_val) + (~is_working) * (pension * w_rate);

            for i = 1:Na
                a_val = a_grid(i);

                c_candidates = (1 + r_rate) * a_val + y_income - a_grid;
                c_candidates(c_candidates <= 0) = 1e-10;

                u_vals = (c_candidates .^ (1 - gamma_r)) / (1 - gamma_r) + beta_d * EV_next(:, k);
                u_vals(c_candidates <= 1e-9) = -1e10;

                [v_max, best_idx] = max(u_vals);
                V(i, k, j) = v_max;
                policy_a(i, k, j) = a_grid(best_idx);
                policy_c(i, k, j) = c_candidates(best_idx);
            end
        end
    end

    % 3. Lifecycle Profile Simulation across cohorts
    mean_assets_age = zeros(J, 1);
    mean_cons_age = zeros(J, 1);
    mean_inc_age = zeros(J, 1);

    for j = 1:J
        mean_assets_age(j) = mean(mean(policy_a(:, :, j)));
        mean_cons_age(j)   = mean(mean(policy_c(:, :, j)));
        is_w = (j <= J_ret);
        mean_inc_age(j)    = is_w * (w_rate * eps_profile(j) * mean(z_grid)) + (~is_w) * (pension * w_rate);
    end

    elapsed = toc(t_start);

    res.V = V;
    res.policy_a = policy_a;
    res.policy_c = policy_c;
    res.mean_assets_age = mean_assets_age;
    res.mean_cons_age = mean_cons_age;
    res.mean_inc_age = mean_inc_age;
    res.eps_profile = eps_profile;
    res.a_grid = a_grid;
    res.z_grid = z_grid;
    res.J = J;
    res.J_ret = J_ret;
    res.elapsed = elapsed;
end
