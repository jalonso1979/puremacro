function res = portfolio_housing(varargin)
% PORTFOLIO_HOUSING Multi-Asset Portfolio Choice with Bonds, Housing & Adjustment Costs.
%
%   RES = puremacro.vfi.portfolio_housing('beta', 0.95, 'gamma', 2.0, 'Nb', 20, 'Nh', 10)

    p_parser = inputParser;
    addParameter(p_parser, 'beta', 0.95);
    addParameter(p_parser, 'gamma', 2.0);
    addParameter(p_parser, 'r_b', 0.02);
    addParameter(p_parser, 'r_s_mean', 0.06);
    addParameter(p_parser, 'p_h', 1.5);
    addParameter(p_parser, 'adj_cost', 0.05);
    addParameter(p_parser, 'Nb', 15);
    addParameter(p_parser, 'Nh', 8);
    addParameter(p_parser, 'Ny', 5);
    parse(p_parser, varargin{:});

    beta_d    = p_parser.Results.beta;
    gamma_r   = p_parser.Results.gamma;
    r_b       = p_parser.Results.r_b;
    r_s_mean  = p_parser.Results.r_s_mean;
    p_h       = p_parser.Results.p_h;
    adj_cost  = p_parser.Results.adj_cost;
    Nb        = p_parser.Results.Nb;
    Nh        = p_parser.Results.Nh;
    Ny        = p_parser.Results.Ny;

    b_grid = linspace(0.1, 15.0, Nb)';
    h_grid = linspace(0.1, 8.0, Nh)';

    [log_y_grid, P_y] = puremacro.vfi.tauchen(Ny, 0.85, 0.15);
    y_grid = exp(log_y_grid);

    t_start = tic;

    V = zeros(Nb, Nh, Ny);
    policy_b = zeros(Nb, Nh, Ny);
    policy_h = zeros(Nb, Nh, Ny);

    max_iter = 300;
    tol = 1e-6;

    for iter = 1:max_iter
        V_old = V;

        % Expected continuation value EV = reshape(V_2d * P_y', [Nb, Nh, Ny])
        V_2d = reshape(V_old, Nb * Nh, Ny);
        EV_2d = V_2d * P_y';
        EV = reshape(EV_2d, Nb, Nh, Ny);

        for k = 1:Ny
            y_val = y_grid(k);
            for ih = 1:Nh
                h_val = h_grid(ih);
                for ib = 1:Nb
                    b_val = b_grid(ib);

                    u_best = -1e10;
                    best_b = b_grid(1);
                    best_h = h_grid(1);

                    for ih_p = 1:Nh
                        hp_val = h_grid(ih_p);
                        cost_h = p_h * (hp_val - h_val) + adj_cost * p_h * abs(hp_val - h_val);

                        for ib_p = 1:Nb
                            bp_val = b_grid(ib_p);
                            c_val = (1 + r_b) * b_val + (1 + r_s_mean) * y_val - bp_val - cost_h;

                            if c_val > 1e-4
                                u_val = (((c_val^0.7) * (hp_val^0.3))^(1 - gamma_r)) / (1 - gamma_r) + beta_d * EV(ib_p, ih_p, k);
                                if u_val > u_best
                                    u_best = u_val;
                                    best_b = bp_val;
                                    best_h = hp_val;
                                end
                            end
                        end
                    end

                    V(ib, ih, k) = u_best;
                    policy_b(ib, ih, k) = best_b;
                    policy_h(ib, ih, k) = best_h;
                end
            end
        end

        dist = max(abs(V(:) - V_old(:)));
        if dist < tol
            break;
        end
    end

    elapsed = toc(t_start);

    res.V = V;
    res.policy_b = policy_b;
    res.policy_h = policy_h;
    res.b_grid = b_grid;
    res.h_grid = h_grid;
    res.y_grid = y_grid;
    res.n_iter = iter;
    res.elapsed = elapsed;
end
