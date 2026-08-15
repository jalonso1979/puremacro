function res = sovereign_default(varargin)
% SOVEREIGN_DEFAULT Arellano (2008) / Eaton-Gersovitz (1981) Sovereign Default Model.
%
%   RES = puremacro.vfi.sovereign_default('beta', 0.95, 'r_star', 0.017, 'Nb', 50, 'Ny', 11)
%
%   Solves:
%       Value of Repayment  V_R(B, y) = max_{B'} { u(y - q(B', y)*B' + B) + beta * E [ V(B', y') ] }
%       Value of Default    V_D(y)    = u(y^{def}) + beta * E [ theta * V(0, y') + (1-theta) * V_D(y') ]
%       Overall Value       V(B, y)   = max(V_R(B, y), V_D(y))
%       Bond Price Schedule q(B', y)   = (1 - E[ default_prob(B', y') ]) / (1 + r*)

    p_parser = inputParser;
    addParameter(p_parser, 'beta', 0.95);      % Discount factor
    addParameter(p_parser, 'gamma', 2.0);     % Risk aversion
    addParameter(p_parser, 'r_star', 0.017);   % Risk-free world interest rate
    addParameter(p_parser, 'theta', 0.28);     % Re-entry probability after default
    addParameter(p_parser, 'pen_def', 0.96);   % Default output penalty factor y_def = pen_def * mean(y)
    addParameter(p_parser, 'Nb', 50);          % Sovereign debt grid points (negative values B <= 0)
    addParameter(p_parser, 'Ny', 11);          % Output endowment shock grid points
    parse(p_parser, varargin{:});

    beta_d  = p_parser.Results.beta;
    gamma_r = p_parser.Results.gamma;
    r_star  = p_parser.Results.r_star;
    theta_r = p_parser.Results.theta;
    pen_def = p_parser.Results.pen_def;
    Nb      = p_parser.Results.Nb;
    Ny      = p_parser.Results.Ny;

    % Sovereign debt grid B <= 0 (borrowing)
    B_grid = linspace(-0.45, 0.0, Nb)';

    % Output endowment shock log(y) ~ AR(1)
    [log_y_grid, P_y] = puremacro.vfi.tauchen(Ny, 0.94, 0.025);
    y_grid = exp(log_y_grid);

    % Initial Value guesses
    V_R = zeros(Nb, Ny);
    V_D = zeros(1, Ny);
    V   = zeros(Nb, Ny);
    q_bond = ones(Nb, Ny) / (1 + r_star);

    t_start = tic;

    max_iter = 400;
    tol = 1e-6;

    default_prob = zeros(Nb, Ny);
    default_choice = zeros(Nb, Ny);
    policy_B = zeros(Nb, Ny);

    for iter = 1:max_iter
        V_old = V;
        q_old = q_bond;

        % 1. Value of Default V_D(y)
        % Penalized output under default
        y_def_grid = min(y_grid, pen_def * mean(y_grid));
        u_def = (y_def_grid .^ (1 - gamma_r)) / (1 - gamma_r);

        EV_reentry = V_old(Nb, :) * P_y'; % Return with 0 debt (B = 0 is index Nb)
        EV_default = V_D * P_y';
        V_D = u_def' + beta_d * (theta_r * EV_reentry + (1 - theta_r) * EV_default);

        % 2. Value of Repayment V_R(B, y)
        EV_cont = V_old * P_y'; % (Nb x Ny)

        for k = 1:Ny
            y_val = y_grid(k);
            for i = 1:Nb
                B_val = B_grid(i);
                % Consumption under candidate B'
                % c = y + B - q(B', y)*B'
                q_col = q_old(:, k); % (Nb x 1)
                c_cand = y_val + B_val - q_col .* B_grid;
                c_cand(c_cand <= 0) = 1e-10;

                u_cand = (c_cand .^ (1 - gamma_r)) / (1 - gamma_r) + beta_d * EV_cont(:, k);
                u_cand(c_cand <= 1e-9) = -1e10;

                [v_max, best_j] = max(u_cand);
                V_R(i, k) = v_max;
                policy_B(i, k) = B_grid(best_j);
            end
        end

        % 3. Sovereign Default Decision d(B, y) = 1 if V_D > V_R
        for k = 1:Ny
            for i = 1:Nb
                if V_D(k) > V_R(i, k)
                    default_choice(i, k) = 1.0; % Default
                    V(i, k) = V_D(k);
                else
                    default_choice(i, k) = 0.0; % Repay
                    V(i, k) = V_R(i, k);
                end
            end
        end

        % 4. Endogenous Bond Price Schedule q(B', y) = (1 - E[ d(B', y') ]) / (1 + r*)
        default_prob = default_choice * P_y'; % (Nb x Ny)
        q_bond = (1.0 - default_prob) / (1 + r_star);

        dist = max(abs(V(:) - V_old(:))) + max(abs(q_bond(:) - q_old(:)));
        if dist < tol
            break;
        end
    end

    elapsed = toc(t_start);

    res.V = V;
    res.V_R = V_R;
    res.V_D = V_D;
    res.q_bond = q_bond;
    res.default_choice = default_choice;
    res.default_prob = default_prob;
    res.policy_B = policy_B;
    res.B_grid = B_grid;
    res.y_grid = y_grid;
    res.n_iter = iter;
    res.elapsed = elapsed;
end
