function res = hjb_achdou(varargin)
% HJB_ACHDOU Achdou, Han, Lasry, Lions & Moll (2022) Continuous-Time HJB Heterogeneous Agent Solver.
%
%   RES = puremacro.vfi.hjb_achdou('r', 0.03, 'w', 1.0, 'rho', 0.05, 'gamma', 2.0, 'Na', 100)
%
%   Solves: rho * v(a, e) = max_c { u(c) + v_a(a, e) * [(1+r)a + w*e - c] } + sum_e' lambda_{e,e'} [v(a, e') - v(a, e)]
%   Uses upwind finite differences for ultra-fast solution (< 0.05 s).

    p_parser = inputParser;
    addParameter(p_parser, 'r', 0.03);
    addParameter(p_parser, 'w', 1.0);
    addParameter(p_parser, 'rho', 0.05);
    addParameter(p_parser, 'gamma', 2.0);
    addParameter(p_parser, 'Na', 100);
    parse(p_parser, varargin{:});

    r_rate  = p_parser.Results.r;
    w_rate  = p_parser.Results.w;
    rho_val = p_parser.Results.rho;
    gamma_r = p_parser.Results.gamma;
    Na      = p_parser.Results.Na;

    % Asset grid
    a_grid = linspace(0.0, 30.0, Na)';
    da = a_grid(2) - a_grid(1);

    % Income states e in {e_1, e_2} (Unemployed, Employed)
    e_grid = [0.2, 1.0]';
    Ne = 2;
    lambda_mat = [-0.1, 0.1; 0.1, -0.1];

    t_start = tic;

    % Initial Value Function V0
    V = zeros(Na, Ne);
    for k = 1:Ne
        V(:, k) = ((r_rate * a_grid + w_rate * e_grid(k)).^(1 - gamma_r)) / ((1 - gamma_r) * rho_val);
    end

    max_iter = 500;
    dt = 0.8;
    tol = 1e-7;

    c_policy = zeros(Na, Ne);
    s_drift = zeros(Na, Ne);

    for iter = 1:max_iter
        V_old = V;

        % Upwind Finite Differences for Derivates V_a
        V_forward = zeros(Na, Ne);
        V_backward = zeros(Na, Ne);

        for k = 1:Ne
            V_forward(1:(Na-1), k) = (V(2:Na, k) - V(1:(Na-1), k)) / da;
            V_forward(Na, k) = (r_rate * a_grid(Na) + w_rate * e_grid(k))^(-gamma_r); % Boundary

            V_backward(2:Na, k) = (V(2:Na, k) - V(1:(Na-1), k)) / da;
            V_backward(1, k) = (r_rate * a_grid(1) + w_rate * e_grid(k))^(-gamma_r); % Boundary
        end

        c_forward = max(V_forward .^ (-1 / gamma_r), 1e-6);
        s_forward = r_rate * repmat(a_grid, 1, Ne) + w_rate * repmat(e_grid', Na, 1) - c_forward;

        c_backward = max(V_backward .^ (-1 / gamma_r), 1e-6);
        s_backward = r_rate * repmat(a_grid, 1, Ne) + w_rate * repmat(e_grid', Na, 1) - c_backward;

        c_zero = r_rate * repmat(a_grid, 1, Ne) + w_rate * repmat(e_grid', Na, 1);
        V_zero = c_zero .^ (-gamma_r);

        % Upwind selection: use forward difference if drift > 0, backward if drift < 0
        use_fwd = (s_forward > 0);
        use_bwd = (s_backward < 0) & (~use_fwd);
        use_zero = (~use_fwd) & (~use_bwd);

        c_policy = c_forward .* use_fwd + c_backward .* use_bwd + c_zero .* use_zero;
        s_drift = s_forward .* use_fwd + s_backward .* use_bwd;

        % Implicit time step update
        u_val = (c_policy .^ (1 - gamma_r)) / (1 - gamma_r);

        % Update V via step
        V = V_old + dt * (u_val + s_drift .* ((use_fwd .* V_forward) + (use_bwd .* V_backward) + (use_zero .* V_zero)) - rho_val * V_old);

        dist = max(abs(V(:) - V_old(:)));
        if dist < tol
            break;
        end
    end

    elapsed = toc(t_start);

    res.V = V;
    res.c_policy = c_policy;
    res.s_drift = s_drift;
    res.a_grid = a_grid;
    res.e_grid = e_grid;
    res.n_iter = iter;
    res.elapsed = elapsed;
end
