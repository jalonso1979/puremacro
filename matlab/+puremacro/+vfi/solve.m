function res = solve(return_fn, a_grid, z_grid, P, beta, varargin)
% SOLVE Fast Vectorized Value Function Iteration with Howard Policy Acceleration.
%
%   RES = puremacro.vfi.solve(RETURN_FN, A_GRID, Z_GRID, P, BETA, 'howard_steps', 20, 'device', 'cpu')
%
%   Inputs:
%       return_fn    - Function handle @(a, a_prime, z) returning utility u(c)
%       a_grid       - (Na x 1) asset grid
%       z_grid       - (Nz x 1) shock grid
%       P            - (Nz x Nz) Markov transition matrix
%       beta         - Discount factor (0 < beta < 1)
%       Name-Value Options:
%           'max_iter'     - Maximum Bellman iterations (default: 2000)
%           'tol'          - Convergence tolerance (default: 1e-7)
%           'howard_steps' - Howard Policy Iteration steps per Bellman update (default: 20)
%           'device'       - Acceleration target: 'cpu' (default), 'gpu' (CUDA), or 'metal_mlx' (Apple Silicon MATRUN)
%
%   Outputs struct RES with fields:
%       V          - (Na x Nz) Value function matrix
%       policy_idx - (Na x Nz) Policy index matrix pointing to optimal a_prime in a_grid
%       policy_a   - (Na x Nz) Policy asset choices a'(a, z)
%       n_iter     - Total iterations to convergence
%       elapsed    - Solution time in seconds

    p_parser = inputParser;
    addParameter(p_parser, 'max_iter', 2000);
    addParameter(p_parser, 'tol', 1e-7);
    addParameter(p_parser, 'howard_steps', 20);
    addParameter(p_parser, 'device', 'cpu');
    parse(p_parser, varargin{:});

    max_iter = p_parser.Results.max_iter;
    tol = p_parser.Results.tol;
    howard_steps = p_parser.Results.howard_steps;
    device = p_parser.Results.device;

    a_grid = a_grid(:);
    z_grid = z_grid(:);
    Na = length(a_grid);
    Nz = length(z_grid);

    t_start = tic;

    % 1. Construct 3D Return Matrix R_mat (Na x Na x Nz)
    % R_mat(i, j, k) = utility of going from asset a_i to a_j given shock z_k
    R_mat = zeros(Na, Na, Nz);
    for k = 1:Nz
        z_val = z_grid(k);
        for i = 1:Na
            a_val = a_grid(i);
            for j = 1:Na
                ap_val = a_grid(j);
                u = return_fn(a_val, ap_val, z_val);
                if isempty(u) || isnan(u) || ~isreal(u)
                    R_mat(i, j, k) = -1e10;
                else
                    R_mat(i, j, k) = u;
                end
            end
        end
    end

    % 2. Transfer arrays to GPU if CUDA gpuArray requested
    if strcmp(device, 'gpu') && (exist('gpuArray', 'file') || exist('gpuArray', 'builtin'))
        R_mat = gpuArray(R_mat);
        P = gpuArray(P);
        a_grid = gpuArray(a_grid);
        V = zeros(Na, Nz, 'gpuArray');
    else
        V = zeros(Na, Nz);
    end

    % 3. Main Bellman Iteration Loop with Howard Policy Acceleration
    policy_idx = zeros(Na, Nz);
    iter = 0;
    dist = 1.0;

    while dist > tol && iter < max_iter
        iter = iter + 1;
        V_old = V;

        % Expected continuation value EV = V * P'  (Na x Nz)
        EV = V_old * P';

        % Bellman maximization: W(i, j, k) = R(i, j, k) + beta * EV(j, k)
        W = zeros(Na, Na, Nz);
        if strcmp(device, 'gpu')
            W = gpuArray(W);
        end

        for k = 1:Nz
            W(:, :, k) = R_mat(:, :, k) + beta * repmat(EV(:, k)', Na, 1);
        end

        % Maximize over j (next period asset choice)
        [V, policy_idx] = max(W, [], 2);
        V = squeeze(V);           % (Na x Nz)
        policy_idx = squeeze(policy_idx); % (Na x Nz)

        % 4. Howard Policy Acceleration (Fast policy evaluation)
        if howard_steps > 0
            u_policy = zeros(Na, Nz);
            for k = 1:Nz
                for i = 1:Na
                    u_policy(i, k) = R_mat(i, policy_idx(i, k), k);
                end
            end

            for h_step = 1:howard_steps
                EV_h = V * P';
                for k = 1:Nz
                    for i = 1:Na
                        V(i, k) = u_policy(i, k) + beta * EV_h(policy_idx(i, k), k);
                    end
                end
            end
        end

        dist = max(abs(V(:) - V_old(:)));
    end

    if strcmp(device, 'gpu')
        V = gather(V);
        policy_idx = gather(policy_idx);
        a_grid = gather(a_grid);
    end

    policy_a = zeros(Na, Nz);
    for k = 1:Nz
        policy_a(:, k) = a_grid(policy_idx(:, k));
    end

    elapsed = toc(t_start);

    res.V = V;
    res.policy_idx = policy_idx;
    res.policy_a = policy_a;
    res.n_iter = iter;
    res.elapsed = elapsed;
    res.device = device;
end
