function res = simulate(policy_a, a_grid, z_grid, P, n_agents, T_periods)
% SIMULATE Simulate heterogeneous agent panel & compute stationary asset distribution.
%
%   RES = puremacro.vfi.simulate(POLICY_A, A_GRID, Z_GRID, P, N_AGENTS, T_PERIODS)

    if nargin < 5 || isempty(n_agents)
        n_agents = 2000;
    end
    if nargin < 6 || isempty(T_periods)
        T_periods = 100;
    end

    Na = length(a_grid);
    Nz = length(z_grid);

    % Stationary Markov distribution of shocks
    [V_p, D_p] = eig(P');
    [~, idx] = min(abs(diag(D_p) - 1.0));
    p_stat = real(V_p(:, idx));
    p_stat = max(p_stat / sum(p_stat), 0);

    % Initial agent allocation
    z_states = discrete_draw(p_stat, n_agents);
    a_states = ones(n_agents, 1) * floor(Na / 2);

    asset_panel = zeros(n_agents, T_periods);

    for t = 1:T_periods
        for i = 1:n_agents
            a_idx = a_states(i);
            z_idx = z_states(i);

            asset_panel(i, t) = a_grid(a_idx);
            a_next_val = policy_a(a_idx, z_idx);

            [~, a_next_idx] = min(abs(a_grid - a_next_val));
            a_states(i) = a_next_idx;

            z_states(i) = discrete_draw(P(z_idx, :)', 1);
        end
    end

    final_assets = sort(asset_panel(:, end));
    mean_assets = mean(final_assets);

    % Gini coefficient
    n_a = length(final_assets);
    cum_a = cumsum(final_assets - min(final_assets));
    gini = 1 - 2 * sum(cum_a) / (n_a * max(cum_a(end), 1e-10)) + 1 / n_a;

    res.mean_assets = mean_assets;
    res.gini = gini;
    res.asset_panel = asset_panel;
end

function idx = discrete_draw(probs, count)
    probs = probs(:) / sum(probs);
    cdf = cumsum(probs);
    u = rand(count, 1);
    idx = zeros(count, 1);
    for i = 1:count
        match = find(u(i) <= cdf, 1, 'first');
        if isempty(match)
            idx(i) = length(probs);
        else
            idx(i) = match;
        end
    end
end
