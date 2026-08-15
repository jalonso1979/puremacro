function test_vfi()
% TEST_VFI Verification test suite for puremacro.vfi module.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR puremacro.vfi DYNAMIC PROGRAMMING MODULE\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % 1. Tauchen (1986) Discretization
    try
        fprintf('[1/5] Tauchen (1986) AR(1) Markov Chain Discretization... ');
        [z_g, P_tau] = puremacro.vfi.tauchen(7, 0.90, 0.10, 3.0);
        assert(length(z_g) == 7);
        assert(size(P_tau, 1) == 7 && size(P_tau, 2) == 7);
        assert(all(abs(sum(P_tau, 2) - 1.0) < 1e-6));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 2. Rouwenhorst (1995) Discretization
    try
        fprintf('[2/5] Rouwenhorst (1995) Persistent AR(1) Discretization... ');
        [z_r, P_rou] = puremacro.vfi.rouwenhorst(5, 0.98, 0.05);
        assert(length(z_r) == 5);
        assert(all(abs(sum(P_rou, 2) - 1.0) < 1e-6));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 3. Vectorized VFI with Howard Acceleration
    try
        fprintf('[3/5] Vectorized VFI with Howard Policy Acceleration... ');
        Na = 30; Nz = 5;
        a_grid = linspace(0.1, 10, Na)';
        [z_grid, P] = puremacro.vfi.tauchen(Nz, 0.85, 0.15);

        % Neoclassical Growth utility: u(c) = log(c)
        % Resource constraint: c = exp(z) * k^alpha + (1-delta)*k - k'
        alpha_k = 0.36; delta_k = 0.10; beta_d = 0.95;

        u_fn = @(a, ap, z) (exp(z)*a^alpha_k + (1-delta_k)*a - ap > 0) * ...
                           log(max(exp(z)*a^alpha_k + (1-delta_k)*a - ap, 1e-6)) + ...
                           (exp(z)*a^alpha_k + (1-delta_k)*a - ap <= 0) * (-1e10);

        vfi_res = puremacro.vfi.solve(u_fn, a_grid, z_grid, P, beta_d, 'howard_steps', 20);
        assert(size(vfi_res.V, 1) == Na);
        assert(size(vfi_res.policy_a, 2) == Nz);
        assert(vfi_res.n_iter < 200); % Howard accelerates convergence

        fprintf('PASSED (%d iterations, %.3f s)\n', vfi_res.n_iter, vfi_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 4. Endogenous Grid Method (EGM)
    try
        fprintf('[4/5] Carroll (2006) Endogenous Grid Method (EGM)... ');
        Na = 40; Nz = 5;
        a_grid = linspace(0.0, 15.0, Na)';
        [z_grid, P] = puremacro.vfi.tauchen(Nz, 0.90, 0.10);

        egm_res = puremacro.vfi.egm(a_grid, z_grid, P, 0.96, 0.03, 2.0);
        assert(size(egm_res.c_policy, 1) == Na);
        assert(all(egm_res.c_policy(:) > 0));

        fprintf('PASSED (%d iterations, %.3f s)\n', egm_res.n_iter, egm_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 5. Heterogeneous Agent Panel Simulation
    try
        fprintf('[5/5] Heterogeneous Agent Panel Simulation & Gini Index... ');
        sim_res = puremacro.vfi.simulate(vfi_res.policy_a, a_grid, z_grid, P, 1000, 100);
        assert(sim_res.mean_assets > 0);
        assert(sim_res.gini >= 0 && sim_res.gini <= 1.0);

        fprintf('PASSED (Mean Assets = %.2f, Gini = %.3f)\n', sim_res.mean_assets, sim_res.gini);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   VFI MODULE TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
