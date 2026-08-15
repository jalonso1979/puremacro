function run_all_tests()
% RUN_ALL_TESTS Master test runner for MATLAB puremacro toolbox.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('====================================================\n');
    fprintf('Running puremacro MATLAB Toolbox Verification Tests\n');
    fprintf('====================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % Test 1: Cycles (Hamilton & HP filter)
    try
        fprintf('[1/10] Testing puremacro.cycles (Hamilton & HP filter)... ');
        rng(42);
        y = cumsum(randn(100, 1));
        [c_h, t_h] = puremacro.cycles.hamilton(y, 8, 4);
        assert(length(c_h) == 100);
        assert(isnan(c_h(1)));
        assert(~isnan(c_h(12)));
        assert(abs((c_h(12) + t_h(12)) - y(12)) < 1e-10);

        [c_hp, t_hp] = puremacro.cycles.hpfilter(y, 1600);
        assert(length(c_hp) == 100);
        assert(abs((c_hp(10) + t_hp(10)) - y(10)) < 1e-10);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 2: Inference (HAC, Driscoll-Kraay, Wild Bootstrap, Sup-t)
    try
        fprintf('[2/10] Testing puremacro.inference (HAC, DK, Bootstrap, Sup-t)... ');
        X = [ones(50, 1), randn(50, 2)];
        u = randn(50, 1);
        [se, cov] = puremacro.inference.hac(X, u, 3);
        assert(length(se) == 3);
        assert(all(se > 0));

        [se_dk, cov_dk] = puremacro.inference.driscoll_kraay(X, u, repmat((1:25)', 2, 1), 1);
        assert(length(se_dk) == 3);

        theta = [1; 2; 3];
        Sigma = eye(3);
        [lo, hi, c, scale, center] = puremacro.inference.supt_bands(theta, Sigma, 'alpha', 0.10, 'n_mc', 1000);
        assert(length(lo) == 3);
        assert(c > 1.5);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 3: VAR & SVAR (Estimate, Cholesky, IRF, FEVD, HD)
    try
        fprintf('[3/10] Testing puremacro.var (Estimate, Cholesky, IRF, FEVD, HD)... ');
        rng(101);
        Y = randn(100, 2);
        res_var = puremacro.var.estimate(Y, 2);
        assert(res_var.p == 2);
        assert(size(res_var.Sigma, 1) == 2);

        B0 = puremacro.var.cholesky(res_var.Sigma);
        assert(norm(B0 * B0' - res_var.Sigma) < 1e-10);

        irfs = puremacro.var.irf(res_var.A_list, B0, 10);
        assert(size(irfs, 1) == 11);

        fev = puremacro.var.fevd(res_var.A_list, B0, 10);
        assert(abs(sum(fev(5, 1, :)) - 1.0) < 1e-8);

        hd_res = puremacro.var.hd(res_var.A_list, B0, res_var.resid);
        assert(size(hd_res.shocks, 1) == res_var.T_eff);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 4: Local Projections (LP)
    try
        fprintf('[4/10] Testing puremacro.lp (Jordà Local Projections)... ');
        y_lp = cumsum(randn(100, 1));
        x_lp = randn(100, 1);
        res_lp = puremacro.lp.estimate(y_lp, x_lp, 0:5, 2, [], 0.10);
        assert(length(res_lp.beta) == 6);
        assert(~isnan(res_lp.beta(1)));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 5: DSGE (gensys)
    try
        fprintf('[5/10] Testing puremacro.dsge (gensys Sims QZ solver)... ');
        G0 = [1, 0; 0, 1];
        G1 = [0.8, 0; 0, 0.5];
        Psi = [1; 0.5];
        Pi = zeros(2, 0); % 0 expectation errors when all roots are stable
        sol = puremacro.dsge.gensys(G0, G1, Psi, Pi);
        assert(sol.eu(1) == 1);
        assert(norm(sol.G - G1) < 1e-8);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 6: Connectedness (Diebold-Yilmaz)
    try
        fprintf('[6/10] Testing puremacro.connectedness (Diebold-Yilmaz)... ');
        returns_data = randn(100, 3);
        dy_res = puremacro.connectedness.diebold_yilmaz(returns_data, 2, 8, 'gfevd');
        assert(dy_res.total >= 0 && dy_res.total <= 100);
        assert(length(dy_res.directional_to) == 3);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 7: Factor Models (PCA)
    try
        fprintf('[7/10] Testing puremacro.factor (PCA)... ');
        X_fac = randn(100, 10);
        fac_res = puremacro.factor.pca(X_fac, 2);
        assert(size(fac_res.factors, 2) == 2);
        assert(length(fac_res.variance_share) == 2);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 8: State Space (Kalman Filter & Smoother)
    try
        fprintf('[8/10] Testing puremacro.statespace (Kalman Filter & Smoother)... ');
        y_ss = randn(50, 1);
        T_mat = 0.9;
        Z_mat = 1.0;
        Q_mat = 0.5;
        H_mat = 0.2;
        R_mat = 1.0;
        sm = puremacro.statespace.kalman_smoother(y_ss, T_mat, Z_mat, Q_mat, H_mat, R_mat);
        assert(length(sm.a_smooth) == 50);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 9: GARCH(1,1)
    try
        fprintf('[9/10] Testing puremacro.garch (GARCH(1,1) MLE)... ');
        r_garch = randn(100, 1);
        g_res = puremacro.garch.estimate(r_garch, 'zero');
        assert(g_res.omega > 0);
        assert(length(g_res.sigma) == 100);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % Test 10: MIDAS
    try
        fprintf('[10/10] Testing puremacro.midas (MIDAS regressions)... ');
        y_lf = randn(30, 1);
        x_hf = randn(90, 1); % K = 3
        m_res = puremacro.midas.estimate(y_lf, x_hf, 3, 'unrestricted');
        assert(length(m_res.beta) == 3);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n====================================================\n');
    fprintf('Test Summary: %d Passed, %d Failed\n', n_pass, n_fail);
    fprintf('====================================================\n');
end
