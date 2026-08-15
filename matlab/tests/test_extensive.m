function test_extensive()
% TEST_EXTENSIVE Comprehensive, multi-scenario test suite for puremacro MATLAB toolbox.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   EXTENSIVE VERIFICATION TEST SUITE FOR puremacro MATLAB TOOLBOX\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % -------------------------------------------------------------
    % Test 1: High-Dimensional VAR(4) with N=5 variables & Cholesky
    % -------------------------------------------------------------
    try
        fprintf('[1/9] High-dimensional 5-variable VAR(4) estimation & IRF/FEVD/HD... ');
        rng(12345);
        T_large = 400;
        N_var = 5;
        Y_hd = cumsum(randn(T_large, N_var) * 0.05);

        res_hd = puremacro.var.estimate(Y_hd, 4);
        assert(res_hd.p == 4);
        assert(size(res_hd.A_list, 2) == 4);
        assert(size(res_hd.Sigma, 1) == 5);
        assert(size(res_hd.resid, 1) == (T_large - 4));

        B0_hd = puremacro.var.cholesky(res_hd.Sigma);
        irf_hd = puremacro.var.irf(res_hd.A_list, B0_hd, 20);
        assert(size(irf_hd, 1) == 21);
        assert(size(irf_hd, 2) == 5 && size(irf_hd, 3) == 5);

        fevd_hd = puremacro.var.fevd(res_hd.A_list, B0_hd, 20);
        for i = 1:5
            assert(abs(sum(fevd_hd(10, i, :)) - 1.0) < 1e-6);
        end

        hd_res = puremacro.var.hd(res_hd.A_list, B0_hd, res_hd.resid);
        assert(size(hd_res.shocks, 1) == (T_large - 4));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 2: Rubio-Ramirez et al. Sign-Restricted SVAR
    % -------------------------------------------------------------
    try
        fprintf('[2/9] Rubio-Ramirez et al. Sign-restricted SVAR sampler... ');
        rng(99);
        Y_sign = randn(200, 2);
        res_var = puremacro.var.estimate(Y_sign, 2);

        restr.h0 = [1; 1];
        sign_res = puremacro.var.sign_restrictions(res_var.A_list, res_var.Sigma, 10, restr, 1000, 1);
        assert(sign_res.n_accepted > 0);
        assert(size(sign_res.irf_median, 1) == 11);

        fprintf('PASSED (%d accepted draws)\n', sign_res.n_accepted);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 3: Blanchard-Quah (1989) Long-Run Restriction SVAR
    % -------------------------------------------------------------
    try
        fprintf('[3/9] Blanchard-Quah (1989) long-run restriction SVAR... ');
        rng(777);
        Y_bq = cumsum(randn(250, 2));
        res_var = puremacro.var.estimate(Y_bq, 2);

        B0_bq = puremacro.var.bq(res_var.A_list, res_var.Sigma, 1);
        assert(size(B0_bq, 1) == 2 && size(B0_bq, 2) == 2);

        A_sum = res_var.A_list{1} + res_var.A_list{2};
        C_lr = inv(eye(2) - A_sum) * B0_bq;
        assert(abs(C_lr(1, 2)) < 1e-8);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 4: Dynamic 3-Equation New Keynesian DSGE (Sims gensys)
    % -------------------------------------------------------------
    try
        fprintf('[4/9] 3-Equation New Keynesian DSGE Gensys solver... ');
        beta = 0.99; kappa = 0.1; phi_pi = 1.5; phi_y = 0.5;
        G0 = [1, 1, 0; 0, beta, 0; -phi_y, -phi_pi, 1];
        G1 = [1, 0, 1; -kappa, 1, 0; 0, 0, 0];
        Psi = [-1, 0; 0, -1; 0, 0];
        Pi  = [1, 1; 0, beta; 0, 0];

        sol_nk = puremacro.dsge.gensys(G0, G1, Psi, Pi);
        assert(sol_nk.eu(1) == 1);
        assert(sol_nk.eu(2) == 1);
        assert(size(sol_nk.G, 1) == 3);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 5: State-Dependent Local Projections (Auerbach & Gorodnichenko)
    % -------------------------------------------------------------
    try
        fprintf('[5/9] Auerbach-Gorodnichenko State-Dependent Local Projections... ');
        rng(123);
        y_sd = cumsum(randn(120, 1));
        x_sd = randn(120, 1);
        z_sd = randn(120, 1); % State variable (e.g. output gap)

        res_sd = puremacro.lp.state_dependent(y_sd, x_sd, z_sd, 0:5, 2, 'logistic', 3.0, 0.0);
        assert(length(res_sd.beta_high) == 6);
        assert(length(res_sd.beta_low) == 6);
        assert(~isnan(res_sd.beta_high(1)));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 6: Kalman Filter & RTS Smoother with Missing Data (NaNs)
    % -------------------------------------------------------------
    try
        fprintf('[6/9] Kalman Filter & RTS Smoother with missing data (NaNs)... ');
        rng(404);
        T_nan = 60;
        y_nan = randn(T_nan, 2);
        y_nan(10:15, 1) = NaN;
        y_nan(40:45, 2) = NaN;
        y_nan(55:60, :) = NaN;

        T_m = [0.9, 0.1; 0.0, 0.8];
        Z_m = eye(2);
        Q_m = 0.5 * eye(2);
        H_m = 0.1 * eye(2);

        sm_nan = puremacro.statespace.kalman_smoother(y_nan, T_m, Z_m, Q_m, H_m);
        assert(size(sm_nan.a_smooth, 1) == T_nan);
        assert(all(~isnan(sm_nan.a_smooth(:))));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 7: Large Panel Driscoll-Kraay Robust SEs (N=50, T=100)
    % -------------------------------------------------------------
    try
        fprintf('[7/9] Panel Driscoll-Kraay robust standard errors (N=50, T=100)... ');
        rng(50100);
        N_units = 50;
        T_periods = 100;
        N_total = N_units * T_periods;

        X_dk = [ones(N_total, 1), randn(N_total, 3)];
        u_dk = randn(N_total, 1);
        time_keys = vec(repmat(1:T_periods, N_units, 1));

        [se_dk, cov_dk] = puremacro.inference.driscoll_kraay(X_dk, u_dk, time_keys, 4);
        assert(length(se_dk) == 4);
        assert(all(se_dk > 0));

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 8: Unrestricted & Beta-Polynomial MIDAS Regressions
    % -------------------------------------------------------------
    try
        fprintf('[8/9] Mixed-data sampling MIDAS (U-MIDAS & Beta-MIDAS)... ');
        rng(314);
        n_low = 50;
        K_sub = 3;
        x_hf = randn(n_low * K_sub, 1);
        y_lf = 1.5 + 0.8 * sum(reshape(x_hf, K_sub, n_low)', 2) + randn(n_low, 1) * 0.5;

        res_u = puremacro.midas.estimate(y_lf, x_hf, K_sub, 'unrestricted');
        assert(res_u.R2 > 0.4);

        res_b = puremacro.midas.estimate(y_lf, x_hf, K_sub, 'beta');
        assert(res_b.R2 > 0.4);
        assert(abs(sum(res_b.weights) - 1.0) < 1e-6);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % -------------------------------------------------------------
    % Test 9: Simultaneous Sup-t Confidence Bands Monte Carlo Coverage
    % -------------------------------------------------------------
    try
        fprintf('[9/9] Montiel Olea & Plagborg-Moller Simultaneous Sup-t Bands... ');
        rng(888);
        H_dim = 15;
        theta_true = sin(linspace(0, pi, H_dim))';
        Sigma_cov = toeplitz(0.8.^(0:(H_dim-1)));

        [lo, hi, c_val, sd_val, ctr] = puremacro.inference.supt_bands(theta_true, Sigma_cov, 'alpha', 0.10, 'n_mc', 5000);
        assert(c_val > 2.0);
        assert(all(hi > lo));

        fprintf('PASSED (Critical value c = %.3f)\n', c_val);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   EXTENSIVE TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
