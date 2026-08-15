function test_frontier_suite()
% TEST_FRONTIER_SUITE Verification suite for newly added frontier macro modules.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR FRONTIER ECONOMETRIC & HJB MACRO MODULES\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % 1. Factor-Augmented VAR (FAVAR)
    try
        fprintf('[1/4] Bernanke-Boivin-Eliasz Factor-Augmented VAR (FAVAR)... ');
        rng(11);
        X_p = randn(150, 15); % Large panel of 15 variables
        Y_m = randn(150, 1);  % Policy interest rate

        favar_res = puremacro.var.favar(X_p, Y_m, 3, 2, 12);
        assert(size(favar_res.factors, 2) == 3);
        assert(size(favar_res.irf_panel, 1) == 15);
        assert(size(favar_res.irf_panel, 2) == 13);

        fprintf('PASSED (15 variables, 3 latent factors extracted)\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 2. Quantile Local Projections (QLP)
    try
        fprintf('[2/4] Quantile Local Projections (Growth-at-Risk)... ');
        rng(22);
        y_q = cumsum(randn(160, 1) * 0.05);
        x_q = randn(160, 1);

        qlp_res = puremacro.lp.quantile(y_q, x_q, 0:10, [0.10, 0.50, 0.90], 2);
        assert(size(qlp_res.beta_q, 1) == 11);
        assert(size(qlp_res.beta_q, 2) == 3);

        fprintf('PASSED (10%%, 50%%, 90%% quantiles fit)\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 3. Achdou Continuous-Time HJB Solver
    try
        fprintf('[3/4] Achdou et al. (2022) Continuous-Time HJB Solver... ');
        hjb_res = puremacro.vfi.hjb_achdou('Na', 80, 'r', 0.035, 'w', 1.2);
        assert(size(hjb_res.V, 1) == 80);
        assert(size(hjb_res.c_policy, 2) == 2);
        assert(hjb_res.n_iter < 500);

        fprintf('PASSED (%d iterations, %.3f s)\n', hjb_res.n_iter, hjb_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 4. Synthetic Control Method
    try
        fprintf('[4/4] Abadie et al. Synthetic Control Method... ');
        rng(44);
        Y_panel = 10.0 + randn(50, 10);
        Y_panel(25:50, 1) = Y_panel(25:50, 1) + 3.0; % Policy effect on unit 1 at t=25

        sc_res = puremacro.causal.synthetic_control(Y_panel, 1, 25);
        assert(length(sc_res.weights) == 9);
        assert(sc_res.att_post > 1.5);

        fprintf('PASSED (Post-treatment ATT = %.2f)\n', sc_res.att_post);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   FRONTIER MODULES TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
