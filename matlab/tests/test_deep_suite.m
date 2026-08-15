function test_deep_suite()
% TEST_DEEP_SUITE Comprehensive verification suite for deep frontier extensions.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR FRONTIER MACROECONOMETRIC EXTENSIONS\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % 1. Barnichon & Brownlees (2019) Smooth LP
    try
        fprintf('[1/3] Barnichon & Brownlees (2019) P-Spline Smooth LP... ');
        rng(11);
        y_sm = cumsum(randn(180, 1) * 0.05);
        x_sm = randn(180, 1);

        res_sm = puremacro.lp.smooth(y_sm, x_sm, 0:15, 2, 20.0, 2, 0.10);
        assert(length(res_sm.beta_smooth) == 16);
        assert(length(res_sm.se) == 16);

        fprintf('PASSED (Smoothing lambda = %.1f)\n', res_sm.lambda);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 2. Krusell & Smith (1998) Heterogeneous Agent Model
    try
        fprintf('[2/3] Krusell & Smith (1998) Aggregate Uncertainty Model... ');
        ks_res = puremacro.vfi.krusell_smith('Na', 20, 'Nk', 6, 'max_outer', 5);
        assert(size(ks_res.policy_a, 1) == 20);
        assert(ks_res.r2_bad > 0.80 && ks_res.r2_good > 0.80);

        fprintf('PASSED (PLOM R^2 Bad = %.4f, Good = %.4f)\n', ks_res.r2_bad, ks_res.r2_good);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 3. Primiceri (2005) TVP-VAR with Stochastic Volatility
    try
        fprintf('[3/3] Primiceri (2005) TVP-VAR-SV... ');
        rng(33);
        Y_tv = randn(120, 2);
        tv_res = puremacro.var.tvp_sv(Y_tv, 2, 10, 50);
        assert(size(tv_res.B_t, 1) == 118);
        assert(size(tv_res.h_t, 2) == 2);
        assert(size(tv_res.irf_t, 2) == 11);

        fprintf('PASSED (%d time-varying periods fit)\n', tv_res.T_eff);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   DEEP EXTENSIONS TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
