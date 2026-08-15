function test_new_functions()
% TEST_NEW_FUNCTIONS Verification suite for newly added advanced MATLAB functions.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR NEW ADVANCED MATLAB ECONOMETRIC FUNCTIONS\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % 1. Proxy-IV SVAR (Stock-Watson / Mertens-Ravn)
    try
        fprintf('[1/5] Proxy-IV External Instrument SVAR... ');
        rng(11);
        T_p = 200;
        Y_p = randn(T_p, 2);
        res_var = puremacro.var.estimate(Y_p, 2);

        % Instrument correlated with first residual
        z_inst = res_var.resid(:, 1) + 0.5 * randn(size(res_var.resid, 1), 1);
        [B0_p, F_stat] = puremacro.var.proxy(res_var.Sigma, res_var.resid, z_inst, 1);
        assert(size(B0_p, 1) == 2);
        assert(F_stat > 10.0);

        fprintf('PASSED (First-stage F = %.1f)\n', F_stat);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 2. FEVD Max-Share SVAR (Barsky-Sims news shock)
    try
        fprintf('[2/5] FEVD Max-Share SVAR (Barsky-Sims news shock)... ');
        rng(22);
        Y_m = randn(150, 3);
        res_var = puremacro.var.estimate(Y_m, 2);

        [B0_ms, fev_share] = puremacro.var.maxshare(res_var.A_list, res_var.Sigma, 1, 20);
        assert(size(B0_ms, 1) == 3);
        assert(fev_share > 0 && fev_share <= 1.0);

        fprintf('PASSED (FEV share at H=20 is %.1f%%)\n', fev_share * 100);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 3. Conjugate Minnesota Bayesian VAR (BVAR)
    try
        fprintf('[3/5] Analytical Conjugate Minnesota BVAR... ');
        rng(33);
        Y_bv = randn(100, 3);
        res_bv = puremacro.var.bvar(Y_bv, 2, 0.2, 1.0);
        assert(res_bv.p == 2);
        assert(size(res_bv.Sigma, 1) == 3);
        assert(size(res_bv.B_post, 1) == 7);

        fprintf('PASSED\n');
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 4. Callaway & Sant'Anna Staggered DiD
    % Syntax: callaway_santanna(y, group_g, period_t, unit_id, x_controls)
    try
        fprintf('[4/5] Callaway & Sant''Anna (2021) Staggered DiD ATT(g,t)... ');
        rng(44);
        N_units = 30;
        T_periods = 5;

        g_assign = [inf * ones(10, 1); 3 * ones(10, 1); 4 * ones(10, 1)];

        y_vec = []; g_vec = []; t_vec = []; u_vec = [];
        for u = 1:N_units
            g_val = g_assign(u);
            for t = 1:T_periods
                y_val = 2.0 + (t > g_val) * 1.5 + randn() * 0.2;
                y_vec(end+1, 1) = y_val;
                g_vec(end+1, 1) = g_val;
                t_vec(end+1, 1) = t;
                u_vec(end+1, 1) = u;
            end
        end

        cs_res = puremacro.did.callaway_santanna(y_vec, g_vec, t_vec, u_vec);
        assert(size(cs_res.att_gt, 1) == 2);
        assert(~isnan(cs_res.overall_att));

        fprintf('PASSED (Overall ATT = %.2f)\n', cs_res.overall_att);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 5. Baruník & Křehlík Frequency-Domain Connectedness
    try
        fprintf('[5/5] Baruník & Křehlík (2018) Frequency-Domain Connectedness... ');
        rng(55);
        ret = randn(120, 3);
        bk_res = puremacro.connectedness.barunik_krehlik(ret, 2, 50, [pi/4, pi]);
        assert(bk_res.total_spillover >= 0 && bk_res.total_spillover <= 100);
        assert(size(bk_res.band_matrix, 1) == 3);

        fprintf('PASSED (High-freq spillover = %.1f%%)\n', bk_res.total_spillover);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   NEW FUNCTIONS TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
