function test_creative_vfi_suite()
% TEST_CREATIVE_VFI_SUITE Verification suite for creative dynamic programming models.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR CREATIVE DYNAMIC PROGRAMMING MODULES\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    % 1. Life-Cycle OLG Model
    try
        fprintf('[1/3] Finite Horizon Life-Cycle OLG Model... ');
        lc_res = puremacro.vfi.life_cycle_olg('J', 40, 'J_ret', 28, 'Na', 30, 'Nz', 3);
        assert(length(lc_res.mean_assets_age) == 40);
        assert(max(lc_res.mean_assets_age) > lc_res.mean_assets_age(1));

        fprintf('PASSED (%d adult ages solved, %.3f s)\n', lc_res.J, lc_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 2. Small Open Economy Sovereign Default Model (Arellano 2008)
    try
        fprintf('[2/3] Arellano (2008) Sovereign Default Model... ');
        sd_res = puremacro.vfi.sovereign_default('Nb', 25, 'Ny', 7);
        assert(size(sd_res.q_bond, 1) == 25);
        assert(size(sd_res.q_bond, 2) == 7);
        assert(min(sd_res.q_bond(:)) < max(sd_res.q_bond(:)));

        fprintf('PASSED (%d iterations, %.3f s)\n', sd_res.n_iter, sd_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    % 3. Multi-Asset Portfolio & Housing Choice Model
    try
        fprintf('[3/3] Multi-Asset Portfolio & Housing Choice... ');
        ph_res = puremacro.vfi.portfolio_housing('Nb', 12, 'Nh', 6, 'Ny', 3);
        assert(size(ph_res.V, 1) == 12);
        assert(size(ph_res.V, 2) == 6);

        fprintf('PASSED (%d iterations, %.3f s)\n', ph_res.n_iter, ph_res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   CREATIVE VFI SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
