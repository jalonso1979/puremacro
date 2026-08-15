function test_dynare_vfi()
% TEST_DYNARE_VFI Verification suite for Dynare-like automated VFI parser.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    fprintf('=================================================================\n');
    fprintf('   VERIFICATION FOR DYNARE-LIKE AUTOMATED VFI MODEL PARSER\n');
    fprintf('=================================================================\n\n');

    n_pass = 0;
    n_fail = 0;

    try
        fprintf('[1/1] Automated Dynare-Like Model Solver... ');
        m = puremacro.vfi.model('Test Model');
        m = m.set_param('beta', 0.95);
        m = m.set_param('gamma', 1.5);
        m = m.add_state('k', 0.2, 10.0, 30);
        m = m.add_shock('z', 0.85, 0.10, 3);
        m = m.set_utility(@(c, p) (c^(1 - p.gamma)) / (1 - p.gamma));
        m = m.set_budget(@(k, kp, z, p) exp(z) * k^0.35 + 0.90 * k - kp);

        res = m.solve('howard_steps', 15);
        assert(size(res.V, 1) == 30);
        assert(size(res.V, 2) == 3);
        assert(res.sim.mean_assets > 0);

        fprintf('PASSED (%d iterations, %.3f s)\n', res.n_iter, res.elapsed);
        n_pass = n_pass + 1;
    catch err
        fprintf('FAILED (%s)\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=================================================================\n');
    fprintf('   AUTOMATED VFI TEST SUITE SUMMARY: %d PASSED, %d FAILED\n', n_pass, n_fail);
    fprintf('=================================================================\n');
end
