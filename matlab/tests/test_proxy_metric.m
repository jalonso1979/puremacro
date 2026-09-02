function test_proxy_metric()
% TEST_PROXY_METRIC Regression test for the proxy-SVAR impact metric.
%
%   The defect this guards against: `proxy.m` returned
%       b_col1 = (Sigma * Pi) / sqrt(Pi' * Sigma * Pi)
%   which is proportional to Sigma*b_1 rather than to b_1. The correct
%   normalisation is b_1' * inv(Sigma) * b_1 = 1, i.e.
%       b_col1 = Pi / sqrt(Pi' * inv(Sigma) * Pi).
%
%   The error VANISHES when Sigma is proportional to the identity, which is
%   what every pre-existing fixture in this directory happened to satisfy --
%   so the bug was structurally unreachable by the test suite that was
%   supposed to cover it. This fixture therefore uses a B with deliberate
%   off-diagonal structure, and that is the whole point of it.
%
%   The wrong vector also satisfies b' * inv(Sigma) * b = 1 exactly, so no
%   internal-consistency check can detect it: only comparison against a known
%   truth can. See docs/ADVISORY.md.

    rand('seed', 42); randn('seed', 42);

    n = 3; T = 200000;
    b1_true = [1.0; 0.8; -0.5];
    B = [b1_true, [0.3; 1.0; 0.4], [-0.2; 0.5; 1.1]];

    eps = randn(T, n);
    u = eps * B';
    z = eps(:, 1) + 0.5 * randn(T, 1);
    Sigma = cov(u);

    % Guard the guard: if Sigma were proportional to the identity this test
    % would pass on the broken code too, and assert nothing.
    off = Sigma - diag(diag(Sigma));
    assert(max(abs(off(:))) > 0.1, ...
           'fixture is degenerate: Sigma is near-diagonal, so the bug is unreachable');

    [B0, ~] = puremacro.var.proxy(Sigma, u, z, 1);
    b = B0(:, 1);

    err = max(abs(b - b1_true));
    printf('  proxy impact metric: max abs error = %.5f (tol 0.02)\n', err);
    assert(err < 0.02, ...
           sprintf(['proxy-SVAR impact vector is off by %.4f. The broken form ' ...
                    '(Sigma*Pi)/sqrt(Pi''*Sigma*Pi) returns ~0.253 here.'], err));

    % The normalisation the broken form ALSO satisfied -- checked so that a
    % future reader can see it is not what distinguishes right from wrong.
    nrm = b' * (Sigma \ b);
    assert(abs(nrm - 1.0) < 1e-8, 'b'' * inv(Sigma) * b should equal 1');

    printf('  test_proxy_metric: PASSED\n');
end
