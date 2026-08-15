function res = gensys(Gamma0, Gamma1, Psi, Pi, div)
% GENSYS Sims (2002) QZ solution for linear rational expectations models.
%
%   RES = puremacro.dsge.gensys(GAMMA0, GAMMA1, PSI, PI, DIV)
%
%   Solves linear RE systems of the form:
%       Gamma0 * z_t = Gamma1 * z_{t-1} + Psi * eps_t + Pi * eta_t

    if nargin < 5 || isempty(div)
        div = 1.0 + 1e-8;
    end

    n = size(Gamma0, 1);
    n_eta = size(Pi, 2);
    n_eps = size(Psi, 2);

    % QZ decomposition: Q * Gamma0 * Z = S, Q * Gamma1 * Z = T
    [S, T, Q, Z] = qz(Gamma0, Gamma1);

    alpha = diag(S);
    beta = diag(T);
    eigvals = abs(beta) ./ max(abs(alpha), 1e-12);
    [eigvals_sorted, ~] = sort(eigvals);

    select = (eigvals < div);
    [S, T, Q, Z] = ordqz(S, T, Q, Z, select);

    n_stable = sum(select);
    n_unstable = n - n_stable;

    eu = [0, 0];
    if n_eta ~= n_unstable
        res.G = zeros(n, n);
        res.Impact = zeros(n, n_eps);
        res.eu = eu;
        res.eigenvalues = eigvals_sorted;
        return;
    end
    eu(1) = 1;

    if n_unstable == 0
        eu(2) = 1;
    else
        Z2 = Z(:, (n_stable + 1):end);
        sv_Z2 = svd(Z2);
        if min(sv_Z2) > 1e-10
            eu(2) = 1;
        end
    end

    if eu(2) == 0
        res.G = zeros(n, n);
        res.Impact = zeros(n, n_eps);
        res.eu = eu;
        res.eigenvalues = eigvals_sorted;
        return;
    end

    if n_stable == 0
        G = zeros(n, n);
    else
        Z1 = Z(:, 1:n_stable);
        S11 = S(1:n_stable, 1:n_stable);
        T11 = T(1:n_stable, 1:n_stable);
        % G = Z1 * S11^{-1} * T11 * Z1'
        G_core = Z1 * (S11 \ T11) * Z1';
        G = real(G_core);
    end

    LHS = Gamma0 - G * Gamma1;
    Impact = real(LHS \ Psi);

    res.G = G;
    res.Impact = Impact;
    res.eu = eu;
    res.eigenvalues = eigvals_sorted;
end
