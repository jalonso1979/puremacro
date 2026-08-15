function res = sign_restrictions(A_list, Sigma, horizon, restrictions, n_draws, target_shock_idx)
% SIGN_RESTRICTIONS Rubio-Ramírez, Waggoner & Zha (2010) sign-restricted SVAR sampler.
%
%   RES = puremacro.var.sign_restrictions(A_LIST, SIGMA, HORIZON, RESTRICTIONS, N_DRAWS, TARGET_SHOCK_IDX)
%
%   Inputs:
%       A_list           - Cell array of (N x N) VAR lag matrices
%       Sigma            - (N x N) reduced-form residual covariance matrix
%       horizon          - IRF horizon H
%       restrictions     - Struct where field 'h<H>' contains a (N x 1) vector of signs (+1, -1, or 0)
%       n_draws          - (Optional) Number of Haar-uniform rotation draws (default: 2000)
%       target_shock_idx - (Optional) Target shock column index (default: 1)
%
%   Outputs struct RES with fields:
%       accepted_B0  - Cell array of accepted structural impact matrices
%       irf_median   - (H+1 x N x N) median impulse response matrix across accepted draws
%       irf_lower    - (H+1 x N x N) 5th percentile impulse response matrix
%       irf_upper    - (H+1 x N x N) 95th percentile impulse response matrix
%       n_accepted   - Number of accepted rotation draws

    if nargin < 5 || isempty(n_draws)
        n_draws = 2000;
    end
    if nargin < 6 || isempty(target_shock_idx)
        target_shock_idx = 1;
    end

    n = size(Sigma, 1);
    P = chol(Sigma, 'lower');

    accepted_irfs = {};
    accepted_B0 = {};

    for d = 1:n_draws
        % Draw random orthogonal matrix R via QR decomposition
        M = randn(n, n);
        [Q, R_mat] = qr(M);
        D = diag(sign(diag(R_mat)));
        D(D == 0) = 1;
        Q_rot = Q * D;

        B0 = P * Q_rot;
        irfs = puremacro.var.irf(A_list, B0, horizon);

        % Check sign restrictions
        passed = true;
        fields = fieldnames(restrictions);
        for f = 1:length(fields)
            h_str = fields{f}; % e.g., 'h0', 'h1'
            h_val = str2double(h_str(2:end));

            if h_val <= horizon
                sign_vec = restrictions.(h_str);
                for i = 1:n
                    s = sign_vec(i);
                    if s ~= 0
                        resp = irfs(h_val + 1, i, target_shock_idx);
                        if sign(resp) ~= s
                            passed = false;
                            break;
                        end
                    end
                end
            end
            if ~passed
                break;
            end
        end

        if passed
            accepted_irfs{end + 1} = irfs;
            accepted_B0{end + 1} = B0;
        end
    end

    n_acc = length(accepted_irfs);
    if n_acc == 0
        error('sign_restrictions:noAcceptedDraws', 'No draws satisfied the sign restrictions out of %d draws.', n_draws);
    end

    % Convert cell array to (N_acc x H+1 x N x N) tensor
    tensor = zeros(n_acc, horizon + 1, n, n);
    for idx = 1:n_acc
        tensor(idx, :, :, :) = accepted_irfs{idx};
    end

    res.accepted_B0 = accepted_B0;
    res.irf_median = squeeze(median(tensor, 1));
    res.irf_lower = squeeze(quantile(tensor, 0.05, 1));
    res.irf_upper = squeeze(quantile(tensor, 0.95, 1));
    res.n_accepted = n_acc;
    res.n_draws = n_draws;
end
