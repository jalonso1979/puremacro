function res = kalman_filter(y, T_mat, Z_mat, Q_mat, H_mat, R_mat, a0, P0)
% KALMAN_FILTER Linear Gaussian Kalman filter for state-space models.
%
%   State:       alpha_{t+1} = T_mat * alpha_t + R_mat * eta_t,  eta_t ~ N(0, Q)
%   Observation: y_t         = Z_mat * alpha_t + eps_t,          eps_t ~ N(0, H)
%
%   RES = puremacro.statespace.kalman_filter(Y, T_MAT, Z_MAT, Q_MAT, H_MAT, R_MAT, A0, P0)

    [T_obs, n] = size(y);
    m = size(T_mat, 1);

    if nargin < 6 || isempty(R_mat)
        R_mat = eye(m);
    end
    if nargin < 7 || isempty(a0)
        a0 = zeros(m, 1);
    else
        a0 = a0(:);
    end
    if nargin < 8 || isempty(P0)
        P0 = 1e6 * eye(m);
    end

    r_dim = size(Q_mat, 1);
    RQR = R_mat * Q_mat * R_mat';

    a_pred = zeros(T_obs + 1, m);
    P_pred = zeros(m, m, T_obs + 1);
    a_filt = zeros(T_obs, m);
    P_filt = zeros(m, m, T_obs);
    innov = nan(T_obs, n);
    loglik = 0.0;
    log2pi = log(2 * pi);

    a_pred(1, :) = a0';
    P_pred(:, :, 1) = P0;

    for t = 1:T_obs
        a_t = a_pred(t, :)';
        P_t = P_pred(:, :, t);
        y_t = y(t, :)';

        obs = ~isnan(y_t);
        n_obs = sum(obs);

        if n_obs == 0
            a_filt(t, :) = a_t';
            P_filt(:, :, t) = P_t;
            a_pred(t + 1, :) = (T_mat * a_t)';
            P_pred(:, :, t + 1) = T_mat * P_t * T_mat' + RQR;
            continue;
        end

        Z_t = Z_mat(obs, :);
        H_t = H_mat(obs, obs);
        v = y_t(obs) - Z_t * a_t;

        F = Z_t * P_t * Z_t' + H_t;
        F = (F + F') / 2;

        [F_chol, p_err] = chol(F, 'lower');
        if p_err > 0
            F = F + 1e-10 * eye(n_obs);
            F_chol = chol(F, 'lower');
        end

        K = (T_mat * P_t * Z_t') / F;

        a_filt(t, :) = (a_t + P_t * Z_t' * (F \ v))';
        P_filt(:, :, t) = P_t - P_t * Z_t' * (F \ (Z_t * P_t));

        a_pred(t + 1, :) = (T_mat * a_t + K * v)';
        P_pred(:, :, t + 1) = T_mat * P_t * T_mat' + RQR - K * F * K';

        innov(t, obs) = v';
        log_det = 2.0 * sum(log(diag(F_chol)));
        loglik = loglik - 0.5 * (n_obs * log2pi + log_det + v' * (F \ v));
    end

    res.a_pred = a_pred;
    res.P_pred = P_pred;
    res.a_filt = a_filt;
    res.P_filt = P_filt;
    res.innov = innov;
    res.loglik = loglik;
end
