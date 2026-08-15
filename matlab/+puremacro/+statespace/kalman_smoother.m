function res = kalman_smoother(y, T_mat, Z_mat, Q_mat, H_mat, R_mat, a0, P0)
% KALMAN_SMOOTHER Rauch-Tung-Striebel (RTS) Kalman smoother.
%
%   RES = puremacro.statespace.kalman_smoother(Y, T_MAT, Z_MAT, Q_MAT, H_MAT, R_MAT, A0, P0)

    m = size(T_mat, 1);
    if nargin < 6 || isempty(R_mat)
        R_mat = eye(m);
    end
    if nargin < 7 || isempty(a0)
        a0 = zeros(m, 1);
    end
    if nargin < 8 || isempty(P0)
        P0 = 1e6 * eye(m);
    end

    filt = puremacro.statespace.kalman_filter(y, T_mat, Z_mat, Q_mat, H_mat, R_mat, a0, P0);

    a_pred = filt.a_pred;
    P_pred = filt.P_pred;
    a_filt = filt.a_filt;
    P_filt = filt.P_filt;

    [T_obs, ~] = size(a_filt);

    a_smooth = zeros(T_obs, m);
    P_smooth = zeros(m, m, T_obs);

    a_smooth(end, :) = a_filt(end, :);
    P_smooth(:, :, end) = P_filt(:, :, end);

    for t = (T_obs - 1):-1:1
        P_pred_next = P_pred(:, :, t + 1);
        J = P_filt(:, :, t) * T_mat' * pinv(P_pred_next);

        a_smooth(t, :) = (a_filt(t, :)' + J * (a_smooth(t + 1, :)' - a_pred(t + 1, :)'))';
        P_smooth(:, :, t) = P_filt(:, :, t) + J * (P_smooth(:, :, t + 1) - P_pred_next) * J';
    end

    res = filt;
    res.a_smooth = a_smooth;
    res.P_smooth = P_smooth;
end
