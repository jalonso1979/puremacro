function fevd_shares = fevd(A_list, B0, horizon)
% FEVD Forecast Error Variance Decomposition for SVAR(p).
%
%   FEVD_SHARES = puremacro.var.fevd(A_LIST, B0, HORIZON)
%
%   Inputs:
%       A_list  - Cell array of (N x N) VAR lag matrices
%       B0      - (N x N) structural impact matrix
%       horizon - Forecast horizon H
%
%   Outputs:
%       fevd_shares - (H+1 x N x N) array where fevd_shares(h+1, i, j) is the
%                     share of Var(y_{i, t+h}) explained by structural shock j.

    irfs = puremacro.var.irf(A_list, B0, horizon); % (H+1 x N x N)
    cum_sq = cumsum(irfs.^2, 1);

    total = sum(cum_sq, 3);
    total(total == 0) = 1.0;

    fevd_shares = cum_sq ./ total;
end
