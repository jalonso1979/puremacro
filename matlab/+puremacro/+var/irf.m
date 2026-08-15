function irfs = irf(A_list, B0, horizon)
% IRF Compute orthogonalized impulse response functions for VAR(p).
%
%   IRFS = puremacro.var.irf(A_LIST, B0, HORIZON)
%
%   Inputs:
%       A_list  - Cell array of (N x N) VAR lag matrices {A1, ..., Ap}
%       B0      - (N x N) structural impact matrix
%       horizon - Forecast horizon H (non-negative integer)
%
%   Outputs:
%       irfs    - (H+1 x N x N) array where irfs(h+1, i, j) is response of variable i
%                 at horizon h to structural shock j.

    p = length(A_list);
    n = size(B0, 1);

    Phi = cell(1, horizon + 1);
    Phi{1} = eye(n);

    for h = 1:horizon
        Ph = zeros(n, n);
        for j = 1:min(h, p)
            Ph = Ph + Phi{h - j + 1} * A_list{j};
        end
        Phi{h + 1} = Ph;
    end

    irfs = zeros(horizon + 1, n, n);
    for h = 0:horizon
        irfs(h + 1, :, :) = Phi{h + 1} * B0;
    end
end
