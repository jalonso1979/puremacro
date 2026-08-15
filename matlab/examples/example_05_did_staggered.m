% EXAMPLE_05_DID_STAGGERED Callaway & Sant'Anna (2021) Staggered DiD
%
% Demonstrates:
%   1. Setting up panel cohort data for staggered policy treatment
%   2. Estimating group-time ATT(g, t) treatment effects
%   3. Computing overall post-treatment average ATT

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 5: Callaway & Sant''Anna (2021) Staggered Difference-in-Differences\n');
fprintf('=================================================================\n\n');

rng(777);
N_units = 40;
T_periods = 6;

% Cohort assignment: 15 never-treated (inf), 15 treated at t=3, 10 treated at t=4
group_assignment = [inf * ones(15, 1); 3 * ones(15, 1); 4 * ones(10, 1)];

y_vec = []; g_vec = []; t_vec = []; u_vec = [];

for u = 1:N_units
    g_val = group_assignment(u);
    for t = 1:T_periods
        % True treatment effect = +2.5 for post-treatment periods
        effect = (t >= g_val) * 2.5;
        y_val = 10.0 + 0.5 * t + effect + randn() * 0.4;

        y_vec(end+1, 1) = y_val;
        g_vec(end+1, 1) = g_val;
        t_vec(end+1, 1) = t;
        u_vec(end+1, 1) = u;
    end
end

% Estimate Callaway & Sant'Anna ATT(g, t)
res_cs = puremacro.did.callaway_santanna(y_vec, g_vec, t_vec, u_vec);

fprintf('Estimated Group-Time ATT(g, t) Matrix:\n');
disp(res_cs.att_gt);

fprintf('Overall Post-Treatment Average Treatment Effect (ATT): %.3f (True = +2.500)\n', res_cs.overall_att);
fprintf('=================================================================\n\n');
