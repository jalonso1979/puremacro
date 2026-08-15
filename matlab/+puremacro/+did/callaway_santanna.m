function res = callaway_santanna(y, group_g, period_t, unit_id, x_controls)
% CALLAWAY_SANTANNA Callaway & Sant'Anna (2021) Staggered DiD ATT(g, t) estimator.
%
%   RES = puremacro.did.callaway_santanna(Y, GROUP_G, PERIOD_T, UNIT_ID, X_CONTROLS)
%
%   Inputs:
%       y          - Outcome variable vector (N_total x 1)
%       group_g    - Vector of treatment cohort periods (inf for never-treated)
%       period_t   - Vector of calendar periods
%       unit_id    - Vector of unit IDs
%       x_controls - (Optional) Pre-treatment control variables (N_total x K)
%
%   Outputs struct RES with fields:
%       att_gt     - Group-time average treatment effects ATT(g, t) table
%       overall_att- Summary overall ATT across post-treatment periods

    y = y(:);
    group_g = group_g(:);
    period_t = period_t(:);
    unit_id = unit_id(:);

    groups = unique(group_g(~isinf(group_g)));
    periods = unique(period_t);

    n_groups = length(groups);
    n_periods = length(periods);

    att_matrix = nan(n_groups, n_periods);
    se_matrix = nan(n_groups, n_periods);

    % Never-treated comparison group mask
    never_treated = isinf(group_g);

    for g_idx = 1:n_groups
        g = groups(g_idx);
        base_period = g - 1;

        if ~ismember(base_period, periods)
            continue;
        end

        % Units in cohort g
        cohort_mask = (group_g == g);

        for t_idx = 1:n_periods
            t = periods(t_idx);
            if t == base_period
                att_matrix(g_idx, t_idx) = 0.0;
                se_matrix(g_idx, t_idx) = 0.0;
                continue;
            end

            % Combine cohort g and never-treated comparison group
            sample_mask = (cohort_mask | never_treated) & (period_t == t | period_t == base_period);

            y_sub = y(sample_mask);
            g_sub = group_g(sample_mask);
            t_sub = period_t(sample_mask);
            u_sub = unit_id(sample_mask);

            units = unique(u_sub);
            n_u = length(units);

            dy = zeros(n_u, 1);
            treat = zeros(n_u, 1);

            for u_i = 1:n_u
                uid = units(u_i);
                m_t = (u_sub == uid & t_sub == t);
                m_b = (u_sub == uid & t_sub == base_period);

                if any(m_t) && any(m_b)
                    dy(u_i) = y_sub(m_t) - y_sub(m_b);
                    treat(u_i) = double(g_sub(find(m_t, 1)) == g);
                end
            end

            % DiD estimation: dy = alpha + att * treat
            n_treat = sum(treat == 1);
            n_ctrl = sum(treat == 0);

            if n_treat > 0 && n_ctrl > 0
                att_val = mean(dy(treat == 1)) - mean(dy(treat == 0));
                se_val = sqrt(var(dy(treat == 1)) / n_treat + var(dy(treat == 0)) / n_ctrl);

                att_matrix(g_idx, t_idx) = att_val;
                se_matrix(g_idx, t_idx) = se_val;
            end
        end
    end

    % Overall post-treatment average ATT
    post_mask = false(size(att_matrix));
    for g_idx = 1:n_groups
        g = groups(g_idx);
        post_mask(g_idx, periods >= g) = true;
    end

    valid_post = post_mask & ~isnan(att_matrix);
    overall_att = mean(att_matrix(valid_post));

    res.groups = groups;
    res.periods = periods;
    res.att_gt = att_matrix;
    res.se_gt = se_matrix;
    res.overall_att = overall_att;
end
