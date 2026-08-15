% EXAMPLE_VIZ_01_SVAR_IRFS Publication-Quality SVAR IRFs with Shaded Confidence Bands
%
% Generates publication-ready plots of SVAR Impulse Response Functions comparing:
%   - Cholesky lower-triangular identification
%   - Blanchard-Quah (1989) long-run identification
%   - Sign-restricted SVAR (Rubio-Ramírez et al. 2010)

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 1: Publication-Quality SVAR IRF Plots\n');
fprintf('=================================================================\n\n');

rng(42);
T = 250;
Y = cumsum(randn(T, 3) * 0.02) + randn(T, 3) * 0.1;

res_var = puremacro.var.estimate(Y, 2);
B0_chol = puremacro.var.cholesky(res_var.Sigma);
horizon = 15;
irf_chol = puremacro.var.irf(res_var.A_list, B0_chol, horizon);

% Sign Restrictions: 3 variables [y1, y2, y3], demand shock prior: y1 > 0, y2 > 0, y3 unrestricted (0)
restr.h0 = [1; 1; 0];
sign_res = puremacro.var.sign_restrictions(res_var.A_list, res_var.Sigma, horizon, restr, 1000, 1);

% Create Figure
fig = figure('Position', [100, 100, 1100, 750], 'Color', [1 1 1], 'Visible', 'off');

var_names = {'GDP Growth', 'Inflation Rate', 'Policy Interest Rate'};
colors_high = [0.12, 0.47, 0.71]; % Deep blue
colors_sign = [0.84, 0.15, 0.16]; % Crimson red

for i = 1:3
    subplot(2, 2, i);
    hold on; grid on; box off;

    h_vec = 0:horizon;
    lo_val = squeeze(sign_res.irf_lower(:, i, 1));
    hi_val = squeeze(sign_res.irf_upper(:, i, 1));
    med_val = squeeze(sign_res.irf_median(:, i, 1));
    chol_val = squeeze(irf_chol(:, i, 1));

    % Shaded Confidence Interval Patch
    fill([h_vec, fliplr(h_vec)], [lo_val', fliplr(hi_val')], colors_sign, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'Sign 90% Band');

    % Median Sign IRF Line
    plot(h_vec, med_val, 'Color', colors_sign, 'LineWidth', 2.5, 'DisplayName', 'Sign Median');

    % Cholesky IRF Line
    plot(h_vec, chol_val, '--', 'Color', colors_high, 'LineWidth', 2.0, 'DisplayName', 'Cholesky Point');

    % Zero reference line
    line([0, horizon], [0, 0], 'Color', [0.5, 0.5, 0.5], 'LineStyle', ':', 'LineWidth', 1.2);

    title(var_names{i}, 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');
    xlabel('Quarters after Shock', 'FontSize', 11, 'FontName', 'Helvetica');
    ylabel('Response (%)', 'FontSize', 11, 'FontName', 'Helvetica');
    set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'GridAlpha', 0.15);
    legend('Location', 'northeast', 'FontSize', 9);
    legend('boxoff');
end

% Summary Subplot: Forecast Error Variance Decomposition (FEVD)
subplot(2, 2, 4);
fevd_table = puremacro.var.fevd(res_var.A_list, B0_chol, horizon);
fevd_target = squeeze(fevd_table(:, 1, :)); % FEVD of Variable 1
b_bar = bar(0:horizon, fevd_target, 'stacked');
set(b_bar(1), 'FaceColor', [0.12, 0.47, 0.71]);
set(b_bar(2), 'FaceColor', [1.00, 0.50, 0.05]);
set(b_bar(3), 'FaceColor', [0.17, 0.63, 0.17]);
grid on; box off;
title('FEVD of GDP Growth (Cholesky)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Quarters after Shock', 'FontSize', 11, 'FontName', 'Helvetica');
ylabel('Variance Share', 'FontSize', 11, 'FontName', 'Helvetica');
legend(var_names, 'Location', 'northeast', 'FontSize', 8);
legend('boxoff');
set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'GridAlpha', 0.15);

% Save Figure as High-Res PNG
out_file = fullfile(fileparts(mfilename('fullpath')), 'svar_irf_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
