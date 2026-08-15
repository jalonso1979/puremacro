% EXAMPLE_VIZ_06_LIFE_CYCLE_PLOTS Life-Cycle Income, Consumption & Asset Profiles
%
% Generates publication-ready plots of:
%   1. Hump-Shaped Life-Cycle Income Profile
%   2. Consumption Smoothing & Asset Accumulation across Ages 20..70

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 6: Life-Cycle OLG Profile Plots\n');
fprintf('=================================================================\n\n');

lc_res = puremacro.vfi.life_cycle_olg('J', 50, 'J_ret', 35);

fig = figure('Position', [100, 100, 1050, 480], 'Color', [1 1 1], 'Visible', 'off');

ages = (20:(20 + lc_res.J - 1))';

% Subplot 1: Life-Cycle Income, Consumption & Asset Paths
subplot(1, 2, 1);
hold on; grid on; box off;
plot(ages, lc_res.mean_inc_age, '--', 'Color', [0.84, 0.15, 0.16], 'LineWidth', 2.2, 'DisplayName', 'Income');
plot(ages, lc_res.mean_cons_age, '-', 'Color', [0.12, 0.47, 0.71], 'LineWidth', 2.5, 'DisplayName', 'Consumption');

% Retirement vertical reference line
line([20 + lc_res.J_ret, 20 + lc_res.J_ret], [0, max(lc_res.mean_inc_age)*1.2], ...
    'Color', [0.5, 0.5, 0.5], 'LineStyle', ':', 'LineWidth', 1.2, 'DisplayName', 'Retirement Age 55');

title('Life-Cycle Income & Consumption Profiles', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Age', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Level ($)', 'FontSize', 10, 'FontName', 'Helvetica');
legend('Location', 'northeast', 'FontSize', 9); legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Subplot 2: Life-Cycle Asset Wealth Accumulation Path
subplot(1, 2, 2);
hold on; grid on; box off;
fill([ages; flipud(ages)], [lc_res.mean_assets_age; zeros(lc_res.J, 1)], [0.12, 0.47, 0.71], ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Wealth Area');
plot(ages, lc_res.mean_assets_age, '-o', 'Color', [0.12, 0.47, 0.71], 'LineWidth', 2.5, ...
    'MarkerSize', 4, 'MarkerFaceColor', [0.12, 0.47, 0.71], 'DisplayName', 'Mean Assets');

title('Life-Cycle Wealth Accumulation & Retirement Decumulation', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Age', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Accumulated Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
legend('Location', 'northeast', 'FontSize', 9); legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

out_file = fullfile(fileparts(mfilename('fullpath')), 'life_cycle_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
