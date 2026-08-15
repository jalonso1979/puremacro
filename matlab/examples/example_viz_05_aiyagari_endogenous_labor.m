% EXAMPLE_VIZ_05_AIYAGARI_ENDOGENOUS_LABOR Visual Plot for Aiyagari Endogenous Labor
%
% Generates publication-ready plots of:
%   1. Endogenous Labor Supply Choice n(a, e)
%   2. Optimal Consumption Policy c(a, e)
%   3. Next-Period Asset Choice a'(a, e)
%   4. Wealth Distribution & Inequality Lorenz Curve

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 5: Aiyagari Model Endogenous Labor Plots\n');
fprintf('=================================================================\n\n');

aiyagari_res = puremacro.vfi.aiyagari_endogenous_labor('Na', 45, 'Ne', 5);

fig = figure('Position', [100, 100, 1100, 750], 'Color', [1 1 1], 'Visible', 'off');

a_grid = aiyagari_res.a_grid;

% Plot 1: Endogenous Labor Supply n(a, e)
subplot(2, 2, 1);
plot(a_grid, aiyagari_res.policy_n, 'LineWidth', 2.0);
grid on; box off;
title('Endogenous Labor Supply n(a, e)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Current Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Labor Hours n(a, e)', 'FontSize', 10, 'FontName', 'Helvetica');
legend({'e_1 (Low)', 'e_2', 'e_3 (Median)', 'e_4', 'e_5 (High)'}, 'Location', 'northeast', 'FontSize', 8);
legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 2: Consumption Policy c(a, e)
subplot(2, 2, 2);
plot(a_grid, aiyagari_res.policy_c, 'LineWidth', 2.0);
grid on; box off;
title('Optimal Consumption c(a, e)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Current Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Consumption c(a, e)', 'FontSize', 10, 'FontName', 'Helvetica');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 3: Asset Policy a'(a, e)
subplot(2, 2, 3);
hold on; grid on; box off;
plot(a_grid, aiyagari_res.policy_a, 'LineWidth', 2.0);
plot(a_grid, a_grid, '--k', 'LineWidth', 1.2, 'DisplayName', '45^\circ Line');
title('Asset Policy a''(a, e)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Current Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Next Period Assets (a'')', 'FontSize', 10, 'FontName', 'Helvetica');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 4: Stationary Wealth Distribution & Lorenz Curve
subplot(2, 2, 4);
hold on; grid on; box off;
final_w = sort(aiyagari_res.sim_res.asset_panel(:, end) - min(aiyagari_res.sim_res.asset_panel(:, end)));
cum_w = cumsum(final_w) / sum(final_w);
pop_share = linspace(0, 1, length(final_w));

fill([pop_share, fliplr(pop_share)], [pop_share, fliplr(cum_w')], [0.85, 0.85, 0.85], ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'Inequality Area');
plot(pop_share, pop_share, '--k', 'LineWidth', 1.2, 'DisplayName', 'Equality');
plot(pop_share, cum_w, 'Color', [0.84, 0.15, 0.16], 'LineWidth', 2.5, 'DisplayName', 'Lorenz Curve');

title(sprintf('Lorenz Curve (Gini = %.3f)', aiyagari_res.sim_res.gini), 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Population Share', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Wealth Share', 'FontSize', 10, 'FontName', 'Helvetica');
legend('Location', 'northeast', 'FontSize', 8); legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

out_file = fullfile(fileparts(mfilename('fullpath')), 'aiyagari_endogenous_labor_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
