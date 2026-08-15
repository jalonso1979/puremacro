% EXAMPLE_VIZ_07_SOVEREIGN_DEFAULT_PLOTS Sovereign Default Bond Schedule Plots
%
% Generates publication-ready plots of:
%   1. Endogenous Bond Price Schedule q(B', y)
%   2. Sovereign Default Decision Boundary d(B, y)

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 7: Sovereign Default Bond Price & Spread Plots\n');
fprintf('=================================================================\n\n');

sd_res = puremacro.vfi.sovereign_default('Nb', 35, 'Ny', 9);

fig = figure('Position', [100, 100, 1050, 480], 'Color', [1 1 1], 'Visible', 'off');

% Subplot 1: Endogenous Bond Price Schedule q(B', y)
subplot(1, 2, 1);
plot(sd_res.B_grid, sd_res.q_bond, 'LineWidth', 2.0);
grid on; box off;
title('Endogenous Bond Price Schedule q(B'', y)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Sovereign Debt Level B'' (Negative = Borrowing)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Bond Price q(B'', y)', 'FontSize', 10, 'FontName', 'Helvetica');
legend({'y_1 (Recession)', 'y_3', 'y_5 (Median)', 'y_7', 'y_9 (Boom)'}, 'Location', 'northwest', 'FontSize', 8);
legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Subplot 2: Sovereign Default Probability Heatmap
subplot(1, 2, 2);
imagesc(sd_res.default_prob * 100);
colormap(gca, 'jet');
colorbar;
axis square;
title('Endogenous Default Probability (%)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Output Endowment Shock State y', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Sovereign Debt Grid B Index', 'FontSize', 10, 'FontName', 'Helvetica');
set(gca, 'FontSize', 9, 'LineWidth', 1.1);

out_file = fullfile(fileparts(mfilename('fullpath')), 'sovereign_default_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
