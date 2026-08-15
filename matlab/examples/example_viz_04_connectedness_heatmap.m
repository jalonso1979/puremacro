% EXAMPLE_VIZ_04_CONNECTEDNESS_HEATMAP Spillover Network Heatmap Plot
%
% Generates publication-ready plots of:
%   1. Diebold-Yilmaz Spillover Matrix Heatmap
%   2. Baruník-Křehlík Frequency-Domain Short vs Long-Run Connectedness

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 4: Spillover Network Heatmap & Frequency Breakdown\n');
fprintf('=================================================================\n\n');

rng(55);
T = 300;
N_assets = 4;
asset_labels = {'Equity', 'Bonds', 'FX', 'Commodities'};
returns = randn(T, N_assets);

dy_res = puremacro.connectedness.diebold_yilmaz(returns, 2, 12, 'gfevd');
bk_short = puremacro.connectedness.barunik_krehlik(returns, 2, 100, [pi/4, pi]);
bk_long = puremacro.connectedness.barunik_krehlik(returns, 2, 100, [0, pi/4]);

fig = figure('Position', [150, 150, 1050, 480], 'Color', [1 1 1], 'Visible', 'off');

% Subplot 1: Diebold-Yilmaz Spillover Heatmap Matrix
subplot(1, 2, 1);
imagesc(dy_res.pairwise_matrix * 100);
colormap(gca, 'jet');
colorbar;
axis square;
title('Diebold-Yilmaz Spillover Matrix (%)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
set(gca, 'XTick', 1:4, 'XTickLabel', asset_labels, 'YTick', 1:4, 'YTickLabel', asset_labels, ...
    'FontSize', 10, 'LineWidth', 1.1);

% Add percentage annotations in each cell
for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%.1f%%', dy_res.pairwise_matrix(i,j)*100), ...
            'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

% Subplot 2: Frequency-Domain Short vs Long-Run Comparison
subplot(1, 2, 2);
short_net = bk_short.total_spillover;
long_net = bk_long.total_spillover;

b_freq = bar([short_net; long_net], 'FaceColor', [0.12, 0.47, 0.71]);
grid on; box off;
title('Frequency-Domain Spillover Index (%)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
set(gca, 'XTickLabel', {'Short-Run (w in [\pi/4, \pi])', 'Long-Run (w in [0, \pi/4])'}, ...
    'FontSize', 10, 'LineWidth', 1.1, 'GridAlpha', 0.15);
ylabel('Total Spillover (%)', 'FontSize', 11, 'FontName', 'Helvetica');

out_file = fullfile(fileparts(mfilename('fullpath')), 'connectedness_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
