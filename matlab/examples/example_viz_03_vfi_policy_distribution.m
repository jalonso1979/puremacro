% EXAMPLE_VIZ_03_VFI_POLICY_DISTRIBUTION Dynamic Programming & Stationary Wealth Plots
%
% Generates publication-ready plots of:
%   1. Value Function V(a, z)
%   2. Asset Policy Function a'(a, z)
%   3. Simulated Stationary Wealth Distribution & Lorenz Curve (Gini Index)

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 3: Value Function & Stationary Wealth Distribution\n');
fprintf('=================================================================\n\n');

Na = 40; Nz = 5;
a_grid = linspace(0.1, 15.0, Na)';
[z_grid, P] = puremacro.vfi.tauchen(Nz, 0.88, 0.12);

u_fn = @(a, ap, z) (exp(z)*a^0.36 + 0.90*a - ap > 0) * ...
                   log(max(exp(z)*a^0.36 + 0.90*a - ap, 1e-6)) + ...
                   (exp(z)*a^0.36 + 0.90*a - ap <= 0) * (-1e10);

vfi_res = puremacro.vfi.solve(u_fn, a_grid, z_grid, P, 0.95, 'howard_steps', 20);
sim_res = puremacro.vfi.simulate(vfi_res.policy_a, a_grid, z_grid, P, 3000, 150);

% Create 2x2 Figure
fig = figure('Position', [100, 100, 1100, 750], 'Color', [1 1 1], 'Visible', 'off');

% Plot 1: Value Function V(a, z)
subplot(2, 2, 1);
plot(a_grid, vfi_res.V, 'LineWidth', 2.0);
grid on; box off;
title('Value Function V(a, z)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Current Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Value V(a, z)', 'FontSize', 10, 'FontName', 'Helvetica');
legend({'z_1 (Low)', 'z_2', 'z_3 (Median)', 'z_4', 'z_5 (High)'}, 'Location', 'northeast', 'FontSize', 8);
legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 2: Asset Policy Function a'(a, z)
subplot(2, 2, 2);
hold on; grid on; box off;
plot(a_grid, vfi_res.policy_a, 'LineWidth', 2.0);
plot(a_grid, a_grid, '--k', 'LineWidth', 1.2, 'DisplayName', '45^\circ Line');
title('Optimal Asset Policy a''(a, z)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Current Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Next Period Assets (a'')', 'FontSize', 10, 'FontName', 'Helvetica');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 3: Stationary Wealth Histogram
subplot(2, 2, 3);
final_wealth = sim_res.asset_panel(:, end);
if exist('histogram', 'file') || exist('histogram', 'builtin')
    histogram(final_wealth, 25, 'FaceColor', [0.12, 0.47, 0.71], 'EdgeColor', 'w', 'FaceAlpha', 0.85);
else
    [n_cnt, x_cnt] = hist(final_wealth, 25);
    bar(x_cnt, n_cnt, 'FaceColor', [0.12, 0.47, 0.71], 'EdgeColor', 'w');
end
grid on; box off;
title('Stationary Wealth Distribution', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Assets (a)', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Agent Count', 'FontSize', 10, 'FontName', 'Helvetica');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

% Plot 4: Lorenz Curve & Gini Index
subplot(2, 2, 4);
hold on; grid on; box off;
sorted_w = sort(final_wealth - min(final_wealth));
cum_w = cumsum(sorted_w) / sum(sorted_w);
pop_share = linspace(0, 1, length(sorted_w));

fill([pop_share, fliplr(pop_share)], [pop_share, fliplr(cum_w')], [0.85, 0.85, 0.85], ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'Inequality Area');
plot(pop_share, pop_share, '--k', 'LineWidth', 1.2, 'DisplayName', 'Perfect Equality');
plot(pop_share, cum_w, 'Color', [0.84, 0.15, 0.16], 'LineWidth', 2.5, 'DisplayName', 'Lorenz Curve');

title(sprintf('Lorenz Curve (Gini = %.3f)', sim_res.gini), 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Cumulative Population Share', 'FontSize', 10, 'FontName', 'Helvetica');
ylabel('Cumulative Wealth Share', 'FontSize', 10, 'FontName', 'Helvetica');
legend('Location', 'northeast', 'FontSize', 8); legend('boxoff');
set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

out_file = fullfile(fileparts(mfilename('fullpath')), 'vfi_wealth_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
