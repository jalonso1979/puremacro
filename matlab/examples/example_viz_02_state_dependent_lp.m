% EXAMPLE_VIZ_02_STATE_DEPENDENT_LP State-Dependent Local Projections Plot
%
% Generates publication-ready plots comparing Local Projection IRFs across:
%   - Expansion / High-Slack Regime (High State)
%   - Recession / Low-Slack Regime (Low State)

if exist('OCTAVE_VERSION', 'builtin')
    try graphics_toolkit('gnuplot'); catch; end
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  VISUAL EXAMPLE 2: State-Dependent LP Regime Plots\n');
fprintf('=================================================================\n\n');

rng(123);
T = 220;
y = cumsum(randn(T, 1) * 0.04);
x = randn(T, 1);
z_slack = randn(T, 1); % State variable (e.g. unemployment gap)

horizon = 12;
res_sd = puremacro.lp.state_dependent(y, x, z_slack, 0:horizon, 2, 'logistic', 3.0, 0.0, 0.10);

% Create Figure
fig = figure('Position', [150, 150, 1000, 500], 'Color', [1 1 1], 'Visible', 'off');

h_vec = 0:horizon;

% Subplot 1: High-Regime (Expansion / High Slack)
subplot(1, 2, 1);
hold on; grid on; box off;
col_high = [0.84, 0.15, 0.16]; % Crimson Red

fill([h_vec, fliplr(h_vec)], [res_sd.lo_high', fliplr(res_sd.hi_high')], col_high, ...
    'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', '90% HAC Band');
plot(h_vec, res_sd.beta_high, '-o', 'Color', col_high, 'LineWidth', 2.5, 'MarkerSize', 5, ...
    'MarkerFaceColor', col_high, 'DisplayName', 'High State IRF');
line([0, horizon], [0, 0], 'Color', [0.5, 0.5, 0.5], 'LineStyle', ':', 'LineWidth', 1.2);

title('Expansion Regime (High State)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Quarters after Policy Shock', 'FontSize', 11, 'FontName', 'Helvetica');
ylabel('Output Cumulative Response (%)', 'FontSize', 11, 'FontName', 'Helvetica');
set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'GridAlpha', 0.15);
legend('Location', 'northeast', 'FontSize', 9); legend('boxoff');

% Subplot 2: Low-Regime (Recession / Low Slack)
subplot(1, 2, 2);
hold on; grid on; box off;
col_low = [0.12, 0.47, 0.71]; % Royal Blue

fill([h_vec, fliplr(h_vec)], [res_sd.lo_low', fliplr(res_sd.hi_low')], col_low, ...
    'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', '90% HAC Band');
plot(h_vec, res_sd.beta_low, '-s', 'Color', col_low, 'LineWidth', 2.5, 'MarkerSize', 5, ...
    'MarkerFaceColor', col_low, 'DisplayName', 'Low State IRF');
line([0, horizon], [0, 0], 'Color', [0.5, 0.5, 0.5], 'LineStyle', ':', 'LineWidth', 1.2);

title('Recession Regime (Low State)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Quarters after Policy Shock', 'FontSize', 11, 'FontName', 'Helvetica');
ylabel('Output Cumulative Response (%)', 'FontSize', 11, 'FontName', 'Helvetica');
set(gca, 'FontSize', 10, 'LineWidth', 1.2, 'GridAlpha', 0.15);
legend('Location', 'northeast', 'FontSize', 9); legend('boxoff');

out_file = fullfile(fileparts(mfilename('fullpath')), 'lp_regimes_demo.png');
try
    saveas(fig, out_file);
    fprintf('Successfully generated and saved figure: %s\n\n', out_file);
catch err
    fprintf('Plot generated cleanly (saveas skipped in headless mode: %s).\n\n', err.message);
end
