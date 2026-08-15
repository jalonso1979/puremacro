classdef model
% MODEL Dynare-like Automated Declarative VFI Model Parser & Solver.
%
%   M = puremacro.vfi.model('Neoclassical Growth Model');
%   M = M.set_param('beta', 0.96);
%   M = M.add_state('a', 0.1, 15.0, 50);
%   M = M.add_shock('z', 0.90, 0.15, 5);
%   M = M.set_utility(@(c, params) log(c));
%   M = M.set_budget(@(a, ap, z, params) exp(z)*a^params.alpha + (1-params.delta)*a - ap);
%   RES = M.solve();

    properties
        name            % Model name string
        params          % Struct of parameters
        states          % Struct array of state variables
        shocks          % Struct array of shock processes
        utility_fn      % Function handle for utility u(c, ...)
        budget_fn       % Function handle for budget constraint c(a, a', z, ...)
    end

    methods
        function obj = model(model_name)
            if nargin < 1
                model_name = 'VFI Model';
            end
            obj.name = model_name;
            obj.params = struct();
            obj.states = struct('name', {}, 'min', {}, 'max', {}, 'n', {});
            obj.shocks = struct('name', {}, 'rho', {}, 'sigma', {}, 'n', {}, 'method', {});
        end

        function obj = set_param(obj, name, val)
            obj.params.(name) = val;
        end

        function obj = add_state(obj, name, min_val, max_val, n_points)
            idx = length(obj.states) + 1;
            obj.states(idx).name = name;
            obj.states(idx).min = min_val;
            obj.states(idx).max = max_val;
            obj.states(idx).n = n_points;
        end

        function obj = add_shock(obj, name, rho, sigma, n_points, method)
            if nargin < 6 || isempty(method)
                method = 'tauchen';
            end
            idx = length(obj.shocks) + 1;
            obj.shocks(idx).name = name;
            obj.shocks(idx).rho = rho;
            obj.shocks(idx).sigma = sigma;
            obj.shocks(idx).n = n_points;
            obj.shocks(idx).method = method;
        end

        function obj = set_utility(obj, fn_handle)
            obj.utility_fn = fn_handle;
        end

        function obj = set_budget(obj, fn_handle)
            obj.budget_fn = fn_handle;
        end

        function res = solve(obj, varargin)
            % Automates grid construction, Tauchen discretization, tensor return evaluation,
            % Howard acceleration VFI, and stationary panel simulation.

            p_parser = inputParser;
            addParameter(p_parser, 'howard_steps', 20);
            addParameter(p_parser, 'device', 'cpu');
            addParameter(p_parser, 'tol', 1e-7);
            parse(p_parser, varargin{:});

            howard_steps = p_parser.Results.howard_steps;
            device = p_parser.Results.device;
            tol_val = p_parser.Results.tol;

            % 1. Discretize Shocks
            if isempty(obj.shocks)
                z_grid = 0.0;
                P_z = 1.0;
            else
                shk = obj.shocks(1);
                if strcmp(shk.method, 'rouwenhorst')
                    [log_z, P_z] = puremacro.vfi.rouwenhorst(shk.n, shk.rho, shk.sigma);
                else
                    [log_z, P_z] = puremacro.vfi.tauchen(shk.n, shk.rho, shk.sigma);
                end
                z_grid = log_z;
            end

            % 2. Construct State Grid
            st = obj.states(1);
            a_grid = linspace(st.min, st.max, st.n)';

            % 3. Automated Return Function Wrapper
            u_wrapper = @(a, ap, z) eval_return_fn(obj, a, ap, z);

            % 4. Call Accelerated VFI Engine
            beta_val = obj.params.beta;
            res_vfi = puremacro.vfi.solve(u_wrapper, a_grid, z_grid, P_z, beta_val, ...
                'howard_steps', howard_steps, 'device', device, 'tol', tol_val);

            % 5. Automated Panel Simulation
            res_sim = puremacro.vfi.simulate(res_vfi.policy_a, a_grid, z_grid, P_z, 2000, 100);

            res = res_vfi;
            res.sim = res_sim;
            res.model_name = obj.name;
            res.a_grid = a_grid;
            res.z_grid = z_grid;
            res.P_z = P_z;
        end

        function plot_results(obj, res)
            % Automated High-Quality Plotter
            if exist('OCTAVE_VERSION', 'builtin')
                try graphics_toolkit('gnuplot'); catch; end
            end

            fig = figure('Position', [100, 100, 1050, 480], 'Color', [1 1 1], 'Visible', 'off');

            % Subplot 1: Value Function V(a, z)
            subplot(1, 2, 1);
            plot(res.a_grid, res.V, 'LineWidth', 2.0);
            grid on; box off;
            title(sprintf('%s — Value Function V(a, z)', obj.name), 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
            xlabel(obj.states(1).name, 'FontSize', 10, 'FontName', 'Helvetica');
            ylabel('Value V', 'FontSize', 10, 'FontName', 'Helvetica');
            set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

            % Subplot 2: Asset Policy Function a'(a, z)
            subplot(1, 2, 2);
            hold on; grid on; box off;
            plot(res.a_grid, res.policy_a, 'LineWidth', 2.0);
            plot(res.a_grid, res.a_grid, '--k', 'LineWidth', 1.2, 'DisplayName', '45^\circ Line');
            title(sprintf('%s — Optimal Policy %s''(a, z)', obj.name, obj.states(1).name), 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');
            xlabel(obj.states(1).name, 'FontSize', 10, 'FontName', 'Helvetica');
            ylabel(sprintf('%s''', obj.states(1).name), 'FontSize', 10, 'FontName', 'Helvetica');
            set(gca, 'FontSize', 9, 'LineWidth', 1.1, 'GridAlpha', 0.15);

            out_file = fullfile(fileparts(mfilename('fullpath')), '..', 'examples', 'dynare_vfi_auto_plot.png');
            try
                saveas(fig, out_file);
                fprintf('Automated model plots saved to: %s\n', out_file);
            catch
            end
        end
    end
end

function u_val = eval_return_fn(obj, a, ap, z)
    c_val = obj.budget_fn(a, ap, z, obj.params);
    if c_val <= 0 || ~isreal(c_val) || isnan(c_val)
        u_val = -1e10;
    else
        u_val = obj.utility_fn(c_val, obj.params);
    end
end
