function run_all_visual_examples()
% RUN_ALL_VISUAL_EXAMPLES Master runner for publication-quality visual examples.

    fprintf('\n');
    fprintf('=================================================================\n');
    fprintf('   puremacro MATLAB TOOLBOX — VISUAL PLOTTING EXAMPLES\n');
    fprintf('=================================================================\n\n');

    % Set headless gnuplot toolkit if running in Octave
    if exist('OCTAVE_VERSION', 'builtin')
        try
            graphics_toolkit('gnuplot');
        catch
        end
    end

    ex_dir = fileparts(mfilename('fullpath'));

    run(fullfile(ex_dir, 'example_viz_01_svar_irfs.m'));
    run(fullfile(ex_dir, 'example_viz_02_state_dependent_lp.m'));
    run(fullfile(ex_dir, 'example_viz_03_vfi_policy_distribution.m'));
    run(fullfile(ex_dir, 'example_viz_04_connectedness_heatmap.m'));

    fprintf('=================================================================\n');
    fprintf('   ALL VISUAL EXAMPLES & PNG FIGURES GENERATED SUCCESSFULLY!\n');
    fprintf('=================================================================\n');
end
