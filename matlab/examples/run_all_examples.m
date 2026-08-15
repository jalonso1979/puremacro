function run_all_examples()
% RUN_ALL_EXAMPLES Executable master runner for all MATLAB puremacro library examples.

    fprintf('\n');
    fprintf('=================================================================\n');
    fprintf('   puremacro MATLAB TOOLBOX — EXAMPLES SUITE\n');
    fprintf('=================================================================\n\n');

    ex_dir = fileparts(mfilename('fullpath'));

    run(fullfile(ex_dir, 'example_01_svar_identification.m'));
    run(fullfile(ex_dir, 'example_02_local_projections.m'));
    run(fullfile(ex_dir, 'example_03_dsge_gensys.m'));
    run(fullfile(ex_dir, 'example_04_connectedness.m'));
    run(fullfile(ex_dir, 'example_05_did_staggered.m'));
    run(fullfile(ex_dir, 'example_06_nowcasting_kalman.m'));

    fprintf('=================================================================\n');
    fprintf('   ALL EXAMPLES EXECUTED SUCCESSFULLY!\n');
    fprintf('=================================================================\n');
end
