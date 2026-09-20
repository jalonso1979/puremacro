function export_dynare(dynare_root, fixture_dir, output_dir)
% Run the same source models independently in Dynare, with shared innovations.
addpath(fullfile(dynare_root, 'matlab'));
start_dir = pwd;
restore_dir = onCleanup(@() cd(start_dir));
run_id = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyyMMdd_HHmmss_SSS'));
completed = {};
models = dir(fullfile(fixture_dir, '*.mod'));
for m = 1:numel(models)
    [~, name] = fileparts(models(m).name);
    for ord = 2:3
        work = fullfile(output_dir, sprintf('%s_order%d', name, ord));
        if ~exist(work, 'dir'), mkdir(work); end
        cd(work);
        src = fileread(fullfile(fixture_dir, models(m).name));
        token = sprintf('%s_order%d', name, ord);
        fid = fopen([token '.mod'], 'w');
        fprintf(fid, '%s\nstoch_simul(order=%d,pruning,irf=0,nograph,noprint,nomoments,nocorr);\n', src, ord);
        fclose(fid);
        dynare([token '.mod'], 'noclearall', 'nolog');
        result = load(fullfile(work,token,'Output',[token '_results.mat']), 'M_', 'oo_', 'options_');
        M_ = result.M_; oo_ = result.oo_; options_ = result.options_;
        innovations = readmatrix(fullfile(fixture_dir, [name '_innovations.csv']));
        reference = struct();
        reference.run_id = run_id;
        reference.model_source = src;
        reference.dr = oo_.dr;
        reference.variable_names = M_.endo_names;
        reference.shock_names = M_.exo_names;
        reference.state_names = M_.endo_names(oo_.dr.order_var(M_.nstatic+(1:M_.nspred)));
        reference.shock_cov = M_.Sigma_e;
        reference.innovations = innovations;
        reference.path = simult_(M_,options_,oo_.dr.ys,oo_.dr,innovations,ord);
        reference.order = ord;
        reference.dynare_version = dynare_version;
        reference.matlab_version = version;
        save(fullfile(output_dir, sprintf('%s_order%d.mat',name,ord)), '-struct', 'reference', '-v7');
        completed{end+1} = token;
    end
end
fid = fopen(fullfile(output_dir, 'export_complete.json'), 'w');
fprintf(fid, '%s\n', jsonencode(struct('run_id',run_id,'cases',{completed})));
fclose(fid);
fprintf('DYNARE_EXPORT_COMPLETE %s: %d cases\n', run_id, numel(completed));
end
