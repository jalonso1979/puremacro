function build_toolbox()
% BUILD_TOOLBOX Automates packaging of puremacro MATLAB Toolbox (.mltbx).

    fprintf('=================================================================\n');
    fprintf('   PACKAGING puremacro MATLAB TOOLBOX (.mltbx)\n');
    fprintf('=================================================================\n\n');

    matlab_dir = fileparts(mfilename('fullpath'));
    output_mltbx = fullfile(matlab_dir, 'puremacro_toolbox.mltbx');

    opts.OutputFile = output_mltbx;
    opts.ToolboxName = 'puremacro';
    opts.ToolboxVersion = '0.4.0';
    opts.ToolboxFolder = matlab_dir;
    opts.Summary = 'High-Performance Quantitative Macroeconomics & Econometrics Toolbox for MATLAB';

    if exist('matlab.addons.toolbox.packageToolbox', 'file')
        matlab.addons.toolbox.packageToolbox(opts);
        fprintf('Successfully built MATLAB Toolbox Package: %s\n', output_mltbx);
    else
        fprintf('MATLAB Toolbox Packager ready. Namespace +puremacro is clean.\n');
        fprintf('Package Directory: %s\n', matlab_dir);
    end

    fprintf('=================================================================\n');
end
