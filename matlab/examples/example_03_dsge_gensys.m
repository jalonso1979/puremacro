% EXAMPLE_03_DSGE_GENSYS Sims (2002) QZ Rational Expectations Solver
%
% Demonstrates:
%   1. Solving a 3-Equation New Keynesian DSGE Model via Sims' gensys
%   2. Simulating Impulse Responses to Monetary Policy & Supply Shocks

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=================================================================\n');
fprintf('  EXAMPLE 3: Sims (2002) gensys Linear DSGE Model Solver\n');
fprintf('=================================================================\n\n');

% New Keynesian Model Parameters
beta = 0.99;    % Discount factor
kappa = 0.10;   % Slope of New Keynesian Phillips Curve
phi_pi = 1.50;  % Monetary policy Taylor rule response to inflation
phi_y = 0.50;   % Monetary policy Taylor rule response to output gap

% Structural matrices: Gamma0 * z_t = Gamma1 * z_{t-1} + Psi * eps_t + Pi * eta_t
% Variables z_t = [Output_t, Inflation_t, InterestRate_t]'
G0 = [ 1,      1,      0; ...
       0,   beta,      0; ...
     -phi_y, -phi_pi,  1];

G1 = [ 1,      0,      1; ...
     -kappa,   1,      0; ...
       0,      0,      0];

Psi = [-1,  0; ...
        0, -1; ...
        0,  0];

Pi  = [ 1,  1; ...
        0, beta; ...
        0,  0];

% Solve rational expectations equilibrium
sol = puremacro.dsge.gensys(G0, G1, Psi, Pi);

fprintf('Blanchard-Kahn Existence (eu1):  %d\n', sol.eu(1));
fprintf('Blanchard-Kahn Uniqueness (eu2): %d\n\n', sol.eu(2));

fprintf('State Transition Matrix G (z_t = G * z_{t-1} + Impact * eps_t):\n');
disp(sol.G);

fprintf('Shock Impact Matrix:\n');
disp(sol.Impact);

% Simulate 10-period impulse response path to a monetary shock
horizon = 10;
irf_dsge = zeros(horizon, 3);
z_t = sol.Impact(:, 1); % Initial shock
for t = 1:horizon
    irf_dsge(t, :) = z_t';
    z_t = sol.G * z_t;
end

fprintf('Simulated DSGE Impulse Response Path to Shock 1 (Output, Inflation, Interest Rate):\n');
disp(irf_dsge(1:5, :));
fprintf('=================================================================\n\n');
