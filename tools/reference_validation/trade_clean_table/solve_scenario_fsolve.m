function solve_scenario_fsolve(scen, chn_hkg)
% Reference solution of one tariff scenario of the legacy trade model on the clean OECD 2020
% table, found with MATLAB's fsolve (trust-region-dogleg) instead of the original hand-written
% Newton loop, whose fixed finite-difference step left the L1 residual oscillating between 30
% and 2,600 after 13 iterations on this table (run_scenarios_clean.log, 2026-10-03). The model
% code (calibrar, ff_equi, ff_eval) is untouched; only the root-finder differs, so the root is
% the same object the original Main77c_11s.m was looking for.
load('data_77c_11s_clean.mat');
nc=77; ns=11; nfd=3; iter_tau=2;
tauf=zeros(1,nc); tauf_fd=zeros(1,nc);
tau_a=ones(ns*nc,ns,nc); taufd_a=ones(ns*nc,nfd,nc);
Reciprocal_Tariff=ones(1,nc).*0.1; Reciprocal_Tariff(1,74)=0; Reciprocal_Tariff(1,13)=chn_hkg; Reciprocal_Tariff(1,29)=chn_hkg;
for ik = 1:nc
    tau_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:ns,74)= ones(ns,ns).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
    taufd_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:3,74)= ones(ns,3).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
end
[ytot, KT, LT, TT, TTfd, theta, alpha, beta, a, afd, tax, tax_fd, invforT,dM0,dMFD,dX0,dXFD,data_calibra,qxX0,qxFD0]=calibrar(ns,nc,nfd,data);
base = load('results_77c_11s_base_clean.mat', 'xx_sol'); x0 = base.xx_sol;   % seed: the zero-tariff solution
f = @(xx) ff_equi(xx,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
opts = optimoptions('fsolve','Algorithm','trust-region-dogleg','Display','iter','FunctionTolerance',1e-12, ...
    'StepTolerance',1e-12,'OptimalityTolerance',1e-10,'MaxIterations',150,'MaxFunctionEvaluations',2e6, ...
    'FiniteDifferenceType','forward','FiniteDifferenceStepSize',1e-7);
t0 = tic;
[xx_sol, ff_end, exitflag, output] = fsolve(f, x0, opts);
[ff_sol,p_sol,pfd_sol,ytot_sol,c_sol,w_sol,r_sol,T_sol,XN_sol,data_model_vf,data_tariff_vf,Pfd_final,qxX0_sol,qxFD0_sol]=...
    ff_eval(xx_sol,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
ff = ff_sol; converged = (exitflag > 0) && (sum(abs(ff_sol)) <= 0.0025); iter = output.iterations; solver = 'fsolve trust-region-dogleg';
fprintf('SCEN %s done converged=%d exitflag=%d iters=%d funcCount=%d final_diff=%.6e max_abs=%.3e elapsed=%.0fs\n', ...
    scen, converged, exitflag, iter, output.funcCount, sum(abs(ff_sol)), max(abs(ff_sol)), toc(t0));
save(sprintf('results_77c_11s_%s_clean.mat', scen), 'xx_sol','ff_sol','p_sol','pfd_sol','ytot_sol','c_sol','w_sol','r_sol','T_sol','XN_sol', ...
    'tau_a','taufd_a','tauf','tauf_fd','ff','converged','iter','exitflag','output','solver','Reciprocal_Tariff','iter_tau', ...
    'a','afd','alpha','beta','KT','LT','invforT','tax','tax_fd','theta','ytot','TT','TTfd','dM0','dMFD','dX0','dXFD','data_calibra');
end
