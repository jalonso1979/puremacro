function polish_rate(scen, rate, chn_hkg, warm_file)
% Legacy MATLAB model on the clean OECD 2020 table: evaluate the residual at a
% puremacro-computed equilibrium and polish it with the original Newton loop
% (relative finite-difference step) until the original L1 tolerance 0.0025 holds.
% The model code (calibrar, ff_equi, ff_eval) is untouched; the warm start only
% chooses where MATLAB's own iteration begins, the saved root is MATLAB's.
load('data_77c_11s_clean.mat');
nc=77; ns=11; nfd=3; iter_tau=2;
tauf=zeros(1,nc); tauf_fd=zeros(1,nc);
tau_a=ones(ns*nc,ns,nc); taufd_a=ones(ns*nc,nfd,nc);
Reciprocal_Tariff=ones(1,nc).*rate; Reciprocal_Tariff(1,74)=0; if ~isnan(chn_hkg), Reciprocal_Tariff(1,13)=chn_hkg; Reciprocal_Tariff(1,29)=chn_hkg; end
for ik = 1:nc
    tau_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:ns,74)= ones(ns,ns).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
    taufd_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:3,74)= ones(ns,3).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
end
[ytot, KT, LT, TT, TTfd, theta, alpha, beta, a, afd, tax, tax_fd, invforT,dM0,dMFD,dX0,dXFD,data_calibra,qxX0,qxFD0]=calibrar(ns,nc,nfd,data);
w = load(warm_file); xx = w.xx_pm(:);
assert(max(abs(w.tau_a(:) - tau_a(:))) == 0 && max(abs(w.taufd_a(:) - taufd_a(:))) == 0, 'warm start was computed for a different tariff schedule');
f = @(x) ff_equi(x,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
ff = f(xx);
fprintf('SCEN %s warm start: L1 residual %.6e, max abs %.3e (puremacro solution evaluated by the MATLAB model)\n', scen, sum(abs(ff)), max(abs(ff)));
niter=15; EPSILON=0.0025; eps_rel=1e-6; n=numel(xx); dff=zeros(n); converged=false; t0=tic;
for iter=1:niter
    ff = f(xx); diff=sum(abs(ff));
    fprintf('SCEN %s polish iter %d diff=%.6e elapsed=%.0fs\n', scen, iter, diff, toc(t0));
    if diff <= EPSILON, converged=true; break; end
    for j=1:n
        h=eps_rel*max(1,abs(xx(j))); xx(j)=xx(j)+h; dff(:,j)=(f(xx)-ff)/h; xx(j)=xx(j)-h;
    end
    xx = xx - dff\ff;
end
xx_sol=xx;
[ff_sol,p_sol,pfd_sol,ytot_sol,c_sol,w_sol,r_sol,T_sol,XN_sol,data_model_vf,data_tariff_vf,Pfd_final,qxX0_sol,qxFD0_sol]=...
    ff_eval(xx,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
ff = ff_sol; converged = converged && (sum(abs(ff_sol)) <= EPSILON); solver = 'legacy Newton, relative finite-difference step, from a puremacro warm start';
fprintf('SCEN %s done converged=%d iters=%d final_diff=%.6e max_abs=%.3e warm_start_rel_change=%.3e elapsed=%.0fs\n', scen, converged, iter, sum(abs(ff_sol)), max(abs(ff_sol)), max(abs(xx_sol - w.xx_pm(:))./(abs(w.xx_pm(:))+1)), toc(t0));
save(sprintf('results_77c_11s_%s_clean.mat', scen), 'xx_sol','ff_sol','p_sol','pfd_sol','ytot_sol','c_sol','w_sol','r_sol','T_sol','XN_sol', ...
    'tau_a','taufd_a','tauf','tauf_fd','ff','converged','iter','solver','Reciprocal_Tariff','iter_tau', ...
    'a','afd','alpha','beta','KT','LT','invforT','tax','tax_fd','theta','ytot','TT','TTfd','dM0','dMFD','dX0','dXFD','data_calibra');
end
