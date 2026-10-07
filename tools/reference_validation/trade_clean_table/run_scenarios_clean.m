% Reference solutions of the legacy MATLAB trade model on the clean OECD 2020 table.
% Model code (calibrar, ff_equi, ff_eval) is the original, untouched. The scenario
% definitions and the Newton loop are transcribed from Main77c_11s.m. Two edits: each
% scenario is a named case instead of a manually uncommented block, and the
% finite-difference step of the Jacobian is relative (1e-6 * max(1, |x_j|)) instead of the
% original absolute 1e-2, with which the L1 residual oscillated between 30 and 110 after
% 176 iterations on the clean table (log 2026-10-03, run_scenarios_clean_v1_eps1e-2.log).
% The root being solved for is unchanged.
load('data_77c_11s_clean.mat');
nc=77; ns=11; nfd=3;
scen_names = {'base','t10','t10_25','t10_54','t10_75','t10_125','t10_145'};
chn_hkg    = [0.1, 0.1, 0.25, 0.54, 0.75, 1.25, 1.45];   % Reciprocal_Tariff(13), (29)
iter_taus  = [1, 2, 2, 2, 2, 2, 2];                      % 1 -> no tariff (base)
xx_base = []; xx_t10 = [];
for s = 1:numel(scen_names)
    scen = scen_names{s}; iter_tau = iter_taus(s);
    t_scen = tic;
    tauf=zeros(1,nc); tauf_fd=zeros(1,nc);
    tau_a=zeros(ns*nc,ns,nc); taufd_a=zeros(ns*nc,nfd,nc);
    tau=zeros(ns,ns);
    for ik=1:nc
        tau(:,:,ik)=ones(ns,ns).*(1+tauf(ik));
        if ik==1
            tau_a(:,:,ik)=[ones(ns,ns);repmat(squeeze(tau(:,:,ik)),nc-ik,1)];
        end
        if ik < nc
            tau_a(:,:,ik)=[repmat(squeeze(tau(:,:,ik)),ik-1,1);ones(ns,ns);repmat(squeeze(tau(:,:,ik)),nc-ik,1)];
        end
        if ik == nc
            tau_a(:,:,ik)=[repmat(squeeze(tau(:,:,ik)),ik-1,1);ones(ns,ns)];
        end
    end
    taufd=zeros(ns,nfd);
    for ik=1:nc
        taufd(:,:,ik)=ones(ns,nfd).*(1+tauf_fd(ik));
        if ik==1
            taufd_a(:,:,ik)=[ones(ns,nfd);repmat(squeeze(taufd(:,:,ik)),nc-ik,1)];
        end
        if ik < nc
            taufd_a(:,:,ik)=[repmat(squeeze(taufd(:,:,ik)),ik-1,1);ones(ns,nfd);repmat(squeeze(taufd(:,:,ik)),nc-ik,1)];
        end
        if ik == nc
            taufd_a(:,:,ik)=[repmat(squeeze(taufd(:,:,ik)),ik-1,1);ones(ns,nfd)];
        end
    end
    Reciprocal_Tariff=ones(1,nc).*0.1;
    Reciprocal_Tariff(1,74)=0; % US
    Reciprocal_Tariff(1,13)=chn_hkg(s); % China
    Reciprocal_Tariff(1,29)=chn_hkg(s); % Hong Kong
    for ik = 1:nc
        tau_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:ns,74)= ones(ns,ns).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
        taufd_a(1+(ik-1)*ns:ns+(ik-1)*ns,1:3,74)= ones(ns,3).*(1+Reciprocal_Tariff(1,ik).*(iter_tau-1));
    end
    [ytot, KT, LT, TT, TTfd, theta, alpha, beta, a, afd, tax, tax_fd, invforT,dM0,dMFD,dX0,dXFD,data_calibra,qxX0,qxFD0]=...
        calibrar(ns,nc,nfd,data);
    if iter_tau==1
        error=(data_calibra-data);
        if max(sum(abs(error))) > 10^(-4)
            disp('ajuste calibra'); return
        end
        Tariffs=dM0.*tauf+dMFD.*tauf_fd;
        TTTotal=TT+TTfd+Tariffs;
        p=ones(ns*nc,1); y=reshape(ytot,nc*ns,1); r=ones(nc,1); w=ones(nc,1);
        T=TTTotal'; XN=invforT(1,1:nc-1)';
        xx=[log(p); log(y); log(r); log(w); T; XN];
    elseif strcmp(scen,'t10')
        xx=xx_base;          % Main77c_11s.m: the base solution seeds t10
    else
        xx=xx_t10;           % Main77c_11s.m: xx_sol_77c_11s_t10.mat seeds the other scenarios
    end
    niter=80; EPSILON=0.0025; eps_rel=1e-6;
    dff=zeros(ns*nc*2+3*nc+nc-1);
    converged=false;
    for iter=1:niter
        [ff]=ff_equi(xx,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
        if sum(isnan(ff)) > 0
            fprintf('SCEN %s iter %d: error isnan ff\n', scen, iter); break
        end
        diff=sum(abs(ff));
        fprintf('SCEN %s iter %d diff=%.6e elapsed=%.0fs\n', scen, iter, diff, toc(t_scen));
        if diff <= EPSILON
            converged=true; break
        end
        for j=1:ns*nc*2+3*nc+nc-1
            h=eps_rel*max(1,abs(xx(j)));
            xx(j)=xx(j)+h;
            [ff1]=ff_equi(xx,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
            if sum(isnan(ff1)) > 0
                fprintf('SCEN %s: error isnan ff1\n', scen); return
            end
            dff(:,j)=(ff1-ff)/h;
            xx(j)=xx(j)-h;
        end
        if iter <=3
            xscal=.5;
        else
            xscal=1;
        end
        dffinv=dff^(-1);
        step = xscal*(dffinv*ff);
        xx = xx - step;
        if max(abs(step)) < 1e-12*max(1,max(abs(xx)))
            fprintf('SCEN %s iter %d: step below machine precision\n', scen, iter); break
        end
    end
    xx_sol=xx;
    [ff_sol,p_sol,pfd_sol,ytot_sol,c_sol,w_sol,r_sol,T_sol,XN_sol,...
        data_model_vf,data_tariff_vf,Pfd_final,qxX0_sol,qxFD0_sol]=...
        ff_eval(xx,nc,ns,nfd,tau_a,taufd_a,tauf,tauf_fd,KT, LT, theta, alpha, beta, a, afd, tax, tax_fd);
    fprintf('SCEN %s done converged=%d iters=%d final_diff=%.6e elapsed=%.0fs\n', scen, converged, iter, sum(abs(ff_sol)), toc(t_scen));
    save(sprintf('results_77c_11s_%s_clean.mat', scen), 'xx_sol','ff_sol','p_sol','pfd_sol','ytot_sol','c_sol','w_sol','r_sol','T_sol','XN_sol', ...
        'tau_a','taufd_a','tauf','tauf_fd','ff','converged','iter','Reciprocal_Tariff','iter_tau', ...
        'a','afd','alpha','beta','KT','LT','invforT','tax','tax_fd','theta','ytot','data_calibra','TT','TTfd','dM0','dMFD','dX0','dXFD', ...
        'data_model_vf','data_tariff_vf','Pfd_final','qxX0_sol','qxFD0_sol','qxX0','qxFD0');
    if strcmp(scen,'base'); xx_base=xx_sol; end
    if strcmp(scen,'t10');  xx_t10=xx_sol;  end
end
disp('DONE_OK')
