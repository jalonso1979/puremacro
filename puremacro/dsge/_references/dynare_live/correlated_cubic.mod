var y z x; varexo e v;
parameters rho beta; rho=.7; beta=.95;
model; x=rho*x(-1)+e; z=.5*z(-1)+v;
y=beta*y(+1)+(x+2*z)^3+.2*x*z; end;
initval; x=0;z=0;y=0;end;
shocks;var e;stderr .1;var v;stderr .2;corr e,v=.2;end;
