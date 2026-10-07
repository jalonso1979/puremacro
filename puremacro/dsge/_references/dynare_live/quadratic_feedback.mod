var x y; varexo e;
parameters rho beta eta; rho=.8;beta=.95;eta=.3;
model;x=rho*x(-1)+e;y=beta*y(+1)+x^2+eta*x*y(+1);end;
initval;x=0;y=0;end;shocks;var e;stderr .1;end;
