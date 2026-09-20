var y x; varexo e;
model;x=.6*x(-1)+.05*x(-1)^2+.02*x(-1)^3+e+.1*e*x(-1);
y=.92*y(+1)+x^2+.15*x*y(+1)+.07*x(+1)^3;end;
initval;y=0;x=0;end;shocks;var e;stderr .05;end;
