var y z x; varexo e v;
model;
x=.6*x(-1)+.05*x(-1)^2+.07*z(-1)^2+e+.1*e*x(-1)+.2*v*z(-1);
z=.4*z(-1)+v+.1*x(-1)*z(-1)+.03*e*v;
y=.92*y(+1)+x^2+.15*x*y(+1)+.07*(x(+1)+z(+1))^3+.1*x*z;
end;
initval;y=0;z=0;x=0;end;
shocks;var e;stderr .05;var v;stderr .03;corr e,v=.3;end;
