// Reference Dynare-style HANK Sequence-Space Model (.mod)
// Auclert, Bardóczy, Rognlie & Straub (2021, Econometrica)
// Compatible with puremacro 3.0.0 Sequence-Space Jacobian (SSJ) bridge

var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50;
  a_max = 30.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
