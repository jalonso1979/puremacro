// Reference Dynare-style Two-Asset HANK Sequence-Space Model (.mod)
// Kaplan, Moll & Violante (2018, AER); Auclert et al. (2021, Econometrica)
// Compatible with puremacro 3.0.0 Sequence-Space Jacobian (SSJ) bridge

var Y C D r_b r_a pi i;
varexo eps_m;

parameters beta gamma r_b_ss r_a_ss phi_pi kappa chi_0 chi_1;
beta = 0.985;
gamma = 1.0;
r_b_ss = 0.01;
r_a_ss = 0.03;
phi_pi = 1.5;
kappa = 0.1;
chi_0 = 0.25;
chi_1 = 1.0;

hetagent_block;
  model = hank_two_asset;
  assets = 2;
  liquid_asset = b;
  illiquid_asset = a;
  n_a = 25;
  n_b = 25;
  a_max = 30.0;
  b_max = 15.0;
  b_min = 0.0;
  chi_0 = 0.25;
  chi_1 = 1.0;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_b_ss + phi_pi * pi + eps_m;
  r_b = i - pi(+1);
  r_a = r_a_ss + 0.5 * (r_b - r_b_ss);
end;
