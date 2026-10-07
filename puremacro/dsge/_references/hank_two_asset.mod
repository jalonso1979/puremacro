// Reference Dynare-style Two-Asset HANK Sequence-Space Model (.mod)
// Kaplan, Moll & Violante (2018, AER); Auclert, Bardoczy, Rognlie & Straub (2021, Econometrica)
// Solved by puremacro.models.hank_sequence_space.solve_two_asset_hank_sequence_space
// through the SSJ bridge (puremacro.dsge.hank.solve_hank_bridge).
//
// Households hold liquid bonds b (return r_b) and an illiquid asset a (return r_a)
// that can only be adjusted at the convex cost
//   chi(d, a) = chi_0/(1+chi_1) |d|^(1+chi_1) / (a + a_bar)^chi_1,  d = a' - (1+r_a) a.
// Quarterly calibration: beta (1 + r_a_ss) = 0.98 * 1.0125 < 1, so the illiquid-wealth
// distribution is interior (no mass at a_max). Asset returns are paid out of aggregate
// income on the holdings households carry into the quarter (z_t N = Y_t - r_b,t B_{t-1}
// - r_a,t A_{t-1}), so household income equals Y; adjustment costs are a resource cost
// (Y = C + CHI), and total wealth A + B stays at its steady state (Walras's law).
// r_b and r_a below are ex-ante real rates, realised by households one quarter later.

var Y C CHI D r_b r_a pi i;
varexo eps_m;

parameters beta gamma r_b_ss r_a_ss phi_pi kappa chi_0 chi_1 alpha_ab;
beta = 0.98;
gamma = 1.0;
r_b_ss = 0.005;
r_a_ss = 0.0125;
phi_pi = 1.5;
kappa = 0.1;
chi_0 = 1.0;
chi_1 = 1.0;
alpha_ab = 0.5;

hetagent_block;
  model = hank_two_asset;
  assets = 2;
  liquid_asset = b;
  illiquid_asset = a;
  n_a = 25;
  n_b = 25;
  a_max = 40.0;
  b_max = 10.0;
  b_min = 0.0;
  chi_0 = 1.0;
  chi_1 = 1.0;
  a_bar = 0.25;
end;

model;
  Y = C + CHI;
  pi = beta * pi(+1) + kappa * Y;
  i = r_b_ss + phi_pi * pi + eps_m;
  r_b = i - pi(+1);
  r_a = r_a_ss + alpha_ab * (r_b - r_b_ss);
end;
