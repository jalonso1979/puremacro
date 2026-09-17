// 3-Equation New Keynesian Model - Constrained Regime (Zero Lower Bound)
// Advanced Macroeconomics (MAV) - Topic 2: DSGE Local Approximations
// Based on Woodford (2003) and Guerrieri & Iacoviello (2015, JME)

var y pi r g;
varexo eps_r eps_g;

parameters beta sigma kappa phi_pi phi_y rho_g r_ss;

beta   = 0.99;
sigma  = 1.0;
kappa  = 0.1;
phi_pi = 1.5;
phi_y  = 0.125;
rho_g  = 0.80;
r_ss   = 0.01;

model(linear);
// 1. Dynamic IS Curve
y = y(+1) - (r - pi(+1))/sigma + g;

// 2. New Keynesian Phillips Curve (Calvo Pricing)
pi = beta*pi(+1) + kappa*y;

// 3. Zero Lower Bound (Nominal Rate Clamped at Floor: r = -r_ss)
r = -r_ss;

// 4. Aggregate Demand / Discount Factor Shock Process
g = rho_g*g(-1) + eps_g;
end;

steady_state_model;
pi = -r_ss;
y  = (1 - beta)*pi/kappa;
r  = -r_ss;
g  = 0;
end;
