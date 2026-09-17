// ==============================================================================
// new_keynesian_3eq.mod
// ==============================================================================
// Macroeconomía Avanzada (MAV) - ITAM, Otoño 2026
// Mazo B1: Modelo Neokeynesiano Canónico de Tres Ecuaciones
// Precios rígidos (Calvo), IS dinámica prospectiva y Regla de Taylor
//
// Variables endógenas (5):
//   y_gap: brecha de producto (log-desviación respecto al nivel natural)
//   pi: tasa de inflación (desviación respecto a la meta)
//   i: tasa de interés nominal de política monetaria
//   rn: tasa de interés real natural (choque de demanda / preferencias)
//   u: choque de costos / empuje inflacionario (cost-push shock)
//
// Innovaciones estructurales (3):
//   eps_i: choque a la tasa de política monetaria (25 bps)
//   eps_rn: innovación a la tasa natural de interés
//   eps_u: innovación de costos (curva de Phillips)
//
// Puede ejecutarse directamente en puremacro (Python puro, sin MATLAB/Dynare):
//   from puremacro.dsge import load_mod
//   m = load_mod("modelos/new_keynesian_3eq.mod")
//   res = m.stoch_simul(order=1, irf=20)
// ==============================================================================

var y_gap pi i rn u;
varexo eps_i eps_rn eps_u;

parameters beta sigma kappa phi_pi phi_y rho_i rho_rn rho_u;
beta   = 0.99;      // Factor de descuento trimestral
sigma  = 1.0;       // Aversión relativa al riesgo / elasticidad intertemporal
kappa  = 0.1275;    // Pendiente de la curva de Phillips neokeynesiana (theta_calvo=0.75)
phi_pi = 1.5;       // Respuesta a la inflación en la regla de Taylor (Principio de Taylor > 1)
phi_y  = 0.125;     // Respuesta a la brecha de producto (0.5 anualizado / 4)
rho_i  = 0.75;      // Suavizamiento de la tasa de interés de política
rho_rn = 0.50;      // Persistencia del choque de tasa natural
rho_u  = 0.50;      // Persistencia del choque de empuje inflacionario

model;
// 1. Curva IS Dinámica Prospectiva (Euler log-linealizada):
//    y_gap_t = E_t[ y_gap_{t+1} ] - (1/sigma) * (i_t - E_t[ pi_{t+1} ] - rn_t)
y_gap = y_gap(+1) - (1/sigma) * (i - pi(+1) - rn);

// 2. Curva de Phillips Neokeynesiana (NKPC con escalonamiento de Calvo):
//    pi_t = beta * E_t[ pi_{t+1} ] + kappa * y_gap_t + u_t
pi = beta * pi(+1) + kappa * y_gap + u;

// 3. Regla de Política Monetaria tipo Taylor con suavizamiento:
//    i_t = rho_i * i_{t-1} + (1 - rho_i) * (phi_pi * pi_t + phi_y * y_gap_t) + eps_i
i = rho_i * i(-1) + (1 - rho_i) * (phi_pi * pi + phi_y * y_gap) + eps_i;

// 4. Proceso autorregresivo del choque de tasa natural (demanda):
rn = rho_rn * rn(-1) + eps_rn;

// 5. Proceso autorregresivo del choque de costos (oferta agregada):
u = rho_u * u(-1) + eps_u;
end;

initval;
y_gap = 0.0;
pi    = 0.0;
i     = 0.0;
rn    = 0.0;
u     = 0.0;
end;

steady;

shocks;
var eps_i;  stderr 0.0025;  // Choque de política de 25 puntos base trimestrales
var eps_rn; stderr 0.0050;  // Choque de demanda de 50 puntos base
var eps_u;  stderr 0.0050;  // Choque de costos de 50 puntos base
end;

stoch_simul(order=1, irf=20);
