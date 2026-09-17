// ==============================================================================
// rbc_canonico.mod
// ==============================================================================
// Macroeconomía Avanzada (MAV) - ITAM, Otoño 2026
// Mazo A4: Modelo Canónico de Ciclos Económicos Reales (RBC) con Trabajo Endógeno
// Calibración trimestral estándar (Kydland-Prescott / Cooley-Prescott):
//   beta=0.99, delta=0.025, alpha=0.33, nu=1.5, rho_a=0.95, sigma_a=0.007
//
// Variables endógenas (8):
//   y: producto, c: consumo, k: capital de fin de periodo, l: horas trabajadas,
//   i: inversión bruta, w: salario real, r: tasa de renta del capital, a: PTF log
//
// Innovación exógena:
//   e_a: choque estructural a la productividad (innovación de Solow)
//
// Puede ejecutarse directamente en puremacro (Python puro, sin MATLAB/Dynare):
//   from puremacro.dsge import load_mod
//   m = load_mod("modelos/rbc_canonico.mod")
//   res = m.stoch_simul(order=1, irf=40)
// ==============================================================================

var y c k l i w r a;
varexo e_a;

parameters alpha beta delta mu nu rho_a sigma_a;
alpha   = 0.33;      // Participación del capital
beta    = 0.99;      // Factor de descuento trimestral
delta   = 0.025;     // Tasa de depreciación trimestral (10% anual)
mu      = 0.822516;  // Calibrado analíticamente para objetivo l_ss = 0.33
nu      = 1.5;       // Curvatura del ocio (1/nu gobierna elasticidad de Frisch)
rho_a   = 0.95;      // Persistencia trimestral de la PTF
sigma_a = 0.007;     // Desviación estándar del choque a la PTF (0.7%)

model;
// 1. Condición intratemporal de oferta laboral (margen consumo-ocio):
//    mu * (1 - l_t)^(-nu) * c_t = w_t
mu * (1 - l)^(-nu) * c = w;

// 2. Ecuación de Euler intertemporal del consumo:
//    c_t^(-1) = beta * E_t[ c_{t+1}^(-1) * (r_{t+1} + 1 - delta) ]
c(+1)/(beta*c) = r(+1) + 1 - delta;

// 3. Demanda de capital de la empresa (precio marginal del capital):
//    r_t = alpha * y_t / k_{t-1}
r = alpha * y / k(-1);

// 4. Demanda de trabajo de la empresa (salario real igual a productividad marginal):
//    w_t = (1 - alpha) * y_t / l_t
w = (1 - alpha) * y / l;

// 5. Función de producción agregada Cobb-Douglas:
//    y_t = exp(a_t) * k_{t-1}^alpha * l_t^(1 - alpha)
y = exp(a) * k(-1)^alpha * l^(1-alpha);

// 6. Restricción de recursos en bienes de la economía cerrada:
//    y_t = c_t + i_t
y = c + i;

// 7. Ley de acumulación del acervo de capital físico:
//    k_t = (1 - delta) * k_{t-1} + i_t
k = (1 - delta)*k(-1) + i;

// 8. Proceso autorregresivo AR(1) de la tecnología:
a = rho_a * a(-1) + e_a;
end;

initval;
y = 0.995058;
c = 0.761184;
i = 0.233874;
k = 9.354978;
l = 0.330000;
w = 2.020299;
r = 0.035101;
a = 0.0;
e_a = 0.0;
end;

steady;

shocks;
var e_a; stderr 0.007;
end;

stoch_simul(order=1, irf=40);
