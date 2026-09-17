// ==============================================================================
// neoclasico_prevision_perfecta.mod
// ==============================================================================
// Macroeconomía Avanzada (MAV) - ITAM, Otoño 2026
// Mazo A2: El modelo neoclásico de crecimiento con elección de trabajo
// Transcripción comentada del modelo de 20 líneas de la transparencia 808
// Calibración anual: beta=0.97, delta=0.06, alpha=0.33, gamma=1.0, n=1.0, K/Y=3.6
//
// Puede ejecutarse directamente en puremacro (Python puro, sin MATLAB/Dynare):
//   from puremacro.dsge import load_mod
//   m = load_mod("modelos/neoclasico_prevision_perfecta.mod")
// ==============================================================================

var y c k l a;
varexo e_a;

parameters alpha beta gamma delta n rho_a;
alpha = 0.33;      // Participación del capital en el producto
beta  = 0.97;      // Factor de descuento intertemporal anual
gamma = 1.0;       // Peso relativo del ocio en la función de utilidad
delta = 0.06;      // Tasa de depreciación anual del capital
n     = 1.0;       // Dotación total de tiempo normalizada
rho_a = 0.95;      // Persistencia autorregresiva de la productividad

model;
// 1. Condición intratemporal de eficiencia trabajo-ocio:
//    u_l / u_c = F_l  =>  gamma*c / (n - l) = (1 - alpha) * y / l
gamma*c/(n-l) = (1-alpha)*y/l;

// 2. Ecuación de Euler intertemporal:
//    u_c(t) = beta * E_t[ u_c(t+1) * (F_k(t+1) + 1 - delta) ]
c(+1)/(beta*c) = alpha*y(+1)/k + 1 - delta;

// 3. Restricción agregada de recursos (acumulación de capital):
//    c_t + k_t - (1 - delta)*k_{t-1} = y_t
c + k - (1-delta)*k(-1) = y;

// 4. Función de producción Cobb-Douglas:
//    y_t = exp(a_t) * k_{t-1}^alpha * l_t^(1 - alpha)
y = exp(a)*k(-1)^alpha*l^(1-alpha);

// 5. Proceso estocástico AR(1) de la productividad total de los factores (PTF):
a = rho_a*a(-1) + e_a;
end;

initval;
y = 0.870494;
c = 0.680939;
k = 3.159242;
l = 0.461355;
a = 0.0;
e_a = 0.0;
end;

steady;

shocks;
var e_a; stderr 0.02;
end;

stoch_simul(order=1, irf=40);
