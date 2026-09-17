/*
 * merz_andolfatto.mod — Ciclos reales con fricciones de búsqueda (Merz 1995; Andolfatto 1996)
 *
 * Problema del planificador DESTENDENCIALIZADO del mazo "Mecanismos de transmisión
 * no competitivos" (Slides08mav): utilidad Cobb-Douglas sobre consumo, ocio de los
 * empleados n(1-l) y ocio de los buscadores (1-n)(1-x); costo material de búsqueda
 * zeta*x^chi*(1-n); costo de vacantes kappa*v; emparejamiento A_m*[x(1-n)]^phi*v^(1-phi).
 *
 * Tendencias: Ahat (productividad) y Phat (población) ya absorbidas:
 *   1+g   = Ahat^(1/(1-alpha))*Phat          (factor bruto de la tendencia agregada)
 *   btil  = beta*Ahat^(muc*(1-sigma)/(1-alpha))*Phat^(1-sigma)
 * Choque estacionario de productividad: log(a_t) AR(1).
 *
 * Convención de fechado tipo Dynare: k y n son variables de FIN de periodo
 * (k = k_{t+1} y n = n_{t+1} de las transparencias; los insumos son k(-1), n(-1)).
 *
 * Punto pedagógico (puzzle de Shimer): comparar sd(log theta)/sd(log prod) del
 * modelo (0.92 en esta corrida: sd(lth)=0.0071 y sd(lpr)=0.0077 con filtro HP)
 * con el cociente de los datos JOLTS/OPHNFB (la literatura cita 15-20; el
 * ejercicio insignia del mazo mide ~25 sobre la muestra 2001-hoy, que es la de
 * la figura). El modelo se queda entre 16 y 27 veces corto: ese es el puzzle.
 *
 * SALIDA COMPROMETIDA: merz_andolfatto.log, +merz_andolfatto/ y
 * merz_andolfatto/ se regeneraron el 8-ago-2026 a partir de ESTA versión del
 * archivo, con Dynare 8-unstable-2026-05-26 + MATLAB R2026a (el mismo par que
 * generó las salidas anteriores). Si tocas una ecuación o un parámetro, vuelve
 * a correr el .mod: los .log del repositorio son la salida vigente y se leen
 * como tal.
 *
 * OJO con 'urate': NO es una tasa de desempleo. Ver la nota en las
 * definiciones observables, más abajo. La variable comparable con la u de las
 * transparencias es 'u_head' (desempleo por cabezas).
 */

var c k n x l v lam mum y m B a theta q p prod urate u_head lth lpr;
varexo e_a;

parameters alpha beta delta sigma s phi chi muc mun mux zeta kappa Am gA Phat G btil rho_a;

alpha = 0.33;      // participación del capital
beta  = 0.99;      // descuento (sin tendencias)
delta = 0.025;     // depreciación trimestral
sigma = 2.0;       // curvatura CRRA del compuesto Cobb-Douglas
s     = 0.10;      // tasa de separación trimestral
phi   = 0.50;      // elasticidad del matching respecto del DESEMPLEO (Hosios: gamma=phi)
chi   = 2.0;       // curvatura del costo material de búsqueda
muc   = 0.30;      // peso del consumo
mun   = 0.35;      // peso del ocio de los empleados
mux   = 1-muc-mun; // peso del ocio de los buscadores
zeta  = 0.50;      // escala del costo material de búsqueda
kappa = 0.30;      // costo de publicar una vacante
Am    = 1.00;      // eficiencia del emparejamiento
gA    = 1.0045;    // Ahat^(1/(1-alpha)): crecimiento per cápita tendencial
Phat  = 1.0025;    // crecimiento poblacional
G     = gA*Phat;   // 1+g
btil  = beta*(gA^(1-alpha))^(muc*(1-sigma)/(1-alpha))*Phat^(1-sigma);
rho_a = 0.95;

model;
// tecnología y emparejamiento
y = a*k(-1)^alpha*(n(-1)*l)^(1-alpha);
m = Am*(x*(1-n(-1)))^phi*v^(1-phi);

// restricciones
c + G*k - (1-delta)*k(-1) + zeta*x^chi*(1-n(-1)) + kappa*v = y;
Phat*n = (1-s)*n(-1) + m;

// compuesto de utilidad y CPO estáticas
B = c^muc*(n(-1)*(1-l))^mun*((1-n(-1))*(1-x))^mux;
lam = muc*B^(1-sigma)/c;
lam*(1-alpha)*y/l = mun*B^(1-sigma)/(1-l);                                   // horas
mum*phi*m/x = lam*zeta*chi*x^(chi-1)*(1-n(-1)) + mux*B^(1-sigma)/(1-x);      // intensidad de búsqueda
lam*kappa = mum*(1-phi)*m/v;                                                 // vacantes

// CPO dinámicas
lam*G = btil*lam(+1)*(alpha*y(+1)/k + 1-delta);                              // Euler del capital
mum*Phat = btil*( (mun/n - mux/(1-n))*B(+1)^(1-sigma)
                 + lam(+1)*((1-alpha)*y(+1)/n + zeta*x(+1)^chi)
                 + mum(+1)*((1-s) - phi*m(+1)/(1-n)) );                      // valor del empleo

// choque
log(a) = rho_a*log(a(-1)) + e_a;

// definiciones observables
theta = v/(x*(1-n(-1)));
q = m/v;
p = m/(x*(1-n(-1)));
prod = y/(n(-1)*l);

// 'urate' son los BUSCADORES EN UNIDADES DE EFICIENCIA sobre la fuerza laboral
// efectiva: pondera a los no ocupados por su intensidad de búsqueda x, no por
// cabezas. Sale 4.17% en el EE solo porque x = 0.0666 es pequeña, no porque el
// desempleo del modelo sea 4%. Como x es procíclica, esta serie es casi
// acíclica: corr(urate,y) = -0.11 y, sobre todo, corr(urate,v) = +0.75. NO se
// puede leer como la u de los datos ni usarla para dibujar la curva de
// Beveridge de Slides08mav.
urate = x*(1-n(-1))/(n(-1) + x*(1-n(-1)));

// Desempleo POR CABEZAS (no ocupación): esta sí es la variable comparable con
// la u de las transparencias. EE = 39.5%. Corrida de referencia (HP 1600):
//   corr(u_head, y)     = -0.78   (dato ~ -0.8)  -> contracíclica, como debe
//   corr(u_head, theta) = -0.79                  -> coherente con p'(theta)>0
//   corr(u_head, v)     = +0.06                  -> Beveridge PLANA
// Es decir: al medir el desempleo por cabezas desaparece la Beveridge
// CRECIENTE que da 'urate', pero el modelo sigue sin reproducir la pendiente
// negativa de los datos. Límite conocido: con un único choque de
// productividad y n muy suave (sd(u_head) = 0.2% del nivel) el desempleo casi
// no se mueve, que es la misma escasez de amplificación del puzzle de Shimer.
u_head = 1-n(-1);

lth = log(theta);
lpr = log(prod);
end;

initval;
n = 0.6048251897;
l = 0.4477749672;
x = 0.0665706552;
v = 0.1460950114;
c = 0.5057069601;
k = 5.1112001355;
y = 0.7140270075;
m = 0.0619945819;
B = 0.3916386199;
lam = 1.5147355296;
mum = 2.1417546262;
theta = 5.5534555721;
q = 0.4243442767;
p = 2.3565770881;
prod = 2.6364828936;
urate = 0.0416823045;
u_head = 0.3951748103;
a = 1.0;
lth = log(5.5534555721);
lpr = log(2.6364828936);
e_a = 0;
end;

steady;
check;

shocks;
var e_a; stderr 0.007;
end;

stoch_simul(order=1, irf=40, hp_filter=1600) y c urate u_head theta prod v n lth lpr;
