// rbc_mexico_dual.mod
// Modelo de Ciclos Económicos Reales con dos sectores: Formal e Informal
// Diseñado para la calibración a México (Mazo A4 - Proyecto)
//
// RECALIBRADO (8-ago-2026). La versión anterior traía A_i_bar = 0.4, que dejaba
// al sector informal en 1.3% de las horas y 1.0% del producto, y hacía que el
// choque informal e_i explicara 0.00% de la varianza del PIB: el "buffer" que
// promete la nota docente del final simplemente no existía en la corrida.
//
// OBJETIVO ALCANZADO (verificado corriendo el archivo con Dynare 7.1 + Octave
// y con Dynare 8 + MATLAB: las dos cadenas dan las mismas cifras al 4o decimal;
// el estado estacionario lo encuentra 'steady' a partir de los initval de
// abajo, y las condiciones de orden y rango quedan verificadas):
//   l_i/l_tot = 0.580   <- informalidad laboral 55-62% (Slides04mav, Slides08mav)
//   k/y_f     = 7.52    <- capital/producto FORMAL trimestral, valor estándar
//
// LÍMITES CONOCIDOS. Esto es un modelo de aula, no una calibración seria de
// México; conviene decirlo en clase antes de que el alumno lo descubra solo:
//   1) y_i/y = 0.532, muy por encima del ~22-24% de las cuentas nacionales. No
//      es un error de calibración: es una restricción del modelo. Las
//      ecuaciones 7 y 8 igualan los productos MARGINALES de las dos
//      actividades, así que el cociente de productividades MEDIAS queda
//      clavado en
//          (y_i/l_i)/(y_f/l_f) = (1-tau)*(1-alpha)/gamma = 0.825
//      cualquiera que sea A_i_bar. Con 58% de las horas en el sector informal
//      su participación en el producto no puede bajar de ~53%. Separar ambas
//      cifras exigiría una cuña extra (que el informal no cobre su producto
//      marginal, un costo de entrada al sector formal, capital informal, ...).
//   2) i/y = 0.088 frente a ~21% observado, y sd(i)/sd(y) = 6.7 SIN FILTRAR
//      (9.7 si se corre con hp_filter=1600, que es la base sobre la que se
//      mide el ~3 de los datos): consecuencia mecánica de (1), porque el
//      sector informal no usa capital y aquí pesa la mitad del producto.
//   3) Ante un choque formal positivo el consumo CAE en el impacto (-0.22%),
//      aunque es procíclico en promedio (corr(c,y) = 0.60): el auge de la
//      inversión se come el aumento de recursos.

var y_f, y_i, y, c, i, k, l_f, l_i, l_tot, a_f, a_i;
varexo e_f, e_i;

parameters alpha, beta, delta, nu, psi, rho_f, rho_i, gamma, A_i_bar, tau;

// Calibración estándar para México + Parámetros informales
alpha = 0.33;    // Elasticidad del capital en el sector formal
beta  = 0.99;    // Factor de descuento
delta = 0.025;   // Tasa de depreciación
nu    = 1.0;     // Inverso de la elasticidad de Frisch
psi   = 2.5;     // Peso del desutilidad laboral
rho_f = 0.90;    // Persistencia del choque formal
rho_i = 0.90;    // Persistencia del choque informal
gamma = 0.65;    // Rendimientos decrecientes del trabajo informal (no usan capital)
A_i_bar = 1.4838; // Escala de la tecnología informal, CALIBRADA para l_i/l_tot = 0.58
                  // (NO es comparable con la TFP formal: con gamma = 0.65 las
                  //  unidades difieren. Lo comparable es la productividad media
                  //  por hora, que sale 0.825 veces la del sector formal.)
tau   = 0.20;    // Impuesto o costo regulatorio al sector formal

model;
// 1. Restricción de recursos
c + i = (1-tau)*y_f + y_i;

// 2. Funciones de producción
y_f = exp(a_f) * k(-1)^alpha * l_f^(1-alpha);
y_i = exp(a_i) * A_i_bar * l_i^gamma;

// 3. Producción total
y = y_f + y_i;

// 4. Acumulación de capital (el capital sólo se usa en el sector formal)
k = (1-delta)*k(-1) + i;

// 5. Horas totales
l_tot = l_f + l_i;

// 6. Ecuación de Euler (Consumo)
1 = beta * (c / c(+1)) * ( (1-tau) * alpha * (y_f(+1)/k) + 1 - delta );

// 7. Oferta laboral al sector formal
psi * l_tot^nu * c = (1-tau) * (1-alpha) * (y_f/l_f);

// 8. Oferta laboral al sector informal (arbitraje entre sectores)
//    Ojo: 7 y 8 igualan productos MARGINALES; de ahí sale el límite 1) de la
//    cabecera sobre la participación del informal en el producto.
psi * l_tot^nu * c = gamma * (y_i/l_i);

// 9. Procesos AR(1) para la TFP
a_f = rho_f * a_f(-1) + e_f;
a_i = rho_i * a_i(-1) + e_i;
end;

// Valores iniciales = estado estacionario de esta calibración, tal como lo
// reporta 'steady' en rbc_mexico_dual.log (resuelto también de forma
// analítica: coinciden en las seis cifras que imprime Dynare).
initval;
k     = 4.6085660;
c     = 1.0726201;
l_f   = 0.2268187;
l_i   = 0.3131662;
l_tot = 0.5399849;
y_f   = 0.6127474;
y_i   = 0.6976363;
y     = 1.3103837;
i     = 0.1152141;
a_f   = 0;
a_i   = 0;
end;

steady;
check;

shocks;
var e_f = 0.01^2;
var e_i = 0.01^2;
end;

stoch_simul(order=1, irf=40);

// --- Nota Docente ---
// Fíjate en cómo una caída en a_f (recesión formal) reduce l_f y AUMENTA l_i:
// el sector informal actúa como buffer/seguro. Con esta calibración el efecto
// ya está ahí y se puede leer directamente en la salida (choque de +1% en e_f,
// respuesta de impacto en % del estado estacionario):
//     l_f  +2.22%      l_i  -0.76%      l_tot  +0.49%      y  +0.90%
// y en la matriz de correlaciones: corr(l_f, l_i) = -0.96, es decir,
// reasignación casi pura entre sectores.
//
// Ese es el mecanismo detrás de que el empleo TOTAL fluctúe poco: las
// desviaciones estándar teóricas (sin filtrar, en % de la media) son
//     l_f 8.11%   l_i 4.98%   l_tot 1.00%   y 2.98%   c 1.74%   i 19.97%
// o sea sd(l_tot)/sd(y) = 0.34. OJO con la comparación: el ~1 que se cita para
// EUA se mide sobre series filtradas, y este stoch_simul NO filtra; corriendo
// el mismo archivo con hp_filter=1600 la razón sube a 0.49, que es la cifra
// comparable con el dato. La brecha sigue siendo de un factor de dos. Como
// l_tot casi no se mueve, casi toda la fluctuación del producto aparece en la
// productividad laboral medida y/l_tot, que resulta muy procíclica y volátil.
//
// El producto informal es contracíclico (corr(y_i, y) = -0.15) y el choque
// informal e_i explica 16.7% de la varianza del PIB (antes, 0.00%).
//
// TODO(profesor): los números de arriba salen de la corrida de esta
// calibración; si tocas cualquier parámetro hay que volver a correr Dynare y
// actualizarlos, no dejarlos como están.
//
// SALIDA COMPROMETIDA: rbc_mexico_dual.log, +rbc_mexico_dual/ y
// rbc_mexico_dual/ se regeneraron el 8-ago-2026 a partir de ESTA versión del
// archivo, con Dynare 8-unstable-2026-05-26 + MATLAB R2026a. Todas las cifras
// citadas en esta nota y en la cabecera se leen directamente de ese .log,
// salvo las dos que llevan la etiqueta "hp_filter=1600" (sd(i)/sd(y) = 9.7 y
// sd(l_tot)/sd(y) = 0.49), que salen de correr este mismo archivo cambiando
// sólo esa opción del stoch_simul.
// Si tocas una ecuación o un parámetro, vuelve a correr el .mod; los 'initval'
// no cuentan, porque son sólo el punto de arranque del solver del EE.
