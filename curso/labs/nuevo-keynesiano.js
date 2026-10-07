/* Laboratorio «Nuevo keynesiano»: el modelo de tres ecuaciones del curso (modelos/new_keynesian_3eq.mod),
   sus impulsos-respuesta, el principio de Taylor y, aparte, la cota inferior cero con OccBin. */

/* ------------------------------------------------------------------ modelo (sin DOM) */
(function () {
  "use strict";
  const N = window.MAVNUM;

  // Calibración del curso: new_keynesian_3eq.mod y T02_C_new_keynesian_local.
  const BASE = {
    beta: 0.99, sigma: 1, theta: 0.75, phiPi: 1.5, phiY: 0.125, rhoI: 0.75,
    rho: { mon: 0, rn: 0.5, u: 0.5 },          // en el .mod el choque monetario es iid
  };
  const SIZE = { mon: 0.0025, rn: 0.005, u: 0.005 }; // desviaciones estándar del bloque shocks del .mod
  const KAPPA0 = 0.1275, THETA0 = 0.75;

  /* Pendiente de Calvo, κ = (1−θ)(1−βθ)/θ · (σ + φ). Como en T02_C, φ (inversa de Frisch) se fija
     para que θ = 0.75 reproduzca el κ = 0.1275 del .mod: φ = 0.4854. */
  const calvo = (th, beta) => ((1 - th) * (1 - beta * th)) / th;
  const FRISCH = KAPPA0 / calvo(THETA0, BASE.beta) - BASE.sigma;
  const kappaOf = (p) => calvo(p.theta, p.beta) * (p.sigma + FRISCH);

  /* Principio de Taylor (Bullard y Mitra 2002; Woodford 2003): con φπ, φy ≥ 0 y 0 ≤ ρi < 1 hay un
     único equilibrio estable si y solo si κ(φπ − 1) + (1 − β)φy > 0. El suavizamiento no cambia la
     frontera porque la regla está escrita en coeficientes de largo plazo, (1 − ρi)(φπ π + φy ỹ). */
  function determinacy(p) {
    const k = kappaOf(p);
    const value = k * (p.phiPi - 1) + (1 - p.beta) * p.phiY;
    return { kappa: k, value, determinate: value > 0, phiPiMin: 1 - ((1 - p.beta) * p.phiY) / k };
  }

  /* A0 x_t = A1 E_t x_{t+1} + A2 x_{t-1} + A3 e_t con x = [ỹ, π, i, rⁿ, u, v] y e = [εi, εrn, εu].
     v es el componente exógeno de la regla: v_t = ρv v_{t-1} + εi; con ρv = 0 es el εi iid del .mod. */
  const X = { y: 0, pi: 1, i: 2, rn: 3, u: 4, v: 5 };
  function system(p) {
    const n = 6, A0 = N.zeros(n, n), A1 = N.zeros(n, n), A2 = N.zeros(n, n), A3 = N.zeros(n, 3);
    const k = kappaOf(p), s = 1 / p.sigma;
    // IS: ỹ + (1/σ) i − (1/σ) rⁿ = E ỹ' + (1/σ) E π'
    A0[0][X.y] = 1; A0[0][X.i] = s; A0[0][X.rn] = -s; A1[0][X.y] = 1; A1[0][X.pi] = s;
    // NKPC: π − κ ỹ − u = β E π'
    A0[1][X.pi] = 1; A0[1][X.y] = -k; A0[1][X.u] = -1; A1[1][X.pi] = p.beta;
    // Taylor: i − (1−ρi)(φπ π + φy ỹ) − v = ρi i_{-1}
    A0[2][X.i] = 1; A0[2][X.pi] = -(1 - p.rhoI) * p.phiPi; A0[2][X.y] = -(1 - p.rhoI) * p.phiY; A0[2][X.v] = -1;
    A2[2][X.i] = p.rhoI;
    // choques
    A0[3][X.rn] = 1; A2[3][X.rn] = p.rho.rn; A3[3][1] = 1;
    A0[4][X.u] = 1; A2[4][X.u] = p.rho.u; A3[4][2] = 1;
    A0[5][X.v] = 1; A2[5][X.v] = p.rho.mon; A3[5][0] = 1;
    return { A0, A1, A2, A3 };
  }

  const SHOCK_INDEX = { mon: 0, rn: 1, u: 2 };

  /* Impulsos-respuesta a un choque de una desviación estándar, horizonte H (t = 0..H).
     Devuelve series en unidades de presentación: ỹ en %, π, i, r = i − E π' y rⁿ en puntos anualizados,
     más las series crudas del modelo en `raw`. */
  function irf(p, shock, H = 24) {
    const det = determinacy(p);
    if (!det.determinate) return { det, ok: false };
    const { A0, A1, A2, A3 } = system(p);
    const sol = N.solveRE(A0, A1, A2, A3, { tol: 1e-13, maxIter: 20000 });
    const e = [0, 0, 0]; e[SHOCK_INDEX[shock]] = SIZE[shock];
    const path = N.irf(sol.P, sol.Q, e, H + 2);            // un periodo extra para E_t π_{t+1}
    const col = (j) => path.map((x) => x[j]);
    const raw = { y: col(X.y), pi: col(X.pi), i: col(X.i), rn: col(X.rn), u: col(X.u), v: col(X.v) };
    const t = Array.from({ length: H + 1 }, (_, h) => h);
    const out = {
      det, ok: sol.ok, sol, t, raw,
      y: t.map((h) => 100 * raw.y[h]),
      pi: t.map((h) => 400 * raw.pi[h]),
      i: t.map((h) => 400 * raw.i[h]),
      r: t.map((h) => 400 * (raw.i[h] - raw.pi[h + 1])),
      rn: t.map((h) => 400 * raw.rn[h]),
    };
    return out;
  }

  /* Cocientes acumulados sobre 24 trimestres (t = 0..23), como el barrido de Calvo de T02_C:
     Σ 100 ỹ / Σ 400 π. */
  function cumulativeRatio(res, T = 24) {
    let sy = 0, sp = 0;
    for (let h = 0; h < T; h++) { sy += 100 * res.raw.y[h]; sp += 400 * res.raw.pi[h]; }
    return { sy, sp, ratio: sy / sp };
  }

  /* ---------------------------------------------------------------- cota inferior cero (OccBin)
     modelos/nk_zlb_ref.mod y nk_zlb_cons.mod, x = [y, π, r, g], e = [εr, εg]:
       y = E y' − (r − E π')/σ + g,  π = β E π' + κ y,  g = ρg g_{-1} + εg
       régimen de referencia: r = φπ π + φy y + εr;  régimen restringido: r = −r_ss.
     Algoritmo lineal por partes con previsión perfecta de Guerrieri e Iacoviello (2015), el mismo
     planteamiento que puremacro.dsge.solve_occbin. */
  const ZLB = { beta: 0.99, sigma: 1, kappa: 0.1, phiPi: 1.5, phiY: 0.125, rhoG: 0.8, rss: 0.01 };

  function zlbSystems(q) {
    const n = 4, Y = 0, P = 1, R = 2, G = 3;
    const make = () => ({ A0: N.zeros(n, n), A1: N.zeros(n, n), A2: N.zeros(n, n), A3: N.zeros(n, 2), c: new Array(n).fill(0) });
    const ref = make(), con = make();
    [ref, con].forEach((m) => {
      m.A0[0][Y] = 1; m.A0[0][R] = 1 / q.sigma; m.A0[0][G] = -1; m.A1[0][Y] = 1; m.A1[0][P] = 1 / q.sigma;
      m.A0[1][P] = 1; m.A0[1][Y] = -q.kappa; m.A1[1][P] = q.beta;
      m.A0[3][G] = 1; m.A2[3][G] = q.rhoG; m.A3[3][1] = 1;
    });
    ref.A0[2][R] = 1; ref.A0[2][P] = -q.phiPi; ref.A0[2][Y] = -q.phiY; ref.A3[2][0] = 1;
    con.A0[2][R] = 1; con.c[2] = -q.rss;
    return { ref, con, idx: { Y, P, R, G } };
  }

  /* Choque εg de tamaño `size` en t = 0 (el t = 1 de puremacro), horizonte H.
     Devuelve la senda lineal (sin restricción), la de OccBin y la secuencia de regímenes. */
  function zlbPath(size, H = 30, q = ZLB) {
    const { ref, con, idx } = zlbSystems(q);
    const sol = N.solveRE(ref.A0, ref.A1, ref.A2, ref.A3, { tol: 1e-13, maxIter: 20000 });
    const e0 = [0, size];
    const lin = N.irf(sol.P, sol.Q, e0, H);
    const shadow = (x) => q.phiPi * x[idx.P] + q.phiY * x[idx.Y];   // tasa nocional de la regla
    let regimes = new Array(H).fill(0), path = lin, iter = 0, converged = false;
    for (; iter < 60; iter++) {
      // recursión hacia atrás: x_t = P_t x_{t-1} + D_t + Q_t e_t, con régimen de referencia después de H
      const Pt = new Array(H), Dt = new Array(H), Qt = new Array(H);
      let Pn = sol.P, Dn = [0, 0, 0, 0];
      const last = regimes.lastIndexOf(1);
      for (let t = H - 1; t >= 0; t--) {
        if (t > last) { Pt[t] = sol.P; Dt[t] = [0, 0, 0, 0]; Qt[t] = sol.Q; Pn = sol.P; Dn = [0, 0, 0, 0]; continue; }
        const m = regimes[t] ? con : ref;
        const M = N.sub(m.A0, N.mul(m.A1, Pn));
        const rhs = N.mulv(m.A1, Dn).map((v, j) => v + m.c[j]);
        Pt[t] = N.solve(M, m.A2); Dt[t] = N.solve(M, rhs); Qt[t] = N.solve(M, m.A3);
        Pn = Pt[t]; Dn = Dt[t];
      }
      const next = []; let x = [0, 0, 0, 0];
      for (let t = 0; t < H; t++) {
        const qe = N.mulv(Qt[t], t === 0 ? e0 : [0, 0]);
        x = N.mulv(Pt[t], x).map((v, j) => v + Dt[t][j] + qe[j]);
        next.push(x);
      }
      // régimen restringido donde la regla (tasa nocional) pediría una tasa por debajo del piso
      const newReg = next.map((xt) => (shadow(xt) < -q.rss ? 1 : 0));
      path = next;
      if (newReg.every((r, t) => r === regimes[t])) { converged = true; break; }
      regimes = newReg;
    }
    const col = (arr, j, k) => arr.map((xt) => k * xt[j]);
    return {
      regimes, converged, iterations: iter + 1, binding: regimes.reduce((a, b) => a + b, 0),
      t: Array.from({ length: H }, (_, h) => h),
      occ: { y: col(path, idx.Y, 100), pi: col(path, idx.P, 400), i: path.map((xt) => 400 * (q.rss + xt[idx.R])), raw: path },
      lin: { y: col(lin, idx.Y, 100), pi: col(lin, idx.P, 400), i: lin.map((xt) => 400 * (q.rss + xt[idx.R])), raw: lin },
    };
  }

  window.MAVNK = { BASE, SIZE, FRISCH, calvo, kappaOf, determinacy, system, irf, cumulativeRatio, ZLB, zlbPath };
})();

/* ------------------------------------------------------------------ interfaz */
(function () {
  "use strict";
  if (typeof document === "undefined" || !window.MAV) return;
  const NK = window.MAVNK, T = MAV.t;
  const H = 24;

  // ---------------------------------------------------------------- estado
  const clone = (p) => Object.assign({}, p, { rho: Object.assign({}, p.rho) });
  const state = { shock: "mon", p: clone(NK.BASE) };
  // Estado inicial desde la URL (?choque=u&theta=0.9&phipi=3&phiy=0.125&rhoi=0.75&rho=0.5): enlaces a un experimento.
  (function fromURL() {
    const q = new URLSearchParams(location.search), num = (k, lo, hi) => { const v = parseFloat(q.get(k)); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null; };
    if (["mon", "rn", "u"].includes(q.get("choque"))) state.shock = q.get("choque");
    const set = (key, k, lo, hi) => { const v = num(k, lo, hi); if (v != null) state.p[key] = v; };
    set("theta", "theta", 0.05, 0.95); set("phiPi", "phipi", 0, 3); set("phiY", "phiy", 0, 1); set("rhoI", "rhoi", 0, 0.95);
    const r = num("rho", 0, 0.9); if (r != null) state.p.rho[state.shock] = r;
  })();
  let reference = clone(NK.BASE), refIsCourse = true;
  const same = (a, b) => ["theta", "phiPi", "phiY", "rhoI"].every((k) => Math.abs(a[k] - b[k]) < 1e-9)
    && ["mon", "rn", "u"].every((k) => Math.abs(a.rho[k] - b.rho[k]) < 1e-9);

  const SHOCKS = {
    mon: { label: { es: "Monetario", en: "Monetary" },
      note: { es: "Alza inesperada de 25 pb trimestrales en la regla (1 pp anualizado): una desviación estándar de εi en el .mod.",
              en: "Surprise 25 bp quarterly increase in the rule (1 pp annualized): one standard deviation of εi in the .mod file." } },
    rn: { label: { es: "Demanda", en: "Demand" },
      note: { es: "La tasa natural rⁿ sube 50 pb trimestrales: el gasto deseado aumenta (una desviación estándar de εrn).",
              en: "The natural rate rⁿ rises by 50 bp quarterly: desired spending increases (one standard deviation of εrn)." } },
    u: { label: { es: "Costos", en: "Cost-push" },
      note: { es: "Choque de costos de 50 pb trimestrales en la curva de Phillips (una desviación estándar de εu).",
              en: "A 50 bp quarterly cost-push shock in the Phillips curve (one standard deviation of εu)." } },
  };

  // ---------------------------------------------------------------- controles
  const ctl = document.getElementById("controls");
  const seg = MAV.segmented(ctl, { key: "choque", label: { es: "Choque", en: "Shock" }, value: state.shock,
    options: Object.entries(SHOCKS).map(([value, s]) => ({ value, label: s.label })) },
  (v) => { state.shock = v; rhoCtl.set(state.p.rho[v]); paintShockNote(); run(); });
  const shockNote = document.createElement("p"); shockNote.className = "ctl-note"; shockNote.style.margin = "-8px 0 16px";
  ctl.appendChild(shockNote);
  const paintShockNote = () => { shockNote.textContent = T(SHOCKS[state.shock].note); };
  MAV.onLang(paintShockNote); paintShockNote();

  const rhoCtl = MAV.slider(ctl, { key: "rho", label: { es: "Persistencia del choque ρ", en: "Shock persistence ρ" },
    min: 0, max: 0.9, step: 0.05, value: state.p.rho[state.shock],
    note: { es: "AR(1). En el .mod: 0 para el monetario, 0.5 para demanda y costos", en: "AR(1). In the .mod file: 0 for monetary, 0.5 for demand and cost-push" } },
  (v) => { state.p.rho[state.shock] = v; run(); });

  const h3 = (es, en) => { const h = document.createElement("h3"); h.className = "ctl-sub";
    h.innerHTML = `<span class="es">${es}</span><span class="en">${en}</span>`; ctl.appendChild(h); };
  h3("Precios rígidos", "Sticky prices");
  const thetaCtl = MAV.slider(ctl, { key: "theta", label: { es: "Rigidez de Calvo θ", en: "Calvo stickiness θ" },
    min: 0.05, max: 0.95, step: 0.01, value: state.p.theta,
    fmt: (v) => `${MAV.num(v, 2)} · κ = ${MAV.num(NK.kappaOf(Object.assign({}, state.p, { theta: v })), 3)}`,
    note: { es: "κ = (1−θ)(1−βθ)/θ · (σ+φ), con φ = 0.485 para que θ = 0.75 dé el κ = 0.1275 del curso",
            en: "κ = (1−θ)(1−βθ)/θ · (σ+φ), with φ = 0.485 so that θ = 0.75 gives the course's κ = 0.1275" } },
  (v) => { state.p.theta = v; run(); });

  h3("Regla de Taylor", "Taylor rule");
  const phiPiCtl = MAV.slider(ctl, { key: "phipi", label: { es: "Respuesta a la inflación φπ", en: "Response to inflation φπ" },
    min: 0, max: 3, step: 0.05, value: state.p.phiPi }, (v) => { state.p.phiPi = v; run(); });
  const phiYCtl = MAV.slider(ctl, { key: "phiy", label: { es: "Respuesta a la brecha φy", en: "Response to the output gap φy" },
    min: 0, max: 1, step: 0.025, value: state.p.phiY, fmt: (v) => MAV.num(v, 3),
    note: { es: "0.125 trimestral = 0.5 anualizado", en: "0.125 quarterly = 0.5 annualized" } }, (v) => { state.p.phiY = v; run(); });
  const rhoICtl = MAV.slider(ctl, { key: "rhoi", label: { es: "Suavizamiento ρi", en: "Interest smoothing ρi" },
    min: 0, max: 0.95, step: 0.05, value: state.p.rhoI }, (v) => { state.p.rhoI = v; run(); });

  const btns = document.createElement("div"); btns.className = "btn-row";
  const bReset = document.createElement("button"); bReset.type = "button"; bReset.className = "btn";
  const bPin = document.createElement("button"); bPin.type = "button"; bPin.className = "btn";
  btns.append(bReset, bPin); ctl.appendChild(btns);
  const pinNote = document.createElement("p"); pinNote.className = "ctl-note"; pinNote.style.marginTop = "8px"; ctl.appendChild(pinNote);
  const paintButtons = () => {
    bReset.textContent = T({ es: "Calibración del curso", en: "Course calibration" });
    bPin.textContent = T({ es: "Fijar como referencia", en: "Pin as reference" });
    pinNote.textContent = T(refIsCourse
      ? { es: "La referencia (línea discontinua) es la calibración del curso.", en: "The reference (dashed line) is the course calibration." }
      : { es: "Referencia fijada: " + describe(reference), en: "Pinned reference: " + describe(reference) });
  };
  function describe(p) {
    return `θ = ${MAV.num(p.theta, 2)}, φπ = ${MAV.num(p.phiPi, 2)}, φy = ${MAV.num(p.phiY, 3)}, ρi = ${MAV.num(p.rhoI, 2)}`;
  }
  bReset.addEventListener("click", () => {
    state.p = clone(NK.BASE); reference = clone(NK.BASE); refIsCourse = true;
    rhoCtl.set(state.p.rho[state.shock]); thetaCtl.set(state.p.theta); phiPiCtl.set(state.p.phiPi);
    phiYCtl.set(state.p.phiY); rhoICtl.set(state.p.rhoI); run();
  });
  bPin.addEventListener("click", () => {
    if (!NK.determinacy(state.p).determinate) return;
    reference = clone(state.p); refIsCourse = same(reference, NK.BASE); run();
  });
  MAV.onLang(paintButtons);

  // ---------------------------------------------------------------- gráficas
  const tiles = document.getElementById("tiles");
  const notice = document.getElementById("indeterminate");
  const irfBox = document.getElementById("irf-charts");
  const chY = MAV.lineChart(document.getElementById("chart-y"), { series: [] });
  const chPi = MAV.lineChart(document.getElementById("chart-pi"), { series: [] });
  const chR = MAV.lineChart(document.getElementById("chart-rates"), { series: [] });
  const detMap = determinacyMap(document.getElementById("chart-det"));

  const xLabel = { es: "Trimestre", en: "Quarter" };
  const common = { xType: "linear", xLabel, zero: true, height: 250, labelRoom: 72,
    xFmt: (v) => String(v), tipFmt: (v) => MAV.num(v, 3), digits: 4 };

  function run() {
    const p = state.p, det = NK.determinacy(p), k = det.kappa;
    const res = NK.irf(p, state.shock, H);
    const showRef = !same(p, reference) && NK.determinacy(reference).determinate;
    const ref = showRef ? NK.irf(reference, state.shock, H) : null;
    bPin.disabled = !det.determinate || !showRef;
    paintButtons();
    const refName = refIsCourse ? { es: "Referencia: calibración del curso", en: "Reference: course calibration" }
                                : { es: "Referencia fijada", en: "Pinned reference" };
    const curName = { es: "Parámetros actuales", en: "Current parameters" };
    const dur = 1 / (1 - p.theta);

    // tiles
    const tKappa = { label: { es: "Pendiente de Phillips κ", en: "Phillips slope κ" }, value: MAV.num(k, 3),
      note: { es: `precios fijos ${MAV.num(dur, 1)} trimestres en promedio`, en: `prices fixed ${MAV.num(dur, 1)} quarters on average` } };
    const tDet = det.determinate
      ? { label: { es: "Equilibrio", en: "Equilibrium" }, value: T({ es: "Determinado", en: "Determinate" }),
          note: { es: `κ(φπ−1)+(1−β)φy = ${MAV.num(det.value, 4)} > 0`, en: `κ(φπ−1)+(1−β)φy = ${MAV.num(det.value, 4)} > 0` } }
      : { label: { es: "Equilibrio", en: "Equilibrium" }, value: T({ es: "Indeterminado", en: "Indeterminate" }),
          note: { es: `se requiere φπ > ${MAV.num(det.phiPiMin, 3)}`, en: `requires φπ > ${MAV.num(det.phiPiMin, 3)}` } };
    if (!det.determinate) {
      MAV.tiles(tiles, [tKappa,
        { label: { es: "Pico de la inflación", en: "Peak inflation" }, value: "—" },
        { label: ratioLabel(), value: "—" }, tDet]);
      irfBox.style.display = "none"; notice.style.display = "";
      notice.innerHTML = "";
      const h = document.createElement("p"); h.className = "notice-title";
      h.textContent = T({ es: "Indeterminado: no hay un equilibrio estable único", en: "Indeterminate: no unique stable equilibrium" });
      const b = document.createElement("p");
      b.textContent = T({
        es: `Con θ = ${MAV.num(p.theta, 2)}, φπ = ${MAV.num(p.phiPi, 2)} y φy = ${MAV.num(p.phiY, 3)}, κ(φπ − 1) + (1 − β)φy = ${MAV.num(det.value, 4)}, que no es positivo. A largo plazo la tasa nominal sube menos que la inflación, así que la tasa real baja justo cuando la inflación sube: la demanda se calienta y la expectativa se cumple sola. Muchas trayectorias estables son compatibles con el mismo choque (manchas solares), de modo que no existe una única función de impulso-respuesta que graficar. Sube φπ por encima de ${MAV.num(det.phiPiMin, 3)}.`,
        en: `With θ = ${MAV.num(p.theta, 2)}, φπ = ${MAV.num(p.phiPi, 2)} and φy = ${MAV.num(p.phiY, 3)}, κ(φπ − 1) + (1 − β)φy = ${MAV.num(det.value, 4)}, which is not positive. In the long run the nominal rate rises by less than inflation, so the real rate falls exactly when inflation rises: demand heats up and the expectation fulfils itself. Many stable paths are consistent with the same shock (sunspots), so there is no unique impulse response to plot. Raise φπ above ${MAV.num(det.phiPiMin, 3)}.`,
      });
      notice.append(h, b);
      detMap.update(p);
      return;
    }
    irfBox.style.display = ""; notice.style.display = "none";

    let peak = 0; res.pi.forEach((v, h) => { if (Math.abs(v) > Math.abs(peak)) peak = v; });
    const peakT = res.pi.findIndex((v) => v === peak);
    const cr = NK.cumulativeRatio(res);
    const ratio = state.shock === "u" ? -cr.ratio : cr.ratio;
    MAV.tiles(tiles, [tKappa,
      { label: { es: "Pico de la inflación", en: "Peak inflation" }, value: (peak > 0 ? "+" : "") + MAV.num(peak, 2),
        note: { es: `pp anualizados, trimestre ${peakT}`, en: `annualized pp, quarter ${peakT}` } },
      { label: ratioLabel(), value: MAV.num(ratio, 2), note: ratioNote() },
      tDet]);

    const t = res.t;
    const pair = (key) => {
      const s = [{ name: curName, short: { es: "actual", en: "current" }, x: t, y: res[key], style: 0 }];
      if (ref) s.push({ name: refName, short: { es: "referencia", en: "reference" }, x: t, y: ref[key], style: 1 });
      return s;
    };
    chY.update(Object.assign({}, common, {
      title: { es: "Brecha del producto", en: "Output gap" },
      sub: { es: "% de desviación respecto al producto natural", en: "% deviation from natural output" },
      series: pair("y"), csvName: `nk-brecha-${state.shock}`, yFmt: (v) => MAV.num(v, 2) }));
    chPi.update(Object.assign({}, common, {
      title: { es: "Inflación", en: "Inflation" },
      sub: { es: "puntos porcentuales anualizados (400 × π trimestral)", en: "annualized percentage points (400 × quarterly π)" },
      series: pair("pi"), csvName: `nk-inflacion-${state.shock}`, yFmt: (v) => MAV.num(v, 2) }));
    const rates = [
      { name: { es: "Nominal i", en: "Nominal i" }, short: "i", x: t, y: res.i, style: 0 },
      { name: { es: "Real r = i − E π′", en: "Real r = i − E π′" }, short: "r", x: t, y: res.r, style: 1 },
    ];
    if (state.shock === "rn") rates.push({ name: { es: "Natural rⁿ", en: "Natural rⁿ" }, short: "rⁿ", x: t, y: res.rn, style: 2 });
    chR.update(Object.assign({}, common, {
      title: { es: "Tasas de interés", en: "Interest rates" },
      sub: { es: "desviación del estado estacionario, puntos anualizados; parámetros actuales", en: "deviation from steady state, annualized points; current parameters" },
      series: rates, csvName: `nk-tasas-${state.shock}`, yFmt: (v) => MAV.num(v, 2) }));
    detMap.update(p);
  }

  function ratioLabel() {
    return state.shock === "mon" ? { es: "Razón de sacrificio", en: "Sacrifice ratio" }
      : state.shock === "u" ? { es: "Disyuntiva producto–inflación", en: "Output–inflation trade-off" }
      : { es: "Brecha por punto de inflación", en: "Output gap per inflation point" };
  }
  function ratioNote() {
    return state.shock === "mon"
      ? { es: "brecha perdida por punto de desinflación, acumuladas en 24 trim.", en: "output gap lost per point of disinflation, cumulated over 24 qtrs" }
      : state.shock === "u"
      ? { es: "brecha perdida por punto de inflación extra, acumuladas en 24 trim.", en: "output gap lost per point of extra inflation, cumulated over 24 qtrs" }
      : { es: "mismo signo: el choque no crea disyuntiva, acumuladas en 24 trim.", en: "same sign: no trade-off, cumulated over 24 qtrs" };
  }

  /* ---------------------------------------------------------------- mapa de determinación
     Plano (φπ, φy) con la región indeterminada sombreada, la frontera κ(φπ−1)+(1−β)φy = 0, el punto
     actual y la calibración del curso. Lectura al pasar el cursor, tabla y CSV como en MAV.lineChart. */
  function determinacyMap(el) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const plot = el.querySelector(".chart"), tip = el.querySelector(".tooltip"), legend = el.querySelector(".legend");
    const tableBtn = el.querySelector('[data-act="table"]'), csvBtn = el.querySelector('[data-act="csv"]'), table = el.querySelector(".table-view");
    const paintTableBtn = () => { tableBtn.textContent = T(table.classList.contains("on") ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" }); };
    tableBtn.addEventListener("click", () => { table.classList.toggle("on"); tableBtn.setAttribute("aria-expanded", String(table.classList.contains("on"))); paintTableBtn(); });
    tableBtn.setAttribute("aria-expanded", "false");
    const svg = d3.select(plot).insert("svg", ".tooltip");
    let p = NK.BASE;

    function draw() {
      paintTableBtn();
      const width = Math.max(300, plot.clientWidth || 640), height = 260;
      const m = { t: 14, r: 18, b: 42, l: 54 };
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img"); svg.selectAll("*").remove();
      const det = NK.determinacy(p), k = det.kappa, slope = (1 - p.beta) / k;   // φπ* = 1 − slope·φy
      const x = d3.scaleLinear().domain([0, 3]).range([m.l, width - m.r]);
      const y = d3.scaleLinear().domain([0, 1]).range([height - m.b, m.t]);
      svg.append("title").text(T({ es: "Región de determinación en el plano (φπ, φy)", en: "Determinacy region in the (φπ, φy) plane" }));
      const clipId = "detclip-" + Math.random().toString(36).slice(2, 8);
      svg.append("clipPath").attr("id", clipId).append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b);
      const fr = (fy) => 1 - slope * fy;
      svg.append("path").attr("class", "shade").attr("clip-path", `url(#${clipId})`)
        .attr("d", d3.line()([[x(-1), y(0)], [x(fr(0)), y(0)], [x(fr(1)), y(1)], [x(-1), y(1)]].map(([a, b]) => [a, b])));
      svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickSize(-(width - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`)
        .call(d3.axisBottom(x).ticks(6).tickFormat((v) => MAV.num(v, 1)).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat((v) => MAV.num(v, 2)).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
      svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end").text("φπ");
      svg.append("text").attr("class", "axis-title").attr("x", m.l).attr("y", m.t - 2).attr("dy", "-0.2em").text("φy");
      svg.append("line").attr("class", "zero").attr("stroke-dasharray", "3 3").attr("x1", x(1)).attr("x2", x(1)).attr("y1", m.t).attr("y2", height - m.b);
      svg.append("path").attr("class", "series").attr("clip-path", `url(#${clipId})`).attr("stroke", MAV.style(0).stroke)
        .attr("d", d3.line()([[x(fr(0)), y(0)], [x(fr(1)), y(1)]]));
      // rótulos de región
      const narrow = width < 480;
      const indLabel = svg.append("text").attr("class", "alabel").text(T({ es: "Indeterminado", en: "Indeterminate" }));
      if (narrow) indLabel.attr("transform", `translate(${x(0.2)},${y(0.5)}) rotate(-90)`).attr("text-anchor", "middle");
      else indLabel.attr("x", x(0.08)).attr("y", y(0.9));
      svg.append("text").attr("class", "alabel").attr("x", x(2.92)).attr("y", y(0.9)).attr("text-anchor", "end").text(T({ es: "Determinado", en: "Determinate" }));
      if (!narrow) {   // rótulo de φπ = 1 a la altura más alejada del punto actual
        const lab = [0.93, 0.5, 0.07].reduce((best, h) => (Math.abs(h - p.phiY) > Math.abs(best - p.phiY) ? h : best), 0.93);
        const clash = Math.abs(p.phiPi - 1) < 0.35 && Math.abs(lab - p.phiY) < 0.12;
        if (!clash) svg.append("text").attr("class", "alabel").attr("x", x(1) + 5).attr("y", y(lab)).text("φπ = 1");
      }
      // puntos
      const base = NK.BASE, cur = [p.phiPi, p.phiY];
      const isBase = Math.abs(p.phiPi - base.phiPi) < 1e-9 && Math.abs(p.phiY - base.phiY) < 1e-9;
      if (!isBase) {
        svg.append("circle").attr("cx", x(base.phiPi)).attr("cy", y(base.phiY)).attr("r", 5).attr("fill", "var(--bg)").attr("stroke", "var(--s2)").attr("stroke-width", 2);
      }
      svg.append("circle").attr("class", "dot").attr("cx", x(cur[0])).attr("cy", y(cur[1])).attr("r", 5.5).attr("fill", "var(--s1)");
      const lx = x(cur[0]) + (cur[0] > 2.3 ? -9 : 9), anchor = cur[0] > 2.3 ? "end" : "start";
      svg.append("text").attr("class", "dlabel").attr("x", lx).attr("y", y(cur[1]) + (cur[1] > 0.85 ? 16 : -9)).attr("text-anchor", anchor)
        .text(T({ es: "actual", en: "current" }));

      legend.innerHTML = "";
      const key = (svgHtml, text) => { const s = document.createElement("span"); s.className = "key"; s.innerHTML = svgHtml;
        const t = document.createElement("span"); t.textContent = text; s.appendChild(t); legend.appendChild(s); };
      key(`<svg width="26" height="10" aria-hidden="true"><line x1="0" x2="26" y1="5" y2="5" stroke="var(--s1)" stroke-width="2"/></svg>`,
        T({ es: "frontera κ(φπ−1)+(1−β)φy = 0", en: "frontier κ(φπ−1)+(1−β)φy = 0" }));
      key(`<svg width="16" height="12" aria-hidden="true"><rect x="1" y="1" width="14" height="10" fill="var(--band)" stroke="var(--rule)"/></svg>`,
        T({ es: "región indeterminada", en: "indeterminate region" }));
      key(`<svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="var(--s1)"/></svg>`, T({ es: "parámetros actuales", en: "current parameters" }));
      if (!isBase) key(`<svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" fill="var(--bg)" stroke="var(--s2)" stroke-width="2"/></svg>`,
        T({ es: "calibración del curso (1.5, 0.125)", en: "course calibration (1.5, 0.125)" }));

      // lectura al pasar el cursor
      const ring = svg.append("circle").attr("r", 4).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).style("display", "none");
      const show = (px, py) => {
        const a = Math.max(0, Math.min(3, x.invert(px))), b = Math.max(0, Math.min(1, y.invert(py)));
        ring.attr("cx", x(a)).attr("cy", y(b)).style("display", null);
        const val = k * (a - 1) + (1 - p.beta) * b;
        tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head";
        head.textContent = val > 0 ? T({ es: "Determinado", en: "Determinate" }) : T({ es: "Indeterminado", en: "Indeterminate" });
        tip.appendChild(head);
        [["φπ", MAV.num(a, 2)], ["φy", MAV.num(b, 3)], ["κ(φπ−1)+(1−β)φy", MAV.num(val, 4)]].forEach(([l, v]) => {
          const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "auto 1fr";
          const bb = document.createElement("b"); bb.textContent = v; const s = document.createElement("span"); s.textContent = l;
          r.append(bb, s); tip.appendChild(r);
        });
        const sc = plot.clientWidth / width, tw = tip.offsetWidth || 170, left = x(a) * sc;
        tip.style.left = (left + 14 + tw > plot.clientWidth ? left - tw - 14 : left + 14) + "px";
        tip.style.top = Math.max(0, y(b) * sc - 30) + "px"; tip.classList.add("on");
      };
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b)
        .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", T({ es: "Explorar valores", en: "Explore values" }))
        .on("pointermove", (ev) => { const [px, py] = d3.pointer(ev); show(px, py); })
        .on("pointerleave", () => { ring.style("display", "none"); tip.classList.remove("on"); })
        .on("focus", () => show(x(p.phiPi), y(p.phiY)))
        .on("blur", () => { ring.style("display", "none"); tip.classList.remove("on"); });

      el.querySelector(".chart-title").textContent = T({ es: "El principio de Taylor", en: "The Taylor principle" });
      el.querySelector(".chart-sub").textContent = T({
        es: `Región de determinación con θ = ${MAV.num(p.theta, 2)} (κ = ${MAV.num(k, 3)}): hace falta φπ > 1 − (1−β)φy/κ`,
        en: `Determinacy region with θ = ${MAV.num(p.theta, 2)} (κ = ${MAV.num(k, 3)}): it takes φπ > 1 − (1−β)φy/κ` });
      const header = ["φy", T({ es: "φπ mínimo para la determinación", en: "minimum φπ for determinacy" })];
      const rows = d3.range(0, 1.0001, 0.125).map((fy) => [MAV.num(fy, 3), MAV.num(fr(fy), 4)]);
      const tb = document.createElement("table"), th = tb.createTHead().insertRow();
      header.forEach((hd) => { const c = document.createElement("th"); c.textContent = hd; th.appendChild(c); });
      const body = tb.createTBody(); rows.forEach((r) => { const tr = body.insertRow(); r.forEach((c) => { tr.insertCell().textContent = c; }); });
      table.textContent = ""; table.appendChild(tb);
      csvBtn.onclick = () => {
        const raw = [header, ...d3.range(0, 1.0001, 0.125).map((fy) => [fy, fr(fy)])].map((r) => r.join(",")).join("\n");
        const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([raw], { type: "text/csv" }));
        a.download = "nk-frontera-determinacion.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      };
    }
    let raf = 0; new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(plot);
    MAV.onLang(draw);
    return { update(next) { p = next; draw(); } };
  }

  MAV.onLang(run);
  run();

  /* ---------------------------------------------------------------- cota inferior cero */
  const zctl = document.getElementById("controls-zlb");
  const zstate = { size: -2.5, rssAnnual: 4 };
  MAV.slider(zctl, { key: "zsize", label: { es: "Choque de demanda εg", en: "Demand shock εg" },
    min: -3.5, max: 3.5, step: 0.25, value: zstate.size, fmt: (v) => (v > 0 ? "+" : "") + MAV.num(v, 2) + "%",
    note: { es: "% del producto, en el trimestre 0; persistencia ρg = 0.8", en: "% of output, in quarter 0; persistence ρg = 0.8" } },
  (v) => { zstate.size = v; runZ(); });
  MAV.slider(zctl, { key: "zrss", label: { es: "Tasa nominal de largo plazo", en: "Steady-state nominal rate" },
    min: 2, max: 6, step: 0.5, value: zstate.rssAnnual, fmt: (v) => MAV.num(v, 1) + "%",
    note: { es: "anual; es la distancia hasta el piso de cero (4% en nk_zlb_ref.mod)", en: "annual; it is the distance to the zero floor (4% in nk_zlb_ref.mod)" } },
  (v) => { zstate.rssAnnual = v; runZ(); });
  const zNote = document.createElement("p"); zNote.className = "ctl-note"; zctl.appendChild(zNote);
  const paintZNote = () => { zNote.textContent = T({
    es: "Calibración de nk_zlb_ref.mod: β = 0.99, σ = 1, κ = 0.1, φπ = 1.5, φy = 0.125, sin suavizamiento. Es otro modelo que el de arriba: los controles de arriba no lo mueven.",
    en: "Calibration from nk_zlb_ref.mod: β = 0.99, σ = 1, κ = 0.1, φπ = 1.5, φy = 0.125, no smoothing. It is a different model from the one above: the controls above do not affect it." }); };
  MAV.onLang(paintZNote); paintZNote();

  const ztiles = document.getElementById("tiles-zlb");
  const zY = MAV.lineChart(document.getElementById("chart-zlb-y"), { series: [] });
  const zPi = MAV.lineChart(document.getElementById("chart-zlb-pi"), { series: [] });
  const zI = MAV.lineChart(document.getElementById("chart-zlb-i"), { series: [] });

  function runZ() {
    const q = Object.assign({}, NK.ZLB, { rss: zstate.rssAnnual / 400 });
    const s = zstate.size / 100;
    const z = NK.zlbPath(s, 30, q);
    const mirror = NK.zlbPath(-s, 30, q);
    const t = z.t;
    const spells = [];
    z.regimes.forEach((r, h) => { if (r && (h === 0 || !z.regimes[h - 1])) spells.push({ from: h, to: h }); if (r) spells[spells.length - 1].to = h; });
    const shades = spells.map((sp, j) => ({ from: Math.max(0, sp.from - 0.5), to: Math.min(t.length - 1, sp.to + 0.5),
      label: j === 0 ? { es: "en la cota", en: "at the bound" } : null }));
    const occName = { es: "Con cota cero (OccBin)", en: "With the zero bound (OccBin)" };
    const linName = { es: "Lineal, sin cota", en: "Linear, no bound" };
    const y0 = z.occ.y[0], y0lin = z.lin.y[0], ym = mirror.occ.y[0];
    const asym = s === 0 ? null : Math.abs(s < 0 ? y0 / ym : ym / y0);
    MAV.tiles(ztiles, [
      { label: { es: "Brecha en el impacto, con cota", en: "Impact output gap, with the bound" }, value: MAV.num(y0, 2) + "%",
        note: { es: `lineal: ${MAV.num(y0lin, 2)}%`, en: `linear: ${MAV.num(y0lin, 2)}%` } },
      { label: { es: "Trimestres en la cota", en: "Quarters at the bound" }, value: String(z.binding),
        note: z.converged ? { es: "duración endógena", en: "endogenous duration" } : { es: "OccBin no convergió", en: "OccBin did not converge" } },
      { label: { es: "Asimetría caída / auge", en: "Bust / boom asymmetry" }, value: asym == null ? "—" : MAV.num(asym, 2) + "×",
        note: { es: "caída con −|ε| entre auge con +|ε|, en el impacto", en: "slump with −|ε| over boom with +|ε|, on impact" } },
      { label: { es: "Inflación en el impacto, con cota", en: "Impact inflation, with the bound" }, value: MAV.num(z.occ.pi[0], 2),
        note: { es: `pp anualizados; lineal: ${MAV.num(z.lin.pi[0], 2)}`, en: `annualized pp; linear: ${MAV.num(z.lin.pi[0], 2)}` } },
    ]);
    const zc = Object.assign({}, common, { shades, height: 230 });
    zY.update(Object.assign({}, zc, { title: { es: "Brecha del producto", en: "Output gap" }, sub: { es: "% de desviación", en: "% deviation" },
      series: [{ name: occName, short: "OccBin", x: t, y: z.occ.y, style: 0 }, { name: linName, short: { es: "lineal", en: "linear" }, x: t, y: z.lin.y, style: 1 }],
      csvName: "zlb-brecha", yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2) }));
    zPi.update(Object.assign({}, zc, { title: { es: "Inflación", en: "Inflation" }, sub: { es: "puntos porcentuales anualizados", en: "annualized percentage points" },
      series: [{ name: occName, short: "OccBin", x: t, y: z.occ.pi, style: 0 }, { name: linName, short: { es: "lineal", en: "linear" }, x: t, y: z.lin.pi, style: 1 }],
      csvName: "zlb-inflacion", yFmt: (v) => MAV.num(v, 1), tipFmt: (v) => MAV.num(v, 2) }));
    zI.update(Object.assign({}, zc, { title: { es: "Tasa nominal de política", en: "Nominal policy rate" },
      sub: { es: "nivel, % anual; la lineal la lleva por debajo de cero", en: "level, % per year; the linear solution takes it below zero" },
      hlines: [{ y: 0, label: { es: "cota cero", en: "zero bound" } }], zero: false,
      series: [{ name: occName, short: "OccBin", x: t, y: z.occ.i, style: 0 }, { name: linName, short: { es: "lineal", en: "linear" }, x: t, y: z.lin.i, style: 1 }],
      csvName: "zlb-tasa", yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2) }));
  }
  MAV.onLang(runZ);
  runZ();
})();
