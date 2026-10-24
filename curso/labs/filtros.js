/* Laboratorio «Filtros»: HP frente a Hamilton, y el ciclo en tiempo real. */
(async function () {
  "use strict";
  const N = window.MAVNUM;
  const T = MAV.t;
  const data = await MAV.json("../data/filtros.json");

  const state = { code: "USA", loglam: Math.log10(1600), h: 8, p: 4, cut: null };
  const ctl = document.getElementById("controls");

  MAV.select(ctl, {
    key: "pais", label: { es: "País", en: "Country" }, value: state.code,
    options: Object.entries(data.paises).map(([code, s]) => ({ value: code, label: s.nombre })),
  }, (v) => { state.code = v; state.cut = null; setupCut(); run(); });

  const LAMBDAS = [10, 50, 100, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 129600, 400000];
  const fmtInt = (v) => Math.round(v).toLocaleString(MAV.lang === "es" ? "es-MX" : "en-US");
  state.loglam = Math.log10(1600);
  const lam = MAV.slider(ctl, {
    key: "lambda", label: { es: "λ de Hodrick–Prescott", en: "Hodrick–Prescott λ" },
    min: 0, max: LAMBDAS.length - 1, step: 1, value: LAMBDAS.indexOf(1600),
    fmt: (i) => fmtInt(LAMBDAS[i]),
    note: { es: "1,600 es el estándar trimestral; 129,600 es el valor de Ravn y Uhlig para comparar con datos anuales",
            en: "1,600 is the quarterly standard; 129,600 is Ravn and Uhlig's value for comparison with annual data" },
  }, (i) => { state.loglam = Math.log10(LAMBDAS[i]); run(); });

  const btns = document.createElement("div"); btns.className = "btn-row"; btns.style.margin = "-8px 0 16px";
  const presetButtons = [100, 1600, 129600].map((v) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "btn";
    b.addEventListener("click", () => { state.loglam = Math.log10(v); lam.set(LAMBDAS.indexOf(v)); run(); });
    btns.appendChild(b); return [b, v];
  });
  const paintPresets = () => presetButtons.forEach(([b, v]) => { b.textContent = "λ = " + fmtInt(v); });
  MAV.onLang(paintPresets); paintPresets();
  ctl.appendChild(btns);

  MAV.slider(ctl, { key: "h", label: { es: "Hamilton: horizonte h (trimestres)", en: "Hamilton: horizon h (quarters)" },
    min: 1, max: 16, step: 1, value: state.h }, (v) => { state.h = v; run(); });
  MAV.slider(ctl, { key: "p", label: { es: "Hamilton: rezagos p", en: "Hamilton: lags p" },
    min: 1, max: 8, step: 1, value: state.p }, (v) => { state.p = v; run(); });

  let cutCtl = null;
  const cutHolder = document.createElement("div"); ctl.appendChild(cutHolder);
  function setupCut() {
    const s = data.paises[state.code];
    const dates = MAV.parseDates(s.fechas);
    const min = Math.max(24, state.h + state.p + 12);
    state.cut = dates.length - 1;
    cutHolder.textContent = "";
    cutCtl = MAV.slider(cutHolder, {
      key: "corte", label: { es: "Fecha de corte (lo que se sabía entonces)", en: "Cutoff date (what was known then)" },
      min, max: dates.length - 1, step: 1, value: state.cut, fmt: (i) => MAV.quarter(dates[i]),
      note: { es: "Las series se recalculan con datos solo hasta esa fecha", en: "Series are recomputed with data up to that date only" },
    }, (v) => { state.cut = v; run(); });
  }
  setupCut();

  const tiles = document.getElementById("tiles");
  const chCycle = MAV.lineChart(document.getElementById("chart-cycle"), { series: [], title: "" });
  const chReal = MAV.lineChart(document.getElementById("chart-realtime"), { series: [], title: "" });
  const chLevel = MAV.lineChart(document.getElementById("chart-level"), { series: [], title: "" });
  document.getElementById("source").textContent = "";

  // Real-time cycle: last value of each filter using data up to s only. Cached per country and parameters.
  const rtCache = new Map();
  function realTime(y, lambda, h, p) {
    const key = [state.code, lambda.toFixed(3), h, p].join("|");
    if (rtCache.has(key)) return rtCache.get(key);
    const hp = new Array(y.length).fill(null), ham = new Array(y.length).fill(null);
    const start = Math.max(16, h + p + 8);
    for (let s = start; s < y.length; s++) {
      const sub = y.slice(0, s + 1);
      hp[s] = sub[s] - N.hp(sub, lambda)[s];
      ham[s] = N.hamilton(sub, h, p).cycle[s];
    }
    const out = { hp, ham };
    rtCache.set(key, out);
    return out;
  }

  function run() {
    const s = data.paises[state.code];
    const dates = MAV.parseDates(s.fechas);
    const y = s.y, lambda = 10 ** state.loglam;
    const cut = state.cut ?? y.length - 1;
    const name = T(s.nombre);

    // full-sample ("final") estimates
    const tr = N.hp(y, lambda);
    const hpc = y.map((v, i) => v - tr[i]);
    const ham = N.hamilton(y, state.h, state.p);
    // estimates with data only up to the cutoff
    const yc = y.slice(0, cut + 1);
    const trc = N.hp(yc, lambda);
    const hpcCut = yc.map((v, i) => v - trc[i]);
    const hamCut = N.hamilton(yc, state.h, state.p).cycle;
    const rt = realTime(y, lambda, state.h, state.p);
    const partial = cut < y.length - 1;

    const lamTxt = fmtInt(lambda);
    const both = hpc.map((v, i) => [v, ham.cycle[i]]).filter(([, b]) => b != null);
    MAV.tiles(tiles, [
      { label: { es: "Volatilidad del ciclo HP", en: "HP cycle volatility" }, value: MAV.num(N.std(hpc), 2),
        note: { es: "desv. est., % del nivel", en: "std. dev., % of level" } },
      { label: { es: "Volatilidad del ciclo Hamilton", en: "Hamilton cycle volatility" }, value: MAV.num(N.std(ham.cycle), 2),
        note: { es: "desv. est., % del nivel", en: "std. dev., % of level" } },
      { label: { es: "Correlación HP–Hamilton", en: "HP–Hamilton correlation" },
        value: MAV.num(N.corr(both.map((b) => b[0]), both.map((b) => b[1])), 2) },
      { label: { es: `Revisión del ciclo HP en ${MAV.quarter(dates[cut])}`, en: `HP cycle revision at ${MAV.quarter(dates[cut])}` },
        value: partial ? (hpc[cut] - hpcCut[cut] >= 0 ? "+" : "") + MAV.num(hpc[cut] - hpcCut[cut], 2) : "—",
        note: partial ? { es: "final menos tiempo real, pp", en: "final minus real time, pp" }
                      : { es: "mueve la fecha de corte", en: "move the cutoff date" } },
    ]);

    const cycleSeries = [
      { name: { es: `HP, λ = ${lamTxt}`, en: `HP, λ = ${lamTxt}` }, short: "HP", x: dates, y: hpc, style: 0 },
      { name: { es: `Hamilton, h = ${state.h}, p = ${state.p}`, en: `Hamilton, h = ${state.h}, p = ${state.p}` }, short: "Hamilton", x: dates, y: ham.cycle, style: 1 },
    ];
    if (partial) {
      cycleSeries.push({ name: { es: `HP con datos hasta ${MAV.quarter(dates[cut])}`, en: `HP with data to ${MAV.quarter(dates[cut])}` },
        short: { es: "HP al corte", en: "HP at cutoff" }, x: dates, y: dates.map((_, i) => (i <= cut ? hpcCut[i] : null)), style: 2 });
      cycleSeries.push({ name: { es: `Hamilton con datos hasta ${MAV.quarter(dates[cut])}`, en: `Hamilton with data to ${MAV.quarter(dates[cut])}` },
        short: { es: "Hamilton al corte", en: "Hamilton at cutoff" }, x: dates, y: dates.map((_, i) => (i <= cut ? hamCut[i] : null)), style: 3, label: false });
    }
    chCycle.update({
      title: { es: `Componente cíclico del PIB real, ${name}`, en: `Cyclical component of real GDP, ${name}` },
      sub: { es: "Desviación de la tendencia, % (100 × log)", en: "Deviation from trend, % (100 × log)" },
      xType: "time", zero: true, series: cycleSeries, csvName: `ciclo-${state.code}`,
      vlines: partial ? [{ x: dates[cut], label: { es: "corte", en: "cutoff" } }] : [],
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2),
    });

    chReal.update({
      title: { es: "Lo que se veía en cada momento frente a lo que se estima hoy", en: "What you saw at each date versus what is estimated today" },
      sub: { es: "Ciclo HP del último trimestre disponible (tiempo real) y del mismo trimestre con toda la muestra (final)",
             en: "HP cycle of the latest available quarter (real time) and of the same quarter using the whole sample (final)" },
      xType: "time", zero: true, csvName: `tiempo-real-${state.code}`,
      series: [
        { name: { es: "HP final", en: "HP final" }, short: { es: "final", en: "final" }, x: dates, y: hpc.map((v, i) => (rt.hp[i] == null ? null : v)), style: 0 },
        { name: { es: "HP en tiempo real", en: "HP real time" }, short: { es: "tiempo real", en: "real time" }, x: dates, y: rt.hp, style: 1 },
        { name: { es: "Hamilton en tiempo real", en: "Hamilton real time" }, short: "Hamilton", x: dates, y: rt.ham, style: 2 },
      ],
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2),
    });

    chLevel.update({
      title: { es: `Nivel y tendencia HP, ${name}`, en: `Level and HP trend, ${name}` },
      sub: { es: "100 × log del PIB real", en: "100 × log real GDP" },
      xType: "time", csvName: `nivel-${state.code}`, height: 260,
      series: [
        { name: { es: "PIB real", en: "Real GDP" }, x: dates, y, style: 2 },
        { name: { es: `Tendencia HP, λ = ${lamTxt}`, en: `HP trend, λ = ${lamTxt}` }, short: { es: "tendencia", en: "trend" }, x: dates, y: tr, style: 0 },
      ],
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2),
    });
    document.getElementById("source").textContent = T({ es: "Datos: ", en: "Data: " }) + T(data.fuente);
  }

  MAV.onLang(run);
  run();
})();
