/* Laboratorio «Riesgo e incertidumbre»: Tauchen (1986) frente a Rouwenhorst (1995) al discretizar un AR(1). */

/* ------------------------------------------------------------ math core (no DOM; node-testable) */
const MKV = (function () {
  "use strict";
  const SQRTPI = Math.sqrt(Math.PI);

  /* erfc(x) for x >= 0 with near machine relative accuracy: a positive-term series for x < 1.5 and a
     continued fraction (modified Lentz) beyond. */
  function erfcPos(x) {
    if (x < 1.5) {
      const x2 = x * x;
      let term = x, sum = x;
      for (let n = 1; n < 200; n++) { term *= (2 * x2) / (2 * n + 1); sum += term; if (term < 1e-17 * sum) break; }
      return 1 - ((2 / SQRTPI) * Math.exp(-x2)) * sum;
    }
    const tiny = 1e-300;
    let f = x, C = x, D = 0;
    for (let n = 1; n < 500; n++) {
      const a = n / 2;
      D = x + a * D; D = Math.abs(D) < tiny ? tiny : D; D = 1 / D;
      C = x + a / C; C = Math.abs(C) < tiny ? tiny : C;
      const delta = C * D; f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    return Math.exp(-x * x) / (SQRTPI * f);
  }
  const upper = (z) => (z >= 0 ? 0.5 * erfcPos(z / Math.SQRT2) : 1 - 0.5 * erfcPos(-z / Math.SQRT2)); // P(Z > z)
  const cdf = (z) => (z <= 0 ? 0.5 * erfcPos(-z / Math.SQRT2) : 1 - 0.5 * erfcPos(z / Math.SQRT2));   // P(Z <= z)
  /* P(lo < Z <= hi), computed in the tail where it is small so tiny probabilities keep their digits. */
  function band(lo, hi) {
    if (lo >= 0) return upper(lo) - upper(hi);
    if (hi <= 0) return cdf(hi) - cdf(lo);
    return 1 - upper(hi) - cdf(lo);
  }
  const pdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

  /* Tauchen (1986): even grid on ±m unconditional sd; mass of each cell under N(rho z_i, sigma^2). */
  function tauchen(n, rho, sigma, m) {
    const su = sigma / Math.sqrt(1 - rho * rho), zmax = m * su;
    const grid = Array.from({ length: n }, (_, j) => -zmax + (2 * zmax * j) / (n - 1));
    const step = grid[1] - grid[0];
    const P = grid.map((zi) => {
      const mu = rho * zi, row = new Array(n);
      row[0] = cdf((grid[0] - mu + step / 2) / sigma);
      row[n - 1] = upper((grid[n - 1] - mu - step / 2) / sigma);
      for (let j = 1; j < n - 1; j++) row[j] = band((grid[j] - mu - step / 2) / sigma, (grid[j] - mu + step / 2) / sigma);
      return row;
    });
    return { grid, P, step };
  }

  /* Rouwenhorst (1995) with p = q = (1+rho)/2; grid on ±sqrt(n-1) unconditional sd (Kopecky and Suen 2010). */
  function rouwenhorst(n, rho, sigma) {
    const p = (1 + rho) / 2;
    let P = [[p, 1 - p], [1 - p, p]];
    for (let k = 3; k <= n; k++) {
      const Q = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let i = 0; i < k - 1; i++) for (let j = 0; j < k - 1; j++) {
        const v = P[i][j];
        Q[i][j] += p * v; Q[i][j + 1] += (1 - p) * v; Q[i + 1][j] += (1 - p) * v; Q[i + 1][j + 1] += p * v;
      }
      for (let i = 1; i < k - 1; i++) for (let j = 0; j < k; j++) Q[i][j] /= 2;
      P = Q;
    }
    const su = sigma / Math.sqrt(1 - rho * rho), psi = Math.sqrt(n - 1) * su;
    const grid = Array.from({ length: n }, (_, j) => -psi + (2 * psi * j) / (n - 1));
    return { grid, P, step: grid[1] - grid[0] };
  }

  /* Stationary distribution by Grassmann–Taksar–Heyman state reduction: no subtractions, so it stays
     accurate when the chain is nearly decomposable (Tauchen with rho close to 1). */
  function stationary(P) {
    const n = P.length, A = P.map((r) => r.slice());
    for (let k = n - 1; k > 0; k--) {
      let s = 0; for (let j = 0; j < k; j++) s += A[k][j];
      if (!(s > 0)) s = 1e-300;
      for (let i = 0; i < k; i++) A[i][k] /= s;
      for (let i = 0; i < k; i++) { const aik = A[i][k]; if (aik) for (let j = 0; j < k; j++) A[i][j] += aik * A[k][j]; }
    }
    const pi = new Array(n).fill(0); pi[0] = 1;
    for (let j = 1; j < n; j++) { let s = 0; for (let i = 0; i < j; i++) s += pi[i] * A[i][j]; pi[j] = s; }
    const tot = pi.reduce((a, b) => a + b, 0);
    return pi.map((v) => v / tot);
  }

  /* Implied moments of the stationary chain: sd, first-order autocorrelation, stay probability. */
  function moments(ch) {
    const { grid, P } = ch, pi = stationary(P), n = grid.length;
    const mu = pi.reduce((s, p, i) => s + p * grid[i], 0);
    let v = 0, cov = 0, k4 = 0;
    for (let i = 0; i < n; i++) {
      const d = grid[i] - mu; v += pi[i] * d * d; k4 += pi[i] * d ** 4;
      let ez = 0; for (let j = 0; j < n; j++) ez += P[i][j] * grid[j];
      cov += pi[i] * d * (ez - mu);
    }
    return { pi, mu, sd: Math.sqrt(v), rho: cov / v, kurt: k4 / (v * v), stay: P[(n - 1) >> 1][(n - 1) >> 1] };
  }

  function discretize(method, n, rho, sigma, m) {
    const ch = method === "tauchen" ? tauchen(n, rho, sigma, m) : rouwenhorst(n, rho, sigma);
    return Object.assign(ch, moments(ch), { target: { sd: sigma / Math.sqrt(1 - rho * rho), rho } });
  }

  /* Common random numbers: eps_t drives the AR(1); u_t = Phi(eps_t) drives both chains by inverse CDF. */
  function simulate(chains, rho, sigma, T, normals) {
    const ar = [0], paths = chains.map(() => []);
    const idx = chains.map((ch) => { let b = 0; ch.grid.forEach((z, i) => { if (Math.abs(z) < Math.abs(ch.grid[b])) b = i; }); return b; });
    chains.forEach((ch, c) => paths[c].push(ch.grid[idx[c]]));
    for (let t = 1; t <= T; t++) {
      const e = normals[t - 1];
      ar.push(rho * ar[t - 1] + sigma * e);
      const u = cdf(e);
      chains.forEach((ch, c) => {
        const row = ch.P[idx[c]]; let s = 0, j = 0;
        for (; j < row.length - 1; j++) { s += row[j]; if (u <= s) break; }
        idx[c] = j; paths[c].push(ch.grid[j]);
      });
    }
    return { ar, paths };
  }

  return { erfcPos, cdf, upper, band, pdf, tauchen, rouwenhorst, stationary, moments, discretize, simulate };
})();
if (typeof module === "object" && module.exports) module.exports = MKV;

/* ------------------------------------------------------------ page */
if (typeof document !== "undefined") (function () {
  "use strict";
  const T = MAV.t;
  const NUM = window.MAVNUM;
  const SIGMAS = [0.002, 0.005, 0.0088, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5];
  const state = { rho: 0.964, sigma: 0.0088, n: 7, m: 3, seed: 1 };
  const ctl = document.getElementById("controls");
  const HORIZON = 200;

  const fmtSigma = (v) => MAV.num(v, v < 0.01 ? 4 : v < 0.1 ? 3 : 2);
  const zero = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v);
  const signed = (v, d) => { const r = zero(v, d); return (r > 0 ? "+" : r < 0 ? "−" : "") + MAV.num(Math.abs(r), d); };
  const halfLife = (r) => (r <= 0 ? 0 : r >= 1 - 1e-12 ? Infinity : Math.log(0.5) / Math.log(r));
  const fmtHL = (h) => (!isFinite(h) || h > 1e4 ? T({ es: "más de 10 000", en: "over 10,000" }) : MAV.num(h, 1));
  const fmtRho = (v) => MAV.num(v, Math.abs(v * 20 - Math.round(v * 20)) < 1e-9 ? 2 : 3);   // ticks 0.05 apart, grid 0.005
  const fmtP = (v) => (v !== 0 && Math.abs(v) < 1e-4 ? v.toExponential(2) : MAV.num(v, 4));

  // presets
  const presetRow = document.createElement("div"); presetRow.className = "btn-row"; presetRow.style.margin = "0 0 18px";
  const PRESETS = [
    { label: { es: "PTF de Fernald", en: "Fernald TFP" }, set: { rho: 0.964, sigma: 0.0088, n: 7, m: 3 } },
    { label: { es: "Ingreso (T06_A)", en: "Income (T06_A)" }, set: { rho: 0.9, sigma: 0.2, n: 5, m: 3 } },
    { label: "ρ = 0.99", set: { rho: 0.99 } },
  ];
  const presetBtns = PRESETS.map((pr) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "btn";
    b.addEventListener("click", () => { Object.assign(state, pr.set); syncControls(); schedule(); });
    presetRow.appendChild(b); return [b, pr];
  });
  const paintPresets = () => presetBtns.forEach(([b, pr]) => { b.textContent = T(pr.label); });
  MAV.onLang(paintPresets); paintPresets();
  ctl.appendChild(presetRow);

  const sl = {};
  sl.rho = MAV.slider(ctl, { key: "rho", label: { es: "Persistencia ρ", en: "Persistence ρ" }, min: 0.5, max: 0.995, step: 0.001, value: state.rho,
    fmt: (v) => MAV.num(v, 3), note: { es: "z′ = ρz + σε, con ε normal estándar", en: "z′ = ρz + σε, with ε standard normal" } },
  (v) => { state.rho = v; schedule(); });
  sl.sigma = MAV.slider(ctl, { key: "sigma", label: { es: "Desviación estándar del choque σ", en: "Shock standard deviation σ" },
    min: 0, max: SIGMAS.length - 1, step: 1, value: SIGMAS.indexOf(state.sigma), fmt: (i) => fmtSigma(SIGMAS[i]),
    note: { es: "0.0088: PTF trimestral de Fernald; 0.20: ingreso anual", en: "0.0088: Fernald quarterly TFP; 0.20: annual income" } },
  (i) => { state.sigma = SIGMAS[i]; schedule(); });
  sl.n = MAV.slider(ctl, { key: "n", label: { es: "Número de estados N", en: "Number of states N" }, min: 3, max: 21, step: 1, value: state.n },
    (v) => { state.n = v; schedule(); });
  sl.m = MAV.slider(ctl, { key: "m", label: { es: "Tauchen: ancho de la malla m", en: "Tauchen: grid width m" }, min: 1, max: 5, step: 0.1, value: state.m,
    note: { es: "La malla cubre ±m desviaciones estándar incondicionales", en: "The grid spans ±m unconditional standard deviations" } },
  (v) => { state.m = v; schedule(); });
  function syncControls() { sl.rho.set(state.rho); sl.sigma.set(SIGMAS.indexOf(state.sigma)); sl.n.set(state.n); sl.m.set(state.m); }
  const simRow = document.createElement("div"); simRow.className = "btn-row";
  const simBtn = document.createElement("button"); simBtn.type = "button"; simBtn.className = "btn";
  simBtn.innerHTML = '<span class="es">Otros choques para la simulación</span><span class="en">New shocks for the simulation</span>';
  simBtn.addEventListener("click", () => { state.seed += 1; schedule(); });
  simRow.appendChild(simBtn); ctl.appendChild(simRow);

  /* ---------------------------------------------------------- chart card (same markup as MAV charts) */
  function card(el) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const q = (s) => el.querySelector(s);
    const parts = { title: q(".chart-title"), sub: q(".chart-sub"), legend: q(".legend"), plot: q(".chart"), tip: q(".tooltip"),
      table: q(".table-view"), tableBtn: q('[data-act="table"]'), csvBtn: q('[data-act="csv"]') };
    const paint = () => { const on = parts.table.classList.contains("on");
      parts.tableBtn.textContent = T(on ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" }); };
    parts.tableBtn.setAttribute("aria-expanded", "false");
    parts.tableBtn.addEventListener("click", () => { parts.tableBtn.setAttribute("aria-expanded", String(parts.table.classList.toggle("on"))); paint(); });
    MAV.onLang(paint); paint();
    return parts;
  }
  const csvText = (rows) => rows.map((r) => r.map((c) => (typeof c === "string" && /[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n");
  function download(name, rows) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csvText(rows)], { type: "text/csv" }));
    a.download = name + ".csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function fillTable(parts, header, rows) {
    const t = document.createElement("table"), h = t.createTHead().insertRow();
    header.forEach((x) => { const th = document.createElement("th"); th.textContent = x; h.appendChild(th); });
    const tb = t.createTBody();
    rows.forEach((r) => { const tr = tb.insertRow(); r.forEach((c) => { tr.insertCell().textContent = c; }); });
    parts.table.textContent = ""; parts.table.appendChild(t);
  }

  /* ---------------------------------------------------------- stationary distributions, two panels */
  function distChart(el) {
    const parts = card(el);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    let o = null;
    function draw() {
      if (!o) return;
      const width = Math.max(300, parts.plot.clientWidth || 640);
      const stacked = width < 600;
      const ph = stacked ? 230 : Math.round(Math.min(320, Math.max(250, width * 0.36)));
      const gap = stacked ? 34 : 28;
      const pw = stacked ? width : (width - gap) / 2;
      const height = stacked ? 2 * ph + gap : ph;
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
      svg.selectAll("*").remove();
      svg.append("title").text(T(o.title));
      const X = o.xmax, Y = o.ymax;
      const dens = d3.range(201).map((j) => { const z = -X + (2 * X * j) / 200; return [z, MKV.pdf(z / o.su) / o.su]; });
      const decX = X < 0.05 ? 3 : X < 0.5 ? 2 : 1;
      o.panels.forEach((pn, k) => {
        const ox = stacked ? 0 : k * (pw + gap), oy = stacked ? k * (ph + gap) : 0;
        const m = { t: 30, r: 10, b: 38, l: 48 };
        const x = d3.scaleLinear().domain([-X, X]).range([ox + m.l, ox + pw - m.r]);
        const y = d3.scaleLinear().domain([0, Y]).range([oy + ph - m.b, oy + m.t]);
        svg.append("g").attr("class", "grid").attr("transform", `translate(${ox + m.l},0)`)
          .call(d3.axisLeft(y).ticks(4).tickSize(-(pw - m.l - m.r)).tickFormat(""));
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${oy + ph - m.b})`)
          .call(d3.axisBottom(x).ticks(pw < 360 ? 4 : 6).tickFormat((v) => MAV.num(v, decX)).tickSizeOuter(0));
        svg.append("g").attr("class", "axis").attr("transform", `translate(${ox + m.l},0)`)
          .call(d3.axisLeft(y).ticks(4).tickFormat((v) => MAV.num(v, Y < 5 ? 1 : 0)).tickSize(0).tickPadding(8)).call((gg) => gg.select(".domain").remove());
        svg.append("text").attr("class", "axis-title").attr("x", ox + pw - m.r).attr("y", oy + ph - 5).attr("text-anchor", "end").text("z");
        svg.append("text").attr("x", ox + m.l).attr("y", oy + 14).attr("fill", "var(--ink)").style("font", "700 13px var(--font)").text(pn.name);
        svg.append("text").attr("class", "alabel").attr("x", ox + pw - m.r).attr("y", oy + 14).attr("text-anchor", "end").text(T(pn.note));
        const st = MAV.style(1);
        svg.append("path").attr("class", "series").attr("stroke", st.stroke).attr("stroke-dasharray", st.dash)
          .attr("d", d3.line().x((d) => x(d[0])).y((d) => y(Math.min(d[1], Y))).curve(d3.curveMonotoneX)(dens));
        pn.grid.forEach((z, i) => {
          const v = pn.pi[i] / pn.step;
          svg.append("line").attr("x1", x(z)).attr("x2", x(z)).attr("y1", y(0)).attr("y2", y(Math.min(v, Y))).attr("stroke", "var(--s1)").attr("stroke-width", 1.5);
          svg.append("circle").attr("class", "dot").attr("cx", x(z)).attr("cy", y(Math.min(v, Y))).attr("r", 4).attr("fill", "var(--ink)");
        });
        const ring = svg.append("circle").attr("r", 7.5).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).style("display", "none");
        const show = (px) => {
          let best = 0; pn.grid.forEach((z, i) => { if (Math.abs(x(z) - px) < Math.abs(x(pn.grid[best]) - px)) best = i; });
          const z = pn.grid[best], v = pn.pi[best] / pn.step;
          ring.attr("cx", x(z)).attr("cy", y(Math.min(v, Y))).style("display", null);
          const tip = parts.tip; tip.textContent = "";
          const head = document.createElement("div"); head.className = "t-head";
          head.textContent = `${pn.name} · ${T({ es: "estado", en: "state" })} ${best + 1} ${T({ es: "de", en: "of" })} ${pn.grid.length}`; tip.appendChild(head);
          [[MAV.num(z, 4), "z"], [fmtP(pn.pi[best]), T({ es: "probabilidad estacionaria πᵢ", en: "stationary probability πᵢ" })],
            [fmtP(v), "πᵢ / Δ"], [fmtP(MKV.pdf(z / o.su) / o.su), T({ es: "densidad normal en z", en: "normal density at z" })]].forEach(([val, name]) => {
            const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "auto 1fr";
            const b = document.createElement("b"); b.textContent = val; const s = document.createElement("span"); s.textContent = name;
            r.append(b, s); tip.appendChild(r);
          });
          const sc = parts.plot.clientWidth / width, left = x(z) * sc, tw = tip.offsetWidth || 190;
          tip.style.left = (left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
          tip.style.top = oy * sc + 8 + "px"; tip.classList.add("on");
        };
        let cur = ox + pw / 2;
        svg.append("rect").attr("x", ox + m.l).attr("y", oy + m.t).attr("width", pw - m.l - m.r).attr("height", ph - m.t - m.b)
          .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", `${pn.name}: ${T({ es: "explorar estados", en: "explore states" })}`)
          .on("pointermove", (ev) => { cur = d3.pointer(ev)[0]; show(cur); })
          .on("pointerleave", () => { ring.style("display", "none"); parts.tip.classList.remove("on"); })
          .on("blur", () => { ring.style("display", "none"); parts.tip.classList.remove("on"); })
          .on("focus", () => show(cur))
          .on("keydown", (ev) => {
            const d = (pw - m.l - m.r) / Math.max(4, pn.grid.length * 2);
            if (ev.key === "ArrowLeft") { cur -= d; show(cur); ev.preventDefault(); }
            if (ev.key === "ArrowRight") { cur += d; show(cur); ev.preventDefault(); }
          });
      });
      parts.title.textContent = T(o.title); parts.sub.textContent = T(o.sub);
      parts.legend.innerHTML = "";
      const st = MAV.style(1);
      [[`<svg width="26" height="10" aria-hidden="true"><line x1="0" x2="26" y1="5" y2="5" stroke="${st.stroke}" stroke-width="2" stroke-dasharray="${st.dash}" stroke-linecap="round"/></svg>`,
        { es: "densidad del AR(1), N(0, σ²/(1 − ρ²))", en: "AR(1) density, N(0, σ²/(1 − ρ²))" }],
       ['<svg width="26" height="12" aria-hidden="true"><line x1="13" x2="13" y1="12" y2="4" stroke="var(--s1)" stroke-width="1.5"/><circle cx="13" cy="4" r="3.5" fill="var(--ink)"/></svg>',
        { es: "cadena: πᵢ / Δ (probabilidad entre el paso de la malla)", en: "chain: πᵢ / Δ (probability divided by the grid step)" }]].forEach(([svgKey, name]) => {
        const k = document.createElement("span"); k.className = "key"; k.innerHTML = svgKey;
        const t = document.createElement("span"); t.textContent = T(name); k.appendChild(t); parts.legend.appendChild(k);
      });
      const n = o.panels[0].grid.length;
      const header = [T({ es: "estado", en: "state" }), "z Tauchen", "π Tauchen", "z Rouwenhorst", "π Rouwenhorst"];
      const raw = d3.range(n).map((i) => [i + 1, o.panels[0].grid[i], o.panels[0].pi[i], o.panels[1].grid[i], o.panels[1].pi[i]]);
      fillTable(parts, header, raw.map((r) => [String(r[0]), MAV.num(r[1], 4), fmtP(r[2]), MAV.num(r[3], 4), fmtP(r[4])]));
      parts.csvBtn.onclick = () => download(o.csvName, [header, ...raw]);
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw);
    return { update(next) { o = next; draw(); } };
  }

  /* ---------------------------------------------------------- charts, tiles, run */
  const tiles = document.getElementById("tiles");
  const chDist = distChart(document.getElementById("chart-dist"));
  const chSd = MAV.lineChart(document.getElementById("chart-sd"), { series: [], title: "" });
  const chRho = MAV.lineChart(document.getElementById("chart-rho"), { series: [], title: "" });
  const chSimT = MAV.lineChart(document.getElementById("chart-sim-t"), { series: [], title: "" });
  const chSimR = MAV.lineChart(document.getElementById("chart-sim-r"), { series: [], title: "" });

  const RHOS = d3.range(0, 100).map((i) => 0.5 + i / 200);   // 0.500, 0.505, ..., 0.995
  const sweepCache = new Map();
  function sweep(n, m) {
    const key = n + "|" + m.toFixed(1);
    if (sweepCache.has(key)) return sweepCache.get(key);
    const out = { tSd: [], tRho: [], rSd: [], rRho: [] };
    RHOS.forEach((r) => {
      const t = MKV.discretize("tauchen", n, r, 1, m), w = MKV.discretize("rouwenhorst", n, r, 1, m);
      out.tSd.push(t.sd / t.target.sd); out.tRho.push(t.rho - r); out.rSd.push(w.sd / w.target.sd); out.rRho.push(w.rho - r);
    });
    sweepCache.set(key, out);
    return out;
  }

  let pending = 0;
  function schedule() { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); }

  function run() {
    const { rho, sigma, n, m } = state;
    const tch = MKV.discretize("tauchen", n, rho, sigma, m), rou = MKV.discretize("rouwenhorst", n, rho, sigma, m);
    const su = tch.target.sd;
    const errSd = (c) => 100 * (c.sd / su - 1);

    MAV.tiles(tiles, [
      { label: { es: "Tauchen: error en ρ", en: "Tauchen: error in ρ" }, value: signed(tch.rho - rho, 4),
        note: { es: `ρ̂ = ${MAV.num(tch.rho, 4)}; vida media de ${fmtHL(halfLife(tch.rho))} periodos frente a ${fmtHL(halfLife(rho))}`,
                en: `ρ̂ = ${MAV.num(tch.rho, 4)}; half-life of ${fmtHL(halfLife(tch.rho))} periods against ${fmtHL(halfLife(rho))}` } },
      { label: { es: "Tauchen: error en σ incondicional", en: "Tauchen: error in unconditional σ" }, value: signed(errSd(tch), 1) + "%",
        note: { es: `σ̂ = ${MAV.num(tch.sd, 4)} frente a ${MAV.num(su, 4)}`, en: `σ̂ = ${MAV.num(tch.sd, 4)} against ${MAV.num(su, 4)}` } },
      { label: { es: "Rouwenhorst: error en ρ", en: "Rouwenhorst: error in ρ" }, value: signed(rou.rho - rho, 4),
        note: { es: `ρ̂ = ${MAV.num(rou.rho, 4)}`, en: `ρ̂ = ${MAV.num(rou.rho, 4)}` } },
      { label: { es: "Rouwenhorst: error en σ incondicional", en: "Rouwenhorst: error in unconditional σ" }, value: signed(errSd(rou), 1) + "%",
        note: { es: `σ̂ = ${MAV.num(rou.sd, 4)}`, en: `σ̂ = ${MAV.num(rou.sd, 4)}` } },
      { label: { es: "Tauchen: paso de la malla", en: "Tauchen: grid step" }, value: MAV.num(tch.step / sigma, 1) + " σ",
        note: { es: `probabilidad de quedarse en el estado central: ${MAV.num(tch.stay, 4)}`, en: `probability of staying in the middle state: ${MAV.num(tch.stay, 4)}` } },
    ]);

    // stationary distributions
    const xmax = 1.08 * Math.max(Math.abs(tch.grid[0]), Math.abs(rou.grid[0]), 3.2 * su);
    const ymax = 1.12 * Math.max(MKV.pdf(0) / su, ...tch.pi.map((p) => p / tch.step), ...rou.pi.map((p) => p / rou.step));
    const panelNote = (c) => ({ es: `σ̂/σ = ${MAV.num(c.sd / su, 3)} · curtosis ${MAV.num(c.kurt, 2)}`, en: `σ̂/σ = ${MAV.num(c.sd / su, 3)} · kurtosis ${MAV.num(c.kurt, 2)}` });
    chDist.update({
      title: { es: "Malla y distribución estacionaria de cada cadena", en: "Grid and stationary distribution of each chain" },
      sub: { es: `N = ${n} estados. La normal tiene curtosis 3. Tauchen cubre ±${MAV.num(m, 1)}σ; Rouwenhorst, ±√(N − 1)σ = ±${MAV.num(Math.sqrt(n - 1), 2)}σ.`,
             en: `N = ${n} states. The normal has kurtosis 3. Tauchen spans ±${MAV.num(m, 1)}σ; Rouwenhorst, ±√(N − 1)σ = ±${MAV.num(Math.sqrt(n - 1), 2)}σ.` },
      su, xmax, ymax: d3.nice(0, ymax, 4)[1], csvName: "distribucion-estacionaria",
      panels: [
        { name: "Tauchen", grid: tch.grid, pi: tch.pi, step: tch.step, note: panelNote(tch) },
        { name: "Rouwenhorst", grid: rou.grid, pi: rou.pi, step: rou.step, note: panelNote(rou) },
      ],
    });

    // moments across rho (independent of sigma)
    const sw = sweep(n, m);
    const vline = [{ x: rho }];
    chSd.update({
      title: { es: "Desviación estándar implícita según ρ", en: "Implied standard deviation across ρ" },
      sub: { es: `σ̂ / σ incondicional (1 = sin error), N = ${n}, m = ${MAV.num(m, 1)}; la vertical marca la ρ elegida`, en: `σ̂ / unconditional σ (1 = no error), N = ${n}, m = ${MAV.num(m, 1)}; the vertical line marks the chosen ρ` },
      xType: "linear", xLabel: "ρ", xFmt: fmtRho, labelRoom: 24, height: 230, csvName: "sigma-implicita", digits: 4,
      hlines: [{ y: 1 }], vlines: vline,
      points: [{ x: rho, y: tch.sd / su, style: 0 }, { x: rho, y: rou.sd / su, style: 1 }],
      series: [
        { name: "Tauchen", x: RHOS, y: sw.tSd, style: 0, label: false },
        { name: "Rouwenhorst", x: RHOS, y: sw.rSd, style: 1, label: false },
      ],
      yFmt: (v) => MAV.num(v, 2), tipFmt: (v) => MAV.num(v, 4),
    });
    chRho.update({
      title: { es: "Error en la persistencia según ρ", en: "Persistence error across ρ" },
      sub: { es: `ρ̂ − ρ: autocorrelación de la cadena menos la del AR(1), N = ${n}, m = ${MAV.num(m, 1)}`, en: `ρ̂ − ρ: chain autocorrelation minus the AR(1)'s, N = ${n}, m = ${MAV.num(m, 1)}` },
      xType: "linear", xLabel: "ρ", xFmt: fmtRho, labelRoom: 24, height: 230, csvName: "error-persistencia", digits: 5, zero: true,
      vlines: vline,
      points: [{ x: rho, y: tch.rho - rho, style: 0 }, { x: rho, y: rou.rho - rho, style: 1 }],
      series: [
        { name: "Tauchen", x: RHOS, y: sw.tRho, style: 0, label: false },
        { name: "Rouwenhorst", x: RHOS, y: sw.rRho, style: 1, label: false },
      ],
      yFmt: (v) => MAV.num(v, 3), tipFmt: (v) => signed(v, 4),
    });

    // simulation with common random numbers
    const rng = NUM.rng(state.seed), eps = Array.from({ length: HORIZON }, () => rng.normal());
    const sim = MKV.simulate([tch, rou], rho, sigma, HORIZON, eps);
    const periods = d3.range(HORIZON + 1);
    const lo = Math.min(...sim.ar, ...sim.paths[0], ...sim.paths[1]), hi = Math.max(...sim.ar, ...sim.paths[0], ...sim.paths[1]);
    const pad = 0.06 * (hi - lo || su);
    const changes = (p) => p.slice(1).filter((v, i) => v !== p[i]).length;
    const simOpts = (name, path, style) => ({
      title: { es: `Simulación: ${name}`, en: `Simulation: ${name}` },
      sub: (() => { const k = changes(path); return { es: `La cadena cambia de estado ${k} ${k === 1 ? "vez" : "veces"} en ${HORIZON} periodos`, en: `The chain changes state ${k} ${k === 1 ? "time" : "times"} in ${HORIZON} periods` }; })(),
      xType: "linear", xLabel: { es: "Periodo", en: "Period" }, xFmt: (v) => MAV.num(v, 0), labelRoom: 16, height: 240, zero: true,
      yDomain: [lo - pad, hi + pad], csvName: "simulacion-" + name.toLowerCase(), digits: 4,
      series: [
        { name: { es: `cadena de ${name}`, en: `${name} chain` }, x: periods, y: path, style },
        { name: { es: "AR(1) continuo", en: "continuous AR(1)" }, x: periods, y: sim.ar, style: 2 },
      ],
      yFmt: (v) => MAV.num(v, Math.abs(hi - lo) < 0.5 ? 2 : 1), tipFmt: (v) => MAV.num(v, 4),
    });
    chSimT.update(simOpts("Tauchen", sim.paths[0], 0));
    chSimR.update(simOpts("Rouwenhorst", sim.paths[1], 1));

    document.getElementById("source").textContent = T({
      es: `Sin datos. Calibraciones del curso: PTF trimestral de Fernald, ρ = 0.964 y σ = 0.0088 (mazo 3); ingreso de T06_A, ρ = 0.90 y σ = 0.20. Distribución estacionaria por reducción de estados (Grassmann–Taksar–Heyman); simulación con la semilla ${state.seed}.`,
      en: `No data. Course calibrations: Fernald quarterly TFP, ρ = 0.964 and σ = 0.0088 (deck 3); T06_A income, ρ = 0.90 and σ = 0.20. Stationary distribution by state reduction (Grassmann–Taksar–Heyman); simulation with seed ${state.seed}.` });
  }

  MAV.onLang(run);
  run();
})();
