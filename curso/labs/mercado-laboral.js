/* Laboratorio «Mercado laboral»: la curva de Beveridge, el desempleo de flujos u* = s/(s+f) y el mercado
   laboral dual mexicano como cadena de Markov de cuatro estados (formal, informal, desempleo, inactividad). */

/* ------------------------------------------------------------------ cálculo (sin DOM) */
(function () {
  "use strict";
  const N = window.MAVNUM;

  /* Promedio móvil hacia atrás de w periodos (null mientras no hay w datos). */
  function trailing(a, w) {
    if (w <= 1) return a.slice();
    const out = new Array(a.length).fill(null);
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      s += a[i]; if (i >= w) s -= a[i - w];
      if (i >= w - 1) out[i] = s / w;
    }
    return out;
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

  /* Desempleo de estado estacionario con flujos E↔U: u* = s/(s+f), y los dos contrafactuales de Shimer
     (solo f varía, con s en su media; solo s varía, con f en su media). Tasas en %, suavizado previo de f y s. */
  function flows(f, s, w = 3) {
    const fm = mean(f), sm = mean(s), fs = trailing(f, w), ss = trailing(s, w);
    const us = fs.map((x, i) => (x == null ? null : (100 * ss[i]) / (ss[i] + x)));
    return {
      f: fs, s: ss, fMean: fm, sMean: sm, ustar: us,
      onlyF: fs.map((x) => (x == null ? null : (100 * sm) / (sm + x))),
      onlyS: ss.map((x) => (x == null ? null : (100 * x) / (x + fm))),
      // vida media (meses) de u_t − u*: u_{t+1} − u* = (1 − f − s)(u_t − u*)
      halfLife: fs.map((x, i) => (x == null ? null : Math.log(0.5) / Math.log(1 - x - ss[i]))),
    };
  }

  /* Ajuste log v = a + b log u antes de 2020 y desplazamiento medio desde 2021 (T01_B, §5.2). */
  function beveridgeFit(dates, u, v) {
    const pre = [], post = [];
    dates.forEach((d, i) => { if (d < "2020-01") pre.push(i); else if (d >= "2021-01") post.push(i); });
    if (pre.length < 12 || post.length < 8) return null;
    const X = pre.map((i) => [1, Math.log(u[i])]), Y = pre.map((i) => Math.log(v[i]));
    const { beta } = N.ols(X, Y);
    const [a, b] = beta;
    const shift = mean(post.map((i) => Math.log(v[i]) - (a + b * Math.log(u[i]))));
    return { a, b, alpha: b < 0 ? b / (b - 1) : null, shift, pre, post };
  }

  /* ---- cadenas de Markov 4×4 */
  const normRows = (P) => P.map((row) => { const s = row.reduce((x, y) => x + y, 0); return row.map((x) => x / s); });
  const toMatrix = (flat) => [0, 1, 2, 3].map((r) => flat.slice(4 * r, 4 * r + 4));

  /* Distribución estacionaria: π (P − I) = 0 con Σπ = 1. */
  function stationary(P) {
    const n = P.length, A = N.zeros(n, n), b = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = P[j][i] - (i === j ? 1 : 0);
    for (let j = 0; j < n; j++) A[n - 1][j] = 1;
    b[n - 1] = 1;
    return N.solve(A, b);
  }

  /* Valores propios de una matriz n×n: polinomio característico (Faddeev–LeVerrier) y raíces por
     Durand–Kerner en aritmética compleja. Suficiente y estable para 4×4. Devuelve [{re, im, abs}]. */
  function eigenvalues(A) {
    const n = A.length;
    let M = N.zeros(n, n); const c = new Array(n + 1).fill(0); c[n] = 1;       // c[k] coef. de λ^k
    for (let k = 1; k <= n; k++) {
      const AM = N.mul(A, M);
      for (let i = 0; i < n; i++) AM[i][i] += c[n - k + 1];
      M = AM;
      const AMk = N.mul(A, M); let tr = 0; for (let i = 0; i < n; i++) tr += AMk[i][i];
      c[n - k] = -tr / k;
    }
    // Durand–Kerner sobre el polinomio mónico
    const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
    const peval = (z) => { let r = [1, 0]; for (let k = n - 1; k >= 0; k--) { r = cmul(r, z); r[0] += c[k]; } return r; };
    let roots = Array.from({ length: n }, (_, k) => { const r = 0.4 + 0.9 * k / n; const th = 0.3 + 2 * Math.PI * k / n; return [r * Math.cos(th), r * Math.sin(th)]; });
    for (let it = 0; it < 500; it++) {
      let move = 0;
      roots = roots.map((z, i) => {
        let den = [1, 0];
        roots.forEach((w, j) => { if (j !== i) den = cmul(den, [z[0] - w[0], z[1] - w[1]]); });
        const step = cdiv(peval(z), den);
        move = Math.max(move, Math.hypot(step[0], step[1]));
        return [z[0] - step[0], z[1] - step[1]];
      });
      if (move < 1e-15) break;
    }
    return roots.map(([re, im]) => ({ re, im: Math.abs(im) < 1e-12 ? 0 : im, abs: Math.hypot(re, im) })).sort((a, b) => b.abs - a.abs);
  }

  /* Vida media (trimestres) de la distancia a la distribución estacionaria: ln ½ / ln|λ₂|. */
  function halfLife(P) {
    const ev = eigenvalues(P);
    return { lambda2: ev[1].abs, quarters: Math.log(0.5) / Math.log(ev[1].abs), ev };
  }

  /* Cambia p_od en `delta` y compensa con la permanencia p_oo para que la fila siga sumando 1. */
  function modify(P, o, d, delta) {
    const Q = P.map((row) => row.slice());
    const target = Math.min(Q[o][d] + Q[o][o], Math.max(0, Q[o][d] + delta));
    Q[o][o] -= target - Q[o][d]; Q[o][d] = target;
    return Q;
  }

  /* x_{t+1} = x_t P, t = 0..H. */
  function markovPath(x0, P, H) {
    const out = [x0.slice()]; let x = x0.slice();
    for (let t = 0; t < H; t++) { x = x.map((_, j) => x.reduce((s, xi, i) => s + xi * P[i][j], 0)); out.push(x); }
    return out;
  }

  /* Matriz y acervos promedio de los 4 trimestres que terminan en k (null si falta alguno). */
  function fourQuarter(mex, k) {
    if (k < 3) return null;
    const idx = [k - 3, k - 2, k - 1, k];
    if (idx.some((j) => !mex.P[j] || !mex.acervos[j])) return null;
    const P = N.zeros(4, 4), x = [0, 0, 0, 0];
    idx.forEach((j) => { const M = normRows(toMatrix(mex.P[j])); for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) P[r][c] += M[r][c] / 4;
      mex.acervos[j].forEach((v, i) => { x[i] += v / 4; }); });
    const xs = x.reduce((a, b) => a + b, 0);
    return { P: normRows(P), x: x.map((v) => v / xs) };
  }

  /* Medidas a partir de una distribución [F, I, U, N], en %. */
  const MEASURES = {
    inf: (x) => (100 * x[1]) / (x[0] + x[1]),
    urate: (x) => (100 * x[2]) / (x[0] + x[1] + x[2]),
    F: (x) => 100 * x[0], I: (x) => 100 * x[1], U: (x) => 100 * x[2], N: (x) => 100 * x[3],
  };

  window.MAVLAB = { trailing, flows, beveridgeFit, stationary, eigenvalues, halfLife, modify, markovPath, fourQuarter, normRows, toMatrix, MEASURES };
})();

/* ------------------------------------------------------------------ interfaz */
(async function () {
  "use strict";
  if (typeof document === "undefined" || !window.MAV) return;
  const L = window.MAVLAB, T = MAV.t;
  const data = await MAV.json("../data/mercado_laboral.json");
  const pct = (v, d = 1) => (v == null || !isFinite(v) ? "—" : MAV.num(v, d) + "%");
  const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + MAV.num(Math.abs(v), d);

  /* Tarjeta con la misma estructura que MAV.lineChart (título, leyenda, lectura, tabla, CSV). */
  function card(el) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const c = { title: el.querySelector(".chart-title"), sub: el.querySelector(".chart-sub"), legend: el.querySelector(".legend"),
      plot: el.querySelector(".chart"), tip: el.querySelector(".tooltip"), table: el.querySelector(".table-view") };
    const tb = el.querySelector('[data-act="table"]'), cb = el.querySelector('[data-act="csv"]');
    const paint = () => { tb.textContent = T(c.table.classList.contains("on") ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" }); };
    tb.setAttribute("aria-expanded", "false");
    tb.addEventListener("click", () => { c.table.classList.toggle("on"); tb.setAttribute("aria-expanded", String(c.table.classList.contains("on"))); paint(); });
    c.svg = d3.select(c.plot).insert("svg", ".tooltip");
    c.setTable = (header, rows, raw, name) => {
      paint();
      const t = document.createElement("table"), h = t.createTHead().insertRow();
      header.forEach((x) => { const th = document.createElement("th"); th.textContent = x; h.appendChild(th); });
      const body = t.createTBody(); rows.forEach((r) => { const tr = body.insertRow(); r.forEach((x) => { tr.insertCell().textContent = x; }); });
      c.table.textContent = ""; c.table.appendChild(t);
      cb.onclick = () => {
        const esc = (x) => (typeof x === "string" && /[",\n]/.test(x) ? `"${x.replace(/"/g, '""')}"` : x);
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([[header, ...raw].map((r) => r.map(esc).join(",")).join("\n")], { type: "text/csv" }));
        a.download = name + ".csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      };
    };
    c.key = (svgHtml, text) => { const s = document.createElement("span"); s.className = "key"; s.innerHTML = svgHtml;
      const t = document.createElement("span"); t.textContent = text; s.appendChild(t); c.legend.appendChild(s); };
    c.tipAt = (rows, head, px, py, width) => {
      const tip = c.tip; tip.textContent = "";
      const hd = document.createElement("div"); hd.className = "t-head"; hd.textContent = head; tip.appendChild(hd);
      rows.forEach(([v, l]) => { const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "auto 1fr";
        const b = document.createElement("b"); b.textContent = v; const s = document.createElement("span"); s.textContent = l; r.append(b, s); tip.appendChild(r); });
      const sc = c.plot.clientWidth / width, tw = tip.offsetWidth || 160, left = px * sc;
      tip.style.left = (left + 14 + tw > c.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
      tip.style.top = Math.max(0, py * sc - 24) + "px"; tip.classList.add("on");
    };
    c.observe = (draw) => { let raf = 0; new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(c.plot); MAV.onLang(draw); };
    return c;
  }

  /* ================================================================ parte 1: EE. UU. y Europa */
  const US = data.eua;
  const countries = [{ value: "USA", label: US.nombre }].concat(Object.entries(data.paises)
    .map(([code, c]) => ({ value: code, label: c.nombre })).sort((a, b) => a.label.es.localeCompare(b.label.es, "es")));
  const s1 = { code: "USA", cursor: US.bev.fechas.length - 1, win: 3, cf: false };
  // Estado inicial desde la URL (?pais=DEU&suav=12&cf=1&flujo=NI&cambio=-5&medida=F): enlaces a un experimento.
  const QS = new URLSearchParams(location.search);
  if (QS.get("pais") === "USA" || data.paises[QS.get("pais")]) s1.code = QS.get("pais");
  if (["1", "3", "12"].includes(QS.get("suav"))) s1.win = +QS.get("suav");
  if (QS.get("cf") === "1") s1.cf = true;
  const series = (code) => (code === "USA" ? { fechas: US.bev.fechas, u: US.bev.u, v: US.bev.v, nombre: US.nombre, monthly: true }
    : Object.assign({ monthly: false }, data.paises[code]));
  const dateLabel = (code, s) => { const d = MAV.parseDates([s])[0]; return code === "USA" ? MAV.month(d) : MAV.quarter(d); };

  function episodes(code, fechas) {
    if (code === "USA") return [
      { from: "0000", to: "2007-12", label: "2001–07", mark: 0 },
      { from: "2008-01", to: "2009-06", label: { es: "2008–09, Gran Recesión", en: "2008–09, Great Recession" }, mark: null },
      { from: "2009-07", to: "2019-12", label: "2009–19", mark: 1 },
      { from: "2020-01", to: "2023-06", label: "2020–23", mark: 2 },
      { from: "2023-07", to: "9999", label: "2023→", mark: 3 }];
    return [
      { from: "0000", to: "2019-12", label: `${fechas[0].slice(0, 4)}–19`, mark: 0 },
      { from: "2020-01", to: "2023-06", label: "2020–23", mark: 1 },
      { from: "2023-07", to: "9999", label: "2023→", mark: 2 }];
  }
  const MARK = [
    { type: d3.symbolCircle, fill: "var(--s1)", stroke: "none" },
    { type: d3.symbolSquare, fill: "var(--bg)", stroke: "var(--s1)" },
    { type: d3.symbolTriangle, fill: "var(--s2)", stroke: "none" },
    { type: d3.symbolDiamond, fill: "var(--bg)", stroke: "var(--s2)" },
  ];
  const glyph = (k) => { const mk = MARK[k]; const p = d3.symbol(mk.type, 46)();
    return `<svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true"><path d="${p}" fill="${mk.fill}" stroke="${mk.stroke}" stroke-width="1.5"/></svg>`; };
  const lineKey = (style, w = 26) => { const st = MAV.style(style);
    return `<svg width="${w}" height="10" aria-hidden="true"><line x1="0" x2="${w}" y1="5" y2="5" stroke="${st.stroke}" stroke-width="2" stroke-dasharray="${st.dash}"/></svg>`; };

  // ---- controles
  const ctl1 = document.getElementById("controls-us");
  MAV.select(ctl1, { key: "pais", label: { es: "Curva de Beveridge de", en: "Beveridge curve for" }, value: s1.code, options: countries },
    (v) => { s1.code = v; setupCursor(); drawBev(); runFlows(); });
  const cursorHolder = document.createElement("div"); ctl1.appendChild(cursorHolder);
  let cursorCtl = null;
  function setupCursor() {
    const c = series(s1.code);
    s1.cursor = c.fechas.length - 1;
    cursorHolder.textContent = "";
    cursorCtl = MAV.slider(cursorHolder, { key: "hasta", label: { es: "Recorrer la curva hasta", en: "Trace the curve up to" },
      min: 0, max: c.fechas.length - 1, step: 1, value: s1.cursor, fmt: (i) => dateLabel(s1.code, series(s1.code).fechas[i]),
      note: { es: "La trayectoria se dibuja desde el primer dato hasta esa fecha", en: "The path is drawn from the first observation up to that date" } },
    (i) => { s1.cursor = i; drawBev(); runFlows(); });
  }
  setupCursor();
  MAV.segmented(ctl1, { key: "suav", label: { es: "Promedio de los flujos f y s", en: "Averaging of flows f and s" }, value: s1.win,
    options: [{ value: 1, label: { es: "Mensual", en: "Monthly" } }, { value: 3, label: { es: "3 meses", en: "3 months" } }, { value: 12, label: { es: "12 meses", en: "12 months" } }] },
  (v) => { s1.win = v; runFlows(); });
  const cfRow = document.createElement("label"); cfRow.className = "check";
  cfRow.innerHTML = `<input type="checkbox" id="ctl-cf"><span class="es">Mostrar contrafactuales: solo cambia f, solo cambia s</span><span class="en">Show counterfactuals: only f moves, only s moves</span>`;
  ctl1.appendChild(cfRow);
  cfRow.querySelector("input").checked = s1.cf;
  cfRow.querySelector("input").addEventListener("change", (ev) => { s1.cf = ev.target.checked; runFlows(); });

  // ---- curva de Beveridge (dispersión conectada)
  const bev = card(document.getElementById("chart-beveridge"));
  function drawBev() {
    const code = s1.code, c = series(code), n = c.fechas.length, cur = Math.min(s1.cursor, n - 1);
    const eps = episodes(code, c.fechas);
    const epOf = (d) => eps.find((e) => d >= e.from && d <= e.to);
    const fit = L.beveridgeFit(c.fechas, c.u, c.v);
    const svg = bev.svg, width = Math.max(300, bev.plot.clientWidth || 640);
    const height = Math.round(Math.min(470, Math.max(320, width * 0.62)));
    const m = { t: 26, r: 18, b: 44, l: 50 };
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img"); svg.selectAll("*").remove();
    const pad = (lo, hi, f) => [lo - f * (hi - lo), hi + f * (hi - lo)];
    const x = d3.scaleLinear().domain(pad(d3.min(c.u), d3.max(c.u), 0.03)).nice().range([m.l, width - m.r]);
    const y = d3.scaleLinear().domain(pad(d3.min(c.v), d3.max(c.v), 0.05)).nice().range([height - m.b, m.t]);
    const tickFmt = (scale, n) => { const st = d3.tickStep(scale.domain()[0], scale.domain()[1], n); const dg = Math.max(0, -Math.floor(Math.log10(st) + 1e-9));
      return (v) => MAV.num(v, dg); };
    const nx = Math.max(4, Math.round(width / 90));
    svg.append("title").text(T({ es: "Curva de Beveridge, ", en: "Beveridge curve, " }) + T(c.nombre));
    const clip = "bevclip-" + Math.random().toString(36).slice(2, 8);
    svg.append("clipPath").attr("id", clip).append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`).call(d3.axisBottom(x).ticks(nx).tickFormat(tickFmt(x, nx)).tickSizeOuter(0));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(tickFmt(y, 6)).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end")
      .text(T({ es: "Tasa de desempleo u (%)", en: "Unemployment rate u (%)" }));
    svg.append("text").attr("class", "axis-title").attr("x", m.l).attr("y", m.t - 12)
      .text(code === "USA" ? T({ es: "Vacantes / fuerza laboral v (%)", en: "Openings / labor force v (%)" }) : T({ es: "Tasa de vacantes v (%)", en: "Job vacancy rate v (%)" }));

    const pts = c.u.map((u, i) => ({ i, u, v: c.v[i], d: c.fechas[i], ep: epOf(c.fechas[i]) }));
    const line = d3.line().x((p) => x(p.u)).y((p) => y(p.v));
    svg.append("path").attr("d", line(pts)).attr("fill", "none").attr("stroke", "var(--rule)").attr("stroke-width", 1);
    svg.append("path").attr("d", line(pts.slice(0, cur + 1))).attr("fill", "none").attr("stroke", "var(--faint)").attr("stroke-width", 1.1);

    // ajustes log-lineales
    const fitStyles = [1, 3];
    if (fit && fit.b < 0) {
      const curve = (idx, shift) => { const us = idx.map((i) => c.u[i]); const [lo, hi] = d3.extent(us);
        return d3.range(0, 41).map((k) => { const u = lo + (hi - lo) * k / 40; return [x(u), y(Math.exp(fit.a + shift + fit.b * Math.log(u)))]; }); };
      [[fit.pre, 0], [fit.post, fit.shift]].forEach(([idx, sh], j) => {
        const st = MAV.style(fitStyles[j]);
        svg.append("path").attr("class", "series").attr("clip-path", `url(#${clip})`).attr("d", d3.line()(curve(idx, sh)))
          .attr("stroke", st.stroke).attr("stroke-dasharray", st.dash);
      });
    }
    // puntos por episodio
    const sym = d3.symbol().size(width < 520 ? 20 : 28);
    svg.append("g").selectAll("path").data(pts.slice(0, cur + 1).filter((p) => p.ep && p.ep.mark != null)).join("path")
      .attr("transform", (p) => `translate(${x(p.u)},${y(p.v)})`).attr("d", (p) => sym.type(MARK[p.ep.mark].type)())
      .attr("fill", (p) => MARK[p.ep.mark].fill).attr("stroke", (p) => MARK[p.ep.mark].stroke).attr("stroke-width", 1.2);
    // inicio y cursor
    const p0 = pts[0], pc = pts[cur];
    const startRight = x(p0.u) > width * 0.75;
    svg.append("text").attr("class", "alabel").attr("x", x(p0.u) + (startRight ? -8 : 8)).attr("y", y(p0.v) + 14)
      .attr("text-anchor", startRight ? "end" : "start").text(dateLabel(code, p0.d));
    svg.append("circle").attr("cx", x(pc.u)).attr("cy", y(pc.v)).attr("r", 8).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 2);
    const right = x(pc.u) > width * 0.7;
    svg.append("text").attr("class", "dlabel").attr("x", x(pc.u) + (right ? -12 : 12)).attr("y", y(pc.v) - 10)
      .attr("text-anchor", right ? "end" : "start").text(dateLabel(code, pc.d));

    // leyenda
    bev.legend.innerHTML = "";
    eps.filter((e) => e.mark != null).forEach((e) => bev.key(glyph(e.mark), T(e.label)));
    if (code === "USA") bev.key(lineKey(0).replace('stroke="var(--s1)"', 'stroke="var(--faint)"').replace('stroke-width="2"', 'stroke-width="1"'),
      T({ es: "trayectoria (incluye 2008–09)", en: "path (includes 2008–09)" }));
    if (fit && fit.b < 0) {
      bev.key(lineKey(fitStyles[0]), T({ es: "ajuste antes de 2020", en: "fit before 2020" }));
      bev.key(lineKey(fitStyles[1]), T({ es: "mismo ajuste desplazado, desde 2021", en: "same fit shifted, since 2021" }));
    }
    bev.title.textContent = T({ es: "Curva de Beveridge: ", en: "Beveridge curve: " }) + T(c.nombre);
    bev.sub.textContent = !fit
      ? T({ es: "Sin ajuste: no hay suficientes datos antes de 2020 y después de 2021.", en: "No fit: not enough data before 2020 and after 2021." })
      : fit.b >= 0
      ? T({ es: `Antes de 2020 la pendiente estimada es positiva (b = ${MAV.num(fit.b, 2)}): esta nube no es una curva de Beveridge.`,
            en: `Before 2020 the estimated slope is positive (b = ${MAV.num(fit.b, 2)}): this cloud is not a Beveridge curve.` })
      : T({ es: `log v = a + b log u antes de 2020: b = ${MAV.num(fit.b, 2)}, α implícito = ${MAV.num(fit.alpha, 2)}. Desde 2021 la curva está ${signed(fit.shift, 2)} log ${fit.shift >= 0 ? "más afuera" : "más adentro"}: ${signed(100 * (Math.exp(fit.shift) - 1), 0)}% de vacantes con el mismo desempleo.`,
            en: `log v = a + b log u before 2020: b = ${MAV.num(fit.b, 2)}, implied α = ${MAV.num(fit.alpha, 2)}. Since 2021 the curve sits ${signed(fit.shift, 2)} log ${fit.shift >= 0 ? "further out" : "further in"}: ${signed(100 * (Math.exp(fit.shift) - 1), 0)}% vacancies at the same unemployment.` });

    // lectura
    const vis = pts.slice(0, cur + 1);
    const del = d3.Delaunay.from(vis, (p) => x(p.u), (p) => y(p.v));
    const ring = svg.append("circle").attr("r", 6).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).style("display", "none");
    const show = (px, py) => { const p = vis[del.find(px, py)]; ring.attr("cx", x(p.u)).attr("cy", y(p.v)).style("display", null);
      bev.tipAt([[pct(p.u, 1), T({ es: "desempleo u", en: "unemployment u" })], [pct(p.v, 2), T({ es: "vacantes v", en: "vacancies v" })],
        [MAV.num(p.v / p.u, 2), T({ es: "tensión θ = v/u", en: "tightness θ = v/u" })], [T(p.ep.label), T({ es: "episodio", en: "episode" })]],
      dateLabel(code, p.d), x(p.u), y(p.v), width); };
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b).attr("fill", "transparent")
      .attr("tabindex", 0).attr("aria-label", T({ es: "Explorar valores", en: "Explore values" }))
      .on("pointermove", (ev) => { const [px, py] = d3.pointer(ev); show(px, py); })
      .on("pointerleave", () => { ring.style("display", "none"); bev.tip.classList.remove("on"); })
      .on("focus", () => show(x(pc.u), y(pc.v))).on("blur", () => { ring.style("display", "none"); bev.tip.classList.remove("on"); });

    const header = [T({ es: "Fecha", en: "Date" }), T({ es: "Desempleo u (%)", en: "Unemployment u (%)" }), T({ es: "Vacantes v (%)", en: "Vacancies v (%)" }), "θ = v/u", T({ es: "Episodio", en: "Episode" })];
    bev.setTable(header, pts.map((p) => [dateLabel(code, p.d), MAV.num(p.u, 2), MAV.num(p.v, 3), MAV.num(p.v / p.u, 3), T(p.ep.label)]),
      pts.map((p) => [p.d, p.u, p.v, p.v / p.u, T(p.ep.label)]), `beveridge-${code}`);
  }
  bev.observe(drawBev);

  // ---- u frente a u*
  const NBER = [["1990-07", "1991-03"], ["2001-03", "2001-11"], ["2007-12", "2009-06"], ["2020-02", "2020-04"]];
  const chFlows = MAV.lineChart(document.getElementById("chart-ustar"), { series: [] });
  const tiles1 = document.getElementById("tiles-us");
  const flDates = MAV.parseDates(US.flujos.fechas);
  function runFlows() {
    const F = L.flows(US.flujos.f, US.flujos.s, s1.win);
    const winTxt = s1.win === 1 ? { es: "mensual", en: "monthly" } : { es: `promedio de ${s1.win} meses`, en: `${s1.win}-month average` };
    const S = [
      { name: { es: "Desempleo observado u", en: "Actual unemployment u" }, short: "u", x: flDates, y: US.flujos.u, style: 0 },
      { name: { es: "Desempleo de flujos u* = s/(s+f)", en: "Flow unemployment u* = s/(s+f)" }, short: "u*", x: flDates, y: F.ustar, style: 1 },
    ];
    if (s1.cf) {
      S.push({ name: { es: "u* si solo cambia f (s en su media)", en: "u* if only f moves (s at its mean)" }, short: { es: "solo f", en: "only f" }, x: flDates, y: F.onlyF, style: 2 });
      S.push({ name: { es: "u* si solo cambia s (f en su media)", en: "u* if only s moves (f at its mean)" }, short: { es: "solo s", en: "only s" }, x: flDates, y: F.onlyS, style: 3 });
    }
    const vlines = [];
    if (s1.code === "USA") vlines.push({ x: MAV.parseDates([US.bev.fechas[s1.cursor]])[0] });
    chFlows.update({ title: { es: "Desempleo observado y desempleo de estado estacionario de los flujos, EE. UU.", en: "Actual and flow steady-state unemployment, United States" },
      sub: { es: `%, mensual; f = p_UE y s = p_EU de la CPS, ${T(winTxt)}. Sombreado: recesiones del NBER; línea punteada: la fecha de la curva de arriba.`,
             en: `%, monthly; f = p_UE and s = p_EU from the CPS, ${T(winTxt)}. Shaded: NBER recessions; dotted line: the date on the curve above.` },
      xType: "time", series: S, shades: NBER.map(([a, b]) => ({ from: MAV.parseDates([a])[0], to: MAV.parseDates([b])[0] })),
      vlines, tipDate: MAV.month, yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2), csvName: `desempleo-flujos-${s1.win}m`, labelRoom: 60 });

    const last = US.flujos.u.length - 1, bl = US.bev.fechas.length - 1;
    const mlab = (s) => MAV.month(MAV.parseDates([s])[0]);
    MAV.tiles(tiles1, [
      { label: { es: "Desempleo, EE. UU.", en: "Unemployment, US" }, value: pct(US.flujos.u[last]), note: mlab(US.flujos.fechas[last]) },
      { label: { es: "Desempleo de flujos u*", en: "Flow unemployment u*" }, value: pct(F.ustar[last]),
        note: { es: `s/(s+f), ${T(winTxt)}`, en: `s/(s+f), ${T(winTxt)}` } },
      { label: { es: "Tensión θ = v/u", en: "Tightness θ = v/u" }, value: MAV.num(US.bev.v[bl] / US.bev.u[bl], 2), note: mlab(US.bev.fechas[bl]) },
      { label: { es: "Vida media hacia u*", en: "Half-life toward u*" }, value: `${MAV.num(F.halfLife[last], 1)} ${T({ es: "meses", en: "months" })}`,
        note: { es: "ln ½ / ln(1 − f − s)", en: "ln ½ / ln(1 − f − s)" } },
    ]);
  }
  MAV.onLang(runFlows);
  drawBev(); runFlows();

  /* ================================================================ parte 2: México, cuatro estados */
  const MX = data.mex, QD = MAV.parseDates(MX.trimestres);
  const valid = MX.trimestres.map((_, k) => k).filter((k) => L.fourQuarter(MX, k));
  const NAMES = [{ es: "Formal", en: "Formal" }, { es: "Informal", en: "Informal" }, { es: "Desempleo", en: "Unemployed" }, { es: "Inactividad", en: "Inactive" }];
  const FLOWS = {
    IF: { o: 1, d: 0, label: { es: "I → F: formalización", en: "I → F: formalization" } },
    FI: { o: 0, d: 1, label: { es: "F → I: informalización", en: "F → I: informalization" } },
    UF: { o: 2, d: 0, label: { es: "U → F: de desempleo a formal", en: "U → F: unemployed to formal" } },
    UI: { o: 2, d: 1, label: { es: "U → I: de desempleo a informal", en: "U → I: unemployed to informal" } },
    IN: { o: 1, d: 3, label: { es: "I → N: de informal a inactivo", en: "I → N: informal to inactive" } },
    NI: { o: 3, d: 1, label: { es: "N → I: de inactivo a informal", en: "N → I: inactive to informal" } },
  };
  const MEAS = {
    inf: { opt: { es: "Informalidad, % del empleo", en: "Informality, % of employment" }, label: { es: "Informalidad (% del empleo)", en: "Informality (% of employment)" }, short: { es: "informalidad", en: "informality" } },
    urate: { opt: { es: "Desempleo abierto, % de la PEA", en: "Open unemployment, % of labor force" }, label: { es: "Desempleo abierto (% de la población económicamente activa)", en: "Open unemployment (% of the labor force)" }, short: { es: "desempleo abierto", en: "open unemployment" } },
    F: { opt: { es: "Empleo formal, % de la PET", en: "Formal employment, % of WAP" }, label: { es: "Empleo formal (% de la población en edad de trabajar)", en: "Formal employment (% of working-age population)" }, short: { es: "empleo formal", en: "formal employment" } },
    I: { opt: { es: "Empleo informal, % de la PET", en: "Informal employment, % of WAP" }, label: { es: "Empleo informal (% de la población en edad de trabajar)", en: "Informal employment (% of working-age population)" }, short: { es: "empleo informal", en: "informal employment" } },
    U: { opt: { es: "Desempleados, % de la PET", en: "Unemployed, % of WAP" }, label: { es: "Desempleados (% de la población en edad de trabajar)", en: "Unemployed (% of working-age population)" }, short: { es: "desempleados", en: "unemployed" } },
    N: { opt: { es: "Inactivos, % de la PET", en: "Inactive, % of WAP" }, label: { es: "Inactivos (% de la población en edad de trabajar)", en: "Inactive (% of working-age population)" }, short: { es: "inactivos", en: "inactive" } },
  };
  const sym = (i) => "FIUN"[i];
  const s2 = { pos: valid.length - 1, flow: "IF", delta: 0, meas: "inf" };
  if (QS.get("flujo") in FLOWS) s2.flow = QS.get("flujo");
  if (QS.get("medida") in MEAS) s2.meas = QS.get("medida");
  if (isFinite(parseFloat(QS.get("cambio")))) s2.delta = Math.max(-10, Math.min(10, Math.round(2 * parseFloat(QS.get("cambio"))) / 2));
  const tq = QS.get("trim"); if (tq && valid.some((k) => MX.trimestres[k] === tq)) s2.pos = valid.findIndex((k) => MX.trimestres[k] === tq);

  const ctl2 = document.getElementById("controls-mx");
  MAV.slider(ctl2, { key: "trim", label: { es: "Trimestre", en: "Quarter" }, min: 0, max: valid.length - 1, step: 1, value: s2.pos,
    fmt: (i) => MAV.quarter(QD[valid[i]]),
    note: { es: "Matriz y acervos: promedio de los cuatro trimestres que terminan en el elegido, sin estacionalidad", en: "Matrix and stocks: average of the four quarters ending in the chosen one, free of seasonality" } },
  (i) => { s2.pos = i; runMx(); });
  MAV.select(ctl2, { key: "flujo", label: { es: "Flujo que cambias", en: "Flow you change" }, value: s2.flow,
    options: Object.entries(FLOWS).map(([value, f]) => ({ value, label: f.label })) }, (v) => { s2.flow = v; runMx(); });
  const deltaOpts = { key: "delta", label: { es: "Cambio en esa probabilidad", en: "Change in that probability" }, min: -10, max: 10, step: 0.5, value: s2.delta,
    fmt: (v) => `${signed(v, 1)} pp`, note: "" };
  const deltaCtl = MAV.slider(ctl2, deltaOpts, (v) => { s2.delta = v; runMx(); });
  const b2 = document.createElement("div"); b2.className = "btn-row"; b2.style.margin = "-6px 0 16px";
  const bReset = document.createElement("button"); bReset.type = "button"; bReset.className = "btn"; b2.appendChild(bReset); ctl2.appendChild(b2);
  const paintReset = () => { bReset.textContent = T({ es: "Sin cambio", en: "No change" }); };
  MAV.onLang(paintReset); paintReset();
  bReset.addEventListener("click", () => { s2.delta = 0; deltaCtl.set(0); runMx(); });
  MAV.select(ctl2, { key: "medida", label: { es: "Medida en las gráficas", en: "Measure in the charts" }, value: s2.meas,
    options: Object.entries(MEAS).map(([value, mm]) => ({ value, label: mm.opt })) }, (v) => { s2.meas = v; runMx(); });
  const measNote = document.createElement("p"); measNote.className = "ctl-note"; measNote.style.marginTop = "-10px"; ctl2.appendChild(measNote);
  const paintMeasNote = () => { measNote.textContent = T({ es: "PEA: F + I + U. PET: población en edad de trabajar, F + I + U + N.", en: "Labor force: F + I + U. WAP: working-age population, F + I + U + N." }); };
  MAV.onLang(paintMeasNote); paintMeasNote();

  const tiles2 = document.getElementById("tiles-mx");
  const heat = card(document.getElementById("chart-matrix"));
  const chLR = MAV.lineChart(document.getElementById("chart-longrun"), { series: [] });
  const chPath = MAV.lineChart(document.getElementById("chart-path"), { series: [] });

  // serie histórica: observado y estado estacionario (promedios de 4 trimestres)
  const hist = MX.trimestres.map((_, k) => { const q = L.fourQuarter(MX, k); return q ? { x: q.x, pi: L.stationary(q.P) } : null; });

  let current = null;
  function state2() {
    const k = valid[s2.pos], q = L.fourQuarter(MX, k), fl = FLOWS[s2.flow];
    const Q = L.modify(q.P, fl.o, fl.d, s2.delta / 100);
    return { k, q, fl, Q, pi0: L.stationary(q.P), pi: L.stationary(Q), hl: L.halfLife(Q), changed: Math.abs(Q[fl.o][fl.d] - q.P[fl.o][fl.d]) > 1e-12 };
  }

  function drawHeat() {
    if (!current) return;
    const { q, Q, fl, changed } = current, svg = heat.svg;
    const width = Math.max(300, heat.plot.clientWidth || 640), narrow = width < 520;
    const lw = narrow ? 84 : 118, top = 30, cw = Math.min(110, (width - lw - 4) / 4), ch = narrow ? 46 : 52;
    const height = top + 4 * ch + 6;
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img"); svg.selectAll("*").remove();
    svg.append("title").text(T({ es: "Matriz de transición", en: "Transition matrix" }));
    const maxOff = 0.35;
    for (let c = 0; c < 4; c++) svg.append("text").attr("class", "axis-title").attr("x", lw + c * cw + cw / 2).attr("y", top - 10).attr("text-anchor", "middle")
      .text(narrow ? sym(c) : `${T(NAMES[c])} ${sym(c)}`);
    svg.append("text").attr("class", "axis-title").attr("x", 0).attr("y", 12).text(T({ es: "de ↓  a →", en: "from ↓  to →" }));
    for (let r = 0; r < 4; r++) {
      svg.append("text").attr("class", "axis-title").attr("x", lw - 10).attr("y", top + r * ch + ch / 2 + 4).attr("text-anchor", "end")
        .text(narrow ? `${T(NAMES[r])}` : `${T(NAMES[r])} ${sym(r)}`);
      for (let c = 0; c < 4; c++) {
        const v = Q[r][c], diag = r === c;
        // tono: la diagonal (0.17–0.84) y las transiciones (0–0.35) en escalas propias, para que se lean ambas
        const t = diag ? 0.25 + 0.75 * Math.min(1, v) : Math.min(1, Math.sqrt(v / maxOff)) * 0.8;
        const g = svg.append("g").attr("transform", `translate(${lw + c * cw},${top + r * ch})`);
        g.append("rect").attr("x", 1).attr("y", 1).attr("width", cw - 2).attr("height", ch - 2).attr("fill", "var(--ink)").attr("fill-opacity", t);
        g.append("rect").attr("x", 1).attr("y", 1).attr("width", cw - 2).attr("height", ch - 2).attr("fill", "none").attr("stroke", "var(--rule)").attr("stroke-width", 0.5);
        const dark = t > 0.58, touched = changed && r === fl.o && (c === fl.d || c === fl.o);
        g.append("text").attr("x", cw / 2).attr("y", touched ? ch / 2 : ch / 2 + 5).attr("text-anchor", "middle")
          .attr("fill", dark ? "var(--bg)" : "var(--ink)").attr("font-weight", touched ? 700 : 600).attr("font-size", narrow ? 13 : 14)
          .style("font-variant-numeric", "tabular-nums").text(MAV.num(v, 2));
        if (touched) {
          g.append("text").attr("x", cw / 2).attr("y", ch / 2 + 15).attr("text-anchor", "middle").attr("fill", dark ? "var(--bg)" : "var(--ink)")
            .attr("font-size", narrow ? 10 : 11).text((narrow ? "" : T({ es: "antes ", en: "was " })) + (narrow ? `(${MAV.num(q.P[r][c], 2)})` : MAV.num(q.P[r][c], 2)));
          g.append("rect").attr("x", 3).attr("y", 3).attr("width", cw - 6).attr("height", ch - 6).attr("fill", "none")
            .attr("stroke", dark ? "var(--bg)" : "var(--ink)").attr("stroke-width", 2).attr("stroke-dasharray", "5 3");
        }
      }
    }
    heat.title.textContent = T({ es: "Matriz de transición trimestral, ", en: "Quarterly transition matrix, " }) + MAV.quarter(QD[current.k]);
    heat.sub.textContent = T({ es: "Probabilidad de pasar del estado de la fila al de la columna en un trimestre; cada fila suma 1. Tono más oscuro, probabilidad más alta (la diagonal y el resto en escalas distintas). Recuadro punteado: el flujo que cambiaste y la permanencia que lo compensa.",
      en: "Probability of moving from the row state to the column state within a quarter; each row sums to 1. Darker means more likely (the diagonal and the rest on separate scales). Dotted box: the flow you changed and the stay probability that offsets it." });
    heat.legend.innerHTML = "";
    const cells = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cells.push({ r, c });
    const hit = svg.append("rect").attr("x", lw).attr("y", top).attr("width", 4 * cw).attr("height", 4 * ch).attr("fill", "transparent")
      .attr("tabindex", 0).attr("aria-label", T({ es: "Explorar valores", en: "Explore values" }));
    const ring = svg.append("rect").attr("width", cw).attr("height", ch).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 2).style("display", "none");
    const show = (px, py) => {
      const c = Math.max(0, Math.min(3, Math.floor((px - lw) / cw))), r = Math.max(0, Math.min(3, Math.floor((py - top) / ch)));
      ring.attr("x", lw + c * cw).attr("y", top + r * ch).style("display", null);
      const rows = [[MAV.num(Q[r][c], 3), T({ es: "probabilidad trimestral", en: "quarterly probability" })]];
      if (Math.abs(Q[r][c] - q.P[r][c]) > 1e-12) rows.push([MAV.num(q.P[r][c], 3), T({ es: "observada", en: "observed" })]);
      heat.tipAt(rows, `${T(NAMES[r])} → ${T(NAMES[c])} (p_${sym(r)}${sym(c)})`, lw + c * cw + cw / 2, top + r * ch + ch / 2, width);
    };
    hit.on("pointermove", (ev) => { const [px, py] = d3.pointer(ev); show(px, py); })
      .on("pointerleave", () => { ring.style("display", "none"); heat.tip.classList.remove("on"); })
      .on("focus", () => show(lw + fl.d * cw + 1, top + fl.o * ch + 1)).on("blur", () => { ring.style("display", "none"); heat.tip.classList.remove("on"); });
    const header = [T({ es: "De \\ a", en: "From \\ to" }), ...[0, 1, 2, 3].map((c) => `${T(NAMES[c])} ${sym(c)}`)];
    heat.setTable(header, Q.map((row, r) => [`${T(NAMES[r])} ${sym(r)}`, ...row.map((v) => MAV.num(v, 3))]),
      Q.map((row, r) => [`${T(NAMES[r])} ${sym(r)}`, ...row]), `enoe-matriz-${MX.trimestres[current.k]}`);
  }
  heat.observe(drawHeat);

  function runMx() {
    current = state2();
    const { k, q, fl, Q, pi0, pi, hl, changed } = current, M = L.MEASURES, qlab = MAV.quarter(QD[k]);
    const from = q.P[fl.o][fl.d], to = Q[fl.o][fl.d];
    deltaOpts.note = changed
      ? { es: `p_${sym(fl.o)}${sym(fl.d)}: ${MAV.num(from, 3)} → ${MAV.num(to, 3)}; p_${sym(fl.o)}${sym(fl.o)} pasa de ${MAV.num(q.P[fl.o][fl.o], 3)} a ${MAV.num(Q[fl.o][fl.o], 3)} para que la fila sume 1`,
          en: `p_${sym(fl.o)}${sym(fl.d)}: ${MAV.num(from, 3)} → ${MAV.num(to, 3)}; p_${sym(fl.o)}${sym(fl.o)} goes from ${MAV.num(q.P[fl.o][fl.o], 3)} to ${MAV.num(Q[fl.o][fl.o], 3)} so the row sums to 1` }
      : { es: `p_${sym(fl.o)}${sym(fl.d)} observada: ${MAV.num(from, 3)}. La permanencia p_${sym(fl.o)}${sym(fl.o)} absorbe el cambio`,
          en: `observed p_${sym(fl.o)}${sym(fl.d)}: ${MAV.num(from, 3)}. The stay probability p_${sym(fl.o)}${sym(fl.o)} absorbs the change` };
    deltaCtl.set(s2.delta);

    MAV.tiles(tiles2, [
      { label: { es: `Informalidad observada, ${qlab}`, en: `Observed informality, ${qlab}` }, value: pct(M.inf(q.x)),
        note: { es: "% del empleo, promedio de 4 trimestres", en: "% of employment, 4-quarter average" } },
      { label: { es: "Informalidad de largo plazo", en: "Long-run informality" }, value: pct(M.inf(pi)),
        note: changed ? { es: `con tu cambio; sin él, ${pct(M.inf(pi0))}`, en: `with your change; without it, ${pct(M.inf(pi0))}` }
                      : { es: "estado estacionario de los flujos observados", en: "steady state of the observed flows" } },
      { label: { es: "Desempleo abierto de largo plazo", en: "Long-run open unemployment" }, value: pct(M.urate(pi)),
        note: changed ? { es: `con tu cambio; sin él, ${pct(M.urate(pi0))}`, en: `with your change; without it, ${pct(M.urate(pi0))}` }
                      : { es: `observado: ${pct(M.urate(q.x))}`, en: `observed: ${pct(M.urate(q.x))}` } },
      { label: { es: "Vida media de la convergencia", en: "Half-life of convergence" }, value: `${MAV.num(hl.quarters, 1)} ${T({ es: "trim.", en: "qtrs" })}`,
        note: { es: `ln ½ / ln|λ₂|, |λ₂| = ${MAV.num(hl.lambda2, 3)}`, en: `ln ½ / ln|λ₂|, |λ₂| = ${MAV.num(hl.lambda2, 3)}` } },
    ]);
    drawHeat();

    const f = M[s2.meas], ml = MEAS[s2.meas];
    const obs = hist.map((h) => (h ? f(h.x) : null)), ss = hist.map((h) => (h ? f(h.pi) : null));
    const firstK = hist.findIndex((h) => h);
    const xs = QD.slice(firstK), dig = s2.meas === "urate" || s2.meas === "U" ? 2 : 1;
    chLR.update({ title: { es: `${T(ml.label)}: observada y de estado estacionario`, en: `${T(ml.label)}: observed and steady state` },
      sub: { es: "Promedios móviles de 4 trimestres. El estado estacionario usa la matriz de flujos del mismo periodo; el punto es tu cambio en el trimestre elegido.",
             en: "4-quarter moving averages. The steady state uses the flow matrix of the same period; the dot is your change in the chosen quarter." },
      xType: "time", csvName: `enoe-${s2.meas}-largo-plazo`, labelRoom: 90,
      series: [
        { name: { es: "Observada", en: "Observed" }, short: { es: "observada", en: "observed" }, x: xs, y: obs.slice(firstK), style: 0 },
        { name: { es: "Estado estacionario de los flujos", en: "Steady state of the flows" }, short: { es: "de los flujos", en: "from flows" }, x: xs, y: ss.slice(firstK), style: 1 },
      ],
      points: changed ? [{ x: QD[k], y: f(pi), label: { es: "con tu cambio", en: "with your change" }, style: 0 }] : [],
      vlines: [{ x: QD[k] }], yDomain: d3.extent([...obs, ...ss, changed ? f(pi) : null].filter((v) => v != null)),
      yFmt: (v) => MAV.num(v, dig === 2 ? 1 : 0), tipFmt: (v) => MAV.num(v, dig) });

    const H = 20, t = d3.range(0, H + 1);
    const pMod = L.markovPath(q.x, Q, H).map(f), pObs = L.markovPath(q.x, q.P, H).map(f);
    const S = changed
      ? [{ name: { es: "Con tu cambio", en: "With your change" }, short: { es: "con cambio", en: "changed" }, x: t, y: pMod, style: 0 },
         { name: { es: "Con los flujos observados", en: "With the observed flows" }, short: { es: "observados", en: "observed" }, x: t, y: pObs, style: 1 }]
      : [{ name: { es: "Con los flujos observados", en: "With the observed flows" }, short: { es: "observados", en: "observed" }, x: t, y: pObs, style: 0 }];
    chPath.update({ title: { es: `Convergencia desde ${qlab}: ${T(ml.short)}`, en: `Convergence from ${qlab}: ${T(ml.short)}` },
      sub: { es: "%, partiendo de la distribución observada y aplicando la matriz trimestre tras trimestre", en: "%, starting from the observed distribution and applying the matrix quarter after quarter" },
      xType: "linear", xLabel: { es: "Trimestres hacia adelante", en: "Quarters ahead" }, xFmt: (v) => String(v), csvName: `enoe-${s2.meas}-convergencia`,
      series: S, labelRoom: 90, height: 250,
      hlines: [{ y: f(pi), label: { es: "estado estacionario", en: "steady state" } }],
      yFmt: (v) => MAV.num(v, dig === 2 ? 2 : 1), tipFmt: (v) => MAV.num(v, dig) });
  }
  MAV.onLang(runMx);
  runMx();

  const src = document.getElementById("source");
  const paintSrc = () => { src.textContent = T({ es: "Datos: ", en: "Data: " }) + T(data.fuente); };
  MAV.onLang(paintSrc); paintSrc();
})();
