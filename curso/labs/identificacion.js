/* Laboratorio «Identificación»: un SVAR conocido, dos ordenamientos de Cholesky y una proyección local
   con instrumento externo (LP-IV), en una muestra y en un Monte Carlo. */

/* ------------------------------------------------------------------ econometría (sin DOM) */
(function () {
  "use strict";
  const N = window.MAVNUM;
  const P_LAGS = 2, H_MAX = 20, BURN = 200;
  // Dinámica verdadera: y_t = A y_{t-1} + B0 ε_t, con y = (producto x, tasa i) y ε = (demanda, monetario).
  const A_TRUE = [[0.8, -0.2], [0.1, 0.85]];
  const B0 = (par) => [[1, par.theta], [par.phi, 1]];

  /* Muestra simulada. El orden de los sorteos (ε_d, ε_m, η) no depende de los parámetros,
     así que mover θ, φ o el ruido cambia la muestra de forma continua. */
  function simulate(par, T, seed) {
    const g = N.rng(seed), b = B0(par), A = A_TRUE;
    let x = 0, i = 0;
    const Y = [], z = [], eps = [];
    for (let t = -BURN; t < T; t++) {
      const ed = g.normal(), em = g.normal(), eta = g.normal();
      const xn = A[0][0] * x + A[0][1] * i + b[0][0] * ed + b[0][1] * em;
      const inn = A[1][0] * x + A[1][1] * i + b[1][0] * ed + b[1][1] * em;
      x = xn; i = inn;
      if (t >= 0) { Y.push([x, i]); z.push(em + par.noise * eta); eps.push([ed, em]); }
    }
    return { Y, z, eps };
  }

  /* IRF verdadera a un choque monetario que sube la tasa 1 pp en el impacto: A^h B0[:,1] / B0[1][1]. */
  function truth(par, H = H_MAX) {
    let v = [par.theta, 1];
    const out = [];
    for (let h = 0; h <= H; h++) { out.push(v.slice()); v = [A_TRUE[0][0] * v[0] + A_TRUE[0][1] * v[1], A_TRUE[1][0] * v[0] + A_TRUE[1][1] * v[1]]; }
    return out;
  }

  /* VAR(p) con constante por MCO. Devuelve A_j (n×n), residuos y Σ con corrección T − p − 1 − n p. */
  function fitVAR(Y, p = P_LAGS) {
    const T = Y.length, n = Y[0].length, k = 1 + n * p;
    const XtX = N.zeros(k, k), XtY = N.zeros(k, n), rows = [];
    for (let t = p; t < T; t++) {
      const r = [1]; for (let j = 1; j <= p; j++) r.push(...Y[t - j]);
      rows.push(r);
      for (let a = 0; a < k; a++) { const ra = r[a]; for (let c = a; c < k; c++) XtX[a][c] += ra * r[c]; for (let q = 0; q < n; q++) XtY[a][q] += ra * Y[t][q]; }
    }
    for (let a = 0; a < k; a++) for (let c = 0; c < a; c++) XtX[a][c] = XtX[c][a];
    const Bc = N.solve(XtX, XtY);                                   // k × n
    const Te = T - p, U = [], S = N.zeros(n, n);
    rows.forEach((r, s) => {
      const u = Y[s + p].map((yv, q) => yv - r.reduce((acc, rv, a) => acc + rv * Bc[a][q], 0));
      U.push(u); for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) S[a][c] += u[a] * u[c];
    });
    const Sigma = S.map((row) => row.map((v) => v / (Te - k)));
    const Alist = Array.from({ length: p }, (_, j) => Array.from({ length: n }, (_, q) => Array.from({ length: n }, (_, c) => Bc[1 + j * n + c][q])));
    const c = Bc[0].slice();
    return { Alist, c, Sigma, U, p };
  }

  /* Respuestas Ψ_h b para un vector de impacto b. */
  function varIRF(Alist, b, H = H_MAX) {
    const p = Alist.length, n = b.length, Psi = [b.slice()];
    for (let h = 1; h <= H; h++) {
      const v = new Array(n).fill(0);
      for (let j = 1; j <= Math.min(h, p); j++) { const Aj = Alist[j - 1], w = Psi[h - j]; for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) v[a] += Aj[a][c] * w[c]; }
      Psi.push(v);
    }
    return Psi;
  }

  /* Vectores de impacto normalizados (la tasa sube 1 pp) para los dos ordenamientos recursivos.
     A: [producto, tasa] -> segunda columna de chol(Σ); B: [tasa, producto] -> Σ[:,i]/Σ_ii. */
  function cholImpacts(Sigma) {
    const s11 = Sigma[0][0], s12 = Sigma[0][1], s22 = Sigma[1][1];
    const pA = Math.sqrt(s22 - s12 * s12 / s11);
    return { A: [0, 1], B: [s12 / s22, 1], scaleA: pA, scaleB: Math.sqrt(s22) };
  }

  /* LP-IV (Jordà con instrumento externo): y_{j,t+h} = α + β_h i_t + γ'w_t + u, con i_t instrumentada por z_t
     y w_t = p rezagos de (producto, tasa). Errores Newey–West (Bartlett, L = h + 1) con residuos de la
     ecuación estructural. F de la primera etapa: t² robusto del coeficiente de z_t (L = h + 1). */
  function lpiv(Y, z, j, H = H_MAX, p = P_LAGS, { se = true } = {}) {
    const T = Y.length, k = 2 + 2 * p;
    const beta = [], sdev = [], F = [];
    for (let h = 0; h <= H; h++) {
      const Wr = [], Xr = [], yv = [];
      for (let t = p; t < T - h; t++) {
        const lags = []; for (let q = 1; q <= p; q++) lags.push(Y[t - q][0], Y[t - q][1]);
        Wr.push([1, z[t], ...lags]); Xr.push([1, Y[t][1], ...lags]); yv.push(Y[t + h][j]);
      }
      const n = yv.length;
      if (n <= k + 2) { beta.push(NaN); sdev.push(NaN); F.push(NaN); continue; }
      const WX = N.zeros(k, k), Wy = new Array(k).fill(0);
      for (let s = 0; s < n; s++) { const w = Wr[s], x = Xr[s]; for (let a = 0; a < k; a++) { Wy[a] += w[a] * yv[s]; for (let c = 0; c < k; c++) WX[a][c] += w[a] * x[c]; } }
      let b;
      try { b = N.solve(WX, Wy); } catch (e) { beta.push(NaN); sdev.push(NaN); F.push(NaN); continue; }
      beta.push(b[1]);
      if (!se) continue;
      const L = h + 1;
      const u = yv.map((v, s) => v - Xr[s].reduce((acc, xv, a) => acc + xv * b[a], 0));
      const iWX = N.inv(WX);
      sdev.push(Math.sqrt(Math.max(0, hacQuad(Wr, u, L, iWX, 1))));
      // primera etapa: i_t sobre W
      const WW = N.zeros(k, k), Wi = new Array(k).fill(0);
      for (let s = 0; s < n; s++) { const w = Wr[s]; for (let a = 0; a < k; a++) { Wi[a] += w[a] * Xr[s][1]; for (let c = 0; c < k; c++) WW[a][c] += w[a] * w[c]; } }
      const iWW = N.inv(WW), pi = N.mulv(iWW, Wi);
      const v = Xr.map((x, s) => x[1] - Wr[s].reduce((acc, wv, a) => acc + wv * pi[a], 0));
      const seF = Math.sqrt(Math.max(0, hacQuad(Wr, v, L, iWW, 1)));
      F.push((pi[1] / seF) ** 2);
    }
    return { beta, se: sdev, F };
  }

  /* Elemento (m, m) de M S M' con S = Σ_ℓ w_ℓ Σ_t (g_t g_{t-ℓ}' + g_{t-ℓ} g_t'), g_t = Z_t u_t, Bartlett con L rezagos. */
  function hacQuad(Z, u, L, M, m) {
    const n = u.length, k = Z[0].length;
    // basta con la combinación lineal q_t = (M g_t)_m
    const q = new Float64Array(n);
    for (let t = 0; t < n; t++) { let acc = 0; for (let a = 0; a < k; a++) acc += M[m][a] * Z[t][a]; q[t] = acc * u[t]; }
    let s = 0; for (let t = 0; t < n; t++) s += q[t] * q[t];
    for (let l = 1; l <= L; l++) { const w = 1 - l / (L + 1); let g = 0; for (let t = l; t < n; t++) g += q[t] * q[t - l]; s += 2 * w * g; }
    return s;
  }

  /* Bandas de Cholesky por bootstrap de residuos (diseño recursivo), percentiles 5 y 95. */
  function bootstrapChol(Y, fit, seed, B = 200, H = H_MAX, level = 0.9) {
    const g = N.rng(seed), p = fit.p, T = Y.length, n = 2, U = fit.U, Te = U.length;
    const drawsA = [], drawsB = [];
    for (let r = 0; r < B; r++) {
      const Ys = Y.slice(0, p).map((v) => v.slice());
      for (let t = p; t < T; t++) {
        const u = U[Math.min(Te - 1, Math.floor(g.u() * Te))];
        const v = fit.c.slice();
        for (let j = 1; j <= p; j++) { const Aj = fit.Alist[j - 1], w = Ys[t - j]; for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) v[a] += Aj[a][c] * w[c]; }
        Ys.push([v[0] + u[0], v[1] + u[1]]);
      }
      const f = fitVAR(Ys, p), im = cholImpacts(f.Sigma);
      drawsA.push(varIRF(f.Alist, im.A, H)); drawsB.push(varIRF(f.Alist, im.B, H));
    }
    const q = (arr, pr) => { const s = arr.filter(isFinite).sort((a, b) => a - b); return s.length ? d3q(s, pr) : NaN; };
    const band = (draws) => [0, 1].map((j) => ({
      lo: Array.from({ length: H + 1 }, (_, h) => q(draws.map((d) => d[h][j]), (1 - level) / 2)),
      hi: Array.from({ length: H + 1 }, (_, h) => q(draws.map((d) => d[h][j]), 1 - (1 - level) / 2)) }));
    return { A: band(drawsA), B: band(drawsB) };
  }
  // cuantil tipo 7 (el de numpy y d3) sobre un arreglo ordenado
  function d3q(s, pr) { const hpos = (s.length - 1) * pr, lo = Math.floor(hpos), hi = Math.ceil(hpos); return s[lo] + (s[hi] - s[lo]) * (hpos - lo); }

  /* Todo lo que la página necesita para una muestra. */
  function estimate(par, T, seed, { boot = 200 } = {}) {
    const sim = simulate(par, T, seed);
    const fit = fitVAR(sim.Y), im = cholImpacts(fit.Sigma);
    const irfA = varIRF(fit.Alist, im.A), irfB = varIRF(fit.Alist, im.B);
    const lpX = lpiv(sim.Y, sim.z, 0), lpI = lpiv(sim.Y, sim.z, 1);
    const bands = boot ? bootstrapChol(sim.Y, fit, seed * 7919 + 17, boot) : null;
    return { sim, fit, irfA, irfB, lpX, lpI, bands, truth: truth(par) };
  }

  /* Una réplica del Monte Carlo: respuesta del producto en el horizonte h con cada método. */
  function mcRep(par, T, seed, h) {
    const sim = simulate(par, T, seed);
    const fit = fitVAR(sim.Y), im = cholImpacts(fit.Sigma);
    const a = varIRF(fit.Alist, im.A, h)[h][0], b = varIRF(fit.Alist, im.B, h)[h][0];
    const lp = lpivPoint(sim.Y, sim.z, 0, h);
    return [a, b, lp];
  }
  function lpivPoint(Y, z, j, h, p = P_LAGS) {
    const T = Y.length, k = 2 + 2 * p, WX = N.zeros(k, k), Wy = new Array(k).fill(0);
    const w = new Array(k), x = new Array(k);
    for (let t = p; t < T - h; t++) {
      w[0] = 1; w[1] = z[t]; x[0] = 1; x[1] = Y[t][1];
      for (let q = 1; q <= p; q++) { w[2 * q] = x[2 * q] = Y[t - q][0]; w[2 * q + 1] = x[2 * q + 1] = Y[t - q][1]; }
      const yv = Y[t + h][j];
      for (let a = 0; a < k; a++) { Wy[a] += w[a] * yv; for (let c = 0; c < k; c++) WX[a][c] += w[a] * x[c]; }
    }
    try { return N.solve(WX, Wy)[1]; } catch (e) { return NaN; }
  }

  /* Semillas del Monte Carlo: fijas, independientes de la muestra que se muestra arriba. */
  const MC_REPS = 300;
  const mcSeed = (r) => 1000003 + 7919 * r;
  function summarize(vals) {
    const s = vals.filter(isFinite).sort((a, b) => a - b);
    const q = (pr) => d3q(s, pr);
    return { n: s.length, q05: q(0.05), q25: q(0.25), q50: q(0.5), q75: q(0.75), q95: q(0.95), mean: s.reduce((a, b) => a + b, 0) / s.length };
  }

  window.MAVIDENT = { MC_REPS, mcSeed, summarize, P_LAGS, H_MAX, A_TRUE, B0, simulate, truth, fitVAR, varIRF, cholImpacts, lpiv, lpivPoint, bootstrapChol, estimate, mcRep, quantile: d3q };
})();

/* ------------------------------------------------------------------ gráficas propias */
(function () {
  "use strict";
  if (typeof document === "undefined") return;
  const G = (window.MAVLABCHARTS = window.MAVLABCHARTS || {});

  function csvText(rows) {
    return rows.map((r) => r.map((c) => (typeof c === "string" && /[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n");
  }
  function download(name, rows) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csvText(rows)], { type: "text/csv" }));
    a.download = name + ".csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  /* Misma tarjeta que MAV.lineChart: título, herramientas (tabla, CSV), leyenda, gráfica, tabla. */
  function card(el) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const parts = { title: el.querySelector(".chart-title"), sub: el.querySelector(".chart-sub"), legend: el.querySelector(".legend"),
      plot: el.querySelector(".chart"), tip: el.querySelector(".tooltip"), table: el.querySelector(".table-view"),
      tableBtn: el.querySelector('[data-act="table"]'), csvBtn: el.querySelector('[data-act="csv"]') };
    const paintBtn = () => { parts.tableBtn.textContent = MAV.t(parts.table.classList.contains("on") ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" }); };
    parts.tableBtn.addEventListener("click", () => {
      const on = parts.table.classList.toggle("on"); parts.tableBtn.setAttribute("aria-expanded", String(on)); paintBtn();
    });
    parts.tableBtn.setAttribute("aria-expanded", "false"); paintBtn(); MAV.onLang(paintBtn);
    return parts;
  }
  function fillTable(parts, header, rows) {
    const t = document.createElement("table");
    const hr = t.createTHead().insertRow();
    header.forEach((h) => { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); });
    const tb = t.createTBody();
    rows.forEach((r) => { const tr = tb.insertRow(); r.forEach((c) => { tr.insertCell().textContent = c; }); });
    parts.table.textContent = ""; parts.table.appendChild(t);
  }
  function keySvg(style, w = 26) {
    const s = MAV.style(style);
    return `<svg width="${w}" height="10" aria-hidden="true"><line x1="0" x2="${w}" y1="5" y2="5" stroke="${s.stroke}" stroke-width="2" stroke-dasharray="${s.dash}" stroke-linecap="round"/></svg>`;
  }
  function legendKey(parts, svg, label) {
    const k = document.createElement("span"); k.className = "key"; k.innerHTML = svg;
    const t = document.createElement("span"); t.textContent = label; k.appendChild(t); parts.legend.appendChild(k);
  }
  function tipRow(tip, keyHtml, value, name) {
    const r = document.createElement("div"); r.className = "t-row";
    const key = document.createElement("span"); key.innerHTML = keyHtml;
    const b = document.createElement("b"); b.textContent = value;
    const s = document.createElement("span"); s.textContent = name;
    r.append(key, b, s); tip.appendChild(r);
  }
  let uid = 0;

  /* Manrope dibuja ν, η y ρ casi iguales a v, n y p: las letras griegas se muestran con la fuente matemática de KaTeX. */
  const GREEK = /[α-ωΓΔΘΛΞΠΣΦΨΩ]/;
  G.greek = (root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) =>
      (GREEK.test(n.nodeValue) && !(n.parentElement && n.parentElement.closest(".katex, .gk, title, option")) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT) });
    const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((n) => {
      const svgCtx = n.parentNode.namespaceURI === "http://www.w3.org/2000/svg";
      const frag = document.createDocumentFragment();
      n.nodeValue.split(/([α-ωΓΔΘΛΞΠΣΦΨΩ])/).forEach((part) => {
        if (!part) return;
        if (GREEK.test(part)) {
          const sp = svgCtx ? document.createElementNS("http://www.w3.org/2000/svg", "tspan") : document.createElement("span");
          sp.setAttribute("class", "gk"); sp.setAttribute("style", "font-family: KaTeX_Math, 'Times New Roman', serif; font-style: italic;");
          sp.textContent = part; frag.appendChild(sp);
        } else frag.appendChild(document.createTextNode(part));
      });
      n.parentNode.replaceChild(frag, n);
    });
  };
  G.watchGreek = (root) => {
    let busy = false;
    const obs = new MutationObserver(() => { if (busy) return; busy = true; obs.disconnect(); G.greek(root); obs.observe(root, opts); busy = false; });
    const opts = { childList: true, subtree: true, characterData: true };
    G.greek(root); obs.observe(root, opts);
  };

  /* Pequeños múltiplos de líneas con eje x común, cursor sincronizado, bandas opcionales, tabla y CSV.
     opts: title, sub, x[], xLabel{es,en}, cols(width) -> n, panelHeight, series:[{name, style}],
           panels:[{name, ys:[[]...] (uno por serie, null = ausente), band:{lo[], hi[], name}, yDomain}],
           yFmt, tipFmt, digits, csvName, bandName, note */
  G.multiples = (el, initial) => {
    let o = initial;
    const parts = card(el);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    const id = "mm" + (++uid);
    let W = 0;
    function draw() {
      const panels = o.panels || [];
      if (!panels.length) { svg.selectAll("*").remove(); return; }
      const width = Math.max(300, parts.plot.clientWidth || 640); W = width;
      const cols = Math.max(1, Math.min(panels.length, o.cols ? o.cols(width) : 2));
      const rows = Math.ceil(panels.length / cols);
      const gapX = 18, gapY = 14, ph = o.panelHeight ? o.panelHeight(width) : 170;
      const m = { t: 22, r: 10, b: 30, l: 44 };
      const pw = (width - gapX * (cols - 1)) / cols;
      const height = rows * (ph + m.t + m.b) + (rows - 1) * gapY;
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").selectAll("*").remove();
      svg.append("title").text(MAV.t(o.title));
      const X = o.x, xd = d3.extent(X);
      const yFmt = o.yFmt || ((v) => MAV.num(v, 1)), tipFmt = o.tipFmt || ((v) => MAV.num(v, 2));
      const active = o.series.map((s, j) => panels.some((p) => p.ys[j])).map((v, j) => (v ? j : -1)).filter((j) => j >= 0);
      const geo = panels.map((p, k) => {
        const c = k % cols, r = Math.floor(k / cols);
        const x0 = c * (pw + gapX), y0 = r * (ph + m.t + m.b + gapY);
        const x = d3.scaleLinear().domain(xd).range([x0 + m.l, x0 + pw - m.r]);
        let vals = active.flatMap((j) => p.ys[j]).filter((v) => v != null && isFinite(v));
        if (p.band && !p.yDomain) vals = vals.concat(p.band.lo, p.band.hi).filter((v) => v != null && isFinite(v));
        let yd = p.yDomain || d3.extent(vals.concat([0]));
        if (yd[0] === yd[1]) yd = [yd[0] - 1, yd[1] + 1];
        const y = d3.scaleLinear().domain(yd).nice(4).range([y0 + m.t + ph, y0 + m.t]);
        return { p, x, y, x0, y0 };
      });
      geo.forEach((g, k) => {
        const { p, x, y, x0, y0 } = g;
        const cid = `${id}-c${k}`;
        svg.append("clipPath").attr("id", cid).append("rect").attr("x", x.range()[0]).attr("y", y0 + m.t - 1)
          .attr("width", x.range()[1] - x.range()[0]).attr("height", ph + 2);
        svg.append("text").attr("class", "dlabel").attr("x", x0 + m.l).attr("y", y0 + 13).text(MAV.t(p.name));
        svg.append("g").attr("class", "grid").attr("transform", `translate(${x.range()[0]},0)`)
          .call(d3.axisLeft(y).ticks(4).tickSize(-(x.range()[1] - x.range()[0])).tickFormat(""));
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y0 + m.t + ph})`)
          .call(d3.axisBottom(x).ticks(Math.max(3, Math.round(pw / 70))).tickFormat((v) => MAV.num(v, 0)).tickSizeOuter(0));
        svg.append("g").attr("class", "axis").attr("transform", `translate(${x.range()[0]},0)`)
          .call(d3.axisLeft(y).ticks(4).tickFormat(yFmt).tickSize(0).tickPadding(6)).call((s) => s.select(".domain").remove());
        if (y.domain()[0] < 0 && y.domain()[1] > 0) svg.append("line").attr("class", "zero").attr("x1", x.range()[0]).attr("x2", x.range()[1]).attr("y1", y(0)).attr("y2", y(0));
        const body = svg.append("g").attr("clip-path", `url(#${cid})`);
        if (p.band) {
          const area = d3.area().defined((d, i) => p.band.lo[i] != null && isFinite(p.band.lo[i]) && isFinite(p.band.hi[i]))
            .x((d, i) => x(X[i])).y0((d, i) => y(p.band.lo[i])).y1((d, i) => y(p.band.hi[i]));
          body.append("path").attr("class", "shade").attr("d", area(X));
        }
        const line = d3.line().defined((d) => d[1] != null && isFinite(d[1])).x((d) => x(d[0])).y((d) => y(d[1]));
        // referencia primero, la serie principal encima
        [...active].reverse().forEach((j) => {
          const ys = p.ys[j]; if (!ys) return;
          const st = MAV.style(o.series[j].style ?? j);
          body.append("path").attr("class", "series").attr("d", line(X.map((xv, i) => [xv, ys[i]])))
            .attr("stroke", st.stroke).attr("stroke-dasharray", st.dash).attr("stroke-width", o.series[j].width || 2);
        });
        if (p.clipped) svg.append("text").attr("class", "alabel").attr("x", x.range()[1]).attr("y", y0 + 13).attr("text-anchor", "end").text(MAV.t(p.clipped));
      });

      // leyenda
      parts.legend.innerHTML = "";
      if (active.length > 1 || panels.some((p) => p.band)) {
        active.forEach((j) => legendKey(parts, keySvg(o.series[j].style ?? j), MAV.t(o.series[j].name)));
        if (panels.some((p) => p.band) && o.bandName) legendKey(parts, `<svg width="18" height="10" aria-hidden="true"><rect width="18" height="10" rx="2" fill="var(--band)" stroke="var(--rule)"/></svg>`, MAV.t(o.bandName));
      }

      // cursor sincronizado
      const hairs = geo.map((g) => svg.append("line").attr("class", "xhair").attr("y1", g.y0 + m.t).attr("y2", g.y0 + m.t + ph).style("display", "none"));
      const dots = geo.map((g) => active.map((j) => svg.append("circle").attr("class", "dot").attr("r", 4)
        .attr("fill", MAV.style(o.series[j].style ?? j).stroke).style("display", "none")));
      let cur = -1;
      const show = (k, idx) => {
        idx = Math.max(0, Math.min(X.length - 1, idx)); cur = idx;
        geo.forEach((g, q) => {
          hairs[q].attr("x1", g.x(X[idx])).attr("x2", g.x(X[idx])).style("display", null);
          active.forEach((j, a) => { const v = g.p.ys[j] && g.p.ys[j][idx];
            const inside = v != null && isFinite(v) && v >= g.y.domain()[0] && v <= g.y.domain()[1];
            dots[q][a].style("display", inside ? null : "none").attr("cx", g.x(X[idx])).attr("cy", inside ? g.y(v) : 0); });
        });
        const g = geo[k], tip = parts.tip; tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head";
        head.textContent = `${MAV.t(g.p.name)} · ${MAV.t(o.xTip || o.xLabel)} ${MAV.num(X[idx], 0)}`; tip.appendChild(head);
        active.forEach((j) => { const v = g.p.ys[j] && g.p.ys[j][idx]; if (g.p.ys[j]) tipRow(tip, keySvg(o.series[j].style ?? j, 14), v == null || !isFinite(v) ? "—" : tipFmt(v), MAV.t(o.series[j].name)); });
        if (g.p.band && g.p.band.lo[idx] != null) {
          const r = document.createElement("div"); r.className = "t-row";
          const key = document.createElement("span"); key.innerHTML = `<svg width="14" height="10" aria-hidden="true"><rect width="14" height="10" rx="2" fill="currentColor" opacity=".25"/></svg>`;
          const b = document.createElement("b"); b.textContent = `${tipFmt(g.p.band.lo[idx])} ${MAV.t({ es: "a", en: "to" })} ${tipFmt(g.p.band.hi[idx])}`;
          const s = document.createElement("span"); s.textContent = MAV.t(o.bandName);
          r.append(key, b, s); tip.appendChild(r);
        }
        const sc = parts.plot.clientWidth / W, left = g.x(X[idx]) * sc, tw = tip.offsetWidth || 170;
        tip.style.left = (left + 14 + tw > parts.plot.clientWidth ? Math.max(0, left - tw - 14) : left + 14) + "px";
        tip.style.top = (g.y0 + m.t) * sc + "px"; tip.classList.add("on");
      };
      const hide = () => { hairs.forEach((h) => h.style("display", "none")); dots.flat().forEach((d) => d.style("display", "none")); parts.tip.classList.remove("on"); };
      geo.forEach((g, k) => {
        svg.append("rect").attr("x", g.x.range()[0]).attr("y", g.y0 + m.t).attr("width", g.x.range()[1] - g.x.range()[0]).attr("height", ph)
          .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", `${MAV.t(g.p.name)}: ${MAV.t({ es: "explorar valores", en: "explore values" })}`)
          .on("pointermove", (ev) => { const xv = g.x.invert(d3.pointer(ev)[0]); show(k, d3.bisector((d) => d).center(X, xv)); })
          .on("pointerleave", hide).on("blur", hide).on("focus", () => show(k, cur < 0 ? 0 : cur))
          .on("keydown", (ev) => {
            if (ev.key === "ArrowLeft") { show(k, cur - 1); ev.preventDefault(); }
            if (ev.key === "ArrowRight") { show(k, cur + 1); ev.preventDefault(); }
          });
      });

      parts.title.textContent = MAV.t(o.title);
      parts.sub.textContent = MAV.t(o.sub || ""); parts.sub.style.display = o.sub ? "" : "none";
      const header = [MAV.t(o.xLabel)];
      const cols2 = [];
      panels.forEach((p) => {
        active.forEach((j) => { if (p.ys[j]) { header.push(`${MAV.t(p.name)} · ${MAV.t(o.series[j].short || o.series[j].name)}`); cols2.push(p.ys[j]); } });
        if (p.band) { header.push(`${MAV.t(p.name)} · ${MAV.t(o.bandShort || o.bandName)} ${MAV.t({ es: "inf.", en: "low" })}`); cols2.push(p.band.lo);
          header.push(`${MAV.t(p.name)} · ${MAV.t(o.bandShort || o.bandName)} ${MAV.t({ es: "sup.", en: "high" })}`); cols2.push(p.band.hi); }
      });
      const d = o.digits ?? 3;
      fillTable(parts, header, X.map((xv, i) => [String(xv), ...cols2.map((c) => (c[i] == null || !isFinite(c[i]) ? "" : MAV.num(c[i], d)))]));
      parts.csvBtn.onclick = () => download(o.csvName || "mav-datos", [header, ...X.map((xv, i) => [xv, ...cols2.map((c) => (c[i] == null || !isFinite(c[i]) ? "" : c[i]))])]);
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw); draw();
    return { update(next) { o = Object.assign({}, o, next); draw(); } };
  };

  /* Distribución de estimaciones por método (Monte Carlo): 90 % central (línea), 50 % central (barra),
     mediana (punto), estimación de la muestra actual (anillo) y la verdad (línea vertical).
     opts: title, sub, rows:[{name, stats:{q05,q25,q50,q75,q95,mean,n}, sample}], truth, fmt, csvName, busy */
  G.intervals = (el, initial) => {
    let o = initial;
    const parts = card(el);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    let W = 0;
    function draw() {
      const rows = o.rows || [];
      if (!rows.length) { svg.selectAll("*").remove(); return; }
      const width = Math.max(300, parts.plot.clientWidth || 640); W = width;
      const narrow = width < 520, lw = narrow ? 92 : 150, rowH = 40, top = 22, axisH = 26, rp = 16;
      const height = top + rows.length * rowH + axisH;
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").selectAll("*").remove();
      svg.append("title").text(MAV.t(o.title));
      const fmt = o.fmt || ((v) => MAV.num(v, 2));
      // dominio: lo central de cada método y la verdad; las colas se recortan si están muy lejos
      const core = [o.truth, ...rows.flatMap((r) => [r.stats.q25, r.stats.q75, r.stats.q50, r.sample])].filter(isFinite);
      let lo = Math.min(...core), hi = Math.max(...core);
      const span = Math.max(0.5, hi - lo);
      lo = Math.max(Math.min(lo, ...rows.map((r) => r.stats.q05).filter(isFinite)), lo - span);
      hi = Math.min(Math.max(hi, ...rows.map((r) => r.stats.q95).filter(isFinite)), hi + span);
      const x = d3.scaleLinear().domain([lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)]).nice(6).range([lw, width - rp]);
      const [x0, x1] = x.range();
      const cx = (v) => Math.max(x0, Math.min(x1, x(v)));
      svg.append("g").attr("class", "grid").attr("transform", `translate(0,${top})`)
        .call(d3.axisBottom(x).ticks(narrow ? 4 : 7).tickSize(rows.length * rowH).tickFormat("")).call((g) => g.select(".domain").remove());
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${top + rows.length * rowH})`)
        .call(d3.axisBottom(x).ticks(narrow ? 4 : 7).tickFormat((v) => MAV.num(v, 1)).tickSizeOuter(0));
      if (x.domain()[0] < 0 && x.domain()[1] > 0) svg.append("line").attr("class", "zero").attr("x1", x(0)).attr("x2", x(0)).attr("y1", top).attr("y2", top + rows.length * rowH);
      if (isFinite(o.truth)) {
        svg.append("line").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).attr("stroke-dasharray", "4 3")
          .attr("x1", x(o.truth)).attr("x2", x(o.truth)).attr("y1", top - 6).attr("y2", top + rows.length * rowH);
        svg.append("text").attr("class", "dlabel").attr("x", x(o.truth)).attr("y", top - 9).attr("text-anchor", x(o.truth) > x1 - 60 ? "end" : "middle")
          .text(`${MAV.t({ es: "verdad", en: "truth" })} ${fmt(o.truth)}`);
      }
      const arrow = (xp, y, dir) => svg.append("path").attr("fill", "var(--muted)")
        .attr("d", `M${xp},${y} l${-dir * 7},-4 l0,8 z`);
      const hits = [];
      rows.forEach((r, k) => {
        const cy = top + k * rowH + rowH / 2, st = r.stats;
        svg.append("text").attr("class", "axis-title").attr("x", 0).attr("y", cy + 4).text(MAV.t(narrow ? (r.short || r.name) : r.name));
        if (!st || !isFinite(st.q50)) return;
        svg.append("line").attr("stroke", "var(--muted)").attr("stroke-width", 2).attr("stroke-linecap", "round")
          .attr("x1", cx(st.q05)).attr("x2", cx(st.q95)).attr("y1", cy).attr("y2", cy);
        if (x(st.q05) < x0) arrow(x0, cy, -1);
        if (x(st.q95) > x1) arrow(x1, cy, 1);
        const bw = Math.max(3, cx(st.q75) - cx(st.q25));
        svg.append("rect").attr("x", cx(st.q25) - (bw === 3 ? 1.5 : 0)).attr("y", cy - 7).attr("width", bw).attr("height", 14).attr("rx", 3)
          .attr("fill", "var(--s3)").attr("stroke", "var(--bg)").attr("stroke-width", 2);
        svg.append("circle").attr("class", "dot").attr("cx", cx(st.q50)).attr("cy", cy).attr("r", 5).attr("fill", "var(--s1)");
        if (isFinite(r.sample)) svg.append("circle").attr("cx", cx(r.sample)).attr("cy", cy - 13).attr("r", 4).attr("fill", "var(--bg)").attr("stroke", "var(--s1)").attr("stroke-width", 1.5);
        hits.push({ r, cy });
      });
      const tip = parts.tip;
      const ring = svg.append("rect").attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-width", 1).attr("rx", 3).style("display", "none");
      const show = (h) => {
        const st = h.r.stats;
        ring.attr("x", 0).attr("y", h.cy - rowH / 2).attr("width", width - 2).attr("height", rowH).style("display", null);
        tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head"; head.textContent = MAV.t(h.r.name); tip.appendChild(head);
        const rowsT = [
          [fmt(st.q50), { es: "mediana", en: "median" }],
          [`${fmt(st.q25)} ${MAV.t({ es: "a", en: "to" })} ${fmt(st.q75)}`, { es: "50 % central", en: "middle 50%" }],
          [`${fmt(st.q05)} ${MAV.t({ es: "a", en: "to" })} ${fmt(st.q95)}`, { es: "90 % central", en: "middle 90%" }],
          [fmt(st.q50 - o.truth), { es: "sesgo de la mediana", en: "median bias" }],
          [fmt(h.r.sample), { es: "esta muestra", en: "this sample" }],
        ];
        rowsT.forEach(([v, l]) => { const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "auto 1fr";
          const b = document.createElement("b"); b.textContent = v; const s = document.createElement("span"); s.textContent = MAV.t(l); r.append(b, s); tip.appendChild(r); });
        const sc = parts.plot.clientWidth / W, tw = tip.offsetWidth || 200, left = cx(st.q50) * sc;
        tip.style.left = Math.max(0, Math.min(parts.plot.clientWidth - tw, left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14)) + "px";
        tip.style.top = (h.cy + rowH / 2 + 2) * sc + "px"; tip.classList.add("on");
      };
      const hide = () => { ring.style("display", "none"); tip.classList.remove("on"); };
      hits.forEach((h) => svg.append("rect").attr("x", 0).attr("y", h.cy - rowH / 2).attr("width", width).attr("height", rowH).attr("fill", "transparent")
        .attr("tabindex", 0).attr("aria-label", MAV.t(h.r.name)).on("pointerenter pointermove", () => show(h)).on("pointerleave", hide).on("focus", () => show(h)).on("blur", hide));

      parts.legend.innerHTML = "";
      legendKey(parts, `<svg width="18" height="12" aria-hidden="true"><rect x="1" y="1" width="16" height="10" rx="3" fill="var(--s3)"/></svg>`, MAV.t({ es: "50 % central", en: "middle 50%" }));
      legendKey(parts, `<svg width="22" height="10" aria-hidden="true"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--muted)" stroke-width="2" stroke-linecap="round"/></svg>`, MAV.t({ es: "90 % central", en: "middle 90%" }));
      legendKey(parts, `<svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="var(--s1)"/></svg>`, MAV.t({ es: "mediana", en: "median" }));
      legendKey(parts, `<svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" fill="var(--bg)" stroke="var(--s1)" stroke-width="1.5"/></svg>`, MAV.t({ es: "esta muestra", en: "this sample" }));
      legendKey(parts, `<svg width="12" height="14" aria-hidden="true"><line x1="6" x2="6" y1="0" y2="14" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="4 3"/></svg>`, MAV.t({ es: "verdad", en: "truth" }));
      parts.title.textContent = MAV.t(o.title);
      parts.sub.textContent = MAV.t(o.sub || ""); parts.sub.style.display = o.sub ? "" : "none";
      const header = [MAV.t({ es: "Método", en: "Method" }), "p5", "p25", MAV.t({ es: "mediana", en: "median" }), "p75", "p95",
        MAV.t({ es: "media", en: "mean" }), MAV.t({ es: "verdad", en: "truth" }), MAV.t({ es: "esta muestra", en: "this sample" }), MAV.t({ es: "réplicas", en: "replications" })];
      const raw = rows.map((r) => [MAV.t(r.name), r.stats.q05, r.stats.q25, r.stats.q50, r.stats.q75, r.stats.q95, r.stats.mean, o.truth, r.sample, r.stats.n]);
      fillTable(parts, header, raw.map((r) => r.map((c, j) => (j === 0 ? c : j === 9 ? String(c) : isFinite(c) ? MAV.num(c, 3) : ""))));
      parts.csvBtn.onclick = () => download(o.csvName || "mav-datos", [header, ...raw]);
      el.style.opacity = o.busy ? "0.55" : "";
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw); draw();
    return { update(next) { o = Object.assign({}, o, next); draw(); } };
  };
})();

/* ------------------------------------------------------------------ página */
(function () {
  "use strict";
  if (typeof document === "undefined" || !window.MAV) return;
  const I = window.MAVIDENT, C = window.MAVLABCHARTS, T = MAV.t, num = MAV.num;
  const H = I.H_MAX, hs = Array.from({ length: H + 1 }, (_, h) => h);
  const state = { theta: 0, phi: 0.5, noise: 1, T: 200, seed: 1, band: "lp", hmc: 4 };
  const ctl = document.getElementById("controls");
  const signed = (v, d = 2) => (!isFinite(v) ? "—" : (v >= 0 ? "+" : "−") + num(Math.abs(v), d));

  MAV.slider(ctl, { key: "theta", label: { es: "Efecto en el impacto θ", en: "Impact effect θ" },
    min: -1, max: 0.5, step: 0.05, value: state.theta, fmt: (v) => signed(v, 2),
    note: { es: "Cuánto mueve al producto, en el mismo trimestre, un choque monetario que sube la tasa 1 pp. Cholesky A (producto primero) supone θ = 0.",
            en: "How much a monetary shock that raises the rate 1 pp moves output within the quarter. Cholesky A (output first) assumes θ = 0." } }, (v) => { state.theta = v; schedule(); });
  MAV.slider(ctl, { key: "phi", label: { es: "Reacción de la tasa φ", en: "Rate reaction φ" },
    min: 0, max: 1.5, step: 0.05, value: state.phi, fmt: (v) => num(v, 2),
    note: { es: "Cuánto sube la tasa, en el mismo trimestre, tras un choque de demanda. Cholesky B (tasa primero) supone φ = 0.",
            en: "How much the rate rises within the quarter after a demand shock. Cholesky B (rate first) assumes φ = 0." } }, (v) => { state.phi = v; schedule(); });
  MAV.slider(ctl, { key: "noise", label: { es: "Ruido en z", en: "Noise in z" },
    min: 0, max: 8, step: 0.1, value: state.noise, fmt: (v) => `${num(v, 1)} · corr ${num(1 / Math.sqrt(1 + v * v), 2)}`,
    note: { es: "z = choque monetario + ruido, como una serie narrativa medida con error; corr es su correlación con el choque verdadero.",
            en: "z = monetary shock + noise, like a narrative series measured with error; corr is its correlation with the true shock." } }, (v) => { state.noise = v; schedule(); });
  MAV.slider(ctl, { key: "T", label: { es: "Tamaño de muestra T", en: "Sample size T" }, min: 60, max: 1000, step: 20, value: state.T,
    fmt: (v) => num(v, 0), note: { es: "Trimestres; 200 son 50 años", en: "Quarters; 200 is 50 years" } }, (v) => { state.T = v; schedule(); });

  const row = document.createElement("div"); row.className = "btn-row"; row.style.margin = "-6px 0 16px"; row.style.alignItems = "center";
  const bNew = document.createElement("button"); bNew.type = "button"; bNew.className = "btn btn-primary";
  const seedLab = document.createElement("span"); seedLab.className = "ctl-note"; seedLab.style.margin = "0";
  row.append(bNew, seedLab); ctl.appendChild(row);
  bNew.addEventListener("click", () => { state.seed += 1; schedule(); });

  MAV.segmented(ctl, { key: "band", label: { es: "Banda de 90 % en la gráfica", en: "90% band on the chart" }, value: state.band,
    options: [{ value: "lp", label: "LP-IV" }, { value: "A", label: "Cholesky A" }, { value: "B", label: "Cholesky B" }] }, (v) => { state.band = v; schedule(); });
  MAV.segmented(ctl, { key: "hmc", label: { es: "Horizonte del Monte Carlo", en: "Monte Carlo horizon" }, value: state.hmc,
    options: [0, 4, 8].map((h) => ({ value: h, label: `h = ${h}` })) }, (v) => { state.hmc = v; schedule(); });

  const bCsv = document.createElement("button"); bCsv.type = "button"; bCsv.className = "btn"; bCsv.style.marginTop = "4px";
  ctl.appendChild(bCsv);
  const paintButtons = () => {
    bNew.textContent = T({ es: "Nueva muestra", en: "New sample" });
    seedLab.textContent = T({ es: `muestra n.º ${state.seed}`, en: `sample no. ${state.seed}` });
    bCsv.textContent = T({ es: "Descargar la muestra (CSV)", en: "Download the sample (CSV)" });
  };
  MAV.onLang(paintButtons);

  const tiles = document.getElementById("tiles");
  const chIrf = C.multiples(document.getElementById("chart-irf"), { panels: [], series: [], x: [] });
  const chMc = C.intervals(document.getElementById("chart-mc"), { rows: [] });
  let current = null;

  bCsv.addEventListener("click", () => {
    if (!current) return;
    const { sim } = current;
    const rows = [["t", "x_producto", "i_tasa", "z_instrumento", "eps_demanda", "eps_monetario"],
      ...sim.Y.map((y, t) => [t + 1, y[0], y[1], sim.z[t], sim.eps[t][0], sim.eps[t][1]])];
    const text = rows.map((r) => r.join(",")).join("\n");
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = `svar-muestra-${state.seed}-T${state.T}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  let pending = false;
  function schedule() {
    if (pending) return; pending = true;
    const go = () => { if (!pending) return; pending = false; run(); };
    requestAnimationFrame(go); setTimeout(go, 60);
  }

  const NAMES = {
    truth: { es: "Verdad", en: "Truth" },
    A: { es: "Cholesky A: producto primero", en: "Cholesky A: output first" },
    B: { es: "Cholesky B: tasa primero", en: "Cholesky B: rate first" },
    lp: { es: "LP-IV con el instrumento z", en: "LP-IV with instrument z" },
  };

  function run() {
    paintButtons();
    const par = { theta: state.theta, phi: state.phi, noise: state.noise };
    const e = I.estimate(par, state.T, state.seed, { boot: 200 });
    current = e;
    const z90 = 1.6448536269514722;
    const lpBand = (res) => ({ lo: res.beta.map((b, h) => b - z90 * res.se[h]), hi: res.beta.map((b, h) => b + z90 * res.se[h]) });
    const bands = state.band === "lp" ? [lpBand(e.lpX), lpBand(e.lpI)] : e.bands[state.band];
    const bandName = {
      lp: { es: "Banda de 90 % de la LP-IV (Newey–West)", en: "LP-IV 90% band (Newey–West)" },
      A: { es: "Banda de 90 % de Cholesky A (bootstrap)", en: "Cholesky A 90% band (bootstrap)" },
      B: { es: "Banda de 90 % de Cholesky B (bootstrap)", en: "Cholesky B 90% band (bootstrap)" },
    }[state.band];

    const F0 = e.lpX.F[0];
    const lo0 = e.lpX.beta[0] - z90 * e.lpX.se[0], hi0 = e.lpX.beta[0] + z90 * e.lpX.se[0];
    const tv = signed(e.truth[0][0]);
    MAV.tiles(tiles, [
      { label: { es: "Cholesky A en el impacto", en: "Cholesky A on impact" }, value: signed(e.irfA[0][0]), note: { es: `verdad: ${tv} pp`, en: `truth: ${tv} pp` } },
      { label: { es: "Cholesky B en el impacto", en: "Cholesky B on impact" }, value: signed(e.irfB[0][0]), note: { es: `verdad: ${tv} pp`, en: `truth: ${tv} pp` } },
      { label: { es: "LP-IV en el impacto", en: "LP-IV on impact" }, value: signed(e.lpX.beta[0]),
        note: { es: `IC 90 %: ${signed(lo0)} a ${signed(hi0)}`, en: `90% CI: ${signed(lo0)} to ${signed(hi0)}` } },
      { label: { es: "F de la primera etapa", en: "First-stage F" }, value: num(F0, F0 < 100 ? 1 : 0),
        note: F0 < 10 ? { es: "débil: menor que 10", en: "weak: below 10" } : F0 < 23.1 ? { es: "menor que 23.1 (Montiel Olea–Pflueger)", en: "below 23.1 (Montiel Olea–Pflueger)" } : { es: "fuerte: mayor que 23.1", en: "strong: above 23.1" } },
    ]);

    // eje y: verdad y estimaciones puntuales; la banda entra hasta cierto punto y, si no cabe, se recorta
    const panelDomain = (j, band) => {
      const pts = [e.truth.map((v) => v[j]), e.irfA.map((v) => v[j]), e.irfB.map((v) => v[j]), (j ? e.lpI : e.lpX).beta].flat().filter(isFinite).concat([0]);
      let lo = Math.min(...pts), hi = Math.max(...pts); const span = Math.max(0.5, hi - lo);
      const blo = Math.min(...band.lo.filter(isFinite)), bhi = Math.max(...band.hi.filter(isFinite));
      const clipped = blo < lo - span || bhi > hi + span;
      return { dom: [Math.max(blo, lo - span), Math.min(bhi, hi + span)].map((v, k) => (k ? Math.max(v, hi) : Math.min(v, lo))), clipped };
    };
    const dX = panelDomain(0, bands[0]), dI = panelDomain(1, bands[1]);
    const clipTxt = { es: "banda recortada", en: "band clipped" };
    chIrf.update({
      title: { es: "Respuesta a un alza de 1 pp en la tasa: verdad frente a estimaciones", en: "Response to a 1 pp rate hike: truth versus estimates" },
      sub: { es: `Una muestra simulada de ${state.T} trimestres (n.º ${state.seed}). Cada método normaliza su choque para que la tasa suba 1 pp en el impacto; VAR y LP con ${I.P_LAGS} rezagos.`,
             en: `One simulated sample of ${state.T} quarters (no. ${state.seed}). Each method scales its shock so that the rate rises 1 pp on impact; VAR and LP with ${I.P_LAGS} lags.` },
      x: hs, xLabel: { es: "trimestres", en: "quarters" }, xTip: { es: "h =", en: "h =" },
      cols: (w) => (w >= 600 ? 2 : 1), panelHeight: (w) => (w >= 600 ? 240 : 200),
      series: [
        { name: NAMES.truth, style: 0 }, { name: NAMES.A, short: "Cholesky A", style: 1 },
        { name: NAMES.B, short: "Cholesky B", style: 2 }, { name: NAMES.lp, short: "LP-IV", style: 3 },
      ],
      panels: [
        { name: { es: "Producto x, pp", en: "Output x, pp" }, ys: [e.truth.map((v) => v[0]), e.irfA.map((v) => v[0]), e.irfB.map((v) => v[0]), e.lpX.beta],
          band: bands[0], yDomain: dX.dom, clipped: dX.clipped ? clipTxt : null },
        { name: { es: "Tasa de política i, pp", en: "Policy rate i, pp" }, ys: [e.truth.map((v) => v[1]), e.irfA.map((v) => v[1]), e.irfB.map((v) => v[1]), e.lpI.beta],
          band: bands[1], yDomain: dI.dom, clipped: dI.clipped ? clipTxt : null },
      ],
      bandName, bandShort: { es: "banda 90 %", en: "90% band" },
      yFmt: (v) => num(v, Math.abs(v - Math.round(v)) < 1e-9 ? 0 : Math.abs(10 * v - Math.round(10 * v)) < 1e-9 ? 1 : 2),
      tipFmt: (v) => signed(v, 2), csvName: `identificacion-irf-muestra${state.seed}`, digits: 4,
    });
    document.getElementById("source").textContent = T({
      es: `Datos simulados: VAR(1) estructural con A = [[0.8, −0.2], [0.1, 0.85]] y B0 = [[1, θ], [φ, 1]], choques normales independientes, ${200} periodos de calentamiento. Semilla de la muestra: ${state.seed}; el Monte Carlo usa ${I.MC_REPS} semillas fijas propias.`,
      en: `Simulated data: structural VAR(1) with A = [[0.8, −0.2], [0.1, 0.85]] and B0 = [[1, θ], [φ, 1]], independent normal shocks, ${200} burn-in periods. Sample seed: ${state.seed}; the Monte Carlo uses its own ${I.MC_REPS} fixed seeds.` });
    startMC(par, e);
  }

  /* Monte Carlo por partes, para no congelar la página; un cambio de controles cancela la corrida anterior. */
  let mcToken = 0;
  const mcCache = new Map();
  function startMC(par, e) {
    const h = state.hmc, key = [par.theta, par.phi, par.noise, state.T, h].join("|");
    const token = ++mcToken;
    const sample = [e.irfA[h][0], e.irfB[h][0], e.lpX.beta[h]];
    const paint = (reps, busy) => {
      const cols = [0, 1, 2].map((j) => I.summarize(reps.map((r) => r[j])));
      chMc.update({
        title: { es: `Monte Carlo: ${I.MC_REPS} muestras del mismo proceso`, en: `Monte Carlo: ${I.MC_REPS} samples from the same process` },
        sub: { es: `Respuesta del producto en h = ${h} a un alza de 1 pp en la tasa, con T = ${state.T}. Cada fila resume las ${reps.length} estimaciones de un método.`,
               en: `Output response at h = ${h} to a 1 pp rate hike, with T = ${state.T}. Each row summarizes one method's ${reps.length} estimates.` },
        truth: e.truth[h][0], busy,
        rows: [
          { name: NAMES.A, short: "Cholesky A", stats: cols[0], sample: sample[0] },
          { name: NAMES.B, short: "Cholesky B", stats: cols[1], sample: sample[1] },
          { name: NAMES.lp, short: "LP-IV", stats: cols[2], sample: sample[2] },
        ],
        fmt: (v) => signed(v, 2), csvName: `identificacion-montecarlo-h${h}-T${state.T}`,
      });
    };
    if (mcCache.has(key)) { paint(mcCache.get(key), false); return; }
    chMc.update({ busy: true });
    const reps = [];
    const step = () => {
      if (token !== mcToken) return;
      const t0 = performance.now();
      while (reps.length < I.MC_REPS && performance.now() - t0 < 24) reps.push(I.mcRep(par, state.T, I.mcSeed(reps.length), h));
      if (reps.length < I.MC_REPS) { setTimeout(step, 0); return; }
      if (mcCache.size > 40) mcCache.clear();
      mcCache.set(key, reps);
      paint(reps, false);
    };
    setTimeout(step, 30);
  }

  MAV.onLang(run);
  run();
  C.watchGreek(document.querySelector("main"));
})();
