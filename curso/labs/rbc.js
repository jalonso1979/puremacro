/* Laboratorio «RBC»: el modelo canónico log-linealizado, sus impulsos-respuesta y los momentos
   de Kydland–Prescott frente a los datos de EE. UU. */

/* ------------------------------------------------------------------ modelo (sin DOM) */
(function () {
  "use strict";
  const N = window.MAVNUM;
  const V = ["y", "c", "k", "l", "i", "w", "r", "a"];
  const ix = Object.fromEntries(V.map((v, j) => [v, j]));
  const L_SS = 0.33;

  /* Estado estacionario con horas fijas en l = 0.33: μ se recalibra para cada ν. */
  function steady(p) {
    const r = 1 / p.beta - 1 + p.delta;
    const kl = (p.alpha / r) ** (1 / (1 - p.alpha));
    const l = L_SS, k = kl * l, y = kl ** p.alpha * l, i = p.delta * k, c = y - i;
    const w = (1 - p.alpha) * y / l;
    const mu = w * (1 - l) ** p.nu / c;
    return { y, c, k, l, i, w, r, a: 0, mu, cy: c / y, iy: i / y, ky: k / y };
  }

  /* A0 x_t = A1 E_t x_{t+1} + A2 x_{t-1} + A3 e_t, x en desviaciones logarítmicas. */
  function system(p, ss) {
    const n = V.length, A0 = N.zeros(n, n), A1 = N.zeros(n, n), A2 = N.zeros(n, n), A3 = N.zeros(n, 1);
    const { y, c, k, l, i, w, r, a } = ix;
    // 0. oferta de trabajo: c + ν l/(1-l) l − w = 0
    A0[0][c] = 1; A0[0][l] = p.nu * ss.l / (1 - ss.l); A0[0][w] = -1;
    // 1. Euler: c_t = E c_{t+1} − β r̄ E r_{t+1}
    A0[1][c] = 1; A1[1][c] = 1; A1[1][r] = -p.beta * ss.r;
    // 2. renta del capital: r = y − k_{t-1}
    A0[2][r] = 1; A0[2][y] = -1; A2[2][k] = -1;
    // 3. salario: w = y − l
    A0[3][w] = 1; A0[3][y] = -1; A0[3][l] = 1;
    // 4. producción: y = a + α k_{t-1} + (1−α) l
    A0[4][y] = 1; A0[4][a] = -1; A0[4][l] = -(1 - p.alpha); A2[4][k] = p.alpha;
    // 5. recursos: y = (c/y) c + (i/y) i
    A0[5][y] = 1; A0[5][c] = -ss.cy; A0[5][i] = -ss.iy;
    // 6. capital: k_t = (1−δ) k_{t-1} + δ i
    A0[6][k] = 1; A0[6][i] = -p.delta; A2[6][k] = 1 - p.delta;
    // 7. PTF: a = ρ a_{t-1} + e
    A0[7][a] = 1; A2[7][a] = p.rho; A3[7][0] = 1;
    return { A0, A1, A2, A3 };
  }

  /* Coeficientes indeterminados (mazo 4): k_t = ψkk k_{t-1} + ψka a_t, c_t = ψck k_{t-1} + ψca a_t.
     Solución cerrada; no depende de iteraciones, así que también sirve en el límite de Hansen (ν = 0),
     donde la iteración hacia atrás de MAVNUM.solveRE converge a la raíz explosiva.
     Devuelve {P, Q} con x_t = P x_{t-1} + Q e_t sobre x = [y, c, k, l, i, w, r, a]. */
  function solve(p) {
    const ss = steady(p);
    const al = p.alpha, chi = p.nu * ss.l / (1 - ss.l), br = p.beta * ss.r;
    // bloque estático como función lineal de (k_{t-1}, a_t, c_t): [coef k, coef a, coef c]
    const l = [al / (al + chi), 1 / (al + chi), -1 / (al + chi)];
    const y = [al + (1 - al) * l[0], 1 + (1 - al) * l[1], (1 - al) * l[2]];
    const inv = [y[0] / ss.iy, y[1] / ss.iy, (y[2] - ss.cy) / ss.iy];
    const kn = [(1 - p.delta) + p.delta * inv[0], p.delta * inv[1], p.delta * inv[2]];
    const g = 1 - br * y[2], h = -br * (y[0] - 1), m = -br * y[1];
    // g kc ψ² + (g kk + h kc − 1) ψ + h kk = 0, raíz estable |kk + kc ψ| < 1
    const qa = g * kn[2], qb = g * kn[0] + h * kn[2] - 1, qc = h * kn[0];
    const disc = qb * qb - 4 * qa * qc;
    let psi_ck = NaN, psi_kk = NaN, ok = false;
    if (disc >= 0 && Math.abs(qa) > 1e-300) {
      const qq = -0.5 * (qb + (qb >= 0 ? 1 : -1) * Math.sqrt(disc));   // fórmula sin cancelación
      const roots = [qq / qa, qc / qq]
        .map((c) => ({ c, kk: kn[0] + kn[2] * c })).filter((o) => Math.abs(o.kk) < 1);
      if (roots.length === 1) { psi_ck = roots[0].c; psi_kk = roots[0].kk; ok = true; }
    }
    const gk = g * psi_ck + h;
    const psi_ca = (gk * kn[1] + m * p.rho) / (1 - gk * kn[2] - g * p.rho);
    const psi_ka = kn[1] + kn[2] * psi_ca;
    // cada variable: v = v_k k_{t-1} + v_a a_t con c sustituida
    const lin = (co) => [co[0] + co[2] * psi_ck, co[1] + co[2] * psi_ca];
    const rows = {
      y: lin(y), c: [psi_ck, psi_ca], k: [psi_kk, psi_ka], l: lin(l), i: lin(inv),
      w: lin([y[0] - l[0], y[1] - l[1], y[2] - l[2]]), r: lin([y[0] - 1, y[1], y[2]]), a: [0, 1],
    };
    const n = V.length, P = N.zeros(n, n), Q = N.zeros(n, 1);
    V.forEach((v, j) => { P[j][ix.k] = rows[v][0]; P[j][ix.a] = rows[v][1] * p.rho; Q[j][0] = rows[v][1]; });
    return { ss, P, Q, ok, psi: { kk: psi_kk, ka: psi_ka, ck: psi_ck, ca: psi_ca } };
  }

  /* IRF a un choque de una desviación estándar, en % (100 × desviación logarítmica). */
  function irf(sol, sigma, H = 40) {
    const path = N.irf(sol.P, sol.Q, [sigma], H + 1);
    return Object.fromEntries(V.map((v, j) => [v, path.map((x) => 100 * x[j])]));
  }

  /* Momentos teóricos del componente cíclico HP(λ) por integración espectral.
     x_t = P x_{t-1} + Q e_t; con estados s = (k, a): x_t = G s_{t-1} + Q e_t, s_t = M s_{t-1} + Nn e_t.
     Respuesta en frecuencia X(ω) = Q + G (I − M z)^{-1} Nn z, z = e^{-iω}.
     Γ_0 = (σ²/π) ∫_0^π |H(ω)|² Re[X X*] dω, γ_ii(1) = (σ²/π) ∫_0^π cos ω |H(ω)|² |X_i|² dω. */
  function moments(sol, sigma, lambda = 1600, nGrid = 4096) {
    const n = V.length, s = [ix.k, ix.a];
    const P = sol.P, Q = sol.Q.map((row) => row[0]);
    const M = [[P[s[0]][s[0]], P[s[0]][s[1]]], [P[s[1]][s[0]], P[s[1]][s[1]]]];
    const Nn = [Q[s[0]], Q[s[1]]];
    const G0 = N.zeros(n, n), g1 = new Array(n).fill(0);
    const dw = Math.PI / nGrid;
    const xr = new Float64Array(n), xi = new Float64Array(n);
    for (let q = 0; q < nGrid; q++) {
      const om = (q + 0.5) * dw, co = Math.cos(om), si = Math.sin(om);
      const u = 1 - co, H = 4 * lambda * u * u / (1 + 4 * lambda * u * u), H2 = H * H;
      // z = e^{-iω} = co − i si ; D = I − M z (2×2 complex); v = D^{-1} Nn z
      const zr = co, zi = -si;
      const d11r = 1 - M[0][0] * zr, d11i = -M[0][0] * zi, d12r = -M[0][1] * zr, d12i = -M[0][1] * zi;
      const d21r = -M[1][0] * zr, d21i = -M[1][0] * zi, d22r = 1 - M[1][1] * zr, d22i = -M[1][1] * zi;
      const detr = d11r * d22r - d11i * d22i - (d12r * d21r - d12i * d21i);
      const deti = d11r * d22i + d11i * d22r - (d12r * d21i + d12i * d21r);
      const dd = detr * detr + deti * deti;
      // b = Nn z (complex 2-vector)
      const b1r = Nn[0] * zr, b1i = Nn[0] * zi, b2r = Nn[1] * zr, b2i = Nn[1] * zi;
      // adj(D) b = [d22 b1 − d12 b2 ; −d21 b1 + d11 b2]
      const t1r = d22r * b1r - d22i * b1i - (d12r * b2r - d12i * b2i);
      const t1i = d22r * b1i + d22i * b1r - (d12r * b2i + d12i * b2r);
      const t2r = -(d21r * b1r - d21i * b1i) + (d11r * b2r - d11i * b2i);
      const t2i = -(d21r * b1i + d21i * b1r) + (d11r * b2i + d11i * b2r);
      const v1r = (t1r * detr + t1i * deti) / dd, v1i = (t1i * detr - t1r * deti) / dd;
      const v2r = (t2r * detr + t2i * deti) / dd, v2i = (t2i * detr - t2r * deti) / dd;
      for (let j = 0; j < n; j++) {
        xr[j] = Q[j] + P[j][s[0]] * v1r + P[j][s[1]] * v2r;
        xi[j] = P[j][s[0]] * v1i + P[j][s[1]] * v2i;
      }
      for (let j = 0; j < n; j++) {
        const m = xr[j] * xr[j] + xi[j] * xi[j];
        G0[j][j] += H2 * m; g1[j] += co * H2 * m;
        for (let k = j + 1; k < n; k++) G0[j][k] += H2 * (xr[j] * xr[k] + xi[j] * xi[k]);
      }
    }
    const f = sigma * sigma / Math.PI * dw * 1e4;          // 1e4: resultados en %²
    for (let j = 0; j < n; j++) { G0[j][j] *= f; g1[j] *= f; for (let k = j + 1; k < n; k++) { G0[j][k] *= f; G0[k][j] = G0[j][k]; } }
    const sd = G0.map((row, j) => Math.sqrt(row[j]));
    const corr = (a, b) => G0[ix[a]][ix[b]] / (sd[ix[a]] * sd[ix[b]]);
    const out = {};
    // la productividad media del trabajo y/l coincide con w en log-desviaciones (Cobb–Douglas)
    [["y", "y"], ["c", "c"], ["i", "i"], ["h", "l"], ["p", "w"]].forEach(([key, v]) => {
      const j = ix[v];
      out[key] = { sd: sd[j], rel: sd[j] / sd[ix.y], corr: corr(v, "y"), ac1: g1[j] / G0[j][j] };
    });
    out.corr_h_p = corr("l", "w");
    return out;
  }

  /* Comprobación con el solucionador genérico compartido (solo para validar; no se usa en la página). */
  function solveGeneric(p) {
    const ss = steady(p), S = system(p, ss);
    return Object.assign({ ss }, N.solveRE(S.A0, S.A1, S.A2, S.A3, { tol: 1e-13, maxIter: 20000 }));
  }

  const lab = { V, L_SS, steady, system, solve, solveGeneric, irf, moments,
    frisch: (nu) => (nu > 0 ? (1 - L_SS) / (nu * L_SS) : Infinity) };
  window.MAVRBC = lab;
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

  /* Gráfica de puntos por facetas (modelo frente a datos), apiladas; cada fila tiene su lectura al pasar el cursor.
     opts: title, sub, marks:[{name, kind:'fill'|'ring'}], facets:[{name, domain, ref:{x,label}, rows:[{label, values:[a,b], sep, emphasize}]}],
           fmt, csvName, table:{header, rows} */
  G.dotFacets = (el, initial) => {
    let o = initial;
    const parts = card(el);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    let W = 0;
    const markSvg = (kind, r = 5) => `<svg width="${2 * r + 4}" height="${2 * r + 4}" aria-hidden="true"><circle cx="${r + 2}" cy="${r + 2}" r="${r}" ${kind === "fill" ? 'fill="var(--s1)"' : 'fill="var(--bg)" stroke="var(--s1)" stroke-width="2"'}/></svg>`;
    function draw() {
      const width = Math.max(300, parts.plot.clientWidth || 640); W = width;
      const narrow = width < 520;
      const lw = narrow ? 104 : 150, rowH = 24, headH = 26, axisH = 26, gap = 18, rp = 14;
      let yy = 0;
      const lay = o.facets.map((f) => { const top = yy; const nSep = f.rows.filter((r) => r.sep).length;
        const h = headH + f.rows.length * rowH + nSep * 8 + axisH; yy += h + gap; return { f, top, h }; });
      const height = yy - gap;
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").selectAll("*").remove();
      svg.append("title").text(MAV.t(o.title));
      const fmt = o.fmt || ((v) => MAV.num(v, 2));
      const hits = [];
      lay.forEach(({ f, top }) => {
        const x = d3.scaleLinear().domain(f.domain).nice(5).range([lw, width - rp]);
        let ry = top + headH;
        const rowsY = f.rows.map((r) => { if (r.sep) ry += 8; const c = ry + rowH / 2; ry += rowH; return c; });
        const bottom = ry;
        svg.append("text").attr("class", "dlabel").attr("x", 0).attr("y", top + 15).text(MAV.t(f.name));
        svg.append("g").attr("class", "grid").attr("transform", `translate(0,${top + headH})`)
          .call(d3.axisBottom(x).ticks(narrow ? 4 : 6).tickSize(bottom - top - headH).tickFormat(""))
          .call((g) => g.select(".domain").remove());
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${bottom})`)
          .call(d3.axisBottom(x).ticks(narrow ? 4 : 6).tickFormat((v) => MAV.num(v, Math.abs(f.domain[1] - f.domain[0]) <= 2.5 ? 1 : 0)).tickSizeOuter(0));
        if (f.ref && f.ref.x >= x.domain()[0] && f.ref.x <= x.domain()[1]) {
          svg.append("line").attr("class", "zero").attr("x1", x(f.ref.x)).attr("x2", x(f.ref.x)).attr("y1", top + headH).attr("y2", bottom);
        }
        f.rows.forEach((r, k) => {
          const cy = rowsY[k];
          if (r.sep) svg.append("line").attr("stroke", "var(--rule)").attr("x1", 0).attr("x2", width - rp).attr("y1", cy - rowH / 2 - 4).attr("y2", cy - rowH / 2 - 4);
          svg.append("text").attr("class", r.emphasize ? "dlabel" : "axis-title").attr("x", 0).attr("y", cy + 4).text(MAV.t(r.label));
          const [a, b] = r.values;
          const clamp = (v) => Math.max(x.range()[0], Math.min(x.range()[1], x(v)));
          if (isFinite(a) && isFinite(b)) svg.append("line").attr("stroke", "var(--faint)").attr("stroke-width", 1).attr("x1", clamp(a)).attr("x2", clamp(b)).attr("y1", cy).attr("y2", cy);
          if (isFinite(b)) svg.append("circle").attr("cx", clamp(b)).attr("cy", cy).attr("r", 5).attr("fill", "var(--bg)").attr("stroke", "var(--s1)").attr("stroke-width", 2);
          if (isFinite(a)) svg.append("circle").attr("class", "dot").attr("cx", clamp(a)).attr("cy", cy).attr("r", 5).attr("fill", "var(--s1)");
          if (r.emphasize && isFinite(a) && isFinite(b) && Math.abs(x(a) - x(b)) > 70) {
            [[a, "model"], [b, "data"]].forEach(([v]) => svg.append("text").attr("class", "alabel").attr("x", clamp(v)).attr("y", cy - 9).attr("text-anchor", "middle").text(fmt(v)));
          }
          hits.push({ f, r, cy, top: cy - rowH / 2, x: x.range()[0], w: x.range()[1] - x.range()[0], xa: clamp(isFinite(a) ? a : b) });
        });
      });
      const tip = parts.tip;
      const ring = svg.append("rect").attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-width", 1).attr("rx", 3).style("display", "none");
      const show = (h) => {
        ring.attr("x", 0).attr("y", h.top).attr("width", width - 2).attr("height", rowH).style("display", null);
        tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head"; head.textContent = `${MAV.t(h.r.label)} · ${MAV.t(h.f.name)}`; tip.appendChild(head);
        o.marks.forEach((mk, j) => tipRow(tip, markSvg(mk.kind, 4), isFinite(h.r.values[j]) ? fmt(h.r.values[j]) : "—", MAV.t(mk.name)));
        const sc = parts.plot.clientWidth / W, tw = tip.offsetWidth || 180, left = h.xa * sc;
        tip.style.left = Math.max(0, Math.min(parts.plot.clientWidth - tw, left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14)) + "px";
        tip.style.top = (h.top + rowH + 4) * sc + "px"; tip.classList.add("on");
      };
      const hide = () => { ring.style("display", "none"); tip.classList.remove("on"); };
      hits.forEach((h) => svg.append("rect").attr("x", 0).attr("y", h.top).attr("width", width).attr("height", rowH).attr("fill", "transparent")
        .attr("tabindex", 0).attr("aria-label", `${MAV.t(h.r.label)}, ${MAV.t(h.f.name)}`)
        .on("pointerenter pointermove", () => show(h)).on("pointerleave", hide).on("focus", () => show(h)).on("blur", hide));

      parts.legend.innerHTML = "";
      o.marks.forEach((mk) => legendKey(parts, markSvg(mk.kind, 5), MAV.t(mk.name)));
      parts.title.textContent = MAV.t(o.title);
      parts.sub.textContent = MAV.t(o.sub || ""); parts.sub.style.display = o.sub ? "" : "none";
      const T = o.table();
      fillTable(parts, T.header, T.rows.map((r) => r.map((c, j) => (j === 0 ? c : (c == null || !isFinite(c) ? "" : MAV.num(c, 3))))));
      parts.csvBtn.onclick = () => download(o.csvName || "mav-datos", [T.header, ...T.rows]);
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw); draw();
    return { update(next) { o = Object.assign({}, o, next); draw(); } };
  };
})();

/* ------------------------------------------------------------------ página */
(async function () {
  "use strict";
  if (typeof document === "undefined" || !window.MAV) return;
  const R = window.MAVRBC, C = window.MAVLABCHARTS, T = MAV.t;
  const data = await MAV.json("../data/rbc.json");

  const BASE = { alpha: 0.33, beta: 0.99, delta: 0.025, rho: 0.95, sigma: 0.007, nu: 1.5 };
  const NU = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10];
  const state = { p: { ...BASE }, ref: { ...BASE }, muestra: "1948-2019" };
  const ctl = document.getElementById("controls");
  const num = MAV.num;
  const frischTxt = (nu) => (nu === 0 ? "∞" : num(R.frisch(nu), 2));
  const PNAMES = { alpha: "α", beta: "β", delta: "δ", rho: "ρa", sigma: "σa", nu: "ν" };
  const pfmt = { alpha: (v) => num(v, 2), beta: (v) => num(v, 3), delta: (v) => num(v, 3), rho: (v) => num(v, 3),
    sigma: (v) => num(100 * v, 2) + " %", nu: (v) => num(v, v % 1 ? (v * 10 % 1 ? 2 : 1) : 0) };

  const sliders = {};
  const add = (key, o) => { sliders[key] = MAV.slider(ctl, Object.assign({ key }, o), (v) => { state.p[key] = o.map ? o.map(v) : v; schedule(); }); };
  add("alpha", { label: { es: "Participación del capital α", en: "Capital share α" }, min: 0.2, max: 0.5, step: 0.01, value: BASE.alpha, fmt: pfmt.alpha });
  add("beta", { label: { es: "Factor de descuento β", en: "Discount factor β" }, min: 0.95, max: 0.995, step: 0.001, value: BASE.beta, fmt: pfmt.beta,
    note: { es: "Trimestral; en el estado estacionario r̄ = 1/β − 1 + δ", en: "Quarterly; in the steady state r̄ = 1/β − 1 + δ" } });
  add("delta", { label: { es: "Depreciación δ", en: "Depreciation δ" }, min: 0.01, max: 0.05, step: 0.001, value: BASE.delta, fmt: pfmt.delta,
    note: { es: "Trimestral: 0.025 es 10 % anual", en: "Quarterly: 0.025 is 10% a year" } });
  add("rho", { label: { es: "Persistencia de la PTF ρa", en: "TFP persistence ρa" }, min: 0, max: 0.995, step: 0.005, value: BASE.rho, fmt: pfmt.rho });
  add("sigma", { label: { es: "Desv. est. del choque σa", en: "Shock std. dev. σa" }, min: 0.001, max: 0.02, step: 0.0001, value: BASE.sigma, fmt: pfmt.sigma,
    note: { es: "Escala las respuestas y σ(y); no cambia cocientes ni correlaciones", en: "Scales the responses and σ(y); leaves ratios and correlations unchanged" } });
  sliders.nu = MAV.slider(ctl, {
    key: "nu", label: { es: "Curvatura del ocio ν", en: "Leisure curvature ν" }, min: 0, max: NU.length - 1, step: 1, value: NU.indexOf(BASE.nu),
    fmt: (i) => `${pfmt.nu(NU[i])} · η = ${frischTxt(NU[i])}`,
    note: { es: "η = (1 − l)/(ν l) es la elasticidad de Frisch con l = 0.33; ν = 0 es el trabajo indivisible de Hansen (1985)",
            en: "η = (1 − l)/(ν l) is the Frisch elasticity at l = 0.33; ν = 0 is Hansen's (1985) indivisible labor" },
  }, (i) => { state.p.nu = NU[i]; schedule(); });
  const presetRow = document.createElement("div"); presetRow.className = "btn-row"; presetRow.style.margin = "-8px 0 18px";
  const PRESETS = [
    { nu: 0, label: { es: "Hansen: ν = 0", en: "Hansen: ν = 0" } },
    { nu: 1.5, label: { es: "Base: ν = 1.5", en: "Base: ν = 1.5" } },
    { nu: 10, label: { es: "Micro: ν = 10", en: "Micro: ν = 10" } },
  ].map((ps) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "btn";
    b.style.padding = "6px 8px"; b.style.fontSize = ".82rem";
    b.addEventListener("click", () => { state.p.nu = ps.nu; sliders.nu.set(NU.indexOf(ps.nu)); schedule(); });
    presetRow.appendChild(b); return [b, ps];
  });
  ctl.appendChild(presetRow);

  MAV.segmented(ctl, { key: "muestra", label: { es: "Datos de EE. UU.", en: "US data" }, value: state.muestra,
    options: [{ value: "1948-2019", label: "1948–2019" }, { value: "1984-2019", label: "1984–2019" }] }, (v) => { state.muestra = v; schedule(); });

  const refBox = document.createElement("div"); refBox.className = "ctl";
  const refRow = document.createElement("div"); refRow.className = "btn-row";
  const bBase = document.createElement("button"); bBase.type = "button"; bBase.className = "btn";
  const bRef = document.createElement("button"); bRef.type = "button"; bRef.className = "btn";
  refRow.append(bBase, bRef);
  const refNote = document.createElement("div"); refNote.className = "ctl-note"; refNote.style.marginTop = "8px";
  const ssNote = document.createElement("div"); ssNote.className = "ctl-note"; ssNote.style.marginTop = "6px";
  refBox.append(refRow, refNote, ssNote); ctl.appendChild(refBox);
  bBase.addEventListener("click", () => {
    state.p = { ...BASE };
    ["alpha", "beta", "delta", "rho", "sigma"].forEach((k) => sliders[k].set(BASE[k])); sliders.nu.set(NU.indexOf(BASE.nu)); schedule();
  });
  bRef.addEventListener("click", () => { state.ref = { ...state.p }; schedule(); });

  const tiles = document.getElementById("tiles");
  const chIrf = C.multiples(document.getElementById("chart-irf"), { panels: [], series: [], x: [] });
  const chMom = C.dotFacets(document.getElementById("chart-moments"), { facets: [], marks: [], table: () => ({ header: [], rows: [] }) });

  const same = (a, b) => Object.keys(BASE).every((k) => Math.abs(a[k] - b[k]) < 1e-12);
  const diffTxt = (a, b) => Object.keys(BASE).filter((k) => Math.abs(a[k] - b[k]) >= 1e-12).map((k) => `${PNAMES[k]} = ${pfmt[k](b[k])}`).join("; ");
  const signed = (v, d = 2) => (v >= 0 ? "+" : "−") + num(Math.abs(v), d);

  /* Un solo recálculo por cuadro; el setTimeout cubre pestañas en segundo plano, donde no corre requestAnimationFrame. */
  let pending = false;
  function schedule() {
    if (pending) return; pending = true;
    const go = () => { if (!pending) return; pending = false; run(); };
    requestAnimationFrame(go); setTimeout(go, 60);
  }

  function run() {
    const p = state.p, ref = state.ref;
    const sol = R.solve(p);
    const H = 40, hs = Array.from({ length: H + 1 }, (_, h) => h);
    const hasRef = !same(p, ref);
    const solRef = hasRef ? R.solve(ref) : null;
    const f = R.irf(sol, p.sigma, H), fr = hasRef ? R.irf(solRef, ref.sigma, H) : null;
    const mom = R.moments(sol, p.sigma);
    const dm = data.muestras[state.muestra], dmom = dm.momentos, etiqueta = dm.etiqueta;

    PRESETS.forEach(([b, ps]) => { b.textContent = T(ps.label); b.classList.toggle("btn-primary", p.nu === ps.nu); b.setAttribute("aria-pressed", String(p.nu === ps.nu)); });
    bBase.textContent = T({ es: "Restaurar la base", en: "Reset to base" });
    bRef.textContent = T({ es: "Fijar como referencia", en: "Pin as reference" });
    refNote.textContent = hasRef
      ? T({ es: "Referencia (línea discontinua): ", en: "Reference (dashed line): " }) + diffTxt(p, ref)
      : T({ es: "La referencia coincide con la calibración actual. Mueve un control para comparar.", en: "The reference equals the current calibration. Move a control to compare." });
    const ss = sol.ss;
    ssNote.textContent = T({ es: "Estado estacionario: ", en: "Steady state: " }) +
      `k/y = ${num(ss.ky, 1)}, i/y = ${num(ss.iy, 2)}, c/y = ${num(ss.cy, 2)}, μ = ${num(ss.mu, 3)}` +
      T({ es: " (l = 0.33 fijo)", en: " (l = 0.33 fixed)" });

    if (!sol.ok) {
      MAV.tiles(tiles, [{ label: { es: "Sin solución estable única", en: "No unique stable solution" }, value: "—" }]);
      return;
    }

    const lImp = f.l[0], lImpRef = hasRef ? fr.l[0] : null;
    MAV.tiles(tiles, [
      { label: { es: "Volatilidad del producto", en: "Output volatility" }, value: num(mom.y.sd, 2) + " %",
        note: { es: `σ(y); datos: ${num(dmom.y.sd, 2)} %`, en: `σ(y); data: ${num(dmom.y.sd, 2)}%` } },
      { label: { es: "Volatilidad relativa de horas", en: "Relative volatility of hours" }, value: num(mom.h.rel, 2),
        note: { es: `σ(h)/σ(y); datos: ${num(dmom.h.rel, 2)}`, en: `σ(h)/σ(y); data: ${num(dmom.h.rel, 2)}` } },
      { label: { es: "Horas y productividad", en: "Hours and productivity" }, value: num(mom.corr_h_p, 2),
        note: { es: `correlación; datos: ${num(dm.corr_h_p, 2)}`, en: `correlation; data: ${num(dm.corr_h_p, 2)}` } },
      { label: { es: "Horas en el impacto", en: "Hours on impact" }, value: signed(lImp, 2) + " %",
        note: hasRef ? { es: `× ${num(lImp / lImpRef, 2)} la referencia (${signed(lImpRef, 2)} %)`, en: `× ${num(lImp / lImpRef, 2)} the reference (${signed(lImpRef, 2)}%)` }
                     : { es: "choque de una desv. est.", en: "one-std.-dev. shock" } },
    ]);

    const PAN = [["y", { es: "Producto y", en: "Output y" }], ["c", { es: "Consumo c", en: "Consumption c" }], ["i", { es: "Inversión i", en: "Investment i" }],
      ["l", { es: "Horas l", en: "Hours l" }], ["w", { es: "Salario real w", en: "Real wage w" }], ["r", { es: "Renta del capital r", en: "Capital rental rate r" }]];
    chIrf.update({
      title: { es: "Respuestas a un choque a la PTF de una desviación estándar", en: "Responses to a one-standard-deviation TFP shock" },
      sub: { es: "Desviación respecto del estado estacionario, % (también la tasa de renta r, en % de su nivel); trimestres después del choque",
             en: "Deviation from steady state, % (the rental rate r too, in % of its level); quarters after the shock" },
      x: hs, xLabel: { es: "trimestres", en: "quarters" }, xTip: { es: "trimestre", en: "quarter" },
      cols: (w) => (w >= 640 ? 3 : 2), panelHeight: (w) => (w >= 640 ? 150 : 120),
      series: [
        { name: { es: "Calibración actual", en: "Current calibration" }, short: { es: "actual", en: "current" }, style: 0 },
        { name: { es: "Referencia", en: "Reference" }, short: { es: "referencia", en: "reference" }, style: 1 },
      ],
      panels: PAN.map(([v, name]) => ({ name, ys: [f[v], hasRef ? fr[v] : null] })),
      yFmt: (v) => num(v, Math.abs(v - Math.round(v)) < 1e-9 ? 0 : Math.abs(10 * v - Math.round(10 * v)) < 1e-9 ? 1 : 2), tipFmt: (v) => num(v, 3) + " %", csvName: "rbc-irf", digits: 4,
    });

    const VARS = [["y", { es: "Producto", en: "Output" }], ["c", { es: "Consumo", en: "Consumption" }], ["i", { es: "Inversión", en: "Investment" }],
      ["h", { es: "Horas", en: "Hours" }], ["p", { es: "Productividad y/h", en: "Productivity y/h" }]];
    const relMax = Math.max(1.2, ...VARS.map(([v]) => Math.max(mom[v].rel, dmom[v].rel))) * 1.05;
    const acMin = Math.min(0, ...VARS.map(([v]) => Math.min(mom[v].ac1, dmom[v].ac1)));
    chMom.update({
      title: { es: "Momentos del ciclo: modelo frente a EE. UU.", en: "Business-cycle moments: model versus the United States" },
      sub: { es: `Componente cíclico HP (λ = 1600) de 100 × log. Modelo: momentos teóricos; datos: ${etiqueta.es}.`,
             en: `HP cyclical component (λ = 1600) of 100 × log. Model: theoretical moments; data: ${etiqueta.en}.` },
      marks: [{ name: { es: "Modelo (calibración actual)", en: "Model (current calibration)" }, kind: "fill" },
              { name: { es: `Datos de EE. UU., ${etiqueta.es}`, en: `US data, ${etiqueta.en}` }, kind: "ring" }],
      facets: [
        { name: { es: "Volatilidad relativa al producto, σx/σy", en: "Volatility relative to output, σx/σy" }, domain: [0, relMax], ref: { x: 1 },
          rows: VARS.map(([v, label]) => ({ label, values: [mom[v].rel, dmom[v].rel] })) },
        { name: { es: "Correlación con el producto", en: "Correlation with output" }, domain: [-1, 1], ref: { x: 0 },
          rows: VARS.map(([v, label]) => ({ label, values: [mom[v].corr, dmom[v].corr] })).concat([
            { label: { es: "Horas con productividad", en: "Hours with productivity" }, values: [mom.corr_h_p, dm.corr_h_p], sep: true, emphasize: true }]) },
        { name: { es: "Autocorrelación de primer orden", en: "First-order autocorrelation" }, domain: [acMin, 1], ref: { x: 0 },
          rows: VARS.map(([v, label]) => ({ label, values: [mom[v].ac1, dmom[v].ac1] })) },
      ],
      csvName: `rbc-momentos-${state.muestra}`,
      table: () => ({
        header: [T({ es: "Variable", en: "Variable" }), T({ es: "σ modelo (%)", en: "σ model (%)" }), T({ es: "σ datos (%)", en: "σ data (%)" }),
          T({ es: "σx/σy modelo", en: "σx/σy model" }), T({ es: "σx/σy datos", en: "σx/σy data" }), T({ es: "corr(x,y) modelo", en: "corr(x,y) model" }),
          T({ es: "corr(x,y) datos", en: "corr(x,y) data" }), T({ es: "autocorr. modelo", en: "autocorr. model" }), T({ es: "autocorr. datos", en: "autocorr. data" })],
        rows: VARS.map(([v, label]) => [T(label), mom[v].sd, dmom[v].sd, mom[v].rel, dmom[v].rel, mom[v].corr, dmom[v].corr, mom[v].ac1, dmom[v].ac1])
          .concat([[T({ es: "corr(horas, productividad)", en: "corr(hours, productivity)" }), null, null, null, null, mom.corr_h_p, dm.corr_h_p, null, null]]),
      }),
    });
    document.getElementById("source").textContent = T({ es: "Datos: ", en: "Data: " }) + T(data.fuente) +
      T({ es: " Modelo: calibración de modelos/rbc_canonico.mod, con μ recalibrada para que l = 0.33.", en: " Model: calibration from modelos/rbc_canonico.mod, with μ recalibrated so that l = 0.33." });
  }

  MAV.onLang(run);
  run();
  C.watchGreek(document.querySelector("main"));
})();
