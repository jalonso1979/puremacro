/* Laboratorio «Modelo neoclásico»: diagrama de fases con la senda de silla global,
   su aproximación de primer orden y la transición al estado estacionario. */

/* ------------------------------------------------------------ math core (no DOM; node-testable) */
const NEO = (function () {
  "use strict";
  const f = (k, p) => p.A * Math.pow(k, p.alpha);
  const fp = (k, p) => p.alpha * p.A * Math.pow(k, p.alpha - 1);
  const fpp = (k, p) => p.alpha * (p.alpha - 1) * p.A * Math.pow(k, p.alpha - 2);
  const g = (k, p) => f(k, p) + (1 - p.delta) * k;        // resources for c_t and k_{t+1}
  const R = (k, p) => fp(k, p) + 1 - p.delta;             // gross return, also g'(k)

  function steady(p) {
    const rk = 1 / p.beta - 1 + p.delta;
    const k = Math.pow((p.alpha * p.A) / rk, 1 / (1 - p.alpha));
    const y = f(k, p), i = p.delta * k, c = y - i;
    return { k, y, c, i, s: i / y, r: 1 / p.beta - 1, w: (1 - p.alpha) * y,
      kGold: Math.pow((p.alpha * p.A) / p.delta, 1 / (1 - p.alpha)) };
  }

  /* First-order solution: k_{t+1}-k* = lam (k_t-k*), c_t-c* = gc (k_t-k*); psi = gc k* / c* is the
     elasticity used by the log-linear version, ln(c_t/c*) = psi ln(k_t/k*). */
  function linear(p, ss) {
    const b = (p.beta * fpp(ss.k, p) * ss.c) / p.sigma;
    const tr = 1 / p.beta + 1 - b, det = 1 / p.beta;
    const lam = (tr - Math.sqrt(tr * tr - 4 * det)) / 2;
    const gc = 1 / p.beta - lam;
    return { lam, lamU: det / lam, gc, psi: (gc * ss.k) / ss.c, halfLife: Math.log(0.5) / Math.log(lam) };
  }

  /* k with A k^alpha + (1-delta) k = m: safeguarded Newton (g is increasing and concave). */
  function invG(m, p, guess) {
    let lo = 0, hi = Math.pow(m / p.A, 1 / p.alpha);
    if (p.delta < 1) hi = Math.min(hi, m / (1 - p.delta));
    let k = guess > lo && guess < hi ? guess : 0.5 * (lo + hi);
    for (let it = 0; it < 200; it++) {
      const h = g(k, p) - m;
      if (h === 0) return k;
      if (h > 0) hi = k; else lo = k;
      let kn = k - h / R(k, p);
      if (!(kn > lo && kn < hi)) kn = 0.5 * (lo + hi);
      if (Math.abs(kn - k) <= 1e-15 * kn) return kn;
      k = kn;
    }
    return k;
  }

  /* Global saddle path by reverse shooting. Start a hair from the steady state on the stable
     eigenvector and iterate the inverse map
        c_t = c_{t+1} [beta R(k_{t+1})]^{-1/sigma},   k_t = g^{-1}(k_{t+1} + c_t),
     which moves away from the steady state along the stable manifold and pulls any error back onto it.
     Interleaved orbits (one per starting offset in a fundamental domain) keep the points dense. */
  function saddle(p, ss, lin, kLo, kHi) {
    const M = Math.max(6, Math.min(300, Math.ceil(300 * Math.log(1 / lin.lam))));
    const pts = [[ss.k, ss.c]];
    const e0 = 1e-4 * ss.k;
    for (const side of [-1, 1]) {
      for (let j = 0; j < M; j++) {
        const eps = side * e0 * Math.pow(lin.lam, -j / M);
        let k1 = ss.k + eps, c1 = ss.c + lin.gc * eps;
        for (let n = 0; n < 50000; n++) {
          const c0 = c1 * Math.pow(p.beta * R(k1, p), -1 / p.sigma);
          const k0 = invG(k1 + c0, p, k1);
          if (!(k0 > 0) || !(c0 > 0) || !isFinite(c0)) break;
          pts.push([k0, c0]);
          if (k0 < kLo || k0 > kHi) break;
          k1 = k0; c1 = c0;
        }
      }
    }
    pts.sort((a, b) => a[0] - b[0]);
    const lk = Float64Array.from(pts, (q) => Math.log(q[0])), lc = Float64Array.from(pts, (q) => Math.log(q[1]));
    /* c(k) on the saddle path: linear interpolation in logs (exact for the closed form delta = 1, sigma = 1). */
    const at = (k) => {
      if (!(k > 0)) return 0;
      const x = Math.log(k); let lo = 0, hi = lk.length - 1;
      if (x <= lk[0]) return Math.exp(lc[0] + ((lc[1] - lc[0]) * (x - lk[0])) / (lk[1] - lk[0]));
      if (x >= lk[hi]) return Math.exp(lc[hi] + ((lc[hi] - lc[hi - 1]) * (x - lk[hi])) / (lk[hi] - lk[hi - 1]));
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (lk[m] <= x) lo = m; else hi = m; }
      return Math.exp(lc[lo] + ((lc[hi] - lc[lo]) * (x - lk[lo])) / (lk[hi] - lk[lo]));
    };
    return { pts, at, orbits: M };
  }

  /* Perfect-foresight transition from k0 to the steady state of p: stacked Newton on the Euler
     equations with k_0 given and k_{T+1} = k*; the Jacobian is tridiagonal. The initial guess
     follows the global saddle-path policy when one is given. */
  function transition(p, k0, ss, lin, T, policy) {
    const K = new Float64Array(T + 2), C = new Float64Array(T + 1), E = new Float64Array(T);
    K[0] = k0; K[T + 1] = ss.k;
    for (let t = 1; t <= T; t++) K[t] = ss.k + Math.pow(lin.lam, t) * (k0 - ss.k);
    if (policy) {
      const G = Float64Array.from(K);
      let ok = true;
      for (let t = 0; t < T; t++) { G[t + 1] = g(G[t], p) - policy(G[t]); if (!(G[t + 1] > 0)) { ok = false; break; } }
      if (ok) K.set(G.subarray(1, T + 1), 1);
    }
    const resid = (X) => {
      let mx = 0;
      for (let t = 0; t <= T; t++) { C[t] = g(X[t], p) - X[t + 1]; if (!(C[t] > 0)) return Infinity; }
      for (let t = 0; t < T; t++) {
        E[t] = p.sigma * (Math.log(C[t + 1]) - Math.log(C[t])) - Math.log(p.beta) - Math.log(R(X[t + 1], p));
        mx = Math.max(mx, Math.abs(E[t]));
      }
      return mx;
    };
    let err = resid(K), it = 0;
    const a = new Float64Array(T), b = new Float64Array(T), u = new Float64Array(T), d = new Float64Array(T), dx = new Float64Array(T);
    const Kn = new Float64Array(T + 2);
    for (; it < 60 && err > 1e-12 && isFinite(err); it++) {
      for (let t = 0; t < T; t++) {
        const kn = K[t + 1];
        b[t] = (p.sigma * R(kn, p)) / C[t + 1] + p.sigma / C[t] - fpp(kn, p) / R(kn, p);
        a[t] = t >= 1 ? (-p.sigma * R(K[t], p)) / C[t] : 0;
        u[t] = t + 1 <= T - 1 ? -p.sigma / C[t + 1] : 0;
        d[t] = -E[t];
      }
      for (let t = 1; t < T; t++) { const w = a[t] / b[t - 1]; b[t] -= w * u[t - 1]; d[t] -= w * d[t - 1]; }
      dx[T - 1] = d[T - 1] / b[T - 1];
      for (let t = T - 2; t >= 0; t--) dx[t] = (d[t] - u[t] * dx[t + 1]) / b[t];
      let s = 1, en = Infinity;
      for (let ls = 0; ls < 40; ls++) {
        Kn.set(K);
        let pos = true;
        for (let t = 0; t < T; t++) { Kn[t + 1] = K[t + 1] + s * dx[t]; if (!(Kn[t + 1] > 0)) pos = false; }
        en = pos ? resid(Kn) : Infinity;
        if (en < err || en < 1e-12) break;
        s /= 2;
      }
      if (!isFinite(en)) break;
      K.set(Kn); err = resid(K);
    }
    let fallback = false;
    if (!(err < 1e-8) && policy) {   // Newton failed: keep the path implied by the global policy
      K[0] = k0; for (let t = 0; t <= T; t++) K[t + 1] = g(K[t], p) - policy(K[t]);
      for (let t = 0; t <= T; t++) C[t] = policy(K[t]);
      fallback = true;
    }
    const out = { k: [], c: [], y: [], i: [], r: [], w: [], err, iters: it, ok: err < 1e-8, fallback };
    for (let t = 0; t <= T; t++) {
      out.k.push(K[t]); out.c.push(C[t]); out.y.push(f(K[t], p)); out.i.push(K[t + 1] - (1 - p.delta) * K[t]);
      out.r.push(fp(K[t], p) - p.delta); out.w.push((1 - p.alpha) * f(K[t], p));
    }
    return out;
  }

  const horizon = (lin) => Math.min(4000, Math.max(200, Math.ceil(Math.log(1e-12) / Math.log(lin.lam)) + 60));

  /* Years until the capital gap first halves along a path (linear interpolation between years). */
  function halfLifePath(k, kss) {
    const g0 = k[0] - kss;
    if (Math.abs(g0) < 1e-9 * kss) return null;
    for (let t = 1; t < k.length; t++) {
      const r1 = (k[t] - kss) / g0, r0 = (k[t - 1] - kss) / g0;
      if (r1 <= 0.5) return t - 1 + (r0 - 0.5) / (r0 - r1);
    }
    return null;
  }

  return { f, fp, g, R, steady, linear, invG, saddle, transition, horizon, halfLifePath };
})();
if (typeof module === "object" && module.exports) module.exports = NEO;

/* ------------------------------------------------------------ page */
if (typeof document !== "undefined") (function () {
  "use strict";
  const T = MAV.t;
  const DEF = { k0: 50, dA: 0, alpha: 0.33, beta: 0.97, delta: 0.06, sigma: 1, approx: "loglin" };
  const state = Object.assign({}, DEF);
  const ctl = document.getElementById("controls");
  const pct = (v, d = 0) => MAV.num(v, d) + "%";
  /* levels: fewer decimals as magnitudes grow (k* reaches millions when alpha is near 0.8) */
  const lvl = (v, d) => MAV.num(v, Math.abs(v) >= 1000 ? 0 : Math.abs(v) >= 100 ? 1 : d);
  const tickFmt = (max) => (max >= 1e4 ? d3.format("~s") : (v) => MAV.num(v, max < 10 ? 1 : 0));
  const signed = (v, d = 0) => { const r = Math.abs(v) < 0.5 * 10 ** -d ? 0 : v; return (r > 0 ? "+" : r < 0 ? "−" : "") + MAV.num(Math.abs(r), d); };

  const sl = {};
  sl.k0 = MAV.slider(ctl, { key: "k0", label: { es: "Capital inicial k₀ (% del k* inicial)", en: "Initial capital k₀ (% of initial k*)" },
    min: 2, max: 200, step: 1, value: state.k0, fmt: (v) => pct(v),
    note: { es: "100% arranca en el estado estacionario", en: "100% starts at the steady state" } }, (v) => { state.k0 = v; schedule(); });
  sl.dA = MAV.slider(ctl, { key: "dA", label: { es: "Cambio permanente de la PTF", en: "Permanent TFP change" },
    min: -30, max: 30, step: 1, value: state.dA, fmt: (v) => signed(v) + "%",
    note: { es: "A pasa de 1 a 1 + cambio en t = 0, sin anuncio previo", en: "A moves from 1 to 1 + change at t = 0, unannounced" } }, (v) => { state.dA = v; schedule(); });

  const sep = document.createElement("h2"); sep.style.marginTop = "22px";
  sep.innerHTML = '<span class="es">Parámetros</span><span class="en">Parameters</span>'; ctl.appendChild(sep);
  sl.alpha = MAV.slider(ctl, { key: "alpha", label: { es: "Participación del capital α", en: "Capital share α" },
    min: 0.2, max: 0.8, step: 0.01, value: state.alpha }, (v) => { state.alpha = v; schedule(); });
  sl.beta = MAV.slider(ctl, { key: "beta", label: { es: "Factor de descuento β (anual)", en: "Discount factor β (annual)" },
    min: 0.9, max: 0.995, step: 0.005, value: state.beta, fmt: (v) => MAV.num(v, 3) }, (v) => { state.beta = v; schedule(); });
  sl.delta = MAV.slider(ctl, { key: "delta", label: { es: "Depreciación δ (anual)", en: "Depreciation δ (annual)" },
    min: 0.02, max: 0.2, step: 0.01, value: state.delta }, (v) => { state.delta = v; schedule(); });
  sl.sigma = MAV.slider(ctl, { key: "sigma", label: { es: "Aversión relativa al riesgo σ", en: "Relative risk aversion σ" },
    min: 0.25, max: 5, step: 0.25, value: state.sigma,
    note: { es: "1/σ es la elasticidad intertemporal; σ = 1 es utilidad logarítmica", en: "1/σ is the intertemporal elasticity; σ = 1 is log utility" } },
  (v) => { state.sigma = v; schedule(); });
  const seg = MAV.segmented(ctl, { key: "approx", label: { es: "Aproximación de primer orden", en: "First-order approximation" }, value: state.approx,
    options: [{ value: "loglin", label: { es: "log-lineal", en: "log-linear" } }, { value: "lin", label: { es: "lineal en niveles", en: "linear in levels" } }] },
  (v) => { state.approx = v; schedule(); });
  const row = document.createElement("div"); row.className = "btn-row";
  const reset = document.createElement("button"); reset.type = "button"; reset.className = "btn";
  reset.innerHTML = '<span class="es">Calibración del curso</span><span class="en">Course calibration</span>';
  reset.addEventListener("click", () => {
    Object.assign(state, DEF);
    ["k0", "dA", "alpha", "beta", "delta", "sigma"].forEach((key) => sl[key].set(state[key]));
    seg.set(state.approx); schedule();
  });
  row.appendChild(reset); ctl.appendChild(row);

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
  const keySvg = (style, w = 26, width = 2, opacity = 1) => { const s = MAV.style(style);
    return `<svg width="${w}" height="10" aria-hidden="true"><line x1="0" x2="${w}" y1="5" y2="5" stroke="${s.stroke}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-dasharray="${s.dash}" stroke-linecap="round"/></svg>`; };
  const dotSvg = (w = 26) => `<svg width="${w}" height="10" aria-hidden="true"><circle cx="${w / 2 - 6}" cy="5" r="2.5" fill="var(--ink)"/><circle cx="${w / 2 + 2}" cy="5" r="2.5" fill="var(--ink)"/><circle cx="${w / 2 + 9}" cy="5" r="2.5" fill="var(--ink)"/></svg>`;
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

  /* ---------------------------------------------------------- phase diagram */
  function phaseChart(el) {
    const parts = card(el);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    const uid = "ph" + Math.random().toString(36).slice(2, 8);
    let o = null;

    function draw() {
      if (!o) return;
      const width = Math.max(300, parts.plot.clientWidth || 640);
      const narrow = width < 560;
      const height = Math.round(Math.min(470, Math.max(300, width * 0.62)));
      const m = { t: 14, r: narrow ? 12 : 22, b: 42, l: 50 };
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
      svg.selectAll("*").remove();
      svg.append("title").text(T(o.title));
      const x = d3.scaleLinear().domain([0, o.kmax]).range([m.l, width - m.r]);
      const y = d3.scaleLinear().domain([0, o.cmax]).range([height - m.b, m.t]);
      const defs = svg.append("defs");
      defs.append("clipPath").attr("id", uid + "c").append("rect").attr("x", m.l).attr("y", m.t)
        .attr("width", width - m.l - m.r).attr("height", height - m.t - m.b);
      defs.append("marker").attr("id", uid + "a").attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5)
        .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,1 L9,5 L0,9 z").attr("fill", "var(--muted)");

      svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickSize(-(width - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`)
        .call(d3.axisBottom(x).ticks(Math.max(3, Math.round((width - m.l - m.r) / 80))).tickFormat(tickFmt(o.kmax)).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat(tickFmt(o.cmax)).tickSize(0).tickPadding(8)).call((gg) => gg.select(".domain").remove());
      svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end")
        .text(T({ es: "capital por trabajador, k", en: "capital per worker, k" }));
      svg.append("text").attr("class", "axis-title").attr("x", m.l + 6).attr("y", m.t + 12)
        .text(T({ es: "consumo, c", en: "consumption, c" }));

      const plot = svg.append("g").attr("clip-path", `url(#${uid}c)`);
      const clampY = (v) => (v == null || !isFinite(v) ? null : Math.max(-0.2 * o.cmax, Math.min(1.4 * o.cmax, v)));
      const line = d3.line().defined((d) => d[1] != null).x((d) => x(d[0])).y((d) => y(d[1]));
      o.series.forEach((s) => {
        const st = MAV.style(s.style);
        plot.append("path").attr("class", "series").attr("d", line(o.grid.map((k, j) => [k, clampY(s.y[j])])))
          .attr("stroke", st.stroke).attr("stroke-dasharray", st.dash).attr("stroke-width", s.width || 2)
          .attr("stroke-opacity", s.faint ? 0.5 : 1);
      });

      // direction of motion, one pair of arrows per region
      const L = 17;
      o.arrows.forEach((a) => {
        const px = x(a.k), py = y(a.c);
        if (px < m.l + L + 4 || px > width - m.r - L - 4 || py < m.t + L + 4 || py > height - m.b - L - 4) return;
        const gA = plot.append("g").attr("stroke", "var(--muted)").attr("stroke-width", 1.4).attr("fill", "none");
        gA.append("line").attr("x1", px).attr("y1", py).attr("x2", px + a.sk * L).attr("y2", py).attr("marker-end", `url(#${uid}a)`);
        gA.append("line").attr("x1", px).attr("y1", py).attr("x2", px).attr("y2", py - a.sc * L).attr("marker-end", `url(#${uid}a)`);
      });

      // transition: one dot per year
      o.path.forEach((q, t) => plot.append("circle").attr("cx", x(q[0])).attr("cy", y(q[1])).attr("r", t === 0 ? 4 : 2.5)
        .attr("fill", "var(--ink)").attr("stroke", "var(--bg)").attr("stroke-width", 1));
      if (o.jump) plot.append("line").attr("x1", x(o.jump[0])).attr("x2", x(o.jump[0])).attr("y1", y(o.jump[1])).attr("y2", y(o.jump[2]))
        .attr("stroke", "var(--muted)").attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      o.points.forEach((q) => {
        plot.append("circle").attr("cx", x(q.k)).attr("cy", y(q.c)).attr("r", 5.5)
          .attr("fill", q.hollow ? "var(--bg)" : "var(--ink)").attr("stroke", q.hollow ? "var(--ink)" : "var(--bg)").attr("stroke-width", q.hollow ? 1.6 : 2);
        if (q.label) svg.append("text").attr("class", "alabel").attr("x", x(q.k) + (q.dx ?? 9)).attr("y", y(q.c) + (q.dy ?? 16))
          .attr("text-anchor", q.anchor || "start").text(T(q.label));
      });
      if (!narrow) o.labels.forEach((lb) => {
        const px = x(lb.k), py = y(lb.c) + (lb.dy || 0);
        if (px < m.l || px > width - m.r || py < m.t + 10 || py > height - m.b - 4) return;
        svg.append("text").attr("class", "dlabel").attr("x", px + (lb.dx || 0)).attr("y", py).attr("text-anchor", lb.anchor || "start").text(T(lb.text));
      });

      // legend
      parts.legend.innerHTML = "";
      const addKey = (html, name) => { const k = document.createElement("span"); k.className = "key"; k.innerHTML = html;
        const t = document.createElement("span"); t.textContent = T(name); k.appendChild(t); parts.legend.appendChild(k); };
      o.series.filter((s) => !s.faint).forEach((s) => addKey(keySvg(s.style, 26, s.width || 2), s.name));
      if (o.series.some((s) => s.faint)) addKey(keySvg(3, 26, 1.2, 0.5), { es: "lugares geométricos iniciales", en: "initial loci" });
      addKey(dotSvg(), { es: "trayectoria, un punto por año", en: "path, one dot per year" });

      // hover readout
      const shown = o.series.filter((s) => !s.faint);
      const hair = svg.append("line").attr("class", "xhair").attr("y1", m.t).attr("y2", height - m.b).style("display", "none");
      const dots = shown.map((s) => svg.append("circle").attr("class", "dot").attr("r", 4).attr("fill", MAV.style(s.style).stroke).style("display", "none"));
      const show = (px) => {
        const kv = x.invert(Math.max(m.l, Math.min(width - m.r, px)));
        const j = Math.max(0, Math.min(o.grid.length - 1, Math.round((kv / o.kmax) * (o.grid.length - 1))));
        const kx = o.grid[j];
        hair.attr("x1", x(kx)).attr("x2", x(kx)).style("display", null);
        shown.forEach((s, i) => { const v = s.y[j];
          if (v == null || v < 0 || v > o.cmax) dots[i].style("display", "none"); else dots[i].attr("cx", x(kx)).attr("cy", y(v)).style("display", null); });
        const tip = parts.tip; tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head"; head.textContent = "k = " + lvl(kx, 2); tip.appendChild(head);
        shown.forEach((s) => {
          const r = document.createElement("div"); r.className = "t-row";
          const key = document.createElement("span"); key.innerHTML = keySvg(s.style, 14);
          const val = document.createElement("b"); val.textContent = s.y[j] == null ? "—" : lvl(s.y[j], 3);
          const nm = document.createElement("span"); nm.textContent = T(s.name);
          r.append(key, val, nm); tip.appendChild(r);
        });
        const sad = o.series[o.saddleIndex].y[j], apx = o.series[o.approxIndex].y[j];
        if (kx > 0 && sad > 0) {
          const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "18px auto 1fr";
          const b = document.createElement("b"); b.textContent = signed(100 * (apx / sad - 1), 1) + "%";
          const nm = document.createElement("span"); nm.textContent = T({ es: "error de la aproximación", en: "approximation error" });
          r.append(document.createElement("span"), b, nm); tip.appendChild(r);
        }
        const scale = parts.plot.clientWidth / width, left = x(kx) * scale, tw = tip.offsetWidth || 180;
        tip.style.left = (left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
        tip.style.top = "8px"; tip.classList.add("on");
      };
      const hide = () => { hair.style("display", "none"); dots.forEach((d) => d.style("display", "none")); parts.tip.classList.remove("on"); };
      let cur = width - m.r;
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b)
        .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", T({ es: "Explorar valores", en: "Explore values" }))
        .on("pointermove", (ev) => { cur = d3.pointer(ev)[0]; show(cur); })
        .on("pointerleave", hide).on("blur", hide).on("focus", () => show(cur))
        .on("keydown", (ev) => {
          const stepPx = (width - m.l - m.r) / 60;
          if (ev.key === "ArrowLeft") { cur = Math.max(m.l, cur - stepPx); show(cur); ev.preventDefault(); }
          if (ev.key === "ArrowRight") { cur = Math.min(width - m.r, cur + stepPx); show(cur); ev.preventDefault(); }
        });

      parts.title.textContent = T(o.title);
      parts.sub.textContent = T(o.sub);
      const header = ["k", ...o.series.map((s) => T(s.name) + (s.faint ? T({ es: " (inicial)", en: " (initial)" }) : ""))];
      const rows = o.grid.map((k, j) => [k, ...o.series.map((s) => s.y[j])]);
      fillTable(parts, header, rows.filter((_, j) => j % 5 === 0).map((r) => r.map((v) => (v == null || !isFinite(v) ? "" : MAV.num(v, 3)))));
      parts.csvBtn.onclick = () => download(o.csvName, [header, ...rows.map((r) => r.map((v) => (v == null || !isFinite(v) ? "" : +v.toPrecision(10))))]);
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw);
    return { update(next) { o = next; draw(); } };
  }

  /* ---------------------------------------------------------- charts and tiles */
  const tiles = document.getElementById("tiles");
  const chPhase = phaseChart(document.getElementById("chart-phase"));
  const chPath = MAV.lineChart(document.getElementById("chart-path"), { series: [], title: "" });
  const chR = MAV.lineChart(document.getElementById("chart-r"), { series: [], title: "" });
  const chW = MAV.lineChart(document.getElementById("chart-w"), { series: [], title: "" });

  let pending = 0;
  function schedule() { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); }

  function run() {
    const base = { alpha: state.alpha, beta: state.beta, delta: state.delta, sigma: state.sigma };
    const p0 = Object.assign({ A: 1 }, base), p1 = Object.assign({ A: 1 + state.dA / 100 }, base);
    const ss0 = NEO.steady(p0), ss1 = NEO.steady(p1), lin = NEO.linear(p1, ss1);
    const shock = state.dA !== 0;
    const k0 = (state.k0 / 100) * ss0.k;
    const kmax = Math.max(1.8 * ss1.k, 1.8 * ss0.k, 1.15 * k0);
    const sad = NEO.saddle(p1, ss1, lin, Math.min(0.002 * kmax, 0.5 * k0), 1.02 * kmax);
    const Tn = NEO.horizon(lin);
    const tr = NEO.transition(p1, k0, ss1, lin, Tn, sad.at);
    const approx = state.approx === "loglin"
      ? (k) => ss1.c * Math.pow(k / ss1.k, lin.psi)
      : (k) => ss1.c + lin.gc * (k - ss1.k);
    const approxName = state.approx === "loglin"
      ? { es: "aproximación log-lineal", en: "log-linear approximation" } : { es: "aproximación lineal", en: "linear approximation" };

    // phase diagram data
    const NG = 300, grid = d3.range(NG + 1).map((j) => (kmax * j) / NG);
    const dk = (k, p) => NEO.f(k, p) - p.delta * k;
    const dc = (k, p, ss) => NEO.g(k, p) - ss.k;
    const ySad = grid.map((k) => (k === 0 ? 0 : sad.at(k)));
    const yApx = grid.map((k) => approx(k));
    const yDk = grid.map((k) => dk(k, p1)), yDc = grid.map((k) => dc(k, p1, ss1));
    const c0 = tr.c[0];
    let cmax = Math.max(d3.max(yDk), sad.at(kmax), ss0.c, c0, ss1.c * 1.3);
    if (shock) cmax = Math.max(cmax, d3.max(grid, (k) => dk(k, p0)));
    cmax = d3.nice(0, cmax * 1.1, 5)[1];

    const series = [
      { name: { es: "senda de silla (global)", en: "saddle path (global)" }, y: ySad, style: 0, width: 2.5 },
      { name: approxName, y: yApx, style: 1 },
      { name: { es: "Δk = 0", en: "Δk = 0" }, y: yDk, style: 3 },
      { name: { es: "Δc = 0", en: "Δc = 0" }, y: yDc, style: 2 },
    ];
    if (shock) {
      series.push({ name: { es: "Δk = 0", en: "Δk = 0" }, y: grid.map((k) => dk(k, p0)), style: 3, width: 1.2, faint: true });
      series.push({ name: { es: "Δc = 0", en: "Δc = 0" }, y: grid.map((k) => dc(k, p0, ss0)), style: 2, width: 1.2, faint: true });
    }
    // arrows: a representative point in each of the four regions
    const arrows = [];
    const addArrow = (k, c) => { if (k > 0 && c > 0 && c < cmax) arrows.push({ k, c, sk: c > dk(k, p1) ? -1 : 1, sc: c > dc(k, p1, ss1) ? 1 : -1 }); };
    const kL = 0.42 * ss1.k, kR = Math.min(1.5 * ss1.k, 0.8 * kmax);
    {
      const top = dk(kL, p1), low = Math.max(0, dc(kL, p1, ss1)), s = sad.at(kL);
      addArrow(kL, top + 0.55 * (cmax - top));
      addArrow(0.62 * ss1.k, (s - low > top - s ? low + 0.5 * (s - low) : s + 0.5 * (top - s)));
      addArrow(kR, 0.45 * dk(kR, p1));
      addArrow(1.3 * ss1.k, (() => { const tp = Math.min(cmax, dc(1.3 * ss1.k, p1, ss1)), lw = dk(1.3 * ss1.k, p1), sv = sad.at(1.3 * ss1.k);
        return tp - sv > sv - lw ? sv + 0.5 * (tp - sv) : lw + 0.5 * (sv - lw); })());
    }
    const nearSS = Math.max(0.004 * ss1.k, 0.01 * Math.abs(k0 - ss1.k));
    const pathPts = [];
    for (let t = 0; t < tr.k.length && pathPts.length < 400; t++) {
      pathPts.push([tr.k[t], tr.c[t]]);
      if (t > 0 && Math.abs(tr.k[t] - ss1.k) < nearSS) break;
    }
    // labels go on opposite sides when the initial steady state and the t = 0 point are close
    const points = [], close = shock && Math.abs(k0 - ss0.k) < 0.05 * ss0.k, up = c0 >= ss0.c;
    if (shock) points.push({ k: ss0.k, c: ss0.c, hollow: true, label: { es: "EE inicial", en: "initial SS" }, dx: -9, dy: close && up ? 16 : -10, anchor: "end" });
    points.push({ k: ss1.k, c: ss1.c, label: shock ? { es: "EE nuevo", en: "new SS" } : { es: "estado estacionario", en: "steady state" }, dx: 10, dy: 18 });
    if (Math.abs(k0 - ss1.k) > 0.03 * ss1.k) points.push({ k: k0, c: c0, label: "t = 0", dx: -9, dy: close && !up ? 16 : -10, anchor: "end" });
    // direct labels, placed where the curves separate
    const labels = [];
    const kLab = 0.93 * kmax;
    labels.push({ k: kLab, c: sad.at(kLab), dy: -9, anchor: "end", text: { es: "senda de silla", en: "saddle path" } });
    labels.push({ k: kLab, c: dk(kLab, p1), dy: 17, anchor: "end", text: "Δk = 0" });
    { // Δc = 0 where it leaves the top of the plot
      let kTop = null; for (let j = NG; j >= 0; j--) if (yDc[j] <= 0.88 * cmax) { kTop = grid[j]; break; }
      if (kTop != null) labels.push({ k: kTop, c: dc(kTop, p1, ss1), dx: -8, anchor: "end", text: "Δc = 0" });
    }
    { // approximation: where it is furthest from the saddle path on the left
      let best = null;
      for (let j = Math.round(0.06 * NG); j <= Math.round((0.7 * ss1.k / kmax) * NG); j++) {
        const gap = Math.abs(yApx[j] - ySad[j]) / cmax;
        if (yApx[j] > 0 && yApx[j] < cmax && (!best || gap > best.gap)) best = { j, gap };
      }
      if (best && best.gap > 0.012) {
        const k = grid[best.j], above = yApx[best.j] > ySad[best.j];
        labels.push({ k, c: yApx[best.j], dy: above ? -10 : 18, anchor: "start", dx: 4, text: state.approx === "loglin" ? { es: "log-lineal", en: "log-linear" } : { es: "lineal", en: "linear" } });
      }
    }

    const cApx0 = approx(k0), errApx = 100 * (cApx0 / c0 - 1);
    const moving = Math.abs(k0 - ss1.k) > 1e-6 * ss1.k;
    chPhase.update({
      title: { es: "Diagrama de fases: senda de silla global y su aproximación", en: "Phase diagram: global saddle path and its approximation" },
      sub: moving
        ? { es: `En k₀ = ${lvl(k0, 2)} la senda global fija c₀ = ${lvl(c0, 3)}; la ${T(approxName)} da ${lvl(cApx0, 3)} (${signed(errApx, 1)}%).`,
            en: `At k₀ = ${lvl(k0, 2)} the global path sets c₀ = ${lvl(c0, 3)}; the ${T(approxName)} gives ${lvl(cApx0, 3)} (${signed(errApx, 1)}%).` }
        : { es: "La economía ya está en su estado estacionario: mueve k₀ o la PTF.", en: "The economy is already at its steady state: move k₀ or TFP." },
      kmax, cmax, grid, series, saddleIndex: 0, approxIndex: 1, arrows, path: pathPts,
      jump: shock && Math.abs(k0 - ss0.k) < 1e-9 * ss0.k ? [k0, ss0.c, c0] : null,
      points, labels, csvName: "diagrama-de-fases",
    });

    // tiles
    const hlPath = NEO.halfLifePath(tr.k, ss1.k);
    const before = (v, d) => (shock ? { es: `inicial: ${lvl(v, d)}`, en: `initial: ${lvl(v, d)}` } : null);
    MAV.tiles(tiles, [
      { label: { es: "Capital k*", en: "Capital k*" }, value: lvl(ss1.k, 2), note: before(ss0.k, 2) || { es: "por trabajador", en: "per worker" } },
      { label: { es: "Producto y*", en: "Output y*" }, value: lvl(ss1.y, 3), note: before(ss0.y, 3) || `k*/y* = ${MAV.num(ss1.k / ss1.y, 2)}` },
      { label: { es: "Consumo c*", en: "Consumption c*" }, value: lvl(ss1.c, 3), note: before(ss0.c, 3) || `c*/y* = ${MAV.pct((100 * ss1.c) / ss1.y, 1)}` },
      { label: { es: "Tasa de ahorro s*", en: "Saving rate s*" }, value: MAV.pct(100 * ss1.s, 1),
        note: { es: `regla de oro: s = α = ${MAV.pct(100 * state.alpha, 0)}`, en: `golden rule: s = α = ${MAV.pct(100 * state.alpha, 0)}` } },
      { label: { es: "Vida media (primer orden)", en: "Half-life (first order)" }, value: MAV.num(lin.halfLife, 1) + T({ es: " años", en: " yrs" }),
        note: { es: `cierra ${MAV.pct(100 * (1 - lin.lam), 1)} de la brecha al año${hlPath != null ? `; trayectoria global: ${MAV.num(hlPath, 1)} años` : ""}`,
                en: `closes ${MAV.pct(100 * (1 - lin.lam), 1)} of the gap per year${hlPath != null ? `; global path: ${MAV.num(hlPath, 1)} yrs` : ""}` } },
    ]);

    // transition paths
    const H = Math.min(Tn, Math.max(30, Math.min(160, Math.ceil(5 * lin.halfLife))));
    const years = d3.range(H + 1);
    const idx = (arr, ref) => years.map((t) => (100 * arr[t]) / ref);
    const newLevel = 100 * Math.pow(p1.A, 1 / (1 - state.alpha));
    chPath.update({
      title: { es: "Transición: capital, consumo, producto e inversión", en: "Transition: capital, consumption, output and investment" },
      sub: shock ? { es: `Índice, estado estacionario inicial = 100; líneas discontinuas finas: EE inicial y nuevo (${MAV.num(newLevel, 1)})`,
                     en: `Index, initial steady state = 100; thin dashed lines: initial and new SS (${MAV.num(newLevel, 1)})` }
                 : { es: "Índice, estado estacionario = 100 (línea discontinua fina)", en: "Index, steady state = 100 (thin dashed line)" },
      xType: "linear", xLabel: { es: "Año", en: "Year" }, csvName: "transicion", labelRoom: 110, digits: 2,
      hlines: shock ? [{ y: 100 }, { y: newLevel }] : [{ y: 100 }],
      series: [
        { name: { es: "capital k", en: "capital k" }, short: "k", x: years, y: idx(tr.k, ss0.k), style: 0 },
        { name: { es: "consumo c", en: "consumption c" }, short: "c", x: years, y: idx(tr.c, ss0.c), style: 1 },
        { name: { es: "producto y", en: "output y" }, short: "y", x: years, y: idx(tr.y, ss0.y), style: 2 },
        { name: { es: "inversión i", en: "investment i" }, short: "i", x: years, y: idx(tr.i, ss0.i), style: 3 },
      ],
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 1),
    });
    chR.update({
      title: { es: "Tasa de interés real rₜ", en: "Real interest rate rₜ" },
      sub: { es: `% anual; línea discontinua fina: r* = 1/β − 1 = ${MAV.num(100 * ss1.r, 1)}%`, en: `% per year; thin dashed line: r* = 1/β − 1 = ${MAV.num(100 * ss1.r, 1)}%` },
      xType: "linear", xLabel: { es: "Año", en: "Year" }, csvName: "tasa-de-interes", labelRoom: 16, height: 240, digits: 3,
      hlines: [{ y: 100 * ss1.r }],
      series: [{ name: { es: "tasa de interés r", en: "interest rate r" }, x: years, y: years.map((t) => 100 * tr.r[t]), style: 0 }],
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2) + "%",
    });
    chW.update({
      title: { es: "Salario real wₜ", en: "Real wage wₜ" },
      sub: { es: `Nivel, wₜ = (1 − α)Akₜ^α; línea discontinua fina: w* = ${lvl(ss1.w, 3)}`, en: `Level, wₜ = (1 − α)Akₜ^α; thin dashed line: w* = ${lvl(ss1.w, 3)}` },
      xType: "linear", xLabel: { es: "Año", en: "Year" }, csvName: "salario", labelRoom: 16, height: 240, digits: 4,
      hlines: [{ y: ss1.w }],
      series: [{ name: { es: "salario w", en: "wage w" }, x: years, y: years.map((t) => tr.w[t]), style: 0 }],
      yFmt: tickFmt(ss1.w * 1.2), tipFmt: (v) => lvl(v, 3),
    });
    document.getElementById("source").textContent = T({
      es: `Modelo sin datos. Calibración anual del curso: α = 0.33, β = 0.97, δ = 0.06, A = 1 (mazo 2 y modelos/neoclasico_prevision_perfecta.mod), con trabajo inelástico. Senda de silla por disparo inverso (${sad.orbits} órbitas por lado); transición por Newton apilado a ${Tn} años, residuo de Euler ${tr.err.toExponential(0)}.`,
      en: `Model only, no data. Course annual calibration: α = 0.33, β = 0.97, δ = 0.06, A = 1 (deck 2 and modelos/neoclasico_prevision_perfecta.mod), with inelastic labor. Saddle path by reverse shooting (${sad.orbits} orbits per side); transition by stacked Newton over ${Tn} years, Euler residual ${tr.err.toExponential(0)}.` });
  }

  MAV.onLang(run);
  run();
})();
