/* Macroeconomía Avanzada: shared lab runtime.
   Language and theme switches, form controls, and grayscale D3 charts with a hover
   readout, a legend, selective direct labels, a table view and CSV download.
   Requires d3 v7 (loaded by each page before this file). */
(function () {
  "use strict";
  const MAV = (window.MAV = window.MAV || {});
  const LS = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
               set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };

  // ------------------------------------------------------------ language
  const listeners = [];
  function initialLang() {
    const q = new URLSearchParams(location.search).get("lang");
    if (q === "es" || q === "en") return q;
    const saved = LS.get("mav-lang");
    if (saved === "es" || saved === "en") return saved;
    return (navigator.language || "es").toLowerCase().startsWith("es") ? "es" : "en";
  }
  MAV.lang = initialLang();
  document.documentElement.dataset.lang = MAV.lang;
  document.documentElement.lang = MAV.lang;

  MAV.t = (s) => (s == null ? "" : typeof s === "string" ? s : (s[MAV.lang] ?? s.es ?? ""));
  MAV.onLang = (fn) => listeners.push(fn);
  MAV.setLang = (lang) => {
    if (lang === MAV.lang) return;
    MAV.lang = lang;
    document.documentElement.dataset.lang = lang;
    document.documentElement.lang = lang;
    LS.set("mav-lang", lang);
    document.querySelectorAll(".lang-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    document.querySelectorAll("[data-title-es]").forEach((el) => { el.textContent = el.dataset["title" + (lang === "es" ? "Es" : "En")]; });
    const u = new URL(location.href); u.searchParams.set("lang", lang); history.replaceState(null, "", u);
    listeners.forEach((fn) => fn(lang));
  };

  const locale = () => (MAV.lang === "es" ? "es-MX" : "en-US");
  MAV.num = (x, d = 2) => (x == null || !isFinite(x) ? "—"
    : x.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }).replace(/^-/, "\u2212"));
  MAV.pct = (x, d = 1) => (x == null || !isFinite(x) ? "—" : MAV.num(x, d) + "%");
  MAV.quarter = (date) => { const q = Math.floor(date.getUTCMonth() / 3) + 1;
    return `${date.getUTCFullYear()}${MAV.lang === "es" ? "T" : "Q"}${q}`; };
  MAV.month = (date) => date.toLocaleDateString(locale(), { month: "short", year: "numeric", timeZone: "UTC" });

  // ------------------------------------------------------------ data
  const cache = new Map();
  MAV.json = (path) => {
    if (!cache.has(path)) cache.set(path, fetch(path).then((r) => { if (!r.ok) throw new Error(path + " " + r.status); return r.json(); }));
    return cache.get(path);
  };
  MAV.parseDates = (arr) => arr.map((s) => new Date(s + (s.length === 7 ? "-01" : "") + "T00:00:00Z"));

  // ------------------------------------------------------------ page chrome
  function wireChrome() {
    document.querySelectorAll(".lang-toggle button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.lang === MAV.lang));
      b.addEventListener("click", () => MAV.setLang(b.dataset.lang));
    });
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, { delimiters: [
        { left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wireChrome); else wireChrome();

  // ------------------------------------------------------------ controls
  /* MAV.slider(parent, {key, label:{es,en}, min, max, step, value, fmt, note:{es,en}}, onInput) -> {get, set} */
  MAV.slider = (parent, o, onInput) => {
    const id = "ctl-" + o.key;
    const row = document.createElement("div");
    row.className = "ctl";
    row.innerHTML = `<div class="ctl-head"><label for="${id}"></label><output for="${id}"></output></div>
      <input type="range" id="${id}"><div class="ctl-note"></div>`;
    const input = row.querySelector("input"), out = row.querySelector("output"),
      label = row.querySelector("label"), note = row.querySelector(".ctl-note");
    Object.assign(input, { min: o.min, max: o.max, step: o.step, value: o.value });
    const fmt = o.fmt || ((v) => MAV.num(v, (String(o.step).split(".")[1] || "").length));
    const paint = () => { label.textContent = MAV.t(o.label); note.textContent = MAV.t(o.note || "");
      out.textContent = fmt(+input.value); input.setAttribute("aria-valuetext", fmt(+input.value)); };
    input.addEventListener("input", () => { paint(); onInput && onInput(+input.value); });
    MAV.onLang(paint); paint();
    parent.appendChild(row);
    return { get: () => +input.value, set: (v) => { input.value = v; paint(); }, input };
  };

  /* MAV.segmented(parent, {key, label, options:[{value, label:{es,en}}], value}, onChange) */
  MAV.segmented = (parent, o, onChange) => {
    const row = document.createElement("div");
    row.className = "ctl";
    const head = document.createElement("div"); head.className = "ctl-head";
    const lab = document.createElement("span"); head.appendChild(lab);
    const seg = document.createElement("div"); seg.className = "seg"; seg.setAttribute("role", "group");
    let value = o.value;
    const buttons = o.options.map((opt) => {
      const b = document.createElement("button"); b.type = "button";
      b.addEventListener("click", () => { value = opt.value; paint(); onChange && onChange(value); });
      seg.appendChild(b); return [b, opt];
    });
    const paint = () => { lab.textContent = MAV.t(o.label); seg.setAttribute("aria-label", MAV.t(o.label));
      buttons.forEach(([b, opt]) => { b.textContent = MAV.t(opt.label); b.setAttribute("aria-pressed", String(opt.value === value)); }); };
    row.append(head, seg); parent.appendChild(row);
    MAV.onLang(paint); paint();
    return { get: () => value, set: (v) => { value = v; paint(); } };
  };

  /* MAV.select(parent, {key, label, options:[{value, label}], value}, onChange) */
  MAV.select = (parent, o, onChange) => {
    const id = "ctl-" + o.key;
    const row = document.createElement("div"); row.className = "ctl";
    row.innerHTML = `<div class="ctl-head"><label for="${id}"></label></div><select id="${id}"></select>`;
    const sel = row.querySelector("select"), lab = row.querySelector("label");
    const paint = () => { lab.textContent = MAV.t(o.label); const v = sel.value || o.value; sel.textContent = "";
      o.options.forEach((opt) => { const el = document.createElement("option"); el.value = opt.value; el.textContent = MAV.t(opt.label); sel.appendChild(el); });
      sel.value = v; };
    sel.addEventListener("change", () => onChange && onChange(sel.value));
    parent.appendChild(row); MAV.onLang(paint); paint(); sel.value = o.value;
    return { get: () => sel.value, set: (v) => { sel.value = v; } };
  };

  /* MAV.tiles(el, [{label, value, note}]) - stat tiles re-painted on every call */
  MAV.tiles = (el, items) => {
    el.classList.add("tiles"); el.textContent = "";
    items.forEach((it) => {
      const d = document.createElement("div"); d.className = "tile";
      const l = document.createElement("div"); l.className = "label"; l.textContent = MAV.t(it.label);
      const v = document.createElement("div"); v.className = "value"; v.textContent = it.value;
      d.append(l, v);
      if (it.note) { const n = document.createElement("div"); n.className = "note"; n.textContent = MAV.t(it.note); d.appendChild(n); }
      el.appendChild(d);
    });
  };

  // ------------------------------------------------------------ charts
  const DASH = ["", "7 5", "2 4", "10 4 2 4"];
  const STROKE = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  MAV.style = (i) => ({ stroke: STROKE[i % 4], dash: DASH[i % 4] });

  function keySvg(style, w = 26) {
    const s = MAV.style(style);
    return `<svg width="${w}" height="10" aria-hidden="true"><line x1="0" x2="${w}" y1="5" y2="5" stroke="${s.stroke}" stroke-width="2" stroke-dasharray="${s.dash}" stroke-linecap="round"/></svg>`;
  }

  function csv(rows) {
    return rows.map((r) => r.map((c) => (typeof c === "string" && /[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n");
  }

  /* Card skeleton shared by all chart types. */
  function card(el, o) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const parts = { title: el.querySelector(".chart-title"), sub: el.querySelector(".chart-sub"), legend: el.querySelector(".legend"),
      plot: el.querySelector(".chart"), tip: el.querySelector(".tooltip"), table: el.querySelector(".table-view"),
      tableBtn: el.querySelector('[data-act="table"]'), csvBtn: el.querySelector('[data-act="csv"]') };
    const paintTableBtn = () => {
      const on = parts.table.classList.contains("on");
      parts.tableBtn.setAttribute("aria-expanded", String(on));
      parts.tableBtn.textContent = MAV.t(on ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" });
    };
    parts.tableBtn.addEventListener("click", () => { parts.table.classList.toggle("on"); paintTableBtn(); });
    MAV.onLang(paintTableBtn);
    paintTableBtn();
    return parts;
  }

  function fillTable(parts, header, rows) {
    const t = document.createElement("table");
    const thead = t.createTHead().insertRow();
    header.forEach((h) => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
    const tb = t.createTBody();
    rows.forEach((r) => { const tr = tb.insertRow(); r.forEach((c) => { tr.insertCell().textContent = c; }); });
    parts.table.textContent = ""; parts.table.appendChild(t);
  }

  /* MAV.lineChart(el, opts) -> {update(opts)}
     opts: title, sub, xType 'time'|'linear', xLabel, yLabel, yFmt, xFmt, height, zero,
           series:[{name, x[], y[], style, label(bool, default true), width}],
           shades:[{from,to,label}], vlines:[{x,label}], hlines:[{y,label}], points:[{x,y,label,style}],
           yDomain, xDomain, csvName, digits */
  MAV.lineChart = (el, initial) => {
    let o = initial;
    const parts = card(el, o);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    let W = 0;

    function draw() {
      if (!o.series || !o.series.length) { svg.selectAll("*").remove(); return; }
      const width = Math.max(300, parts.plot.clientWidth || 640);
      W = width;
      const height = o.height || Math.round(Math.min(360, Math.max(240, width * 0.5)));
      const m = { t: 12, r: o.labelRoom ?? 110, b: 42, l: 54 };
      if (width < 520) m.r = Math.min(m.r, 64);
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
      svg.selectAll("*").remove();

      const time = o.xType === "time";
      const all = o.series.filter((s) => !s.hidden);
      const xs = all.flatMap((s) => s.x), ys = all.flatMap((s) => s.y).filter((v) => v != null && isFinite(v));
      (o.hlines || []).forEach((h) => ys.push(h.y));
      const x = (time ? d3.scaleUtc() : d3.scaleLinear()).domain(o.xDomain || d3.extent(xs)).range([m.l, width - m.r]);
      let yd = o.yDomain || d3.extent(ys);
      if (o.zero && !o.yDomain) yd = [Math.min(0, yd[0]), Math.max(0, yd[1])];
      const y = d3.scaleLinear().domain(yd).nice().range([height - m.b, m.t]);
      const yFmt = o.yFmt || ((v) => MAV.num(v, Math.abs(yd[1] - yd[0]) < 5 ? 1 : 0));
      const xFmt = o.xFmt || (time ? (d) => String(d.getUTCFullYear()) : (v) => MAV.num(v, 0));

      svg.append("title").text(MAV.t(o.title));
      // shading first
      (o.shades || []).forEach((s) => {
        const x0 = x(s.from), x1 = x(s.to);
        svg.append("rect").attr("class", "shade").attr("x", Math.min(x0, x1)).attr("y", m.t)
          .attr("width", Math.max(1, Math.abs(x1 - x0))).attr("height", height - m.b - m.t);
        if (s.label) svg.append("text").attr("class", "alabel").attr("x", Math.min(x0, x1) + 4).attr("y", m.t + 12).text(MAV.t(s.label));
      });
      const ticksX = Math.max(3, Math.round((width - m.l - m.r) / 90));
      svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickSize(-(width - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`)
        .call(d3.axisBottom(x).ticks(ticksX).tickFormat(xFmt).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat(yFmt).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end").text(MAV.t(o.xLabel));
      if (o.yLabel) svg.append("text").attr("class", "axis-title").attr("x", m.l).attr("y", m.t - 2).attr("dy", "-0.2em").text(MAV.t(o.yLabel));
      if (o.zero && y.domain()[0] < 0 && y.domain()[1] > 0) svg.append("line").attr("class", "zero").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y(0)).attr("y2", y(0));
      (o.hlines || []).forEach((h) => {
        svg.append("line").attr("class", "zero").attr("stroke-dasharray", "3 3").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y(h.y)).attr("y2", y(h.y));
        if (h.label) svg.append("text").attr("class", "alabel").attr("x", width - m.r).attr("y", y(h.y) - 5).attr("text-anchor", "end").text(MAV.t(h.label));
      });
      (o.vlines || []).forEach((v) => {
        svg.append("line").attr("class", "zero").attr("stroke-dasharray", "3 3").attr("x1", x(v.x)).attr("x2", x(v.x)).attr("y1", m.t).attr("y2", height - m.b);
        if (v.label) svg.append("text").attr("class", "alabel").attr("x", x(v.x) + 4).attr("y", height - m.b - 6).text(MAV.t(v.label));
      });

      const line = d3.line().defined((d) => d[1] != null && isFinite(d[1])).x((d) => x(d[0])).y((d) => y(d[1]));
      const ends = [];
      all.forEach((s, i) => {
        const st = MAV.style(s.style ?? i);
        const pts = s.x.map((xv, k) => [xv, s.y[k]]);
        svg.append("path").attr("class", "series").attr("d", line(pts)).attr("stroke", st.stroke)
          .attr("stroke-dasharray", st.dash).attr("stroke-width", s.width || 2);
        const last = [...pts].reverse().find((p) => p[1] != null && isFinite(p[1]));
        if (last && s.label !== false) ends.push({ y: y(last[1]), x: x(last[0]), text: MAV.t(s.short || s.name) });
      });
      (o.points || []).forEach((p) => {
        svg.append("circle").attr("class", "dot").attr("cx", x(p.x)).attr("cy", y(p.y)).attr("r", 4.5).attr("fill", MAV.style(p.style || 0).stroke);
        if (p.label) svg.append("text").attr("class", "alabel").attr("x", x(p.x) + 8).attr("y", y(p.y) - 8).text(MAV.t(p.label));
      });
      // direct end labels, only when they don't collide
      ends.sort((a, b) => a.y - b.y);
      const clash = ends.some((e, k) => k && e.y - ends[k - 1].y < 15);
      if (!clash && m.r >= 60) ends.forEach((e) => svg.append("text").attr("class", "dlabel").attr("x", e.x + 8).attr("y", e.y + 4).text(e.text));

      // legend (always, for 2+ series)
      parts.legend.innerHTML = "";
      if (all.length > 1) all.forEach((s, i) => {
        const k = document.createElement("span"); k.className = "key";
        k.innerHTML = keySvg(s.style ?? i); const t = document.createElement("span"); t.textContent = MAV.t(s.name);
        k.appendChild(t); parts.legend.appendChild(k);
      });

      // hover layer: crosshair snaps to the nearest x of the first series
      const base = all[0];
      const hair = svg.append("line").attr("class", "xhair").attr("y1", m.t).attr("y2", height - m.b).style("display", "none");
      const dots = all.map((s, i) => svg.append("circle").attr("class", "dot").attr("r", 4).attr("fill", MAV.style(s.style ?? i).stroke).style("display", "none"));
      const bis = d3.bisector((d) => +d).center;
      const show = (px) => {
        if (!base) return;
        const xv = x.invert(px);
        const hits = all.map((s) => { const k = Math.max(0, Math.min(s.x.length - 1, bis(s.x, +xv))); return [s.x[k], s.y[k]]; });
        const xh = hits[0][0];
        hair.attr("x1", x(xh)).attr("x2", x(xh)).style("display", null);
        hits.forEach((h, i) => (h[1] == null || !isFinite(h[1]) ? dots[i].style("display", "none")
          : dots[i].attr("cx", x(h[0])).attr("cy", y(h[1])).style("display", null)));
        const tip = parts.tip; tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head";
        head.textContent = time ? (o.tipDate || MAV.quarter)(xh) : `${MAV.t(o.xLabel)} ${xFmt(xh)}`; tip.appendChild(head);
        all.forEach((s, i) => {
          const r = document.createElement("div"); r.className = "t-row";
          const key = document.createElement("span"); key.innerHTML = keySvg(s.style ?? i, 14);
          const val = document.createElement("b"); val.textContent = hits[i][1] == null ? "—" : (o.tipFmt || yFmt)(hits[i][1]);
          const nm = document.createElement("span"); nm.textContent = MAV.t(s.name);
          r.append(key, val, nm); tip.appendChild(r);
        });
        const scale = parts.plot.clientWidth / W;
        const left = x(xh) * scale; const tw = tip.offsetWidth || 160;
        tip.style.left = (left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
        tip.style.top = "8px"; tip.classList.add("on");
      };
      const hide = () => { hair.style("display", "none"); dots.forEach((d) => d.style("display", "none")); parts.tip.classList.remove("on"); };
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b)
        .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", MAV.t({ es: "Explorar valores", en: "Explore values" }))
        .on("pointermove", (ev) => show(d3.pointer(ev)[0]))
        .on("pointerleave", hide).on("blur", hide)
        .on("focus", () => show(width - m.r))
        .on("keydown", (ev) => {
          if (!base) return; const cur = +hair.attr("x1") || width - m.r;
          if (ev.key === "ArrowLeft") { show(cur - (width - m.l - m.r) / Math.max(1, base.x.length - 1)); ev.preventDefault(); }
          if (ev.key === "ArrowRight") { show(cur + (width - m.l - m.r) / Math.max(1, base.x.length - 1)); ev.preventDefault(); }
        });

      // text + table
      parts.title.textContent = MAV.t(o.title);
      parts.sub.textContent = MAV.t(o.sub || "");
      parts.sub.style.display = o.sub ? "" : "none";
      const xs0 = base ? base.x : [];
      const header = [MAV.t(o.xLabel) || (time ? MAV.t({ es: "Fecha", en: "Date" }) : "x"), ...all.map((s) => MAV.t(s.name))];
      const d = o.digits ?? 3;
      const rows = xs0.map((xv, k) => [time ? (o.tipDate || MAV.quarter)(xv) : String(xv),
        ...all.map((s) => (s.y[k] == null || !isFinite(s.y[k]) ? "" : MAV.num(s.y[k], d)))]);
      fillTable(parts, header, rows);
      parts.csvBtn.onclick = () => {
        const raw = [header, ...xs0.map((xv, k) => [time ? xv.toISOString().slice(0, 10) : xv, ...all.map((s) => (s.y[k] == null ? "" : s.y[k]))])];
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([csv(raw)], { type: "text/csv" }));
        a.download = (o.csvName || "mav-datos") + ".csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      };
    }

    let raf = 0;
    const redraw = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); };
    new ResizeObserver(redraw).observe(parts.plot);
    MAV.onLang(draw);
    draw();
    return { update(next) { o = Object.assign({}, o, next); draw(); }, get opts() { return o; } };
  };

  /* MAV.scatter(el, opts): points:[{x,y,label?}], fit:{a,b,label}, highlight: index set, xLabel, yLabel, xFmt, yFmt */
  MAV.scatter = (el, initial) => {
    let o = initial;
    const parts = card(el, o);
    const svg = d3.select(parts.plot).insert("svg", ".tooltip");
    function draw() {
      const width = Math.max(300, parts.plot.clientWidth || 640);
      const height = o.height || Math.round(Math.min(380, Math.max(260, width * 0.56)));
      const m = { t: 14, r: 24, b: 42, l: 54 };
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img"); svg.selectAll("*").remove();
      svg.append("title").text(MAV.t(o.title));
      const P = o.points;
      const x = d3.scaleLinear().domain(o.xDomain || d3.extent(P, (p) => p.x)).nice().range([m.l, width - m.r]);
      const y = d3.scaleLinear().domain(o.yDomain || d3.extent(P, (p) => p.y)).nice().range([height - m.b, m.t]);
      const xFmt = o.xFmt || ((v) => MAV.num(v, 1)), yFmt = o.yFmt || ((v) => MAV.num(v, 1));
      svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(xFmt).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(yFmt).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end").text(MAV.t(o.xLabel));
      if (o.yLabel) svg.append("text").attr("class", "axis-title").attr("x", m.l).attr("y", m.t - 4).text(MAV.t(o.yLabel));
      if (y.domain()[0] < 0 && y.domain()[1] > 0) svg.append("line").attr("class", "zero").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y(0)).attr("y2", y(0));
      if (x.domain()[0] < 0 && x.domain()[1] > 0) svg.append("line").attr("class", "zero").attr("y1", m.t).attr("y2", height - m.b).attr("x1", x(0)).attr("x2", x(0));
      svg.append("g").selectAll("circle").data(P).join("circle").attr("class", "dot")
        .attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", (p) => (p.hi ? 5 : 3.5))
        .attr("fill", (p) => (p.hi ? "var(--s1)" : "var(--s3)")).attr("fill-opacity", (p) => (p.hi ? 1 : 0.75));
      if (o.fit) {
        const [x0, x1] = x.domain();
        svg.append("path").attr("class", "series").attr("stroke", "var(--s1)")
          .attr("d", d3.line()([[x(x0), y(o.fit.a + o.fit.b * x0)], [x(x1), y(o.fit.a + o.fit.b * x1)]]));
        if (o.fit.label) svg.append("text").attr("class", "dlabel").attr("x", x(x1) - 4).attr("y", y(o.fit.a + o.fit.b * x1) - 8).attr("text-anchor", "end").text(MAV.t(o.fit.label));
      }
      P.filter((p) => p.label).forEach((p) => svg.append("text").attr("class", "alabel").attr("x", x(p.x) + 7).attr("y", y(p.y) - 7).text(MAV.t(p.label)));
      const vor = d3.Delaunay.from(P, (p) => x(p.x), (p) => y(p.y));
      const ring = svg.append("circle").attr("r", 7).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 1.5).style("display", "none");
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b).attr("fill", "transparent")
        .on("pointermove", (ev) => {
          const [px, py] = d3.pointer(ev); const k = vor.find(px, py); const p = P[k];
          ring.attr("cx", x(p.x)).attr("cy", y(p.y)).style("display", null);
          const tip = parts.tip; tip.textContent = "";
          const h = document.createElement("div"); h.className = "t-head"; h.textContent = p.name ? MAV.t(p.name) : ""; tip.appendChild(h);
          [[o.xLabel, xFmt(p.x)], [o.yLabel, yFmt(p.y)]].forEach(([l, v]) => {
            const r = document.createElement("div"); r.className = "t-row"; r.style.gridTemplateColumns = "auto 1fr";
            const b = document.createElement("b"); b.textContent = v; const s = document.createElement("span"); s.textContent = MAV.t(l);
            r.append(b, s); tip.appendChild(r); });
          const sc = parts.plot.clientWidth / width; const left = x(p.x) * sc; const tw = tip.offsetWidth || 150;
          tip.style.left = (left + 14 + tw > parts.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
          tip.style.top = Math.max(0, y(p.y) * sc - 20) + "px"; tip.classList.add("on");
        })
        .on("pointerleave", () => { ring.style("display", "none"); parts.tip.classList.remove("on"); });
      parts.title.textContent = MAV.t(o.title); parts.sub.textContent = MAV.t(o.sub || ""); parts.sub.style.display = o.sub ? "" : "none";
      parts.legend.innerHTML = "";
      const header = [MAV.t({ es: "Observación", en: "Observation" }), MAV.t(o.xLabel), MAV.t(o.yLabel)];
      fillTable(parts, header, P.map((p) => [MAV.t(p.name || ""), MAV.num(p.x, 3), MAV.num(p.y, 3)]));
      parts.csvBtn.onclick = () => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([csv([header, ...P.map((p) => [MAV.t(p.name || ""), p.x, p.y])])], { type: "text/csv" }));
        a.download = (o.csvName || "mav-datos") + ".csv"; a.click();
      };
    }
    let raf = 0; new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(parts.plot);
    MAV.onLang(draw); draw();
    return { update(next) { o = Object.assign({}, o, next); draw(); } };
  };
})();
