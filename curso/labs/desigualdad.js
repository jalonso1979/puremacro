/* Laboratorio «Desigualdad»: Aiyagari (1994) resuelto en el navegador.
   Hogar por grillas endógenas (Carroll 2006), distribución estacionaria por el método
   de la lotería (Young 2010) y r* por Brent sobre el exceso de oferta de activos.
   El núcleo numérico (aiyagariKernel) no toca el DOM: corre en un Web Worker creado
   desde un Blob y también se puede cargar en node para validarlo. */

function aiyagariKernel() {
  "use strict";

  // ------------------------------------------------------------ income process
  /* erf by its all-positive series erf(x) = 2/√π e^{-x²} Σ 2^n x^{2n+1}/(2n+1)!!.
     No cancellation, so the absolute error is ~1e-16 for every x: exactly what
     Tauchen's differences of CDFs need. */
  function erf(x) {
    const ax = Math.abs(x);
    if (ax > 8.5) return x > 0 ? 1 : -1;
    const x2 = 2 * ax * ax;
    let term = ax, sum = ax;
    for (let n = 1; n < 400; n++) {
      term *= x2 / (2 * n + 1); sum += term;
      if (term < 1e-17 * sum) break;
    }
    const v = Math.min(1, (2 / Math.sqrt(Math.PI)) * Math.exp(-ax * ax) * sum);
    return x < 0 ? -v : v;
  }
  const Phi = (u) => 0.5 * (1 + erf(u / Math.SQRT2));

  /* Tauchen (1986), m standard deviations: the same construction as puremacro.vfi.tauchen. */
  function tauchen(n, rho, sigma, m = 3) {
    const sz = sigma / Math.sqrt(1 - rho * rho), zmax = m * sz;
    const z = Array.from({ length: n }, (_, i) => -zmax + (2 * zmax * i) / (n - 1));
    const step = z[1] - z[0], P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      const mu = rho * z[i];
      P[i * n] = Phi((z[0] - mu + step / 2) / sigma);
      P[i * n + n - 1] = 1 - Phi((z[n - 1] - mu - step / 2) / sigma);
      for (let j = 1; j < n - 1; j++) P[i * n + j] = Phi((z[j] - mu + step / 2) / sigma) - Phi((z[j] - mu - step / 2) / sigma);
    }
    return { z, P };
  }

  /* Rouwenhorst (1995): the same recursion as puremacro.vfi.rouwenhorst. */
  function rouwenhorst(n, rho, sigma) {
    const p = (1 + rho) / 2;
    let M = [[p, 1 - p], [1 - p, p]];
    for (let k = 3; k <= n; k++) {
      const Q = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let i = 0; i < k - 1; i++) for (let j = 0; j < k - 1; j++) {
        Q[i][j] += p * M[i][j]; Q[i][j + 1] += (1 - p) * M[i][j];
        Q[i + 1][j] += (1 - p) * M[i][j]; Q[i + 1][j + 1] += p * M[i][j];
      }
      for (let i = 1; i < k - 1; i++) for (let j = 0; j < k; j++) Q[i][j] /= 2;
      M = Q;
    }
    const psi = Math.sqrt(n - 1) * sigma / Math.sqrt(1 - rho * rho);
    const z = Array.from({ length: n }, (_, i) => -psi + (2 * psi * i) / (n - 1));
    return { z, P: Float64Array.from(M.flat()) };
  }

  function markovStationary(P, n) {
    let pi = new Float64Array(n).fill(1 / n);
    for (let it = 0; it < 100000; it++) {
      const nx = new Float64Array(n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) nx[j] += pi[i] * P[i * n + j];
      let d = 0; for (let j = 0; j < n; j++) d = Math.max(d, Math.abs(nx[j] - pi[j]));
      pi = nx; if (d < 1e-15) break;
    }
    const s = pi.reduce((a, b) => a + b, 0);
    return pi.map((v) => v / s);
  }

  // ------------------------------------------------------------ setup
  const DEFAULTS = { alpha: 0.36, delta: 0.08, beta: 0.96, gamma: 2, rho: 0.9, sigma: 0.2, nz: 7,
    abar: 0, disc: "tauchen", na: 300, amax: 100, curv: 3 };

  const prices = (p, L, r) => {
    const kl = Math.pow(p.alpha / (r + p.delta), 1 / (1 - p.alpha));
    return { w: (1 - p.alpha) * Math.pow(kl, p.alpha), Kd: kl * L };
  };

  function setup(params) {
    const p = Object.assign({}, DEFAULTS, params);
    const mk = p.disc === "rouwenhorst" ? rouwenhorst(p.nz, p.rho, p.sigma) : tauchen(p.nz, p.rho, p.sigma, 3);
    const pi = markovStationary(mk.P, p.nz);
    const ez = mk.z.map(Math.exp);
    const L = ez.reduce((s, e, j) => s + e * pi[j], 0);
    /* The ad hoc limit cannot exceed the natural one, w·min e^z / r: beyond it the worst
       income state could never repay. It is tightest at the highest rate the solver tries,
       r → 1/β − 1, so clamp there (with a 10% margin) and keep one grid for every r. */
    const rMax = 1 / p.beta - 1;
    const natural = (prices(p, L, rMax).w * Math.min(...ez)) / rMax;
    p.abarAsked = p.abar;
    p.abarNatural = natural;
    if (p.abar > 0.9 * natural) p.abar = Math.floor(0.9 * natural * 100) / 100;
    const aMin = -p.abar, a = new Float64Array(p.na);
    for (let i = 0; i < p.na; i++) a[i] = aMin + (p.amax - aMin) * Math.pow(i / (p.na - 1), p.curv);
    return { p, z: mk.z, P: mk.P, pi, ez, L, a, aMin, na: p.na, nz: p.nz };
  }

  // ------------------------------------------------------------ household: EGM
  /* Euler: u'(c) = β(1+r) Σ Π(z,z') u'(c(a',z')) on a grid of a'; invert for c and for
     today's a; interpolate back on the exogenous grid. Borrowing constraint where
     a <= a_endog(a_min). Linear extrapolation above the top endogenous point. */
  function egm(S, r, w, c0, tol = 1e-9, maxIter = 4000) {
    const { a, na, nz, P, ez, aMin } = S, { beta, gamma } = S.p, R = 1 + r, N = na * nz;
    const y = ez.map((e) => w * e);
    const coh = new Float64Array(N);
    let c = new Float64Array(N);
    for (let i = 0; i < na; i++) for (let z = 0; z < nz; z++) {
      const k = i * nz + z; coh[k] = R * a[i] + y[z];
      c[k] = c0 ? c0[k] : Math.max(coh[k] - aMin, 1e-10);
    }
    let cn = new Float64Array(N);
    const mu = new Float64Array(N), ce = new Float64Array(N), ae = new Float64Array(N);
    const bR = beta * R, ig = -1 / gamma;
    let it = 0, sup = Infinity;
    for (it = 1; it <= maxIter; it++) {
      for (let k = 0; k < N; k++) mu[k] = Math.pow(c[k], -gamma);
      for (let i = 0; i < na; i++) {
        const row = i * nz;
        for (let z = 0; z < nz; z++) {
          let s = 0; const pz = z * nz;
          for (let zp = 0; zp < nz; zp++) s += P[pz + zp] * mu[row + zp];
          const cc = Math.pow(bR * s, ig);
          ce[row + z] = cc; ae[row + z] = (cc + a[i] - y[z]) / R;
        }
      }
      sup = 0;
      for (let z = 0; z < nz; z++) {
        let j = 0; const a0 = ae[z];
        for (let i = 0; i < na; i++) {
          const k = i * nz + z, ai = a[i];
          let v;
          if (ai <= a0) v = coh[k] - aMin;
          else {
            while (j < na - 2 && ae[(j + 1) * nz + z] < ai) j++;
            const xl = ae[j * nz + z], xh = ae[(j + 1) * nz + z];
            const yl = ce[j * nz + z], yh = ce[(j + 1) * nz + z];
            v = yl + ((yh - yl) * (ai - xl)) / (xh - xl);
          }
          if (v < 1e-10) v = 1e-10;
          const d = Math.abs(v - c[k]); if (d > sup) sup = d;
          cn[k] = v;
        }
      }
      const t = c; c = cn; cn = t;
      if (sup < tol) break;
    }
    const ap = new Float64Array(N), amax = a[na - 1];
    for (let k = 0; k < N; k++) {
      let v = coh[k] - c[k];
      if (v < aMin) v = aMin; else if (v > amax) v = amax;
      ap[k] = v; c[k] = coh[k] - v;
    }
    return { c, ap, iters: it, sup, converged: sup < tol };
  }

  // ------------------------------------------------------------ distribution: Young (2010)
  /* Each (a,z) sends its mass to the two grid nodes around a'(a,z), with weights that
     preserve E[a']; then z moves with Π. Iterated to a sup-norm fixed point. */
  function lottery(S, ap, mu0, tol = 1e-12, maxIter = 200000) {
    const { a, na, nz, P, pi } = S, N = na * nz;
    const jl = new Int32Array(N), wl = new Float64Array(N);
    for (let z = 0; z < nz; z++) {
      let j = 0;
      for (let i = 0; i < na; i++) {
        const k = i * nz + z, x = ap[k];
        // policy is increasing in a, so the bracket pointer only moves right
        if (x <= a[0]) { jl[k] = 0; wl[k] = 1; continue; }
        if (x >= a[na - 1]) { jl[k] = na - 2; wl[k] = 0; continue; }
        if (a[j] > x) j = 0;
        while (j < na - 2 && a[j + 1] < x) j++;
        jl[k] = j; wl[k] = Math.min(1, Math.max(0, (a[j + 1] - x) / (a[j + 1] - a[j])));
      }
    }
    const half = new Float64Array(N);
    /* out = T x: one forward step of the distribution (linear in x). Returns sup|out − x|. */
    const push = (x, out) => {
      half.fill(0);
      for (let k = 0; k < N; k++) {
        const m = x[k]; if (m === 0) continue;
        const z = k % nz, j = jl[k], wv = wl[k];
        half[j * nz + z] += m * wv; half[(j + 1) * nz + z] += m * (1 - wv);
      }
      let sup = 0;
      for (let i = 0; i < na; i++) {
        const row = i * nz;
        for (let zp = 0; zp < nz; zp++) {
          let s = 0;
          for (let z = 0; z < nz; z++) s += half[row + z] * P[z * nz + zp];
          const d = Math.abs(s - x[row + zp]); if (d > sup) sup = d;
          out[row + zp] = s;
        }
      }
      return sup;
    };
    let mu = new Float64Array(N);
    if (mu0) mu.set(mu0); else for (let i = 0; i < na; i++) for (let z = 0; z < nz; z++) mu[i * nz + z] = pi[z] / na;
    let nx = new Float64Array(N);
    let it = 0, sup = Infinity, krylov = 0;
    const plain = Math.min(maxIter, 1500);
    for (it = 1; it <= plain; it++) {
      sup = push(mu, nx);
      const t = mu; mu = nx; nx = t;
      if (sup < tol) break;
    }
    if (sup >= tol) {
      /* Near r = 1/β − 1 wealth drifts slowly and plain iteration needs ~1/gap steps.
         Solve (I − T + v 1ᵀ) μ = v instead, v uniform: its solution is the stationary μ with
         unit mass. Restarted GMRES needs far fewer applications of T. */
      const v = 1 / N;
      const Ax = (x, out) => { push(x, out); let s = 0; for (let k = 0; k < N; k++) s += x[k];
        for (let k = 0; k < N; k++) out[k] = x[k] - out[k] + v * s; };
      krylov = gmres(Ax, v, mu, N, 1e-13, 60, 40);
      for (let k = 0; k < N; k++) if (mu[k] < 0) mu[k] = 0;
      let tot = 0; for (let k = 0; k < N; k++) tot += mu[k];
      for (let k = 0; k < N; k++) mu[k] /= tot;
      for (let p2 = 0; p2 < 400; p2++) {        // polish with plain steps
        sup = push(mu, nx); const t = mu; mu = nx; nx = t; it++;
        if (sup < tol) break;
      }
    }
    let tot = 0; for (let k = 0; k < N; k++) tot += mu[k];
    for (let k = 0; k < N; k++) mu[k] /= tot;
    return { mu, iters: it, krylov, sup, converged: sup < Math.max(tol, 1e-11) };
  }

  /* Restarted GMRES(m) for A x = b with b the constant vector bval; x is updated in place.
     Returns the number of matrix–vector products. */
  function gmres(A, bval, x, N, tol, m, cycles) {
    const bnorm = Math.abs(bval) * Math.sqrt(N);
    const V = Array.from({ length: m + 1 }, () => new Float64Array(N));
    const H = Array.from({ length: m + 1 }, () => new Float64Array(m));
    const cs = new Float64Array(m), sn = new Float64Array(m), g = new Float64Array(m + 1);
    const w = new Float64Array(N);
    let mv = 0;
    for (let cyc = 0; cyc < cycles; cyc++) {
      A(x, w); mv++;
      let beta = 0;
      for (let k = 0; k < N; k++) { const r = bval - w[k]; V[0][k] = r; beta += r * r; }
      beta = Math.sqrt(beta);
      if (beta / bnorm < tol) return mv;
      for (let k = 0; k < N; k++) V[0][k] /= beta;
      g.fill(0); g[0] = beta;
      let kUsed = 0;
      for (let j = 0; j < m; j++) {
        A(V[j], w); mv++;
        for (let i = 0; i <= j; i++) {
          let h = 0; const Vi = V[i];
          for (let k = 0; k < N; k++) h += w[k] * Vi[k];
          H[i][j] = h;
          for (let k = 0; k < N; k++) w[k] -= h * Vi[k];
        }
        let hn = 0; for (let k = 0; k < N; k++) hn += w[k] * w[k]; hn = Math.sqrt(hn);
        H[j + 1][j] = hn;
        if (hn > 0) for (let k = 0; k < N; k++) V[j + 1][k] = w[k] / hn;
        for (let i = 0; i < j; i++) {
          const t = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
          H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j]; H[i][j] = t;
        }
        const den = Math.hypot(H[j][j], H[j + 1][j]) || 1e-300;
        cs[j] = H[j][j] / den; sn[j] = H[j + 1][j] / den;
        H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j]; H[j + 1][j] = 0;
        g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j];
        kUsed = j + 1;
        if (Math.abs(g[j + 1]) / bnorm < tol || hn === 0) break;
      }
      const y = new Float64Array(kUsed);
      for (let i = kUsed - 1; i >= 0; i--) {
        let s = g[i]; for (let l = i + 1; l < kUsed; l++) s -= H[i][l] * y[l];
        y[i] = s / H[i][i];
      }
      for (let i = 0; i < kUsed; i++) { const Vi = V[i], yi = y[i]; for (let k = 0; k < N; k++) x[k] += yi * Vi[k]; }
    }
    return mv;
  }

  // ------------------------------------------------------------ one price r
  function household(S, r, warm) {
    const { w, Kd } = prices(S.p, S.L, r);
    const h = egm(S, r, w, warm && warm.c);
    const d = lottery(S, h.ap, warm && warm.mu);
    let A = 0; const { a, na, nz } = S;
    for (let i = 0; i < na; i++) { let m = 0; for (let z = 0; z < nz; z++) m += d.mu[i * nz + z]; A += m * a[i]; }
    return { r, w, Kd, A, c: h.c, ap: h.ap, mu: d.mu, egmIters: h.iters, lotIters: d.iters, krylov: d.krylov,
      ok: h.converged && d.converged };
  }

  // ------------------------------------------------------------ general equilibrium
  /* Φ(r) = A(r) − K^d(r) rises with r and diverges as r → 1/β − 1. Brent's method on a
     bracket found by stepping out from a guess. Every evaluation warm-starts the next. */
  /* The asset grid must not bind: if households pile up in the top 3% of nodes, stretch
     a_max and solve again from the previous r*. */
  function equilibrium(params, opts = {}) {
    const t0 = Date.now();
    let prm = Object.assign({}, DEFAULTS, params), eq = null, tries = 0;
    if (params.amax == null) {
      // scale the top of the grid with the best income state (the course's 100 at its calibration)
      const sz = prm.sigma / Math.sqrt(1 - prm.rho * prm.rho);
      const zmax = (prm.disc === "rouwenhorst" ? Math.sqrt(prm.nz - 1) : 3) * sz;
      prm.amax = Math.ceil((100 * Math.max(1, Math.exp(zmax - 3 * 0.2 / Math.sqrt(1 - 0.81)))) / 50) * 50;
    }
    for (;;) {
      tries++;
      try { eq = equilibriumOnGrid(prm, opts); }
      catch (e) {
        if (!e || !e.regrid || tries >= 5) throw e;
        opts = Object.assign({}, opts, { guess: e.r });
        prm = Object.assign({}, prm, { amax: prm.amax * 2 });
        continue;
      }
      if (eq.massTop < TOPTOL || tries >= 5) break;
      opts = Object.assign({}, opts, { guess: eq.r });
      prm = Object.assign({}, prm, { amax: prm.amax * 2 });
    }
    eq.ms = Date.now() - t0; eq.gridTries = tries;
    return eq;
  }
  const TOPTOL = 1e-5;
  const topMass = (S, mu) => { let m = 0; for (let k = Math.floor(S.na * 0.97) * S.nz; k < S.na * S.nz; k++) m += mu[k]; return m; };

  function equilibriumOnGrid(params, opts = {}) {
    const t0 = Date.now();
    const S = setup(params), p = S.p;
    const rMax = 1 / p.beta - 1, rFloor = -p.delta + 0.01;
    let warm = null, evals = 0;
    const log = [];
    const F = (r) => {
      const h = household(S, r, warm); warm = h; evals++;
      log.push({ r, A: h.A, Kd: h.Kd, egm: h.egmIters, lot: h.lotIters, kr: h.krylov });
      if (!isFinite(h.A)) throw new Error("numerical failure at r = " + r);
      // below r* wealth is lower than in equilibrium: if the grid already binds, stretch it now
      if (h.A < h.Kd && topMass(S, h.mu) > TOPTOL) throw { regrid: true, r };
      return h;
    };
    const xtol = opts.xtol ?? 1e-7;
    let rTop = rMax - 1e-3;
    let g = Math.min(Math.max(opts.guess ?? 0.02, rFloor + 0.002), rTop - 0.0005);
    let hg = F(g), fg = hg.A - hg.Kd;
    let lo, hi, flo, fhi, hlo, hhi;
    // step out toward the sign change
    let step = 0.004;
    if (fg > 0) {
      hi = g; fhi = fg; hhi = hg;
      for (;;) {
        const x = Math.max(rFloor, hi - step), h = F(x), f = h.A - h.Kd;
        if (f <= 0) { lo = x; flo = f; hlo = h; break; }
        hi = x; fhi = f; hhi = h; step *= 2;
        if (x <= rFloor) throw new Error("no bracket below");
      }
    } else {
      lo = g; flo = fg; hlo = hg;
      for (;;) {
        let x = lo + step;
        if (x >= rTop) { // squeeze toward 1/β − 1 when risk is small
          x = rTop; if (x - lo < 1e-9) { rTop = rMax - (rMax - lo) / 4; x = rTop; }
        }
        const h = F(x), f = h.A - h.Kd;
        if (f > 0) { hi = x; fhi = f; hhi = h; break; }
        lo = x; flo = f; hlo = h; step *= 2;
        if (x === rTop) rTop = rMax - (rMax - rTop) / 4;
        if (evals > 60) throw new Error("no bracket above");
      }
    }
    // Brent
    let aB = lo, fa = flo, bB = hi, fb = fhi, cB = aB, fc = fa, dB = bB - aB, eB = dB;
    let best = Math.abs(flo) < Math.abs(fhi) ? hlo : hhi;
    for (let k = 0; k < 60; k++) {
      if (fb * fc > 0) { cB = aB; fc = fa; dB = bB - aB; eB = dB; }
      if (Math.abs(fc) < Math.abs(fb)) { aB = bB; bB = cB; cB = aB; fa = fb; fb = fc; fc = fa; }
      const tol1 = 2 * Number.EPSILON * Math.abs(bB) + 0.5 * xtol, xm = 0.5 * (cB - bB);
      if (Math.abs(xm) <= tol1 || fb === 0) break;
      if (Math.abs(eB) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
        const s = fb / fa; let pp, qq;
        if (aB === cB) { pp = 2 * xm * s; qq = 1 - s; }
        else { const q0 = fa / fc, r0 = fb / fc; pp = s * (2 * xm * q0 * (q0 - r0) - (bB - aB) * (r0 - 1)); qq = (q0 - 1) * (r0 - 1) * (s - 1); }
        if (pp > 0) qq = -qq; pp = Math.abs(pp);
        if (2 * pp < Math.min(3 * xm * qq - Math.abs(tol1 * qq), Math.abs(eB * qq))) { eB = dB; dB = pp / qq; }
        else { dB = xm; eB = dB; }
      } else { dB = xm; eB = dB; }
      aB = bB; fa = fb;
      bB += Math.abs(dB) > tol1 ? dB : (xm > 0 ? tol1 : -tol1);
      const h = F(bB); fb = h.A - h.Kd;
      if (Math.abs(fb) < Math.abs(best.A - best.Kd)) best = h;
    }
    // final solve at the root (warm, cheap) so every object is consistent with r*
    const rStar = bB;
    const sol = Math.abs(best.r - rStar) < 1e-12 ? best : F(rStar);
    const out = summarize(S, sol);
    out.evals = evals; out.ms = Date.now() - t0; out.log = log;
    return out;
  }

  /* Aggregates and the distributional statistics the lab reports. */
  function summarize(S, h) {
    const { p, a, na, nz, L, aMin } = S;
    const K = h.A, Y = Math.pow(K, p.alpha) * Math.pow(L, 1 - p.alpha);
    const muA = new Float64Array(na);
    for (let i = 0; i < na; i++) { let m = 0; for (let z = 0; z < nz; z++) m += h.mu[i * nz + z]; muA[i] = m; }
    // Lorenz and Gini (wealth depends on a only, and the grid is sorted)
    const pop = new Float64Array(na + 1), val = new Float64Array(na + 1);
    let cp = 0, cv = 0;
    for (let i = 0; i < na; i++) { cp += muA[i]; cv += muA[i] * a[i]; pop[i + 1] = cp; val[i + 1] = cv / K; }
    let gini = 1;
    for (let i = 1; i <= na; i++) gini -= (pop[i] - pop[i - 1]) * (val[i] + val[i - 1]);
    let cons = 0, massTop = 0;
    for (let k = 0; k < na * nz; k++) if (h.ap[k] <= aMin + 1e-9) cons += h.mu[k];
    for (let i = Math.floor(na * 0.97); i < na; i++) massTop += muA[i];
    return { params: p, r: h.r, w: h.w, K, L, Y, KY: K / Y, rCM: 1 / p.beta - 1, gini,
      constrained: cons, massTop, ok: h.ok, a: Array.from(a), z: S.z, ez: S.ez, pi: Array.from(S.pi),
      c: h.c, ap: h.ap, mu: h.mu, muA: Array.from(muA), pop: Array.from(pop), val: Array.from(val) };
  }

  /* K^s(r) = A(r) for the classic supply–demand figure, on the equilibrium's own grid.
     Reuses the root finder's evaluations and walks away from r* in both directions, warm
     starting each solve from its neighbour, until the curve leaves the plotted window. */
  function supplyCurve(eq) {
    const S = setup(eq.params), p = S.p, rMax = 1 / p.beta - 1;
    const rBottom = Math.max(-p.delta + 0.012, eq.r - 0.03);
    const Kcap = 3 * Math.max(eq.K, 1);
    const pts = [{ r: eq.r, A: eq.K }];
    (eq.log || []).forEach((l) => { if (l.r >= rBottom && l.r < rMax && Math.abs(l.r - eq.r) > 1e-6) pts.push({ r: l.r, A: l.A }); });
    const have = (r, tol) => pts.some((q) => Math.abs(q.r - r) < tol);
    let warm = { c: eq.c, mu: eq.mu };
    for (let k = 1; k <= 12; k++) {                     // up toward 1/β − 1
      const r = rMax - (rMax - eq.r) * Math.pow(0.7, k);
      if (have(r, (rMax - r) * 0.12)) { const q = pts.find((q) => Math.abs(q.r - r) < (rMax - r) * 0.12); if (q.A > Kcap) break; continue; }
      const h = household(S, r, warm); warm = h;
      pts.push({ r, A: h.A });
      if (h.A > Kcap) break;
    }
    warm = { c: eq.c, mu: eq.mu };
    const stepDn = Math.max(0.004, (eq.r - rBottom) / 5);
    for (let k = 1; k <= 8; k++) {                      // down to the bottom of the window
      const r = eq.r - stepDn * k;
      if (r < rBottom - stepDn) break;
      if (have(r, stepDn * 0.3)) continue;
      const h = household(S, r, warm); warm = h;
      pts.push({ r, A: h.A });
    }
    pts.sort((x, y) => x.r - y.r);
    return { pts, rMax, rBottom, Kcap, L: S.L, alpha: p.alpha, delta: p.delta };
  }

  /* MPC(a,z) = [c(a + T/(1+r), z) − c(a,z)] / T: a windfall of T goods today is worth
     T/(1+r) units of beginning-of-period assets. Linear interpolation of the continuous EGM
     policy, extrapolated with the terminal slope above the grid (as figuras_curso._a6_pmc). */
  function mpcStats(eq, T) {
    const a = eq.a, na = a.length, nz = eq.z.length, N = na * nz, aMin = a[0];
    const { c, mu, ap, r } = eq, shift = T / (1 + r);
    const mpc = new Float64Array(N);
    for (let z = 0; z < nz; z++) {
      let j = 0;
      for (let i = 0; i < na; i++) {
        const x = a[i] + shift;
        while (j < na - 2 && a[j + 1] <= x) j++;
        const cl = c[j * nz + z], ch = c[(j + 1) * nz + z];
        const cx = cl + ((ch - cl) * (x - a[j])) / (a[j + 1] - a[j]);
        mpc[i * nz + z] = (cx - c[i * nz + z]) / T;
      }
    }
    let mean = 0, mc = 0, wc = 0, mu0 = 0, wu = 0;
    for (let k = 0; k < N; k++) {
      mean += mu[k] * mpc[k];
      if (ap[k] <= aMin + 1e-9) { mc += mu[k] * mpc[k]; wc += mu[k]; } else { mu0 += mu[k] * mpc[k]; wu += mu[k]; }
    }
    // wealth quintiles: each node's mass covers [P_{i-1}, P_i] of the population
    const quint = [0, 0, 0, 0, 0];
    let P0 = 0;
    for (let i = 0; i < na; i++) {
      let m = 0, s = 0;
      for (let z = 0; z < nz; z++) { m += mu[i * nz + z]; s += mu[i * nz + z] * mpc[i * nz + z]; }
      if (m <= 0) continue;
      const P1 = P0 + m, mbar = s / m;
      for (let q = 0; q < 5; q++) {
        const ov = Math.min(P1, (q + 1) / 5) - Math.max(P0, q / 5);
        if (ov > 0) quint[q] += ov * mbar * 5;
      }
      P0 = P1;
    }
    const { beta, gamma } = eq.params;
    return { mpc, mean, constrained: wc > 0 ? mc / wc : null, unconstrained: wu > 0 ? mu0 / wu : null, quint,
      kappa: 1 - Math.pow(beta * (1 + r), 1 / gamma) / (1 + r) };
  }

  return { erf, tauchen, rouwenhorst, markovStationary, setup, prices, egm, lottery, household,
    equilibrium, summarize, supplyCurve, mpcStats, DEFAULTS };
}

if (typeof module !== "undefined" && module.exports) module.exports = aiyagariKernel;

/* ======================================================================== page */
if (typeof window !== "undefined" && window.document) (async function () {
  "use strict";
  const T = MAV.t;
  const KER = aiyagariKernel();
  const ref = await MAV.json("../data/desigualdad.json").catch(() => null);

  const BASE = { beta: 0.96, gamma: 2, rho: 0.9, sigma: 0.2, abar: 0, disc: "tauchen" };
  const state = Object.assign({}, BASE, { T: 0.1 });
  const pct = (x, d = 2) => MAV.num(100 * x, d) + "%";
  const sigmaMax = (rho) => Math.min(0.35, Math.round(100 * Math.sqrt(1 - rho * rho)) / 100);

  // ------------------------------------------------------------ controls
  const ctl = document.getElementById("controls");
  const group = (es, en) => { const p = document.createElement("p"); p.className = "ctl-group";
    p.innerHTML = `<span class="es">${es}</span><span class="en">${en}</span>`; ctl.appendChild(p); };
  const extraNote = (sl) => { const n = document.createElement("div"); n.className = "ctl-note"; sl.input.parentNode.appendChild(n); return n; };

  group("Preferencias", "Preferences");
  const optBeta = { key: "beta", label: { es: "Factor de descuento β", en: "Discount factor β" }, min: 0.92, max: 0.98, step: 0.005,
    value: state.beta, fmt: (v) => MAV.num(v, 3) };
  const slBeta = MAV.slider(ctl, optBeta, (v) => { state.beta = v; paintNotes(); schedule(); });
  const noteBeta = extraNote(slBeta);
  const slGamma = MAV.slider(ctl, { key: "gamma", label: { es: "Aversión al riesgo γ", en: "Risk aversion γ" }, min: 1, max: 5, step: 0.5,
    value: state.gamma, fmt: (v) => MAV.num(v, 1),
    note: { es: "γ = 1 es utilidad logarítmica; también gobierna la prudencia", en: "γ = 1 is log utility; it also governs prudence" } },
  (v) => { state.gamma = v; schedule(); });

  group("Riesgo de ingreso", "Income risk");
  const slRho = MAV.slider(ctl, { key: "rho", label: { es: "Persistencia ρ", en: "Persistence ρ" }, min: 0.8, max: 0.98, step: 0.01,
    value: state.rho, fmt: (v) => MAV.num(v, 2) }, (v) => { state.rho = v; clampSigma(); paintNotes(); schedule(); });
  const slSig = MAV.slider(ctl, { key: "sigma", label: { es: "Desv. est. del choque σ_ε", en: "Shock std. dev. σ_ε" }, min: 0.05, max: 0.35, step: 0.01,
    value: state.sigma, fmt: (v) => MAV.num(v, 2) }, (v) => { state.sigma = Math.min(v, sigmaMax(state.rho)); paintNotes(); schedule(); });
  const noteSig = extraNote(slSig);
  const discCtl = MAV.segmented(ctl, { key: "disc", label: { es: "Discretización (7 estados)", en: "Discretization (7 states)" }, value: state.disc,
    options: [{ value: "tauchen", label: { es: "Tauchen (mazo)", en: "Tauchen (deck)" } }, { value: "rouwenhorst", label: "Rouwenhorst" }] },
  (v) => { state.disc = v; schedule(); });

  group("Crédito", "Credit");
  const slAbar = MAV.slider(ctl, { key: "abar", label: { es: "Límite de endeudamiento ā", en: "Borrowing limit ā" }, min: 0, max: 4, step: 0.25,
    value: state.abar, fmt: (v) => MAV.num(v, 2), note: { es: "a′ ≥ −ā, en bienes", en: "a′ ≥ −ā, in goods" } },
  (v) => { state.abar = v; schedule(); });
  const noteAbar = extraNote(slAbar);

  group("Propensión a consumir", "Propensity to consume");
  const tctl = MAV.segmented(ctl, { key: "transfer", label: { es: "Transferencia T para medir la PMC", en: "Transfer T used for the MPC" }, value: state.T,
    options: [0.01, 0.1, 1].map((v) => ({ value: v, label: MAV.num(v, 2) })) }, (v) => { state.T = v; if (lastEq) drawMPC(lastEq); });
  const noteT = document.createElement("div"); noteT.className = "ctl-note"; ctl.appendChild(noteT);

  const btns = document.createElement("div"); btns.className = "btn-row"; btns.style.marginTop = "18px";
  const reset = document.createElement("button"); reset.type = "button"; reset.className = "btn";
  reset.innerHTML = '<span class="es">Calibración del curso</span><span class="en">Course calibration</span>';
  reset.addEventListener("click", () => {
    Object.assign(state, BASE, { T: 0.1 });
    slBeta.set(state.beta); slGamma.set(state.gamma); slRho.set(state.rho); clampSigma(); slSig.set(state.sigma); slAbar.set(state.abar);
    discCtl.set(state.disc); tctl.set(state.T); paintNotes(); solve();
  });
  btns.appendChild(reset); ctl.appendChild(btns);

  function clampSigma() {
    const smax = sigmaMax(state.rho);
    slSig.input.max = smax;
    if (state.sigma > smax) state.sigma = smax;
    slSig.set(state.sigma);
  }
  function paintNotes() {
    noteBeta.textContent = T({ es: `Con mercados completos r = 1/β − 1 = ${pct(1 / state.beta - 1)}`,
      en: `With complete markets r = 1/β − 1 = ${pct(1 / state.beta - 1)}` });
    const sz = state.sigma / Math.sqrt(1 - state.rho * state.rho);
    noteSig.textContent = T({ es: `Desv. est. incondicional del log-ingreso: ${MAV.num(sz, 2)}. Máximo σ_ε con este ρ: ${MAV.num(sigmaMax(state.rho), 2)}`,
      en: `Unconditional std. dev. of log income: ${MAV.num(sz, 2)}. Largest σ_ε at this ρ: ${MAV.num(sigmaMax(state.rho), 2)}` });
    noteT.textContent = T({ es: `En bienes; el ingreso laboral medio es ≈ ${lastEq ? MAV.num(lastEq.w * lastEq.L, 2) : "1.52"} al año`,
      en: `In goods; average labor income is ≈ ${lastEq ? MAV.num(lastEq.w * lastEq.L, 2) : "1.52"} a year` });
    if (lastEq) {
      const p = lastEq.params, clamped = p.abar < p.abarAsked - 1e-9;
      noteAbar.textContent = clamped
        ? T({ es: `Se usa ā = ${MAV.num(p.abar, 2)}: el límite natural con r = 1/β − 1 es ${MAV.num(p.abarNatural, 2)}`,
              en: `Using ā = ${MAV.num(p.abar, 2)}: the natural limit at r = 1/β − 1 is ${MAV.num(p.abarNatural, 2)}` })
        : T({ es: `Límite natural con r = 1/β − 1: ${MAV.num(p.abarNatural, 2)}`, en: `Natural limit at r = 1/β − 1: ${MAV.num(p.abarNatural, 2)}` });
    } else noteAbar.textContent = "";
  }

  // ------------------------------------------------------------ charts
  const $ = (id) => document.getElementById(id);
  const tiles = $("tiles"), status = $("status");
  const chMarket = MAV.lineChart($("chart-market"), { series: [], title: "" });
  const chDist = barChart($("chart-dist"));
  const chLorenz = MAV.lineChart($("chart-lorenz"), { series: [], title: "" });
  const chPolicy = MAV.lineChart($("chart-policy"), { series: [], title: "" });
  const chMPC = MAV.lineChart($("chart-mpc"), { series: [], title: "" });
  const chQuint = barChart($("chart-quint"));
  const decomp = $("decomp");
  const busyEls = [tiles, $("chart-market"), $("chart-dist"), $("chart-lorenz"), $("chart-policy"), $("chart-mpc"), $("chart-quint"), decomp];

  const Z = [{ j: 0, es: "Ingreso bajo", en: "Low income" }, { j: 3, es: "Ingreso medio", en: "Middle income" }, { j: 6, es: "Ingreso alto", en: "High income" }];
  let lastEq = null, lastCurve = null, lastMPC = null;

  function quantileA(eq, q) {
    let c = 0;
    for (let i = 0; i < eq.a.length; i++) { c += eq.muA[i]; if (c >= q) return eq.a[i]; }
    return eq.a[eq.a.length - 1];
  }
  const niceUp = (v) => { const s = [1, 2, 2.5, 5, 10]; const e = 10 ** Math.floor(Math.log10(v)); return s.find((k) => k * e >= v) * e; };

  function drawAll(eq) {
    const m = KER.mpcStats(eq, state.T); m.T = state.T; lastMPC = m;
    const p = eq.params, KYcm = p.alpha / (eq.rCM + p.delta);
    MAV.tiles(tiles, [
      { label: { es: "Tasa de interés r*", en: "Interest rate r*" }, value: pct(eq.r),
        note: { es: `1/β − 1 = ${pct(eq.rCM)} sin fricciones`, en: `1/β − 1 = ${pct(eq.rCM)} frictionless` } },
      { label: { es: "Capital / producto", en: "Capital / output" }, value: MAV.num(eq.KY, 2),
        note: { es: `${MAV.num(KYcm, 2)} sin fricciones`, en: `${MAV.num(KYcm, 2)} frictionless` } },
      { label: { es: "Gini de la riqueza", en: "Wealth Gini" }, value: MAV.num(eq.gini, 2),
        note: ref ? { es: `EUA: ${MAV.num(ref.scf.gini, 2)} (SCF ${ref.scf.anio})`, en: `US: ${MAV.num(ref.scf.gini, 2)} (SCF ${ref.scf.anio})` } : null },
      { label: { es: "En la restricción", en: "At the constraint" }, value: pct(eq.constrained, 1),
        note: p.abar > 0 ? { es: `eligen a′ = −${MAV.num(p.abar, 2)}`, en: `choose a′ = −${MAV.num(p.abar, 2)}` } : { es: "eligen a′ = 0", en: "choose a′ = 0" } },
      { label: { es: `PMC media, T = ${MAV.num(state.T, 2)}`, en: `Average MPC, T = ${MAV.num(state.T, 2)}` }, value: MAV.num(m.mean, 3),
        note: { es: `1 − β = ${MAV.num(1 - p.beta, 3)} sin fricciones`, en: `1 − β = ${MAV.num(1 - p.beta, 3)} frictionless` } },
    ]);
    drawDist(eq); drawLorenz(eq); drawPolicy(eq); drawMPC(eq, m);
    drawMarket(eq, null);
    paintNotes();
  }

  // --- capital market: A(r) against K^d(r), r on the vertical axis (Aiyagari's figure)
  function monotone(xs, ys) {
    const n = xs.length, d = [], mm = new Array(n);
    for (let i = 0; i < n - 1; i++) d[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    mm[0] = d[0]; mm[n - 1] = d[n - 2];
    for (let i = 1; i < n - 1; i++) mm[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
      if (d[i] === 0) { mm[i] = 0; mm[i + 1] = 0; continue; }
      const a = mm[i] / d[i], b = mm[i + 1] / d[i], s = a * a + b * b;
      if (s > 9) { const t = 3 / Math.sqrt(s); mm[i] = t * a * d[i]; mm[i + 1] = t * b * d[i]; }
    }
    return (x) => {
      if (x < xs[0] || x > xs[n - 1]) return null;
      let i = 0; while (i < n - 2 && xs[i + 1] < x) i++;
      const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
      return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * mm[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * mm[i + 1];
    };
  }

  function drawMarket(eq, curve) {
    const p = eq.params, L = eq.L, rMax = eq.rCM;
    const rBottom = curve ? curve.rBottom : Math.max(-p.delta + 0.012, eq.r - 0.03);
    const Kcap = curve ? curve.Kcap : 3 * Math.max(eq.K, 1);
    const yTop = rMax + 0.006, yBot = rBottom - 0.002;
    let sup = [];
    if (curve) {
      const pts = curve.pts.filter((q) => isFinite(q.A)).sort((x, y) => x.r - y.r);
      let k = 0; pts.forEach((q, i) => { if (Math.abs(q.r - eq.r) < Math.abs(pts[k].r - eq.r)) k = i; });
      const branch = [pts[k]];
      for (let i = k + 1; i < pts.length; i++) if (pts[i].A > branch[branch.length - 1].A + 1e-4 * Kcap) branch.push(pts[i]); else if (pts[i].A < branch[branch.length - 1].A) break;
      for (let i = k - 1; i >= 0; i--) if (pts[i].A < branch[0].A - 1e-4 * Kcap) branch.unshift(pts[i]); else if (pts[i].A > branch[0].A) break;
      sup = branch;
    }
    const xMin = Math.min(0, sup.length ? sup[0].A : 0);
    const n = 181, xs = [];
    for (let i = 0; i < n; i++) xs.push(xMin + ((Kcap - xMin) * i) / (n - 1));
    xs.push(eq.K); xs.sort((a, b) => a - b);
    const rd = (K) => { if (K <= 0) return null; const r = p.alpha * Math.pow(K / L, p.alpha - 1) - p.delta; return r > yTop || r < yBot ? null : 100 * r; };
    const rsF = sup.length > 1 ? monotone(sup.map((q) => q.A), sup.map((q) => q.r)) : null;
    const rs = (K) => { if (!rsF) return null; const r = rsF(K); return r == null || r > yTop || r < yBot ? null : 100 * r; };
    const series = [];
    if (rsF) series.push({ name: { es: "Oferta de los hogares A(r)", en: "Household supply A(r)" }, x: xs, y: xs.map(rs), style: 0, label: false });
    series.push({ name: { es: "Demanda de las empresas Kᵈ(r)", en: "Firm demand Kᵈ(r)" }, x: xs, y: xs.map(rd), style: 1, label: false });
    const Kcm = L * Math.pow(p.alpha / (rMax + p.delta), 1 / (1 - p.alpha));
    const cw = $("chart-market").clientWidth || 800, narrow = cw < 600;
    // the two reference lines share one label when they would sit on top of each other
    const plotH = Math.round(Math.min(360, Math.max(240, cw * 0.5))) - 54;
    const close = (Math.abs(rMax - eq.r) / (yTop - yBot)) * plotH < 18;
    chMarket.update({
      title: { es: "El mercado de capital", en: "The capital market" },
      sub: curve ? { es: "Tasa de interés anual contra capital. La oferta se vuelve vertical al acercarse a 1/β − 1: ahí está el ahorro precautorio",
                     en: "Annual interest rate against capital. Supply turns vertical as r approaches 1/β − 1: that is precautionary saving" }
                 : { es: "Calculando la oferta de capital…", en: "Computing capital supply…" },
      xType: "linear", xLabel: { es: "Capital K", en: "Capital K" }, yLabel: { es: "r, %", en: "r, %" },
      xDomain: [xMin, Kcap], yDomain: [100 * yBot, 100 * yTop], labelRoom: 24, zero: true,
      xFmt: (v) => MAV.num(v, Number.isInteger(Math.round(v * 100) / 100) ? 0 : 1),
      yFmt: (v) => MAV.num(v, 0), tipFmt: (v) => MAV.num(v, 2) + "%", digits: 3,
      series,
      hlines: close
        ? [{ y: 100 * rMax, label: narrow ? `1/β − 1 = ${pct(rMax)} · r* = ${pct(eq.r)}`
            : { es: `1/β − 1 = ${pct(rMax)} (completos) · r* = ${pct(eq.r)} (incompletos)`, en: `1/β − 1 = ${pct(rMax)} (complete) · r* = ${pct(eq.r)} (incomplete)` } },
           { y: 100 * eq.r }]
        : narrow
        ? [{ y: 100 * rMax, label: `1/β − 1 = ${pct(rMax)}` }, { y: 100 * eq.r, label: `r* = ${pct(eq.r)}` }]
        : [{ y: 100 * rMax, label: { es: `1/β − 1 = ${pct(rMax)}: mercados completos`, en: `1/β − 1 = ${pct(rMax)}: complete markets` } },
           { y: 100 * eq.r, label: { es: `r* = ${pct(eq.r)}: mercados incompletos`, en: `r* = ${pct(eq.r)}: incomplete markets` } }],
      points: [{ x: eq.K, y: 100 * eq.r }, ...(Kcm <= Kcap ? [{ x: Kcm, y: 100 * rMax, style: 2 }] : [])],
      csvName: "aiyagari-mercado-capital",
    });
  }

  // --- wealth distribution: histogram, with the point mass at the constraint drawn apart
  function drawDist(eq) {
    const a = eq.a, na = a.length, lo = a[0];
    const aHi = Math.max(lo + 5, niceUp(quantileA(eq, 0.99)));
    const w = niceUp((aHi - lo) / 60);
    const nb = Math.ceil((aHi - lo) / w - 1e-9);
    const bins = Array.from({ length: nb }, (_, k) => ({ x0: lo + k * w, x1: lo + (k + 1) * w, spike: 0, rest: 0 }));
    let above = 0;
    for (let i = 1; i < na; i++) {                      // spread each node's mass over its cell
      const c0 = (a[i - 1] + a[i]) / 2, c1 = i < na - 1 ? (a[i] + a[i + 1]) / 2 : a[i];
      const m = eq.muA[i];
      if (c1 <= c0) continue;
      for (let k = Math.max(0, Math.floor((c0 - lo) / w)); k < nb; k++) {
        const ov = Math.min(c1, bins[k].x1) - Math.max(c0, bins[k].x0);
        if (bins[k].x0 >= c1) break;
        if (ov > 0) bins[k].rest += (m * ov) / (c1 - c0);
      }
      if (c1 > lo + nb * w) above += (m * (c1 - Math.max(c0, lo + nb * w))) / (c1 - c0);
    }
    bins[0].spike = eq.muA[0];
    let debt = 0; for (let i = 0; i < na; i++) if (a[i] < -1e-9) debt += eq.muA[i];
    const mean = eq.K, med = quantileA(eq, 0.5);
    chDist.update({
      kind: "hist", title: { es: "Distribución de la riqueza", en: "Wealth distribution" },
      sub: { es: `% de hogares por tramo de ${MAV.num(w, w < 1 ? 2 : 0)} de riqueza. ${lo < 0 ? `${pct(debt, 1)} tiene deuda; ` : ""}${pct(above, 1)} tiene más de ${MAV.num(aHi, 0)}`,
             en: `% of households per wealth bin of width ${MAV.num(w, w < 1 ? 2 : 0)}. ${lo < 0 ? `${pct(debt, 1)} are in debt; ` : ""}${pct(above, 1)} hold more than ${MAV.num(aHi, 0)}` },
      xDomain: [lo, lo + nb * w], xLabel: { es: "Riqueza a", en: "Wealth a" }, yFmt: (v) => MAV.num(v, 0),
      yMax: 1.3 * Math.max(...bins.map((b) => 100 * (b.spike + b.rest))),
      data: bins.map((b) => ({ x0: b.x0, x1: b.x1, segs: [{ v: 100 * b.spike, fill: "var(--s1)" }, { v: 100 * b.rest, fill: "var(--s3)" }] })),
      legend: [{ name: { es: `En la restricción, a = −ā: ${pct(eq.muA[0], 1)}`, en: `At the constraint, a = −ā: ${pct(eq.muA[0], 1)}` }, fill: "var(--s1)" },
               { name: { es: "Resto de los hogares", en: "Other households" }, fill: "var(--s3)" }],
      segNames: [{ es: "en la restricción", en: "at the constraint" }, { es: "resto", en: "other" }],
      vlines: [{ x: med, label: { es: `mediana ${MAV.num(med, 1)}`, en: `median ${MAV.num(med, 1)}` } }, { x: mean, label: { es: `media ${MAV.num(mean, 1)}`, en: `mean ${MAV.num(mean, 1)}` } }],
      csvName: "aiyagari-distribucion",
      table: {
        header: [T({ es: "Tramo de riqueza", en: "Wealth bin" }), T({ es: "% en la restricción", en: "% at constraint" }), T({ es: "% resto", en: "% other" })],
        rows: bins.map((b) => [`${MAV.num(b.x0, 2)} – ${MAV.num(b.x1, 2)}`, MAV.num(100 * b.spike, 3), MAV.num(100 * b.rest, 3)]),
        raw: bins.map((b) => [`${b.x0}–${b.x1}`, 100 * b.spike, 100 * b.rest]),
      },
    });
  }

  // --- Lorenz: model, DFA polyline and the 45° line on a common grid of population shares
  function drawLorenz(eq) {
    const xs = Array.from({ length: 201 }, (_, i) => i / 2);
    const interp = (px, py, x) => { let j = 0; while (j < px.length - 2 && px[j + 1] < x) j++;
      const dx = px[j + 1] - px[j]; return dx > 0 ? py[j] + ((py[j + 1] - py[j]) * (x - px[j])) / dx : py[j + 1]; };
    const model = xs.map((x) => 100 * interp(eq.pop, eq.val, x / 100));
    const series = [{ name: { es: `Modelo, Gini ${MAV.num(eq.gini, 2)}`, en: `Model, Gini ${MAV.num(eq.gini, 2)}` }, x: xs, y: model, style: 0, label: false }];
    if (ref) {
      const q = MAV.quarter(MAV.parseDates([ref.dfa.fecha])[0]);
      series.push({ name: { es: `EUA ${q} (DFA), Gini ≥ ${MAV.num(ref.dfa.gini_min, 2)}`, en: `US ${q} (DFA), Gini ≥ ${MAV.num(ref.dfa.gini_min, 2)}` },
        x: xs, y: xs.map((x) => 100 * interp(ref.dfa.lorenz_pop, ref.dfa.lorenz_val, x / 100)), style: 1, label: false });
    }
    series.push({ name: { es: "Igualdad perfecta", en: "Perfect equality" }, x: xs, y: xs.slice(), style: 2, label: false });
    const ymin = Math.min(0, ...model);
    const top1 = 100 - interp(eq.pop, eq.val, 0.99) * 100, b50 = interp(eq.pop, eq.val, 0.5) * 100;
    chLorenz.update({
      title: { es: "Curva de Lorenz de la riqueza", en: "Wealth Lorenz curve" },
      sub: ref ? { es: `Mitad inferior: ${MAV.num(b50, 1)}% en el modelo, ${MAV.num(ref.dfa.b50, 1)}% en EUA. 1% más rico: ${MAV.num(top1, 1)}% contra ${MAV.num(ref.dfa.top1, 1)}%`,
                   en: `Bottom half: ${MAV.num(b50, 1)}% in the model, ${MAV.num(ref.dfa.b50, 1)}% in the US. Top 1%: ${MAV.num(top1, 1)}% versus ${MAV.num(ref.dfa.top1, 1)}%` } : "",
      xType: "linear", xLabel: { es: "Hogares, % acumulado", en: "Households, cumulative %" }, yLabel: { es: "Riqueza, % acumulado", en: "Wealth, cumulative %" },
      xDomain: [0, 100], yDomain: [ymin, 100], labelRoom: 20, zero: ymin < 0, series, xFmt: (v) => MAV.num(v, 0), yFmt: (v) => MAV.num(v, 0),
      tipFmt: (v) => MAV.num(v, 1) + "%", digits: 2, csvName: "aiyagari-lorenz",
    });
  }

  function policyRange(eq) {
    const hi = Math.min(eq.a[eq.a.length - 1], Math.max(eq.a[0] + 5, niceUp(quantileA(eq, 0.9) * 1.2)));
    const idx = []; eq.a.forEach((v, i) => { if (v <= hi + 1e-9) idx.push(i); });
    return { idx, hi };
  }

  function drawPolicy(eq) {
    const { idx, hi } = policyRange(eq), nz = eq.z.length;
    const xs = idx.map((i) => eq.a[i]);
    chPolicy.update({
      title: { es: "Consumo según riqueza e ingreso", en: "Consumption by wealth and income" },
      sub: { es: "c(a, z) en bienes al año. Junto a la restricción el hogar consume todo lo que tiene; después la curva se dobla: es cóncava",
             en: "c(a, z), goods per year. Near the constraint the household consumes everything it has; then the curve bends: it is concave" },
      xType: "linear", xLabel: { es: "Riqueza a", en: "Wealth a" }, yLabel: { es: "c", en: "c" }, xDomain: [xs[0], hi], labelRoom: 24,
      series: Z.map((zz, s) => ({ name: { es: `${zz.es}, eᶻ = ${MAV.num(eq.ez[zz.j], 2)}`, en: `${zz.en}, eᶻ = ${MAV.num(eq.ez[zz.j], 2)}` },
        x: xs, y: idx.map((i) => eq.c[i * nz + zz.j]), style: s, label: false })),
      xFmt: (v) => MAV.num(v, 0), yFmt: (v) => MAV.num(v, 1), tipFmt: (v) => MAV.num(v, 3), digits: 3, csvName: "aiyagari-consumo",
    });
  }

  function drawMPC(eq, m) {
    if (!m || m.T !== state.T) { m = KER.mpcStats(eq, state.T); m.T = state.T; }
    lastMPC = m;
    const { idx, hi } = policyRange(eq), nz = eq.z.length;
    const xs = idx.map((i) => eq.a[i]);
    chMPC.update({
      title: { es: "Propensión marginal a consumir", en: "Marginal propensity to consume" },
      sub: { es: `Fracción de una transferencia de ${MAV.num(state.T, 2)} que se consume hoy. La línea punteada es la media ponderada por la distribución`,
             en: `Share of a transfer of ${MAV.num(state.T, 2)} consumed today. The dotted line is the average weighted by the distribution` },
      xType: "linear", xLabel: { es: "Riqueza a", en: "Wealth a" }, yLabel: { es: "PMC", en: "MPC" }, xDomain: [xs[0], hi], labelRoom: 24,
      yDomain: [0, Math.min(1, Math.max(0.2, ...Z.flatMap((zz) => idx.map((i) => m.mpc[i * nz + zz.j]))))],
      series: Z.map((zz, s) => ({ name: { es: zz.es, en: zz.en }, x: xs, y: idx.map((i) => m.mpc[i * nz + zz.j]), style: s, label: false })),
      hlines: [{ y: m.mean, label: { es: `media ${MAV.num(m.mean, 3)}`, en: `average ${MAV.num(m.mean, 3)}` } }],
      xFmt: (v) => MAV.num(v, 0), yFmt: (v) => MAV.num(v, 2), tipFmt: (v) => MAV.num(v, 3), digits: 3, csvName: "aiyagari-pmc",
    });
    const names = [{ es: "20% más pobre", en: "Poorest 20%" }, { es: "Segundo", en: "Second" }, { es: "Tercero", en: "Third" },
      { es: "Cuarto", en: "Fourth" }, { es: "20% más rico", en: "Richest 20%" }];
    const short = [{ es: "Q1 pobre", en: "Q1 poor" }, "Q2", "Q3", "Q4", { es: "Q5 rico", en: "Q5 rich" }];
    chQuint.update({
      kind: "cat", title: { es: "PMC por quintil de riqueza", en: "MPC by wealth quintile" },
      sub: { es: `Transferencia de ${MAV.num(state.T, 2)}; promedio dentro de cada quintil`, en: `Transfer of ${MAV.num(state.T, 2)}; average within each quintile` },
      yFmt: (v) => MAV.num(v, 2), valueFmt: (v) => MAV.num(v, 3), yMax: Math.max(0.1, ...m.quint) * 1.15,
      data: m.quint.map((v, q) => ({ key: "q" + q, name: names[q], short: short[q], segs: [{ v, fill: "var(--s2)" }] })),
      segNames: [{ es: "PMC", en: "MPC" }],
      hlines: [{ y: m.mean, label: { es: `media ${MAV.num(m.mean, 3)}`, en: `average ${MAV.num(m.mean, 3)}` } }],
      csvName: "aiyagari-pmc-quintiles",
      table: { header: [T({ es: "Quintil de riqueza", en: "Wealth quintile" }), T({ es: "PMC", en: "MPC" })],
        rows: m.quint.map((v, q) => [T(names[q]), MAV.num(v, 4)]), raw: m.quint.map((v, q) => [T(names[q]), v]) },
    });
    const c = eq.constrained;
    const ratio = m.quint[4] > 0 ? m.quint[0] / m.quint[4] : null;
    decomp.innerHTML = "";
    const h = document.createElement("h3"); h.textContent = T({ es: "La PMC agregada depende de la distribución", en: "The aggregate MPC depends on the distribution" });
    const big = document.createElement("div"); big.className = "big";
    big.textContent = m.constrained == null ? MAV.num(m.mean, 3)
      : `${MAV.num(m.mean, 3)} = ${pct(c, 1)} × ${MAV.num(m.constrained, 2)} + ${pct(1 - c, 1)} × ${MAV.num(m.unconstrained, 3)}`;
    const under = document.createElement("div"); under.className = "under";
    under.textContent = T({ es: "media = masa restringida × su PMC + resto × su PMC", en: "average = constrained mass × its MPC + the rest × its MPC" });
    const p1 = document.createElement("p");
    p1.textContent = T({ es: `Con mercados completos y β(1 + r) = 1, todos tendrían la misma PMC, 1 − β = ${MAV.num(1 - eq.params.beta, 3)}. Aquí la media la fija cuántos hogares viven cerca de la restricción, y eso lo decide la distribución: por eso el multiplicador deja de ser un parámetro libre.`,
      en: `With complete markets and β(1 + r) = 1, everyone would share the same MPC, 1 − β = ${MAV.num(1 - eq.params.beta, 3)}. Here the average is set by how many households live near the constraint, and the distribution decides that: this is why the multiplier stops being a free parameter.` });
    const p2 = document.createElement("p");
    p2.textContent = ratio ? T({ es: `El 20% más pobre consume ${MAV.num(m.quint[0], 3)} de cada unidad extra; el 20% más rico, ${MAV.num(m.quint[4], 3)}: ${MAV.num(ratio, 1)} veces menos.`,
      en: `The poorest 20% consume ${MAV.num(m.quint[0], 3)} of each extra unit; the richest 20%, ${MAV.num(m.quint[4], 3)}: ${MAV.num(ratio, 1)} times less.` }) : "";
    decomp.append(h, big, under, p1, p2);
    // the MPC tile follows the transfer size
    const tileVals = tiles.querySelectorAll(".tile");
    if (tileVals[4]) {
      tileVals[4].querySelector(".value").textContent = MAV.num(m.mean, 3);
      tileVals[4].querySelector(".label").textContent = T({ es: `PMC media, T = ${MAV.num(state.T, 2)}`, en: `Average MPC, T = ${MAV.num(state.T, 2)}` });
    }
  }

  // ------------------------------------------------------------ bar chart (histogram or categories)
  function barChart(el) {
    el.classList.add("chart-card");
    el.innerHTML = `<div class="chart-head"><div><p class="chart-title"></p><p class="chart-sub"></p></div>
      <div class="chart-tools"><button type="button" data-act="table"></button><button type="button" data-act="csv">CSV</button></div></div>
      <div class="legend"></div><div class="chart"><div class="tooltip" role="status" aria-live="polite"></div></div><div class="table-view"></div>`;
    const P = { title: el.querySelector(".chart-title"), sub: el.querySelector(".chart-sub"), legend: el.querySelector(".legend"),
      plot: el.querySelector(".chart"), tip: el.querySelector(".tooltip"), table: el.querySelector(".table-view"),
      tableBtn: el.querySelector('[data-act="table"]'), csvBtn: el.querySelector('[data-act="csv"]') };
    const paintBtn = () => { P.tableBtn.textContent = T(P.table.classList.contains("on") ? { es: "Ocultar tabla", en: "Hide table" } : { es: "Tabla", en: "Table" }); };
    P.tableBtn.addEventListener("click", () => { const on = P.table.classList.toggle("on"); P.tableBtn.setAttribute("aria-expanded", String(on)); paintBtn(); });
    P.tableBtn.setAttribute("aria-expanded", "false"); paintBtn();
    const svg = d3.select(P.plot).insert("svg", ".tooltip");
    let o = null;
    const colPath = (x0, x1, yb, yt, r) => {
      r = Math.max(0, Math.min(r, (x1 - x0) / 2, yb - yt));
      return `M${x0},${yb}V${yt + r}Q${x0},${yt} ${x0 + r},${yt}H${x1 - r}Q${x1},${yt} ${x1},${yt + r}V${yb}Z`;
    };
    function draw() {
      if (!o || !o.data) return;
      paintBtn();
      const width = Math.max(280, P.plot.clientWidth || 600);
      const height = o.height || Math.round(Math.min(320, Math.max(230, width * 0.55)));
      const m = { t: 22, r: 16, b: 42, l: 50 };
      svg.attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img"); svg.selectAll("*").remove();
      svg.append("title").text(T(o.title));
      const cat = o.kind === "cat";
      const totals = o.data.map((d) => d.segs.reduce((s, g) => s + Math.max(0, g.v), 0));
      const y = d3.scaleLinear().domain([0, o.yMax || d3.max(totals) * 1.08 || 1]).nice().range([height - m.b, m.t]);
      let x, bx;
      if (cat) {
        x = d3.scaleBand().domain(o.data.map((d) => d.key)).range([m.l, width - m.r]).padding(0.3);
        const bw = Math.min(24, x.bandwidth());
        bx = (d) => { const c = x(d.key) + x.bandwidth() / 2; return [c - bw / 2, c + bw / 2]; };
      } else {
        x = d3.scaleLinear().domain(o.xDomain).range([m.l, width - m.r]);
        bx = (d) => [x(d.x0) + 1, x(d.x1) - 1];           // a 2px surface gap between neighbours
      }
      svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width - m.l - m.r)).tickFormat(""));
      const xa = svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - m.b})`);
      if (cat) xa.call(d3.axisBottom(x).tickSizeOuter(0).tickFormat((k) => T(o.data.find((d) => d.key === k).short)));
      else xa.call(d3.axisBottom(x).ticks(Math.max(3, Math.round((width - m.l - m.r) / 80))).tickFormat((v) => MAV.num(v, 0)).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat(o.yFmt || ((v) => MAV.num(v, 1))).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", width - m.r).attr("y", height - 6).attr("text-anchor", "end").text(T(o.xLabel));
      const bars = svg.append("g");
      const groups = o.data.map((d, i) => {
        const [x0, x1] = bx(d); const g = bars.append("g").attr("class", "bar");
        let base = y(0); const segs = d.segs.filter((s) => s.v > 0);
        segs.forEach((s, k) => {
          const top = y(0) - (y(0) - y(s.v)) - (y(0) - base);
          const last = k === segs.length - 1;
          const yb = base, yt = Math.min(yb, top + (last ? 0 : 0));
          const gap = k > 0 ? 2 : 0;                      // 2px surface gap between stacked segments
          if (yb - gap - yt < 0.5) { base = yt; return; }
          g.append("path").attr("d", last ? colPath(x0, x1, yb - gap, yt, 4) : `M${x0},${yb - gap}V${yt}H${x1}V${yb - gap}Z`).attr("fill", s.fill);
          base = yt;
        });
        if (cat && o.valueFmt) svg.append("text").attr("class", "dlabel").attr("x", (x0 + x1) / 2).attr("y", y(totals[i]) - 7).attr("text-anchor", "middle").text(o.valueFmt(totals[i]));
        return g;
      });
      (o.hlines || []).forEach((h) => {
        svg.append("line").attr("class", "zero").attr("stroke-dasharray", "3 3").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y(h.y)).attr("y2", y(h.y));
        if (h.label) svg.append("text").attr("class", "alabel").attr("x", width - m.r).attr("y", y(h.y) - 5).attr("text-anchor", "end").text(T(h.label));
      });
      const vl = (o.vlines || []).slice().sort((a, b) => a.x - b.x);
      vl.forEach((v, k) => {
        if (v.x < x.domain()[0] || v.x > x.domain()[1]) return;
        svg.append("line").attr("class", "zero").attr("stroke-dasharray", "3 3").attr("x1", x(v.x)).attr("x2", x(v.x)).attr("y1", m.t).attr("y2", height - m.b);
        // labels right of their lines, stepped down so neighbours never collide
        svg.append("text").attr("class", "alabel").attr("x", x(v.x) + 4).attr("y", m.t + 4 + 15 * k).text(T(v.label));
      });
      if (o.topLabel && !cat) {
        const d = o.data[o.topLabel.i], [x0] = bx(d);
        svg.append("text").attr("class", "dlabel").attr("x", x0).attr("y", y(totals[o.topLabel.i]) - 7).text(o.topLabel.text);
      }
      // hover: the bar under the pointer, with a tooltip
      const show = (i) => {
        if (i == null || i < 0 || i >= o.data.length) return;
        groups.forEach((g, k) => g.style("opacity", k === i ? 1 : 0.5));
        const d = o.data[i], tip = P.tip; tip.textContent = "";
        const head = document.createElement("div"); head.className = "t-head";
        head.textContent = cat ? T(d.name) : `${T(o.xLabel)} ${MAV.num(d.x0, 2)} – ${MAV.num(d.x1, 2)}`;
        tip.appendChild(head);
        d.segs.forEach((s, k) => {
          if (d.segs.length > 1 && !(s.v > 0)) return;
          const r = document.createElement("div"); r.className = "t-row";
          const key = document.createElement("span"); key.innerHTML = `<svg width="10" height="10" aria-hidden="true"><rect width="10" height="10" rx="2" fill="${s.fill}"/></svg>`;
          const b = document.createElement("b"); b.textContent = (o.valueFmt || ((v) => MAV.num(v, 2) + "%"))(s.v);
          const nm = document.createElement("span"); nm.textContent = T((o.segNames || [])[k] || "");
          r.append(key, b, nm); tip.appendChild(r);
        });
        const [x0, x1] = bx(d), sc = P.plot.clientWidth / width, left = ((x0 + x1) / 2) * sc, tw = tip.offsetWidth || 160;
        tip.style.left = (left + 14 + tw > P.plot.clientWidth ? left - tw - 14 : left + 14) + "px";
        tip.style.top = "8px"; tip.classList.add("on");
      };
      let cur = null;
      const hide = () => { groups.forEach((g) => g.style("opacity", 1)); P.tip.classList.remove("on"); cur = null; };
      const pick = (px) => {
        if (cat) { const k = Math.floor((px - m.l) / ((width - m.l - m.r) / o.data.length)); return Math.max(0, Math.min(o.data.length - 1, k)); }
        const xv = x.invert(px); return Math.max(0, Math.min(o.data.length - 1, o.data.findIndex((d) => xv < d.x1)));
      };
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", width - m.l - m.r).attr("height", height - m.t - m.b)
        .attr("fill", "transparent").attr("tabindex", 0).attr("aria-label", T({ es: "Explorar valores", en: "Explore values" }))
        .on("pointermove", (ev) => { cur = pick(d3.pointer(ev)[0]); show(cur); })
        .on("pointerleave", hide).on("blur", hide).on("focus", () => { cur = 0; show(cur); })
        .on("keydown", (ev) => {
          if (ev.key === "ArrowRight") { cur = Math.min(o.data.length - 1, (cur ?? -1) + 1); show(cur); ev.preventDefault(); }
          if (ev.key === "ArrowLeft") { cur = Math.max(0, (cur ?? 1) - 1); show(cur); ev.preventDefault(); }
        });
      // text, legend and table
      P.title.textContent = T(o.title); P.sub.textContent = T(o.sub || ""); P.sub.style.display = o.sub ? "" : "none";
      P.legend.innerHTML = "";
      (o.legend || []).forEach((l) => {
        const k = document.createElement("span"); k.className = "key";
        k.innerHTML = `<svg width="12" height="12" aria-hidden="true"><rect width="12" height="12" rx="2" fill="${l.fill}"/></svg>`;
        const t = document.createElement("span"); t.textContent = T(l.name); k.appendChild(t); P.legend.appendChild(k);
      });
      const tb = document.createElement("table"), hr = tb.createTHead().insertRow();
      o.table.header.forEach((h) => { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); });
      const body = tb.createTBody(); o.table.rows.forEach((r) => { const tr = body.insertRow(); r.forEach((c) => { tr.insertCell().textContent = c; }); });
      P.table.textContent = ""; P.table.appendChild(tb);
      P.csvBtn.onclick = () => {
        const rows = [o.table.header, ...o.table.raw].map((r) => r.map((c) => (typeof c === "string" && /[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n");
        const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows], { type: "text/csv" }));
        a.download = (o.csvName || "mav-datos") + ".csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      };
    }
    let raf = 0;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(P.plot);
    MAV.onLang(draw);
    return { update(next) { o = Object.assign({}, o, next); draw(); } };
  }

  // ------------------------------------------------------------ solver in a Web Worker
  const src = `const K = (${aiyagariKernel.toString()})();
self.onmessage = (e) => {
  const { id, params, guess } = e.data;
  try {
    const eq = K.equilibrium(params, { guess });
    self.postMessage({ id, type: "eq", eq });
    const curve = K.supplyCurve(eq);
    self.postMessage({ id, type: "curve", curve });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String((err && err.message) || "no bracket") });
  }
};`;
  let workerURL = null, worker = null, running = false, reqId = 0, t0 = 0, timer = 0;
  try { workerURL = URL.createObjectURL(new Blob([src], { type: "text/javascript" })); } catch (e) { workerURL = null; }
  function newWorker() {
    if (!workerURL) return null;
    try { const w = new Worker(workerURL); w.onmessage = (e) => onMessage(e.data); w.onerror = () => onMessage({ id: reqId, type: "error", message: "worker" }); return w; }
    catch (e) { return null; }
  }
  worker = newWorker();

  const setBusy = (els, on) => els.forEach((e) => e.classList.toggle("is-busy", on));
  function statusText(kind, eq) {
    status.textContent = "";
    if (kind === "busy") {
      const s = document.createElement("span"); s.className = "spin"; s.setAttribute("aria-hidden", "true"); status.appendChild(s);
      status.appendChild(document.createTextNode(T({ es: "Calculando el equilibrio…", en: "Computing the equilibrium…" })));
    } else if (kind === "error") {
      status.textContent = T({ es: "No convergió con estos parámetros; se muestra el último equilibrio.", en: "No convergence at these parameters; showing the last equilibrium." });
    } else if (eq) {
      const secs = MAV.num(eq.ms / 1000, 1);
      status.textContent = T({ es: `Equilibrio en ${secs} s: ${eq.evals} evaluaciones de r, ${eq.a.length} nodos hasta a = ${MAV.num(eq.params.amax, 0)}.`,
        en: `Equilibrium in ${secs} s: ${eq.evals} evaluations of r, ${eq.a.length} nodes up to a = ${MAV.num(eq.params.amax, 0)}.` })
        + (eq.params.amax > 600 ? T({ es: " Con tanta dispersión del ingreso la malla se estiró mucho: tómalo como aproximación.",
                                      en: " With this much income dispersion the grid had to stretch far: treat it as approximate." }) : "")
        + (kind === "curve-busy" ? T({ es: " Calculando la oferta de capital…", en: " Computing capital supply…" }) : "");
    }
  }

  function params() { return { beta: state.beta, gamma: state.gamma, rho: state.rho, sigma: state.sigma, abar: state.abar, disc: state.disc }; }
  function schedule() { clearTimeout(timer); timer = setTimeout(solve, 220); }
  function solve() {
    const id = ++reqId; t0 = performance.now();
    if (running && worker) { worker.terminate(); worker = newWorker(); }
    running = true;
    setBusy(busyEls, true); statusText("busy");
    const msg = { id, params: params(), guess: lastEq ? lastEq.r : 0.02 };
    if (worker) worker.postMessage(msg);
    else setTimeout(() => {                               // no workers: solve on the main thread
      try { const eq = KER.equilibrium(msg.params, { guess: msg.guess }); onMessage({ id, type: "eq", eq });
        setTimeout(() => { if (id === reqId) onMessage({ id, type: "curve", curve: KER.supplyCurve(eq) }); }, 30); }
      catch (e) { onMessage({ id, type: "error", message: String(e.message || e) }); }
    }, 30);
  }
  function onMessage(d) {
    if (d.id !== reqId) return;
    if (d.type === "eq") {
      lastEq = d.eq; lastCurve = null;
      drawAll(lastEq);
      setBusy(busyEls, false); setBusy([$("chart-market")], true);
      statusText("curve-busy", lastEq);
    } else if (d.type === "curve") {
      lastCurve = d.curve; running = false;
      drawMarket(lastEq, lastCurve);
      setBusy(busyEls, false); statusText("done", lastEq);
      window.__mavDesigualdad = { eq: lastEq, curve: lastCurve, mpc: lastMPC, wallMs: performance.now() - t0 };
    } else {
      running = false; setBusy(busyEls, false); statusText("error");
    }
  }

  let rsz = 0, lastNarrow = null;
  window.addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(() => {
    const nw = $("chart-market").clientWidth < 600;
    if (lastEq && nw !== lastNarrow) drawMarket(lastEq, lastCurve);
    lastNarrow = nw;
  }, 150); });
  MAV.onLang(() => {
    paintNotes();
    if (lastEq) { drawAll(lastEq); if (lastCurve) drawMarket(lastEq, lastCurve); statusText(running ? (lastCurve ? "busy" : "curve-busy") : "done", lastEq); }
  });
  const srcEl = $("source");
  const paintSource = () => {
    const c = ref && ref.curso;
    srcEl.textContent = (ref ? T(ref.fuente) + " " : "")
      + T({ es: "Validación: en la misma malla, r*, K, Gini y PMC difieren de puremacro.vfi (solve_egm + lottery_distribution) en menos de una millonésima",
            en: "Validation: on the same grid, r*, K, Gini and MPC differ from puremacro.vfi (solve_egm + lottery_distribution) by less than one millionth" })
      + (c ? T({ es: `; con ${c.n_a} nodos se reproduce la solución del mazo (r* = ${pct(c.r)}, K/Y = ${MAV.num(c.KY, 3)}, Gini = ${MAV.num(c.gini, 3)}).`,
                 en: `; with ${c.n_a} nodes it reproduces the deck's solution (r* = ${pct(c.r)}, K/Y = ${MAV.num(c.KY, 3)}, Gini = ${MAV.num(c.gini, 3)}).` }) : ".");
  };
  MAV.onLang(paintSource); paintSource();
  clampSigma(); paintNotes();
  solve();
})();
