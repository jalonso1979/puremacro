/* Macroeconomía Avanzada: small numerical kit for the labs.
   Dense linear algebra, OLS, the HP and Hamilton filters, a seeded RNG and a
   first-order linear rational-expectations solver. Plain arrays, no dependencies. */
(function () {
  "use strict";
  const N = (window.MAVNUM = {});

  // ------------------------------------------------------------ linear algebra
  N.zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  N.eye = (n) => { const I = N.zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; };
  N.mul = (A, B) => {
    const r = A.length, c = B[0].length, k = B.length, C = N.zeros(r, c);
    for (let i = 0; i < r; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; if (a) for (let j = 0; j < c; j++) C[i][j] += a * B[p][j]; }
    return C;
  };
  N.mulv = (A, v) => A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
  N.sub = (A, B) => A.map((row, i) => row.map((a, j) => a - B[i][j]));
  N.add = (A, B) => A.map((row, i) => row.map((a, j) => a + B[i][j]));
  N.T = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  N.maxabs = (A) => Math.max(...A.flat().map(Math.abs));

  /* Solve A X = B (B matrix or vector) by Gaussian elimination with partial pivoting. */
  N.solve = (A, B) => {
    const n = A.length, vec = !Array.isArray(B[0]);
    const M = A.map((r, i) => [...r, ...(vec ? [B[i]] : B[i])]);
    const m = M[0].length - n;
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-14) throw new Error("singular matrix");
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) if (r !== c) {
        const f = M[r][c] / M[c][c]; if (f) for (let j = c; j < n + m; j++) M[r][j] -= f * M[c][j];
      }
    }
    const X = M.map((row, i) => row.slice(n).map((v) => v / M[i][i]));
    return vec ? X.map((r) => r[0]) : X;
  };
  N.inv = (A) => N.solve(A, N.eye(A.length));

  /* Spectral radius by power iteration on A^T A-free repeated squaring bound (fine for small matrices). */
  N.spectralRadius = (A, iters = 200) => {
    let v = A.map((_, i) => 1 + i * 1e-3), lam = 0;
    for (let k = 0; k < iters; k++) {
      const w = N.mulv(A, v); const nrm = Math.hypot(...w);
      if (nrm < 1e-300) return 0;
      lam = nrm / Math.hypot(...v); v = w.map((x) => x / nrm);
    }
    return lam;
  };

  // ------------------------------------------------------------ regression
  /* OLS of y on columns of X (add a column of ones yourself). Returns {beta, resid, r2, se} (homoskedastic se). */
  N.ols = (X, y) => {
    const XT = N.T(X), XtX = N.mul(XT, X), Xty = N.mulv(XT, y);
    const beta = N.solve(XtX, Xty);
    const fit = N.mulv(X, beta), resid = y.map((v, i) => v - fit[i]);
    const n = y.length, k = beta.length, ybar = y.reduce((a, b) => a + b, 0) / n;
    const sse = resid.reduce((s, e) => s + e * e, 0), sst = y.reduce((s, v) => s + (v - ybar) ** 2, 0);
    const s2 = sse / Math.max(1, n - k), V = N.inv(XtX);
    return { beta, resid, fit, r2: 1 - sse / sst, se: V.map((row, i) => Math.sqrt(s2 * row[i])) };
  };

  // ------------------------------------------------------------ filters
  /* Hodrick-Prescott trend: (I + λ K'K) τ = y, solved as a symmetric pentadiagonal system. */
  N.hp = (y, lambda = 1600) => {
    const n = y.length; if (n < 5) return y.slice();
    // band[i][j - i + 2] holds A[i][j] for |j - i| <= 2
    const band = Array.from({ length: n }, () => new Float64Array(5));
    for (let i = 0; i < n; i++) {
      const c = (i === 0 || i === n - 1) ? 1 : (i === 1 || i === n - 2) ? 5 : 6;
      band[i][2] = 1 + lambda * c;
      if (i + 1 < n) { const b = lambda * ((i === 0 || i === n - 2) ? -2 : -4); band[i][3] = b; band[i + 1][1] = b; }
      if (i + 2 < n) { band[i][4] = lambda; band[i + 2][0] = lambda; }
    }
    const rhs = Float64Array.from(y);
    for (let k = 0; k < n - 1; k++) {           // symmetric positive definite: no pivoting needed
      for (let i = k + 1; i <= Math.min(k + 2, n - 1); i++) {
        const f = band[i][k - i + 2] / band[k][2];
        if (!f) continue;
        for (let j = k; j <= Math.min(k + 2, n - 1); j++) band[i][j - i + 2] -= f * band[k][j - k + 2];
        rhs[i] -= f * rhs[k];
      }
    }
    const t = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = rhs[i];
      if (i + 1 < n) s -= band[i][3] * t[i + 1];
      if (i + 2 < n) s -= band[i][4] * t[i + 2];
      t[i] = s / band[i][2];
    }
    return Array.from(t);
  };

  /* Hamilton (2018): regress y_{t+h} on a constant and y_t..y_{t-p+1}; the cycle is the residual,
     dated t+h. Returns {cycle, trend} aligned with y (null where undefined). */
  N.hamilton = (y, h = 8, p = 4) => {
    const n = y.length, X = [], Y = [], idx = [];
    for (let t = p - 1; t + h < n; t++) {
      X.push([1, ...Array.from({ length: p }, (_, j) => y[t - j])]); Y.push(y[t + h]); idx.push(t + h);
    }
    const cycle = new Array(n).fill(null), trend = new Array(n).fill(null);
    if (Y.length <= p + 2) return { cycle, trend };
    const r = N.ols(X, Y);
    idx.forEach((t, k) => { cycle[t] = r.resid[k]; trend[t] = r.fit[k]; });
    return { cycle, trend, beta: r.beta };
  };

  N.std = (a) => { const v = a.filter((x) => x != null && isFinite(x)); const m = v.reduce((s, x) => s + x, 0) / v.length;
    return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1)); };
  N.corr = (a, b) => { const P = a.map((x, i) => [x, b[i]]).filter(([x, y]) => x != null && y != null && isFinite(x) && isFinite(y));
    const ma = P.reduce((s, p) => s + p[0], 0) / P.length, mb = P.reduce((s, p) => s + p[1], 0) / P.length;
    let sab = 0, saa = 0, sbb = 0; P.forEach(([x, y]) => { sab += (x - ma) * (y - mb); saa += (x - ma) ** 2; sbb += (y - mb) ** 2; });
    return sab / Math.sqrt(saa * sbb); };

  // ------------------------------------------------------------ random numbers
  N.rng = (seed = 1) => { let s = seed >>> 0;
    const u = () => { s += 0x6d2b79f5; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const g = () => { let a = 0, b = 0; while (a === 0) a = u(); while (b === 0) b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };
    return { u, normal: g }; };

  // ------------------------------------------------------------ linear rational expectations
  /* Solve A0 x_t = A1 E_t x_{t+1} + A2 x_{t-1} + A3 e_t for x_t = P x_{t-1} + Q e_t by iterating
     P <- (A0 - A1 P)^{-1} A2 from P = 0. Converges to the stable solution when the model is determinate;
     callers should check determinacy analytically where they can. Returns {P, Q, ok, iters, resid, rho}. */
  N.solveRE = (A0, A1, A2, A3, { tol = 1e-11, maxIter = 5000 } = {}) => {
    const n = A0.length; let P = N.zeros(n, n), it = 0, diff = Infinity;
    try {
      for (; it < maxIter && diff > tol; it++) {
        const Pn = N.solve(N.sub(A0, N.mul(A1, P)), A2);
        diff = N.maxabs(N.sub(Pn, P)); P = Pn;
      }
      const Q = N.solve(N.sub(A0, N.mul(A1, P)), A3);
      const resid = N.maxabs(N.add(N.sub(N.mul(A1, N.mul(P, P)), N.mul(A0, P)), A2));
      const rho = N.spectralRadius(P);
      return { P, Q, ok: diff <= tol && resid < 1e-8 && rho < 1 + 1e-9, iters: it, resid, rho };
    } catch (e) {
      return { P, Q: null, ok: false, iters: it, resid: Infinity, rho: NaN, error: String(e) };
    }
  };

  /* Impulse responses: x_0 = Q e, x_t = P x_{t-1}. Returns array [T][n]. */
  N.irf = (P, Q, shock, T = 40) => {
    const out = []; let x = N.mulv(Q, shock);
    for (let t = 0; t < T; t++) { out.push(x); x = N.mulv(P, x); }
    return out;
  };
})();
