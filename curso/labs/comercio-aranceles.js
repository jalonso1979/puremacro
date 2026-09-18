/* Laboratorio interactivo: Guerra arancelaria y desvío de comercio (Caliendo & Parro 2015).
   Equilibrio general cuantitativo entre México, EE. UU. y China con enlaces insumo-producto. */
(function () {
  "use strict";

  const N = window.MAVNUM;

  // Calibración canónica del modelo de comercio (3 países, 2 sectores)
  const COUNTRIES = [
    { code: "MEX", name: { es: "México", en: "Mexico" } },
    { code: "USA", name: { es: "Estados Unidos", en: "United States" } },
    { code: "CHN", name: { es: "China", en: "China" } }
  ];

  const BASE_TRADE_SHARES = [
    // Sector 0: Manufacturas (Tradable)
    // [Destino n (Fila)] x [Origen i (Columna)]
    [
      [0.45, 0.42, 0.13], // MEX importa: 45% propio, 42% USA, 13% CHN
      [0.14, 0.72, 0.14], // USA importa: 14% MEX, 72% propio, 14% CHN
      [0.08, 0.10, 0.82]  // CHN importa: 8% MEX, 10% USA, 82% propio
    ],
    // Sector 1: Servicios (No transable)
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ]
  ];

  const BASE_ALPHA = [
    [0.35, 0.65],
    [0.35, 0.65],
    [0.35, 0.65]
  ];

  // Ingreso laboral consistente con equilibrio general de línea base
  const BASE_LABOR_INCOME = [7322.9791099, 17824.61398728, 19152.40690282];
  const WORLD_INCOME = BASE_LABOR_INCOME.reduce((a, b) => a + b, 0);

  /* ------------------------------------------------------------------ Núcleo numérico */
  function solveCaliendoParro(tariffs, thetaMfg = 5.0, gVaMfg = 0.45) {
    const numC = 3, numJ = 2;
    const theta = [thetaMfg, 4.0];
    const gammaVa = [
      [gVaMfg, 0.60],
      [gVaMfg, 0.60],
      [gVaMfg, 0.60]
    ];

    const gammaIo = [];
    for (let n = 0; n < numC; n++) {
      const remM = 1.0 - gammaVa[n][0];
      const remS = 1.0 - gammaVa[n][1];
      gammaIo.push([
        [remM * 0.65, remM * 0.35],
        [remS * 0.35, remS * 0.65]
      ]);
    }

    let wHat = [1.0, 1.0, 1.0];
    let nu = 0.25;
    let XPrime, YPrime, tariffRevPrime, piPrime, pHat, cHat;

    // Iteración de equilibrio general
    for (let it = 0; it < 180; it++) {
      // 1. Contracción no lineal de precios sectoriales
      let lnP = [[0, 0], [0, 0], [0, 0]];
      for (let pit = 0; pit < 35; pit++) {
        let lnPNext = [[0, 0], [0, 0], [0, 0]];
        let lnC = [];
        for (let n = 0; n < numC; n++) {
          const lnWn = Math.log(Math.max(wHat[n], 1e-15));
          const c0 = gammaVa[n][0] * lnWn + gammaIo[n][0][0] * lnP[n][0] + gammaIo[n][0][1] * lnP[n][1];
          const c1 = gammaVa[n][1] * lnWn + gammaIo[n][1][0] * lnP[n][0] + gammaIo[n][1][1] * lnP[n][1];
          lnC.push([c0, c1]);
        }

        // Manufacturas (j = 0)
        for (let n = 0; n < numC; n++) {
          let sumExp = 0;
          for (let i = 0; i < numC; i++) {
            const cost = lnC[i][0] + Math.log(tariffs[0][n][i]);
            sumExp += BASE_TRADE_SHARES[0][n][i] * Math.exp(-theta[0] * cost);
          }
          lnPNext[n][0] = -(1.0 / theta[0]) * Math.log(Math.max(sumExp, 1e-300));
          // Servicios no transables (j = 1)
          lnPNext[n][1] = lnC[n][1];
        }

        let diffP = 0;
        for (let n = 0; n < numC; n++) {
          diffP = Math.max(diffP, Math.abs(lnPNext[n][0] - lnP[n][0]), Math.abs(lnPNext[n][1] - lnP[n][1]));
        }
        lnP = lnPNext;
        if (diffP < 1e-10) break;
      }

      pHat = lnP.map(row => row.map(Math.exp));
      cHat = [];
      for (let n = 0; n < numC; n++) {
        const lnWn = Math.log(Math.max(wHat[n], 1e-15));
        cHat.push([
          Math.exp(gammaVa[n][0] * lnWn + gammaIo[n][0][0] * lnP[n][0] + gammaIo[n][0][1] * lnP[n][1]),
          Math.exp(gammaVa[n][1] * lnWn + gammaIo[n][1][0] * lnP[n][0] + gammaIo[n][1][1] * lnP[n][1])
        ]);
      }

      // 2. Actualización de cuotas bilaterales
      piPrime = [N.zeros(numC, numC), N.eye(numC)];
      for (let n = 0; n < numC; n++) {
        let rsum = 0;
        for (let i = 0; i < numC; i++) {
          const term = (cHat[i][0] * tariffs[0][n][i]) / Math.max(pHat[n][0], 1e-15);
          piPrime[0][n][i] = BASE_TRADE_SHARES[0][n][i] * Math.pow(term, -theta[0]);
          rsum += piPrime[0][n][i];
        }
        for (let i = 0; i < numC; i++) piPrime[0][n][i] /= Math.max(rsum, 1e-15);
      }

      // 3. Sistema Leontief de gasto
      const dim = numC * numJ;
      const M = N.eye(dim);
      const b = new Array(dim).fill(0);
      const mu = N.zeros(numC, numJ);
      for (let n = 0; n < numC; n++) {
        for (let j = 0; j < numJ; j++) {
          for (let i = 0; i < numC; i++) {
            const tau = tariffs[j][n][i];
            mu[n][j] += ((tau - 1.0) / tau) * piPrime[j][n][i];
          }
        }
      }

      for (let n = 0; n < numC; n++) {
        const dispInc = wHat[n] * BASE_LABOR_INCOME[n];
        for (let j = 0; j < numJ; j++) {
          const row = n * numJ + j;
          b[row] = BASE_ALPHA[n][j] * dispInc;
          for (let k = 0; k < numJ; k++) M[row][n * numJ + k] -= BASE_ALPHA[n][j] * mu[n][k];
          for (let k = 0; k < numJ; k++) {
            const gkj = gammaIo[n][k][j];
            if (!gkj) continue;
            for (let m = 0; m < numC; m++) {
              M[row][m * numJ + k] -= gkj * (piPrime[k][m][n] / tariffs[k][m][n]);
            }
          }
        }
      }

      const xVec = N.solve(M, b);
      XPrime = N.zeros(numC, numJ);
      for (let n = 0; n < numC; n++) for (let j = 0; j < numJ; j++) XPrime[n][j] = xVec[n * numJ + j];

      YPrime = N.zeros(numC, numJ);
      for (let n = 0; n < numC; n++) {
        for (let j = 0; j < numJ; j++) {
          for (let m = 0; m < numC; m++) {
            YPrime[n][j] += (piPrime[j][m][n] / tariffs[j][m][n]) * XPrime[m][j];
          }
        }
      }

      tariffRevPrime = new Array(numC).fill(0);
      for (let n = 0; n < numC; n++) tariffRevPrime[n] = mu[n][0] * XPrime[n][0] + mu[n][1] * XPrime[n][1];

      // 4. Exceso de demanda laboral
      let maxDiff = 0;
      const wHatNew = [...wHat];
      for (let n = 0; n < numC; n++) {
        const ld = gammaVa[n][0] * YPrime[n][0] + gammaVa[n][1] * YPrime[n][1];
        const ls = wHat[n] * BASE_LABOR_INCOME[n];
        maxDiff = Math.max(maxDiff, Math.abs(ld - ls) / BASE_LABOR_INCOME[n]);
        const ratio = Math.max(0.05, Math.min(20.0, ld / ls));
        wHatNew[n] = wHat[n] * Math.pow(ratio, nu);
      }

      if (maxDiff < 1e-9) break;

      let norm = 0;
      for (let n = 0; n < numC; n++) norm += wHatNew[n] * BASE_LABOR_INCOME[n];
      for (let n = 0; n < numC; n++) wHat[n] = wHatNew[n] * (WORLD_INCOME / norm);
    }

    // Resultados finales
    const cpiHat = [];
    const realWageHat = [];
    const welfarePct = [];
    const totDecomp = [];
    const ioDecomp = [];
    const revDecomp = [];

    for (let n = 0; n < numC; n++) {
      const cpi = Math.pow(pHat[n][0], BASE_ALPHA[n][0]) * Math.pow(pHat[n][1], BASE_ALPHA[n][1]);
      cpiHat.push(cpi);
      realWageHat.push(wHat[n] / cpi);

      const incPrime = wHat[n] * BASE_LABOR_INCOME[n] + tariffRevPrime[n];
      const welHat = (incPrime / BASE_LABOR_INCOME[n]) / cpi;
      welfarePct.push((welHat - 1.0) * 100.0);

      // Descomposición
      const lnShare = Math.log(BASE_TRADE_SHARES[0][n][n] / Math.max(piPrime[0][n][n], 1e-15));
      const tot = (BASE_ALPHA[n][0] / theta[0]) * lnShare * 100.0;
      const io = (BASE_ALPHA[n][0] * (1.0 - gammaVa[n][0]) / (gammaVa[n][0] * theta[0])) * lnShare * 100.0;
      const rev = welfarePct[n] - (tot + io);

      totDecomp.push(tot);
      ioDecomp.push(io);
      revDecomp.push(rev);
    }

    return {
      wHat,
      pHat,
      cpiHat,
      realWageHat,
      welfarePct,
      piPrime,
      tariffRevPrime,
      totDecomp,
      ioDecomp,
      revDecomp
    };
  }

  /* ------------------------------------------------------------------ UI y Controladores */
  document.addEventListener("DOMContentLoaded", () => {
    const ctl = document.getElementById("controls");
    const tilesEl = document.getElementById("tiles");

    const state = {
      preset: "baseline",
      tauUsaMex: 0,
      tauUsaChn: 25,
      tauMexUsa: 0,
      tauChnUsa: 25,
      theta: 5.0,
      gammaVa: 0.45
    };

    // Presets
    const presets = {
      baseline: { tauUsaMex: 0, tauUsaChn: 0, tauMexUsa: 0, tauChnUsa: 0 },
      us_chn: { tauUsaMex: 0, tauUsaChn: 25, tauMexUsa: 0, tauChnUsa: 25 },
      total_war: { tauUsaMex: 25, tauUsaChn: 60, tauMexUsa: 25, tauChnUsa: 60 },
      nearshoring: { tauUsaMex: 0, tauUsaChn: 60, tauMexUsa: 0, tauChnUsa: 0 }
    };

    const presetSeg = MAV.segmented(ctl, {
      key: "preset",
      label: { es: "Escenario predeterminado", en: "Preset scenario" },
      options: [
        { value: "baseline", label: { es: "Línea base", en: "Baseline" } },
        { value: "us_chn", label: { es: "Guerra EE. UU.–China", en: "US–China War" } },
        { value: "nearshoring", label: { es: "Boom Nearshoring", en: "Nearshoring" } },
        { value: "total_war", label: { es: "Guerra total", en: "Total War" } }
      ],
      value: state.preset
    }, (v) => {
      state.preset = v;
      const p = presets[v];
      if (p) {
        sliderUsaMex.set(p.tauUsaMex);
        sliderUsaChn.set(p.tauUsaChn);
        sliderMexUsa.set(p.tauMexUsa);
        sliderChnUsa.set(p.tauChnUsa);
        state.tauUsaMex = p.tauUsaMex;
        state.tauUsaChn = p.tauUsaChn;
        state.tauMexUsa = p.tauMexUsa;
        state.tauChnUsa = p.tauChnUsa;
        run();
      }
    });

    const sub1 = document.createElement("div"); sub1.className = "ctl-sub";
    sub1.textContent = "Aranceles bilaterales / Bilateral tariffs"; ctl.appendChild(sub1);

    const sliderUsaMex = MAV.slider(ctl, {
      key: "tau_usa_mex",
      label: { es: "Arancel EE. UU. a México", en: "US tariff on Mexico" },
      min: 0, max: 50, step: 1, value: state.tauUsaMex,
      fmt: (v) => v + "%"
    }, (v) => { state.tauUsaMex = v; run(); });

    const sliderUsaChn = MAV.slider(ctl, {
      key: "tau_usa_chn",
      label: { es: "Arancel EE. UU. a China", en: "US tariff on China" },
      min: 0, max: 100, step: 5, value: state.tauUsaChn,
      fmt: (v) => v + "%"
    }, (v) => { state.tauUsaChn = v; run(); });

    const sliderMexUsa = MAV.slider(ctl, {
      key: "tau_mex_usa",
      label: { es: "Represalia México a EE. UU.", en: "Mexico retaliation on US" },
      min: 0, max: 50, step: 1, value: state.tauMexUsa,
      fmt: (v) => v + "%"
    }, (v) => { state.tauMexUsa = v; run(); });

    const sliderChnUsa = MAV.slider(ctl, {
      key: "tau_chn_usa",
      label: { es: "Represalia China a EE. UU.", en: "China retaliation on US" },
      min: 0, max: 100, step: 5, value: state.tauChnUsa,
      fmt: (v) => v + "%"
    }, (v) => { state.tauChnUsa = v; run(); });

    const sub2 = document.createElement("div"); sub2.className = "ctl-sub";
    sub2.textContent = "Parámetros estructurales / Structural parameters"; ctl.appendChild(sub2);

    MAV.slider(ctl, {
      key: "theta",
      label: { es: "Elasticidad de comercio θ", en: "Trade elasticity θ" },
      min: 2.0, max: 10.0, step: 0.5, value: state.theta,
      fmt: (v) => MAV.num(v, 1)
    }, (v) => { state.theta = v; run(); });

    MAV.slider(ctl, {
      key: "gamma_va",
      label: { es: "Valor agregado manufactura γ_va", en: "Mfg value-added share γ_va" },
      min: 0.20, max: 0.80, step: 0.05, value: state.gammaVa,
      fmt: (v) => MAV.num(v, 2)
    }, (v) => { state.gammaVa = v; run(); });

    // Tarjetas y gráficas
    const chWelfare = MAV.lineChart(document.getElementById("chart-welfare"), { series: [] });
    const chDiversion = MAV.lineChart(document.getElementById("chart-diversion"), { series: [] });
    const chDecomp = MAV.lineChart(document.getElementById("chart-decomp"), { series: [] });

    function run() {
      // Construir matriz de aranceles (2 sectores x 3 países x 3 países)
      // 0: MEX, 1: USA, 2: CHN
      const tariffs = [
        [
          [1.0, 1.0 + state.tauMexUsa / 100.0, 1.0], // MEX importa de [MEX, USA, CHN]
          [1.0 + state.tauUsaMex / 100.0, 1.0, 1.0 + state.tauUsaChn / 100.0], // USA importa de [MEX, USA, CHN]
          [1.0, 1.0 + state.tauChnUsa / 100.0, 1.0]  // CHN importa de [MEX, USA, CHN]
        ],
        // Servicios
        [
          [1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0]
        ]
      ];

      const res = solveCaliendoParro(tariffs, state.theta, state.gammaVa);

      // Stat tiles
      const mexW = res.welfarePct[0];
      const usaW = res.welfarePct[1];
      const chnW = res.welfarePct[2];
      const baseShareUsaMex = BASE_TRADE_SHARES[0][1][0] * 100.0;
      const primeShareUsaMex = res.piPrime[0][1][0] * 100.0;
      const diversionPct = ((primeShareUsaMex - baseShareUsaMex) / baseShareUsaMex) * 100.0;

      MAV.tiles(tilesEl, [
        {
          label: { es: "Bienestar México", en: "Mexico Welfare" },
          value: (mexW >= 0 ? "+" : "") + MAV.num(mexW, 2) + "%",
          note: { es: `Salario real: ${(res.realWageHat[0] - 1.0) >= 0 ? "+" : ""}${MAV.num((res.realWageHat[0] - 1.0) * 100, 2)}%`,
                  en: `Real wage: ${(res.realWageHat[0] - 1.0) >= 0 ? "+" : ""}${MAV.num((res.realWageHat[0] - 1.0) * 100, 2)}%` }
        },
        {
          label: { es: "Bienestar EE. UU.", en: "US Welfare" },
          value: (usaW >= 0 ? "+" : "") + MAV.num(usaW, 2) + "%",
          note: { es: `Recaudación: ${MAV.num(res.tariffRevPrime[1], 1)}`,
                  en: `Tariff revenue: ${MAV.num(res.tariffRevPrime[1], 1)}` }
        },
        {
          label: { es: "Bienestar China", en: "China Welfare" },
          value: (chnW >= 0 ? "+" : "") + MAV.num(chnW, 2) + "%",
          note: { es: `Salario real: ${(res.realWageHat[2] - 1.0) >= 0 ? "+" : ""}${MAV.num((res.realWageHat[2] - 1.0) * 100, 2)}%`,
                  en: `Real wage: ${(res.realWageHat[2] - 1.0) >= 0 ? "+" : ""}${MAV.num((res.realWageHat[2] - 1.0) * 100, 2)}%` }
        },
        {
          label: { es: "Desvío hacia México", en: "Diversion to Mexico" },
          value: (diversionPct >= 0 ? "+" : "") + MAV.num(diversionPct, 1) + "%",
          note: { es: `Cuota en EE. UU.: ${MAV.num(primeShareUsaMex, 1)}% (base 14.0%)`,
                  en: `US import share: ${MAV.num(primeShareUsaMex, 1)}% (base 14.0%)` }
        }
      ]);

      // Gráfica 1: Bienestar y Salarios Reales
      chWelfare.update({
        title: { es: "Impacto en Bienestar y Salarios Reales", en: "Welfare and Real Wage Impact" },
        sub: { es: "Variación porcentual contrafactual por país (Caliendo & Parro 2015)",
               en: "Counterfactual percentage change by country (Caliendo & Parro 2015)" },
        xLabel: { es: "País", en: "Country" },
        yLabel: { es: "% de cambio", en: "% change" },
        xFmt: (v) => v === 0 ? "MEX" : v === 1 ? "USA" : "CHN",
        series: [
          {
            name: { es: "Bienestar (% ΔW)", en: "Welfare (% ΔW)" },
            x: [0, 1, 2],
            y: [res.welfarePct[0], res.welfarePct[1], res.welfarePct[2]],
            style: 0
          },
          {
            name: { es: "Salario Real (% Δω)", en: "Real Wage (% Δω)" },
            x: [0, 1, 2],
            y: [(res.realWageHat[0] - 1.0) * 100.0, (res.realWageHat[1] - 1.0) * 100.0, (res.realWageHat[2] - 1.0) * 100.0],
            style: 1
          }
        ],
        csvName: "bienestar-comercio",
        yFmt: (v) => MAV.num(v, 2) + "%"
      });

      // Gráfica 2: Desvío de Comercio en EE. UU.
      chDiversion.update({
        title: { es: "Desvío de Comercio en EE. UU. (Manufacturas)", en: "US Trade Diversion (Manufactures)" },
        sub: { es: "Cuota de mercado en la absorción manufacturera de EE. UU. (Línea base vs Contrafactual)",
               en: "Market shares in US manufacturing absorption (Baseline vs Counterfactual)" },
        xLabel: { es: "Origen", en: "Origin" },
        yLabel: { es: "Cuota de mercado (%)", en: "Market share (%)" },
        xFmt: (v) => v === 0 ? "MEX" : v === 1 ? "USA (doméstico)" : "CHN",
        series: [
          {
            name: { es: "Línea base 2026", en: "Baseline 2026" },
            x: [0, 1, 2],
            y: [BASE_TRADE_SHARES[0][1][0] * 100.0, BASE_TRADE_SHARES[0][1][1] * 100.0, BASE_TRADE_SHARES[0][1][2] * 100.0],
            style: 1
          },
          {
            name: { es: "Contrafactual con aranceles", en: "Counterfactual with tariffs" },
            x: [0, 1, 2],
            y: [res.piPrime[0][1][0] * 100.0, res.piPrime[0][1][1] * 100.0, res.piPrime[0][1][2] * 100.0],
            style: 0
          }
        ],
        csvName: "desvio-comercio",
        yFmt: (v) => MAV.num(v, 1) + "%"
      });

      // Gráfica 3: Descomposición de Bienestar
      chDecomp.update({
        title: { es: "Descomposición del Bienestar (Caliendo & Parro 2015)", en: "Welfare Decomposition (Caliendo & Parro 2015)" },
        sub: { es: "Canales económicos: Términos de intercambio, insumo-producto y recaudación arancelaria",
               en: "Economic channels: Terms of trade, input-output, and tariff revenue" },
        xLabel: { es: "País", en: "Country" },
        yLabel: { es: "Contribución al bienestar (%)", en: "Welfare contribution (%)" },
        xFmt: (v) => v === 0 ? "MEX" : v === 1 ? "USA" : "CHN",
        series: [
          {
            name: { es: "Términos de intercambio", en: "Terms of trade" },
            x: [0, 1, 2],
            y: res.totDecomp,
            style: 0
          },
          {
            name: { es: "Multiplicador insumo-producto", en: "Input-output multiplier" },
            x: [0, 1, 2],
            y: res.ioDecomp,
            style: 2
          },
          {
            name: { es: "Efecto recaudación fiscal", en: "Tariff revenue effect" },
            x: [0, 1, 2],
            y: res.revDecomp,
            style: 3
          }
        ],
        csvName: "descomposicion-bienestar",
        yFmt: (v) => MAV.num(v, 2) + "%"
      });
    }

    MAV.onLang(run);
    run();
  });
})();
