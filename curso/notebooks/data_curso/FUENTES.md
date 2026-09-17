# Fuentes de `data_curso/`

*English summary at the end.*

Estos archivos son **copias congeladas** (paquete `bundle_2026A`, agosto de 2026)
de datos públicos de terceros, más algunos cálculos del curso hechos a partir de
ellos. Se incluyen para que los cuadernos corran sin red y den siempre las mismas
cifras. `bundle_2026A/manifest.csv` registra, archivo por archivo, la fuente, la
fecha de descarga, la última observación y el SHA-256. Los archivos de la raíz de
`data_curso/` son idénticos a los de `bundle_2026A/`.

**La licencia MIT de `puremacro` cubre su código, no estos datos.** Cada editor
conserva sus derechos y sus términos de uso. Donde abajo no se indica una licencia
concreta, consulte los términos del editor antes de redistribuir. **Si usa estos
datos, cite a los editores originales**, no a este curso ni a `puremacro`. Ningún
editor está afiliado al curso ni lo respalda.

Las series cambian después de publicadas (revisiones): para un trabajo propio,
descargue la versión vigente de la fuente.

## Por editor

### U.S. Bureau of Economic Analysis (BEA), vía FRED

<https://www.bea.gov>

Obra del gobierno federal de EUA. Cite a BEA y a FRED (Federal Reserve Bank of St. Louis).

`A191RL1Q225SBEA.csv`, `GDPA.csv`, `GDPC1.csv`, `GPDI.csv`, `GPDIC1.csv`, `PCECC96.csv`, `PNFI.csv`, `Y001RC1Q027SBEA.csv`

### U.S. Bureau of Labor Statistics (BLS), vía FRED

<https://www.bls.gov>

Obra del gobierno federal de EUA. Cite a BLS y a FRED. Incluye CPS, JOLTS, LAUS, productividad.

`AKUR.csv`, `ALUR.csv`, `ARUR.csv`, `AWHI.csv`, `AZUR.csv`, `CAUR.csv`, `CE16OV.csv`, `CLF16OV.csv`, `CNP16OV.csv`, `COUR.csv`, `CPIAUCSL.csv`, `CPILFESL.csv`, `CTUR.csv`, `DCUR.csv`, `DEUR.csv`, `FLUR.csv`, `GAUR.csv`, `HIUR.csv`, `HOANBS.csv`, `IAUR.csv`, `IDUR.csv`, `ILUR.csv`, `INUR.csv`, `JTSHIL.csv`, `JTSJOL.csv`, `KSUR.csv`, `KYUR.csv`, `LAUR.csv`, `LEU0252916700Q.csv`, `LEU0252917300Q.csv`, `LEU0252918500Q.csv`, `LNS12027660.csv`, `LNS12027662.csv`, `LNS15000000.csv`, `LNS17000000.csv`, `LNS17100000.csv`, `LNS17200000.csv`, `LNS17400000.csv`, `LNS17500000.csv`, `LNS17600000.csv`, `LNS17800000.csv`, `LNS17900000.csv`, `LNU02027660.csv`, `LNU02027662.csv`, `MAUR.csv`, `MDUR.csv`, `MEUR.csv`, `MIUR.csv`, `MNUR.csv`, `MOUR.csv`, `MSUR.csv`, `MTUR.csv`, `NCUR.csv`, `NDUR.csv`, `NEUR.csv`, `NHUR.csv`, `NJUR.csv`, `NMUR.csv`, `NVUR.csv`, `NYUR.csv`, `OHUR.csv`, `OKUR.csv`, `OPHNFB.csv`, `ORUR.csv`, `OUTNFB.csv`, `PAUR.csv`, `PRS85006173.csv`, `RIUR.csv`, `SCUR.csv`, `SDUR.csv`, `TNUR.csv`, `TXUR.csv`, `UEMPLT5.csv`, `UNEMPLOY.csv`, `UNRATE.csv`, `USPRIV.csv`, `UTUR.csv`, `VAUR.csv`, `VTUR.csv`, `WAUR.csv`, `WIUR.csv`, `WVUR.csv`, `WYUR.csv`, `cps_transiciones_mensuales.csv`, `paro_estatal_mensual.csv`

### Board of Governors of the Federal Reserve System, vía FRED

<https://www.federalreserve.gov>

Incluye G.17, Z.1 (Financial Accounts) y las Distributional Financial Accounts. Cite a la Junta de la Reserva Federal y a FRED.

`FEDFUNDS.csv`, `INDPRO.csv`, `NCBEILQ027S.csv`, `TCU.csv`, `TNWMVBSNNCB.csv`, `WFRBSB50215.csv`, `WFRBSN09161.csv`, `WFRBSN40188.csv`, `WFRBST01134.csv`

### Federal Reserve Bank of Chicago (NFCI), vía FRED

<https://www.chicagofed.org>

Consulte los términos del editor. Cite al Chicago Fed.

`NFCI.csv`

### NBER (fechas de recesión), vía FRED

<https://www.nber.org/research/business-cycle-dating>

Consulte los términos del editor. Cite al NBER Business Cycle Dating Committee.

`USREC.csv`

### Cboe Global Markets (VIX), vía FRED

<https://www.cboe.com>

Serie con derechos de un tercero: consulte los términos de Cboe antes de redistribuirla.

`VIXCLS.csv`

### Otras series de FRED

<https://fred.stlouisfed.org>

FRED redistribuye series de muchos editores y algunas tienen derechos de terceros: consulte la ficha de cada serie en FRED y los términos de su editor.

`K1PTOTL1ES000.csv`, `M1NTOTL1ES000.csv`, `M1PTOTL1ES000.csv`, `PERIC.csv`, `PIRIC.csv`

### OCDE (QNA, STES, MEI, ANA, KEI, LFS)

<https://data-explorer.oecd.org>

Reutilización permitida con atribución según los términos de la OCDE; consúltelos. Cite a la OCDE, no a este curso. Varios archivos son paneles derivados (rebases, empalmes) construidos con puremacro.fetch a partir de estos datos.

`CPALTT01MXM659N.csv`, `CPGRLE01MXM659N.csv`, `IR3TIB01MXM156N.csv`, `LRUN64TTMXQ156S.csv`, `oecd_ana_actividad.csv`, `oecd_fx_lcu_usd_q.csv`, `oecd_lfs_mex_prate_Q.csv`, `oecd_mei_mex_log_ip.csv`, `oecd_qna_ag_hogares.csv`, `oecd_qna_apertura.csv`, `oecd_qna_base_fija_Q.csv`, `oecd_qna_bkk.csv`, `oecd_qna_dglp17.csv`, `oecd_qna_kaldor.csv`, `oecd_qna_labor_margenes.csv`, `oecd_qna_labor_ramas.csv`, `oecd_qna_largo.csv`, `oecd_qna_largo_seams.csv`, `oecd_qna_mex_componentes_VQ.csv`, `oecd_qna_panel_meta.csv`, `oecd_qna_panel_nominal.csv`, `oecd_qna_scb.csv`, `oecd_stes_comp_con_real.csv.gz`, `oecd_stes_comp_deflator.csv.gz`, `oecd_stes_comp_employment.csv.gz`, `oecd_stes_comp_exports_real.csv.gz`, `oecd_stes_comp_gdp_nom.csv.gz`, `oecd_stes_comp_gfcf_real.csv.gz`, `oecd_stes_comp_govcon_real.csv.gz`, `oecd_stes_comp_imports_real.csv.gz`, `oecd_stes_revisiones_mex.csv`, `oecd_stes_vintages_gdp.csv.gz`, `oecd_urate_q.csv`, `qna_gasto.csv`, `qna_mercado_laboral.csv`, `qna_meta.csv`, `qna_produccion.csv`, `qna_renta.csv`

### Banco Mundial (World Development Indicators)

<https://data.worldbank.org>

Licencia CC BY 4.0. Cite al Banco Mundial.

`POPTOTMXA647NWDB.csv`, `POPTOTUSA647NWDB.csv`, `wb_credito_privado.csv`

### Penn World Table (Universidad de Groningen), vía FRED

<https://www.rug.nl/ggdc/productivity/pwt/>

Licencia CC BY 4.0. Cite Feenstra, Inklaar y Timmer (2015).

`RGDPNAMXA666NRUG.csv`, `RKNANPMXA666NRUG.csv`

### INEGI — ENOE (procesada)

<https://www.inegi.org.mx/programas/enoe/>

Consulte los Términos de libre uso del INEGI. Cite al INEGI. Los archivos son agregados procesados, no microdatos.

`enoe_stocks_monthly.csv`, `enoe_stocks_monthly.parquet`, `enoe_transitions_quarterly_observed.parquet`

### OIT — ILOSTAT

<https://ilostat.ilo.org>

Consulte los términos del editor. Cite a ILOSTAT (OIT).

`ilo_informalidad.csv`, `ilo_informalidad_por_situacion.csv`, `ilo_informalidad_trimestral.csv`, `ilostat_urate_mex_usa.csv`

### Eurostat (vacantes y paro), con JOLTS y UNRATE para EUA

<https://ec.europa.eu/eurostat>

Consulte los términos de Eurostat y de BLS. Cite a ambos.

`beveridge_uv.csv`

### Economic Policy Uncertainty — policyuncertainty.com

<https://www.policyuncertainty.com>

Consulte los términos del sitio. Cite Baker, Bloom y Davis (2016).

`epu_mex_mensual.csv`, `epu_usa_mensual.csv`

### John Fernald — PTF trimestral ajustada por utilización (Federal Reserve Bank of San Francisco)

<https://www.frbsf.org/research-and-insights/data-and-indicators/total-factor-productivity-tfp/>

Consulte los términos del editor. Cite Fernald (2014).

`fernald_tfp.csv`, `quarterly_tfp.xlsx`

### Federal Reserve Bank of Philadelphia — Real-Time Data Set for Macroeconomists (y ALFRED)

<https://www.philadelphiafed.org/surveys-and-data/real-time-data-research>

Consulte los términos del editor. Cite a la Reserva Federal de Filadelfia.

`alfred_pib_primeras_publicaciones.csv`, `philfed_routputMvQd.csv`, `philfed_routputMvQd.xlsx`

### Archivos de réplica de artículos (Ramey–Zubairy 2018, Romer–Romer 2010, Galí 1999, Devries–Guajardo–Leigh–Pescatori 2011)

Consulte los términos de cada archivo de réplica. Cite el artículo original. Galí (1999) se reconstruye con series de BLS en FRED.

`dglp_consolidaciones_trimestrales.csv`, `gali1999.csv`, `rz2018.csv`, `tax14_narrative_tax_shocks.csv`, `tax14_us_fiscal.csv`

### Jurado, Ludvigson y Ng — incertidumbre macro

<https://www.sydneyludvigson.com>

Consulte los términos del editor. Cite Jurado, Ludvigson y Ng (2015).

`jln_macro_usa.csv`

### EU KLEMS

<https://euklems-intanprod-llee.luiss.it>

Consulte los términos del editor. Cite la base EU KLEMS.

`klems_labor_share.csv`

### Banco de México — anuncios de política monetaria, clasificados para el curso

<https://www.banxico.org.mx>

Clasificación propia del curso a partir de comunicados públicos de Banxico. Cite a Banxico.

`banxico_stance_monthly.csv`

### Cálculos del curso

Resultados calculados con puremacro (Aiyagari, contabilidad del ciclo) a partir de los datos de arriba. Cite las fuentes de origen.

`a6_aiyagari_egm.npz`, `a6_aiyagari_ss.npz`, `wedges_bca_curso.csv`, `wedges_bca_slim.csv`, `wedges_panel_BGP_clean_nu2.csv`

---

## English summary

These files are frozen copies (bundle `bundle_2026A`, August 2026) of public
third-party data, plus a few course computations derived from them, so the
notebooks run offline and reproduce the same numbers. `bundle_2026A/manifest.csv`
records each file's source, download date, last observation and SHA-256.

**The MIT licence of `puremacro` covers its code, not this data.** Each publisher
keeps its rights and terms of use; the licence is stated above only where it is
known (World Bank and Penn World Table: CC BY 4.0; U.S. federal agencies: U.S.
government works). Otherwise, check the publisher's terms before redistributing.
**If you use this data, cite the original publishers** listed above, not this
course or `puremacro`. No publisher is affiliated with or endorses this course.
