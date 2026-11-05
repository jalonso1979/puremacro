import copy
import json
import warnings
import numpy as np, pandas as pd, pytest
import puremacro.examples.first_brief_fiscal as fb


# rows kept by RZ's war-exclusion rule on the bundled data: a row drops when its outcome window, shock or lagged controls touch 1941Q3-1945Q4
RZ_RULE_ROWS = [(7, 464), (15, 448), (19, 440), (20, 438)]


@pytest.fixture(scope="module")
def computed():
    return fb.compute_brief(exclude=fb.WWII)          # (record, paths)


def test_parse_exclude_valid():
    assert fb.parse_exclude("1941Q3:1945Q4") == (pd.Period("1941Q3", "Q"), pd.Period("1945Q4", "Q"))


@pytest.mark.parametrize("text", ["1941Q3", "1945Q4:1941Q3", "1941-07:1945-12", "", "1941Q5:1945Q4"])
def test_parse_exclude_rejects(text):
    with pytest.raises(ValueError):
        fb.parse_exclude(text)


def test_exclude_period_masks_y_g_newsy_and_rebuilds_lags():
    df = fb.load_brief_data()
    m = fb.exclude_period(df, fb.parse_exclude(fb.WWII))
    q = df.index.to_period("Q")
    inside = (q >= "1941Q3") & (q <= "1945Q4")
    assert len(m) == len(df) and m.loc[inside, ["y", "g", "newsy"]].isna().all().all()
    pd.testing.assert_frame_equal(m.loc[~inside, ["y", "g", "newsy"]], df.loc[~inside, ["y", "g", "newsy"]])
    after = df.index[q == "1946Q1"][0]
    assert np.isnan(m.loc[after, "y_l1"]) and np.isnan(m.loc[after, "newsy_l1"])   # lags rebuilt from masked series
    assert m.loc[after, "y"] == df.loc[after, "y"]
    assert "cumy" not in m and "cumg" not in m                                       # stale running sums dropped


def test_exclude_period_rejects_period_outside_sample():
    with pytest.raises(ValueError, match="does not overlap"):
        fb.exclude_period(fb.load_brief_data(), fb.parse_exclude("1700Q1:1701Q4"))


@pytest.mark.parametrize(("h", "n"), RZ_RULE_ROWS)   # (h, n): rows kept at h = 7, 15, 19, 20 under RZ's war-exclusion rule
def test_exclusion_selects_rz_rows(h, n):
    from puremacro.examples.ramey_zubairy_2018_multipliers import RZ_CONTROLS
    from puremacro.lp._common import cumulative_windows
    m = fb.exclude_period(fb.load_brief_data(), fb.parse_exclude(fb.WWII))
    yw, xw = cumulative_windows(m["y"], m["g"], h)
    used = pd.concat([yw, xw, m["newsy"], m[RZ_CONTROLS]], axis=1).notna().all(axis=1)
    assert int(used.sum()) == n


def test_concentration_finds_a_planted_spike():
    df = fb.load_brief_data().copy()
    spike = df.index[300]
    df.loc[spike, "newsy"] = df["newsy"].abs().max() * 50
    regime = float(df.loc[spike, "slack_l"])
    out = fb.concentration(df, regime, 8)
    assert out["share"] > 0.9 and out["date"] == str(spike.to_period("Q"))


def test_weak_flags_follow_the_cutoff(computed):
    record, _ = computed
    cutoff = record["specification"]["weak_cutoff"]
    assert abs(cutoff - 23.11) < 0.01
    for sample in ("full", "experiment"):
        for cells in record[sample]["horizons"].values():
            for name in ("linear", "slack", "no_slack"):
                assert cells[name]["weak"] == (cells[name]["f_eff"] < cutoff)
            assert cells["difference"]["reliable"] == (not cells["slack"]["weak"] and not cells["no_slack"]["weak"])


def test_record_and_paths_shape(computed):
    record, paths = computed
    assert set(record) == {"puremacro_version", "data", "specification", "full", "experiment"}
    assert record["experiment"]["exclude"] == ["1941Q3", "1945Q4"]
    assert set(record["full"]["horizons"]) == {"7", "15"}
    assert list(paths.groupby("sample").size()) == [21, 21]
    assert json.loads(json.dumps(record)) == record                  # JSON-ready, no NaN or inf


def test_record_matches_independent_measurements(computed):
    # Full-sample values measured independently with the manual running-sum recipe:
    # one estimator call per horizon, a running sum of y as the outcome, the window
    # sum of g as the policy variable, n_lags=0 and the pre-lagged controls, without
    # cumulative=True. The multipliers are recorded at full precision.
    full = computed[0]["full"]["horizons"]
    np.testing.assert_allclose(full["7"]["slack"]["multiplier"], 0.6028751048822653, rtol=1e-8)
    np.testing.assert_allclose(full["7"]["no_slack"]["multiplier"], 0.5949443896394699, rtol=1e-8)
    np.testing.assert_allclose(full["15"]["slack"]["multiplier"], 0.6819723644803134, rtol=1e-8)
    np.testing.assert_allclose(full["15"]["no_slack"]["multiplier"], 0.6683477316895448, rtol=1e-8)
    np.testing.assert_allclose(full["7"]["linear"]["multiplier"], 0.6637153416055578, rtol=1e-8)
    np.testing.assert_allclose(full["15"]["linear"]["multiplier"], 0.7133618907912175, rtol=1e-8)
    # effective F, measured independently to one decimal: slack strong, no slack and linear weak
    np.testing.assert_allclose([full["7"]["slack"]["f_eff"], full["7"]["no_slack"]["f_eff"], full["7"]["linear"]["f_eff"]],
                               [210.2, 7.3, 16.0], atol=0.051)
    assert full["7"]["slack"]["n_eff"] == 180 and full["7"]["no_slack"]["n_eff"] == 313
    assert computed[0]["experiment"]["horizons"]["15"]["slack"]["ar_set_type"] == "unbounded_rays"


def test_default_run_has_no_experiment_and_keeps_estimator_warnings(monkeypatch):
    real = fb.estimate

    def noisy(df):
        warnings.warn("horizon h=3 could not be estimated", RuntimeWarning)
        return real(df)

    monkeypatch.setattr(fb, "estimate", noisy)                       # looked up at call time
    with pytest.warns(RuntimeWarning, match="could not be estimated"):  # re-issued after capture
        record, paths = fb.compute_brief()
    assert record["experiment"] is None and list(paths["sample"]) == ["full"] * 21
    assert fb._LAST_WARNINGS == ["full: RuntimeWarning: horizon h=3 could not be estimated"]


FILES = {"headline.md", "headline.csv", "headline.tex", "paths.csv", "multipliers.png",
         "headline.json", "manifest.json"}


def test_outputs_and_manifest(tmp_path):
    study = fb.run_brief(tmp_path, lang="es")
    assert FILES <= {p.name for p in tmp_path.iterdir()}
    man = study["manifest"]
    assert set(man) == {"lang", "puremacro_version", "data", "specification", "experiment",
                        "warnings", "timings", "environment", "artifacts"} and man["lang"] == "es"
    for name, art in man["artifacts"].items():
        assert fb.sha256(tmp_path / name) == art["sha256"]
    assert "Con holgura" in (tmp_path / "headline.md").read_text(encoding="utf-8")
    assert "/Users/" not in (tmp_path / "manifest.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", ["en", "es"])
def test_reading_quotes_ar_set_not_point_for_weak_cells(computed, lang):
    record = copy.deepcopy(computed[0])
    cells = record["full"]["horizons"]["7"]
    cells["no_slack"].update(multiplier=9.87, weak=True, f_eff=5.0,
                             ar_lo=0.11, ar_hi=2.22, ar_set_type="bounded")
    cells["slack"].update(multiplier=1.23, weak=False, f_eff=99.0)
    cells["difference"]["reliable"] = False
    text = fb.page_block(record, "headline_reading", lang)
    assert "9.87" not in text and "1.23" in text                       # weak point never quoted
    assert fb._fmt(0.11) in text and fb._fmt(2.22) in text and fb._fmt(5.0, 1) in text


@pytest.mark.parametrize("lang", ["en", "es"])
def test_headline_table_marks_each_weak_cell(computed, lang):
    record = computed[0]
    weak = sum(c[n]["weak"] for c in record["full"]["horizons"].values() for n in ("linear", "slack", "no_slack"))
    note = 1 if weak else 0                                    # the note appears only when a cell is weak
    assert fb.page_block(record, "headline", lang).count("†") == weak + note


@pytest.mark.parametrize("lang", ["en", "es"])
def test_readings_report_concentration_by_the_rule(computed, lang):
    record = computed[0]
    for sample, block in (("full", "headline_reading"), ("experiment", "experiment_reading")):
        text = fb.page_block(record, block, lang)
        for cells in record[sample]["horizons"].values():
            for name in ("slack", "no_slack"):
                conc = cells["concentration"][name]
                if conc["share"] >= fb.CONCENTRATION_REPORT_SHARE:
                    assert conc["date"] in text and f"{round(100 * conc['share'])}" in text, (sample, name)


@pytest.mark.parametrize("argv", [["--exclude", "1941Q3"], ["--exclude", "1945Q4:1941Q3"],
                                  ["--exclude", "1700Q1:1701Q4"], ["--lang", "fr"]])
def test_cli_rejects_bad_arguments(argv, tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        fb.main([*argv, "--output", str(out)])
    assert exc.value.code == 2 and not out.exists()


def test_no_outputs_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "estimate", lambda df: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        fb.run_brief(tmp_path / "out")
    assert not (tmp_path / "out" / "headline.json").exists()


# The reading copy is fixed: what the rule writes for the bundled data, word for word.
READINGS = {
    ("full", "en"): (
        'Multipliers are dollars of GDP per dollar of government purchases, both cumulated over the '
        'horizon. Linear: weak at 2 years (effective F 16.0, cutoff 23.1), with a '
        'weak-instrument-robust 90% set of 0.57 to 0.83; weak at 4 years (effective F 12.0, cutoff '
        '23.1), with a weak-instrument-robust 90% set of 0.65 to 0.81. Slack: 0.60 at 2 years (90% '
        'band 0.41 to 0.79); 0.68 at 4 years (90% band 0.59 to 0.77). No slack: weak at 2 years '
        '(effective F 7.3, cutoff 23.1), with a weak-instrument-robust 90% set of 0.44 to 0.82; weak '
        'at 4 years (effective F 10.8, cutoff 23.1), with a weak-instrument-robust 90% set of 0.40 to'
        ' 0.89. The comparison of the multipliers with and without slack is not reliable at either '
        'horizon, because at least one regime is weak at each. Slack: one quarter, 1941Q4, carries '
        "52% of the instrument's residual variation at both horizons. No slack: one quarter, 1950Q3, "
        "carries 36% of the instrument's residual variation at both horizons."
    ),
    ("full", "es"): (
        'Los multiplicadores son dólares de PIB por dólar de compras del gobierno, ambos acumulados '
        'en el horizonte. Lineal: débil a 2 años (F efectiva 16.0, umbral 23.1), con un conjunto al '
        '90% robusto a instrumentos débiles de 0.57 a 0.83; débil a 4 años (F efectiva 12.0, umbral '
        '23.1), con un conjunto al 90% robusto a instrumentos débiles de 0.65 a 0.81. Con holgura: '
        '0.60 a 2 años (banda al 90%: 0.41 a 0.79); 0.68 a 4 años (banda al 90%: 0.59 a 0.77). Sin '
        'holgura: débil a 2 años (F efectiva 7.3, umbral 23.1), con un conjunto al 90% robusto a '
        'instrumentos débiles de 0.44 a 0.82; débil a 4 años (F efectiva 10.8, umbral 23.1), con un '
        'conjunto al 90% robusto a instrumentos débiles de 0.40 a 0.89. La comparación de los '
        'multiplicadores con y sin holgura no es fiable en ninguno de los dos horizontes, porque en '
        'cada uno al menos un régimen es débil. Con holgura: un solo trimestre, 1941Q4, concentra el '
        '52% de la variación residual del instrumento en ambos horizontes. Sin holgura: un solo '
        'trimestre, 1950Q3, concentra el 36% de la variación residual del instrumento en ambos '
        'horizontes.'
    ),
    ("experiment", "en"): (
        'Excluding 1941Q3-1945Q4: every quarter whose window, shock or lags touch the period is '
        'dropped, as Ramey and Zubairy do for World War II. Multipliers are dollars of GDP per dollar'
        ' of government purchases, both cumulated over the horizon. Linear: weak at 2 years '
        '(effective F 4.8, cutoff 23.1), with a weak-instrument-robust 90% set of 0.54 to 2.04; weak '
        'at 4 years (effective F 19.6, cutoff 23.1), with a weak-instrument-robust 90% set of 0.50 to'
        ' 1.07. Slack: weak at 2 years (effective F 20.8, cutoff 23.1), with a weak-instrument-robust'
        ' 90% set of 0.07 to 3.61; weak at 4 years (effective F 3.2, cutoff 23.1), and its '
        'weak-instrument-robust 90% set is unbounded: the data cannot bound it. No slack: weak at 2 '
        'years (effective F 5.7, cutoff 23.1), with a weak-instrument-robust 90% set of 0.25 to 1.00;'
        ' weak at 4 years (effective F 22.3, cutoff 23.1), with a weak-instrument-robust 90% set of '
        '0.18 to 0.90. The comparison of the multipliers with and without slack is not reliable at '
        'either horizon, because at least one regime is weak at each. Slack: one quarter, 1961Q2, '
        "carries 41% of the instrument's residual variation at 2 years and 48% at 4 years. No slack: "
        "one quarter, 1950Q3, carries 49% of the instrument's residual variation at both horizons. No"
        ' estimate clears the cutoff in this sample, so none is compared with the full sample. Ramey '
        "and Zubairy's Table 3 values without World War II are three-step estimates (summed impulse "
        "responses); the brief's are one-step, so the two need not match."
    ),
    ("experiment", "es"): (
        'Excluyendo 1941Q3-1945Q4: se descarta todo trimestre cuya ventana, choque o rezagos tocan el'
        ' periodo, como hacen Ramey y Zubairy con la Segunda Guerra Mundial. Los multiplicadores son '
        'dólares de PIB por dólar de compras del gobierno, ambos acumulados en el horizonte. Lineal: '
        'débil a 2 años (F efectiva 4.8, umbral 23.1), con un conjunto al 90% robusto a instrumentos '
        'débiles de 0.54 a 2.04; débil a 4 años (F efectiva 19.6, umbral 23.1), con un conjunto al '
        '90% robusto a instrumentos débiles de 0.50 a 1.07. Con holgura: débil a 2 años (F efectiva '
        '20.8, umbral 23.1), con un conjunto al 90% robusto a instrumentos débiles de 0.07 a 3.61; '
        'débil a 4 años (F efectiva 3.2, umbral 23.1), y su conjunto al 90% robusto a instrumentos '
        'débiles no está acotado: los datos no pueden acotarlo. Sin holgura: débil a 2 años (F '
        'efectiva 5.7, umbral 23.1), con un conjunto al 90% robusto a instrumentos débiles de 0.25 a '
        '1.00; débil a 4 años (F efectiva 22.3, umbral 23.1), con un conjunto al 90% robusto a '
        'instrumentos débiles de 0.18 a 0.90. La comparación de los multiplicadores con y sin holgura'
        ' no es fiable en ninguno de los dos horizontes, porque en cada uno al menos un régimen es '
        'débil. Con holgura: un solo trimestre, 1961Q2, concentra el 41% de la variación residual del'
        ' instrumento a 2 años y el 48% a 4 años. Sin holgura: un solo trimestre, 1950Q3, concentra '
        'el 49% de la variación residual del instrumento en ambos horizontes. Ninguna estimación '
        'supera el umbral en esta muestra, así que ninguna se compara con la muestra completa. Los '
        'valores sin la Segunda Guerra Mundial de la Tabla 3 de Ramey y Zubairy son estimaciones en '
        'tres pasos (respuestas al impulso sumadas); las del informe son de un paso, así que no '
        'tienen por qué coincidir.'
    ),
}


@pytest.mark.parametrize(("sample", "lang"), list(READINGS))
def test_readings_reproduce_the_fixed_copy(computed, sample, lang):
    block = "headline_reading" if sample == "full" else "experiment_reading"
    assert fb.page_block(computed[0], block, lang) == READINGS[sample, lang]


def _record_for_every_branch(record):
    """The bundled record, edited so that the rule's other branches fire."""
    record = copy.deepcopy(record)
    full, exp = record["full"]["horizons"], record["experiment"]["horizons"]
    full["7"]["linear"].update(ar_set_type="empty", ar_lo=None, ar_hi=None)
    full["15"]["linear"].update(ar_set_type="all_real", ar_lo=None, ar_hi=None)
    full["7"]["no_slack"].update(weak=False, f_eff=30.0, multiplier=0.55, lo=0.40, hi=0.70)
    full["7"]["difference"].update(reliable=True, estimate=0.05, p_value=0.80)
    full["15"]["concentration"]["slack"].update(date="1942Q1", share=0.30)
    full["7"]["concentration"]["no_slack"]["share"] = 0.20             # below the reporting share
    exp["7"]["slack"].update(weak=False, f_eff=30.0, multiplier=1.77)    # full sample strong
    exp["15"]["no_slack"].update(weak=False, f_eff=30.0, multiplier=0.54)  # full sample weak
    for cells in exp.values():
        for top in cells["concentration"].values():
            top["share"] = 0.10
    return record


BRANCHES = {
    "en": ("Linear: weak at 2 years (effective F 16.0, cutoff 23.1), and its weak-instrument-robust "
           "90% set is empty; weak at 4 years (effective F 12.0, cutoff 23.1), and its "
           "weak-instrument-robust 90% set is unbounded: the data cannot bound it. "
           "Slack: 0.60 at 2 years (90% band 0.41 to 0.79); 0.68 at 4 years (90% band 0.59 to 0.77). "
           "No slack: 0.55 at 2 years (90% band 0.40 to 0.70); weak at 4 years (effective F 10.8, "
           "cutoff 23.1), with a weak-instrument-robust 90% set of 0.40 to 0.89. "
           "With and without slack, the multipliers differ by 0.05 at 2 years (p = 0.80). "
           "The comparison of the multipliers with and without slack is not reliable at 4 years, "
           "because a regime is weak there. "
           "Slack: 1941Q4 carries 52% of the instrument's residual variation at 2 years, and 1942Q1 "
           "carries 30% at 4 years. "
           "No slack: one quarter, 1950Q3, carries 36% of the instrument's residual variation at 4 years.",
           "Slack at 2 years: 1.77 here against 0.60 in the full sample. "
           "No slack at 4 years: 0.54 here, while the full-sample estimate is weak, with a "
           "weak-instrument-robust 90% set of 0.40 to 0.89. "
           "Ramey and Zubairy's Table 3"),
    "es": ("Lineal: débil a 2 años (F efectiva 16.0, umbral 23.1), y su conjunto al 90% robusto a "
           "instrumentos débiles está vacío; débil a 4 años (F efectiva 12.0, umbral 23.1), y su "
           "conjunto al 90% robusto a instrumentos débiles no está acotado: los datos no pueden "
           "acotarlo. "
           "Con holgura: 0.60 a 2 años (banda al 90%: 0.41 a 0.79); 0.68 a 4 años (banda al 90%: 0.59 "
           "a 0.77). "
           "Sin holgura: 0.55 a 2 años (banda al 90%: 0.40 a 0.70); débil a 4 años (F efectiva 10.8, "
           "umbral 23.1), con un conjunto al 90% robusto a instrumentos débiles de 0.40 a 0.89. "
           "Con y sin holgura, los multiplicadores difieren en 0.05 a 2 años (p = 0.80). "
           "La comparación de los multiplicadores con y sin holgura no es fiable a 4 años, porque un "
           "régimen es débil ahí. "
           "Con holgura: 1941Q4 concentra el 52% de la variación residual del instrumento a 2 años, y "
           "1942Q1 el 30% a 4 años. "
           "Sin holgura: un solo trimestre, 1950Q3, concentra el 36% de la variación residual del "
           "instrumento a 4 años.",
           "Con holgura a 2 años: 1.77 aquí frente a 0.60 en la muestra completa. "
           "Sin holgura a 4 años: 0.54 aquí, mientras que la estimación de la muestra completa es "
           "débil, con un conjunto al 90% robusto a instrumentos débiles de 0.40 a 0.89. "
           "Los valores sin la Segunda Guerra Mundial"),
}


@pytest.mark.parametrize("lang", ["en", "es"])
def test_readings_follow_the_rule_in_every_branch(computed, lang):
    record = _record_for_every_branch(computed[0])
    headline, comparisons = BRANCHES[lang]
    assert headline in fb.page_block(record, "headline_reading", lang)
    text = fb.page_block(record, "experiment_reading", lang)
    assert comparisons in text                       # strong cells compared; a weak one by its set
    assert "1961Q2" not in text and "1950Q3" not in text      # no share reaches the threshold
    table = fb.page_block(record, "headline", lang)
    empty = {"en": "empty set", "es": "conjunto vacío"}[lang]
    assert f"{empty} †" in table and "(-∞, ∞) †" in table and "0.05 (p = 0.80)" in table
    assert table.count("†") == 3 + 1                 # three weak cells and the note


@pytest.mark.parametrize("lang", ["en", "es"])
def test_experiment_blocks_say_how_to_run_it(computed, lang):
    record = {**computed[0], "experiment": None}
    for block in ("experiment", "experiment_reading"):
        assert f"--exclude {fb.WWII}" in fb.page_block(record, block, lang)


def test_nothing_is_written_when_the_record_cannot_be_rendered(computed, tmp_path):
    record = copy.deepcopy(computed[0])
    record["full"]["horizons"]["7"]["linear"]["se"] = float("nan")       # not JSON-ready
    with pytest.raises(ValueError):
        fb.write_outputs(record, computed[1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("exclude", "message"), [("1941Q3", "debe ser INICIO:FIN"),
                                                   ("0000Q1:1945Q4", "debe ser INICIO:FIN"),
                                                   ("1945Q4:1941Q3", "empieza después de terminar"),
                                                   ("1700Q1:1701Q4", "no se solapa con la muestra")])
def test_cli_explains_a_bad_exclude_in_spanish(exclude, message, tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        fb.main(["--exclude", exclude, "--lang", "es", "--output", str(out)])
    assert exc.value.code == 2 and message in capsys.readouterr().err and not out.exists()


def test_cli_prints_the_summary_and_writes_the_files(tmp_path, capsys):
    out = tmp_path / "out"
    assert fb.main(["--output", str(out), "--lang", "es"]) == 0
    printed = capsys.readouterr().out
    assert "Con holgura" in printed and f"--exclude {fb.WWII}" in printed and str(out) in printed
    assert FILES <= {p.name for p in out.iterdir()}


def test_manifest_records_the_experiment_definition_not_its_results(computed, tmp_path):
    record, paths = computed
    man = fb.write_outputs(record, paths, tmp_path / "experiment")
    assert man["experiment"] == {"exclude": ["1941Q3", "1945Q4"]}            # results live in headline.json
    full = paths[paths["sample"] == "full"]
    assert fb.write_outputs({**record, "experiment": None}, full, tmp_path / "default")["experiment"] is None


def test_headline_csv_carries_no_point_estimate_for_weak_cells(computed, tmp_path):
    record = copy.deepcopy(computed[0])
    h7 = record["full"]["horizons"]["7"]              # a strong no-slack cell and a reliable test,
    h7["no_slack"].update(weak=False, f_eff=30.0)     # so that every kind of row is checked
    h7["difference"]["reliable"] = True
    fb.write_outputs(record, computed[1], tmp_path)
    table = pd.read_csv(tmp_path / "headline.csv")
    point = ["estimate", "se", "lo", "hi"]
    for sample in ("full", "experiment"):
        for h, cells in record[sample]["horizons"].items():
            rows = table[(table["sample"] == sample) & (table["h"] == int(h))].set_index("row")
            for name in ("linear", "slack", "no_slack"):
                cell, row = cells[name], rows.loc[name]
                if cell["weak"]:                     # as in the md and tex tables: its set, not its estimate
                    assert row[point].isna().all(), (sample, h, name)
                    assert row[["ar_lo", "ar_hi", "ar_set_type", "f_eff"]].notna().all(), (sample, h, name)
                else:
                    np.testing.assert_allclose(row[point].to_numpy(dtype=float),
                                               [cell["multiplier"], cell["se"], cell["lo"], cell["hi"]],
                                               rtol=1e-12)
            test, quoted = cells["difference"], rows.loc["difference", ["estimate", "se", "p_value"]]
            if test["reliable"]:
                np.testing.assert_allclose(quoted.to_numpy(dtype=float),
                                           [test["estimate"], test["se"], test["p_value"]], rtol=1e-12)
            else:
                assert quoted.isna().all(), (sample, h)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_a_set_that_could_not_be_computed_is_called_unavailable(computed, lang):
    record = copy.deepcopy(computed[0])
    record["full"]["horizons"]["7"]["no_slack"]["ar_lo"] = None                    # bounded, a bound missing
    record["experiment"]["horizons"]["7"]["no_slack"].update(weak=False, f_eff=30.0)  # compared with it
    unavailable = {"en": "and its weak-instrument-robust 90% set is not available",
                   "es": "y su conjunto al 90% robusto a instrumentos débiles no está disponible"}[lang]
    reading = fb.page_block(record, "headline_reading", lang)
    assert unavailable in reading and "n/a" not in reading
    assert f"{unavailable}." in fb.page_block(record, "experiment_reading", lang)
    assert "| n/a † |" in fb.page_block(record, "headline", lang)
    assert "n/a$^\\dagger$" in fb._latex_page(record, lang)
