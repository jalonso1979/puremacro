"""Independent author-code reference for production and capital valuation."""
import hashlib
import json
from pathlib import Path

import pytest

from puremacro.models.hank_capital import capital_production_block

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "tests/fixtures/hank_capital_reference"
RECORD = json.loads((FOLDER / "record.json").read_text())


def test_official_capital_reference_sources_and_generator_are_bound():
    assert RECORD["reference_commit"] == "eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9"
    for name, expected in RECORD["source_sha256"].items():
        assert hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == expected
    generator = ROOT / "tools/reference_validation/hank_capital_reference.py"
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == RECORD["generator_sha256"]
    assert len(RECORD["cases"]) == 25


@pytest.mark.pyodide_smoke
@pytest.mark.parametrize("case", RECORD["cases"], ids=lambda c: c["name"])
def test_production_and_realized_returns_match_author_blocks(case):
    reference = case["reference"]
    native = capital_production_block(**case["inputs"])
    for field in ("N", "mc", "Knext", "r_a", "adj", "investment_goods"):
        assert native[field] == pytest.approx(reference[field], abs=2e-12, rel=1e-12), field
    assert abs(reference["investment_foc"]) < 2e-12
    assert abs(reference["valuation_foc"]) < 2e-12
    # The reference firm retains rentals and pays for investment. Our direct
    # capital owners receive that part of its payoff separately.
    dividend = native["div"] + native["rental"]*native["K"] - native["investment_goods"]
    assert dividend == pytest.approx(reference["dividend"], abs=2e-12, rel=1e-12)
    assert (1+native["r_a"])*native["lag_q"]*native["K"] == pytest.approx(
        native["rental"]*native["K"] - native["investment_goods"] + native["q"]*native["Knext"],
        abs=2e-12, rel=1e-12)


def test_published_partial_calibration_is_exact_and_distinct_from_rounded_defaults():
    published = RECORD["published_calibration"]
    partial = RECORD["published_partial_ss"]
    assert published["K"] == 10. and published["r"] == .0125
    assert published["delta"] == .02 and published["epsI"] == 4.
    assert partial["alpha"] == pytest.approx(.325/.985, abs=1e-15)
    assert partial["mu"] == pytest.approx(1/.985, abs=1e-15)
    steady = capital_production_block(**RECORD["cases"][0]["inputs"])
    assert steady["N"] == pytest.approx(1., abs=1e-14)
    assert steady["r_a"] == pytest.approx(published["r"], abs=1e-14)
    assert steady["Knext"] == pytest.approx(published["K"], abs=1e-14)
    assert steady["investment_goods"] == pytest.approx(.2, abs=1e-14)
