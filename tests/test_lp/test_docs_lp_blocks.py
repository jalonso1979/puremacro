"""Every ```python block of docs/lp.md and docs/es/lp.md must run verbatim,
in order, sharing one namespace per page.

Regression for the 2.3.x audit: block 1 and 3 used columns no earlier
block defined (M35), block 2 called lp_state_dep with state_var/horizon/
lags and selected beta_high columns (C1, C2), block 4 called panel_lp_dk
with unit_col/time_col (C11) and the Spanish page called
panel_lp(cov_type='driscoll-kraay') (C12).
"""
import contextlib
import io
import pathlib
import re

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGES = ["docs/lp.md", "docs/es/lp.md"]


def _blocks(page: str) -> list[str]:
    text = (ROOT / page).read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("page", PAGES)
def test_doc_page_runs_top_to_bottom(page):
    blocks = _blocks(page)
    assert len(blocks) >= 4, f"{page}: expected the tutorial blocks, found {len(blocks)}"
    ns: dict = {"__name__": "__doc__"}
    for i, src in enumerate(blocks):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"{page}:block{i}", "exec"), ns)
        except Exception as exc:  # pragma: no cover - the message is the diagnostic
            pytest.fail(f"{page} block {i} failed: {type(exc).__name__}: {exc}")
        finally:
            plt.close("all")
    # The pages promise the state-dependent and panel examples ran
    assert "res_panel" in ns
    assert "res_state" in ns or "res_regime" in ns
