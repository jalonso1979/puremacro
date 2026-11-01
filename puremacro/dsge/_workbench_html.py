"""Small dependency-free notebook/report styles, scoped to workbench output."""
from html import escape

STYLE = """<style>
.pm-dsge { font: 14px/1.5 system-ui, sans-serif; color: #233142; background: #fafbf9;
  border: 1px solid #d8e0df; border-radius: 12px; padding: 22px; max-width: 1100px; }
.pm-dsge * { box-sizing: border-box; }
.pm-dsge h2 { font-size: 24px; margin: 0 0 8px; }
.pm-dsge h3 { font-size: 17px; margin: 20px 0 10px; }
.pm-dsge p { margin: 8px 0; }
.pm-dsge .pm-meta { color: #536775; font-size: 12px; }
.pm-dsge .pm-scroll { overflow-x: auto; }
.pm-dsge table { border-collapse: collapse; width: 100%; font: inherit; text-align: left; }
.pm-dsge th { color: #536775; font-size: 12px; font-weight: 600; }
.pm-dsge th, .pm-dsge td { padding: 9px 8px; border-bottom: 1px solid #dfe5e4; text-align: left; vertical-align: top; }
.pm-dsge tr:last-child td { border-bottom: 0; }
.pm-dsge summary { cursor: pointer; }
.pm-dsge details { margin: 8px 0; }
.pm-dsge td details { margin: 0; min-width: 170px; max-width: 370px; }
.pm-dsge pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px; background: #eef2f1; border-radius: 6px; }
.pm-dsge .pm-badge { display: inline-block; white-space: nowrap; padding: 2px 8px; border-radius: 5px; background: #e9edee; color: #3e505c; font-size: 12px; }
.pm-dsge .pm-passed { background: #e0f1e6; color: #21573a; }
.pm-dsge .pm-failed { background: #fae6e4; color: #8c2c28; }
.pm-dsge .pm-reported { background: #f6edcf; color: #73530b; }
.pm-dsge .pm-note { border-left: 3px solid #d4a540; padding-left: 12px; }
.pm-dsge .pm-run { margin: 20px 0; padding-top: 16px; border-top: 1px solid #d8e0df; }
</style>"""


def badge(status: str) -> str:
    known = status if status in {"passed", "failed", "reported"} else "neutral"
    return f'<span class="pm-badge pm-{known}">{escape(status.replace("_", " "))}</span>'


def panel(body: str) -> str:
    return STYLE + '<section class="pm-dsge">' + body + '</section>'


def document(title: str, fragment: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'
            + escape(title) + '</title></head><body style="margin:24px">' + fragment + '</body></html>')
