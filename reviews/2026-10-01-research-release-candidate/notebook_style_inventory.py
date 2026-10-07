"""Read-only inventory of the notebook style contract's exact source patterns."""
from datetime import datetime, timezone
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
paths = sorted(path for path in (ROOT / "notebooks").glob("*.py") if not path.name.startswith("_"))
issues = []
syntax = []
colors = {"color", "c", "edgecolor", "facecolor", "colors", "ec", "fc", "fillcolor", "linecolor"}
grayscale = re.compile(r"^0?\.\d+$|^1\.0+$")
for path in paths:
    source = path.read_text()
    for number, line in enumerate(source.splitlines(), 1):
        for label, pattern in (("show", r"\bplt\.show\s*\("), ("tight_layout", r"\bplt\.tight_layout\s*\(")):
            if re.search(pattern, line):
                issues.append({"path": str(path.relative_to(ROOT)), "line": number,
                               "issue": label, "text": line.strip()})
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        syntax.append({"path": str(path.relative_to(ROOT)), "line": exc.lineno, "error": str(exc)})
        continue
    for node in ast.walk(tree):
        if (isinstance(node, ast.keyword) and node.arg in colors
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                and grayscale.match(node.value.value)):
            issues.append({"path": str(path.relative_to(ROOT)), "line": node.lineno,
                           "issue": "literal_grayscale", "keyword": node.arg, "value": node.value.value})
result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "showcase_sources": len(paths),
          "syntax_issues": syntax, "style_issues": issues,
          "note": "Read-only inventory. Course-subdirectory files are outside this particular existing test's scope."}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"sources": len(paths), "syntax_issues": len(syntax),
                  "style_counts": {kind: sum(item["issue"] == kind for item in issues)
                                   for kind in ("show", "tight_layout", "literal_grayscale")},
                  "affected_sources": len({item["path"] for item in issues})}, indent=2))
