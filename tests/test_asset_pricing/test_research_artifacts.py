"""Executed notebook freshness and English/Spanish computational parity."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_executed_notebooks_match_their_sources():
    for suffix in ("", "_es"):
        source = ROOT / "notebooks" / f"investment_research{suffix}.py"
        artifact = json.loads(source.with_suffix(".ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in artifact["cells"] if cell["cell_type"] == "code"]
        code = "\n".join("".join(cell["source"]) for cell in cells)
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(source.read_text(encoding="utf-8")))
        assert artifact["metadata"]["investment_research"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert all(cell["execution_count"] is not None for cell in cells)
        outputs = [output for cell in cells for output in cell["outputs"]]
        assert not any(output["output_type"] == "error" for output in outputs)
        assert sum("image/png" in output.get("data", {}) for output in outputs) == 2


class _WithoutDisplayLanguage(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("set_ylabel", "set_xlabel", "set_title"):
            node.args = [ast.Constant("translated label")]
        for keyword in node.keywords:
            if keyword.arg in ("title", "xlabel", "ylabel"):
                keyword.value = ast.Constant("translated label")
        return node


def test_bilingual_notebooks_have_the_same_computations():
    trees = []
    for suffix in ("", "_es"):
        code = (ROOT / "notebooks" / f"investment_research{suffix}.py").read_text(encoding="utf-8")
        trees.append(ast.dump(_WithoutDisplayLanguage().visit(ast.parse(code))))
    assert trees[0] == trees[1]
