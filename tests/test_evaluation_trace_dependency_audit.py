"""F3-04: lock that EvaluationTrace and its builders are gone from production."""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval"

_REMOVED_BUILDERS = {
    "build_langfuse_evaluation_trace",
    "build_mcp_evaluation_trace",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def _source_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def test_evaluation_trace_module_is_removed():
    assert not (_SRC_ROOT / "domain" / "trace.py").exists()


def test_no_production_module_imports_evaluation_trace():
    offenders: list[str] = []
    for path in _source_files():
        imported = _imported_modules(path)
        names = _imported_names(path)
        if "ai_qe_eval.domain.trace" in imported or "EvaluationTrace" in names:
            offenders.append(str(path.relative_to(_SRC_ROOT.parent)))
    assert offenders == []


def test_deleted_builders_are_not_imported_or_defined():
    offenders: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        names = _imported_names(path)
        if names & _REMOVED_BUILDERS:
            offenders.append(f"import:{path.name}")
        for builder in _REMOVED_BUILDERS:
            if f"def {builder}(" in text:
                offenders.append(f"def:{path.name}:{builder}")
    assert offenders == []


def test_active_request_helpers_remain():
    mcp = (_SRC_ROOT / "capture" / "mcp_trace.py").read_text(encoding="utf-8")
    langfuse = (_SRC_ROOT / "capture" / "langfuse_trace.py").read_text(encoding="utf-8")
    assert "def mcp_p0_request(" in mcp
    assert "def langfuse_tool_correctness_request(" in langfuse
