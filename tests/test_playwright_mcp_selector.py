"""Focused tests for the provider-neutral SequenceToolSelector."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.integrations.playwright_mcp_selector import SequenceToolSelector

_SELECTOR_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "integrations"
    / "playwright_mcp_selector.py"
)

_STEPS = [
    ("browser_navigate", {"url": "https://demo.playwright.dev/todomvc"}),
    ("browser_snapshot", {}),
    ("browser_type", {"text": "Buy milk", "submit": True}),
]


def test_selects_the_next_tool_and_preserves_arguments():
    first_args = _STEPS[0][1]
    selector = SequenceToolSelector(_STEPS)

    name, arguments = selector.select_next("Add Buy milk", [], None)

    assert name == "browser_navigate"
    assert arguments == {"url": "https://demo.playwright.dev/todomvc"}
    assert arguments is first_args


def test_progresses_through_the_sequence_in_order():
    selector = SequenceToolSelector(_STEPS)

    names = []
    for _ in range(len(_STEPS)):
        selected = selector.select_next("goal", names, None)
        assert selected is not None
        names.append(selected[0])

    assert names == ["browser_navigate", "browser_snapshot", "browser_type"]
    assert selector.index == 3


def test_returns_none_after_the_sequence_is_exhausted():
    selector = SequenceToolSelector(_STEPS[:1])

    assert selector.select_next("goal", [], None) == _STEPS[0]
    assert selector.select_next("goal", ["observed"], {"isError": False}) is None
    assert selector.select_next("goal", ["observed"], {"isError": False}) is None


def test_arguments_are_passed_unchanged_including_empty_dict():
    empty = {}
    selector = SequenceToolSelector([("browser_snapshot", empty)])

    _name, arguments = selector.select_next("goal", [], None)

    assert arguments is empty
    assert arguments == {}


def test_selector_module_does_not_import_evaluation_or_vendor_stack():
    tree = ast.parse(_SELECTOR_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ai_qe_eval",
        "deepeval",
        "mcp",
        "langgraph",
        "langfuse",
        "openai",
        "ragas",
    }
    assert forbidden.isdisjoint(imported_roots)
