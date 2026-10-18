"""Focused tests for P2-03 EvaluationRun.

Does not import RAGAS, LangChain, HTTP helpers, or Phase 1 mapping.
Stores request maps, not EvaluationTrace.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.run import EvaluationRun

_RUN_SOURCE = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "run.py"


def _request(*pairs: tuple[str, tuple]) -> dict:
    return {name: {"args": list(args), "kwargs": {}} for name, args in pairs}


def test_evaluation_run_can_be_created_with_run_id():
    run = EvaluationRun(run_id="run-001")
    assert run.run_id == "run-001"


def test_evaluation_run_can_contain_zero_requests():
    run = EvaluationRun(run_id="run-001", requests=[])
    assert run.requests == []
    defaulted = EvaluationRun(run_id="run-002")
    assert defaulted.requests == []


def test_evaluation_run_can_contain_one_request():
    request = _request(("exact_match", ("answer-1", "expected-1")))
    run = EvaluationRun(run_id="run-001", requests=[request])
    assert len(run.requests) == 1
    assert run.requests[0] is request
    assert run.requests[0]["exact_match"]["args"] == ["answer-1", "expected-1"]


def test_evaluation_run_can_contain_multiple_requests_in_order():
    first = _request(("exact_match", ("answer-1", "expected-1")))
    second = _request(("exact_match", ("answer-2", "expected-2")))
    run = EvaluationRun(run_id="run-001", requests=[first, second])
    assert run.run_id == "run-001"
    assert len(run.requests) == 2
    assert run.requests[0] is first
    assert run.requests[1] is second
    assert [item["exact_match"]["args"][0] for item in run.requests] == [
        "answer-1",
        "answer-2",
    ]


def test_grouped_requests_retain_identity_and_values():
    request = {
        "faithfulness": {
            "args": ["question-1", "answer-1", [{"page_content": "chunk"}]],
            "kwargs": {},
        }
    }
    run = EvaluationRun(run_id="run-001", requests=[request])
    grouped = run.requests[0]
    assert grouped is request
    assert grouped["faithfulness"]["args"][0] == "question-1"
    assert grouped["faithfulness"]["args"][1] == "answer-1"
    assert grouped["faithfulness"]["args"][2] == [{"page_content": "chunk"}]


def test_serialization_preserves_nested_request_maps():
    requests = [
        _request(("exact_match", ("answer-1", "expected-1"))),
        {
            "tool_correctness": {
                "args": [["observed"], ["expected"]],
                "kwargs": {"input": "question-2"},
            }
        },
    ]
    run = EvaluationRun(run_id="run-001", requests=requests)
    restored = EvaluationRun.from_dict(run.to_dict())
    assert restored.run_id == "run-001"
    assert restored.requests[0]["exact_match"]["args"] == ["answer-1", "expected-1"]
    assert restored.requests[1]["tool_correctness"]["args"] == [["observed"], ["expected"]]
    assert restored.requests[1]["tool_correctness"]["kwargs"] == {"input": "question-2"}


def test_default_request_lists_are_not_shared_across_runs():
    run_a = EvaluationRun(run_id="run-a")
    run_b = EvaluationRun(run_id="run-b")
    run_a.requests.append(_request(("exact_match", ("answer-1",))))
    assert run_a.requests[0]["exact_match"]["args"] == ["answer-1"]
    assert run_b.requests == []
    assert run_a.requests is not run_b.requests


def test_run_does_not_duplicate_request_payload_fields():
    payload = EvaluationRun(
        run_id="run-001",
        requests=[_request(("exact_match", ("answer", "expected")))],
    ).to_dict()
    assert "args" not in payload
    assert "kwargs" not in payload
    assert payload["requests"][0]["exact_match"]["args"] == ["answer", "expected"]


def test_run_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_RUN_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            imported.update(names)
            imported_roots.update(name.split(".", 1)[0] for name in names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ragas",
        "deepeval",
        "langchain",
        "langchain_openai",
        "requests",
        "httpx",
        "pytest",
        "mcp",
        "langfuse",
        "langgraph",
        "opentelemetry",
        "uuid",
    }
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.domain.trace" not in imported
