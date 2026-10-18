"""Focused tests for RAG demo capture request maps.

Does not call the live RAG API, RAGAS, DeepEval, Runner, Policy, or Gate.
Does not change Phase 1 tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.capture.rag_demo import live_rag_demo_request, rag_demo_request

_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "capture"
    / "rag_demo.py"
)

QUESTION = "How many articles are there in the Selenium webdriver python course?"
ANSWER = "There are 23 articles."
CONTEXTS = ["The course contains 23 articles."]
REFERENCE = "23"


def test_rag_demo_request_maps_supported_evaluator_args():
    request = rag_demo_request(
        user_input=QUESTION,
        response=ANSWER,
        retrieved_contexts=CONTEXTS,
        reference=REFERENCE,
    )
    assert request["faithfulness"]["args"] == [QUESTION, ANSWER, CONTEXTS]
    assert request["answer_relevancy"]["args"] == [QUESTION, ANSWER]
    assert request["contextual_relevancy"]["args"] == [QUESTION, CONTEXTS]
    assert request["hallucination"]["args"] == [QUESTION, ANSWER, CONTEXTS]
    assert request["contextual_precision"]["args"] == [QUESTION, REFERENCE, CONTEXTS]
    assert request["contextual_recall"]["args"] == [QUESTION, REFERENCE, CONTEXTS]
    assert request["correctness"]["args"] == [QUESTION, ANSWER, REFERENCE]
    assert "kwargs" not in request["faithfulness"]


def test_rag_demo_request_omits_reference_capabilities_without_gold():
    request = rag_demo_request(
        user_input=QUESTION,
        response=ANSWER,
        retrieved_contexts=CONTEXTS,
    )
    assert set(request) == {
        "faithfulness",
        "answer_relevancy",
        "contextual_relevancy",
        "hallucination",
    }


def test_rag_demo_request_copies_retrieval_list():
    contexts = ["ctx-1"]
    request = rag_demo_request(
        user_input=QUESTION,
        response=ANSWER,
        retrieved_contexts=contexts,
        evaluations=["faithfulness"],
    )
    mapped = request["faithfulness"]["args"][2]
    assert mapped == ["ctx-1"]
    assert mapped is not contexts


def test_rag_demo_request_rejects_unknown_evaluation():
    with pytest.raises(ValueError, match="Unsupported RAG demo evaluation"):
        rag_demo_request(
            user_input=QUESTION,
            response=ANSWER,
            retrieved_contexts=CONTEXTS,
            evaluations=["tool_correctness"],
        )


def test_rag_demo_request_rejects_reference_capability_without_gold():
    with pytest.raises(ValueError, match="require reference"):
        rag_demo_request(
            user_input=QUESTION,
            response=ANSWER,
            retrieved_contexts=CONTEXTS,
            evaluations=["correctness"],
        )


def test_live_rag_demo_request_uses_injected_response_without_http(monkeypatch):
    def fail_http(_passed):
        raise AssertionError("get_api_response should not be called")

    monkeypatch.setattr(
        "ai_qe_eval.capture.rag_demo.get_api_response",
        fail_http,
    )
    request = live_rag_demo_request(
        QUESTION,
        reference=REFERENCE,
        evaluations=["faithfulness", "correctness"],
        response_data={
            "answer": ANSWER,
            "retrieved_docs": [{"page_content": CONTEXTS[0]}],
        },
    )
    assert request["faithfulness"]["args"] == [QUESTION, ANSWER, CONTEXTS]
    assert request["correctness"]["args"] == [QUESTION, ANSWER, REFERENCE]


def test_live_rag_demo_request_calls_existing_client(monkeypatch):
    seen = {}

    def fake_get_api_response(passed_data):
        seen["passed"] = passed_data
        return {
            "answer": ANSWER,
            "retrieved_docs": [{"page_content": CONTEXTS[0]}],
        }

    monkeypatch.setattr(
        "ai_qe_eval.capture.rag_demo.get_api_response",
        fake_get_api_response,
    )
    request = live_rag_demo_request(
        QUESTION,
        reference=REFERENCE,
        evaluations=["faithfulness"],
    )
    assert seen["passed"]["question"] == QUESTION
    assert request["faithfulness"]["args"][1] == ANSWER
    assert request["faithfulness"]["args"][2] == CONTEXTS


def test_live_rag_demo_request_rejects_empty_question():
    with pytest.raises(ValueError, match="non-empty question"):
        live_rag_demo_request("   ")


def test_rag_demo_capture_does_not_import_evaluators_or_runner():
    tree = ast.parse(_CAPTURE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "deepeval" not in imported
    assert "ragas" not in imported
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "ai_qe_eval.runner.evaluation_runner" not in modules
    assert "ai_qe_eval.evaluators.ragas" not in modules
    assert "ai_qe_eval.evaluators.deepeval" not in modules
