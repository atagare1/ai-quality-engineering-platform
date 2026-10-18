"""Focused tests for P2-09 RAGAS Faithfulness evaluator.

Mocks the RAGAS metric boundary. Does not call Together or the RAG API.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)

RAGAS_INPUT = "How many articles are there?"
RAGAS_OUTPUT = "There are 23 articles."
RAGAS_RETRIEVAL = [
    {"file_name": "course.docx", "page_content": "23 articles"},
    {"file_name": "outline.docx", "page_content": "Selenium WebDriver Python"},
]


class RecordingFaithfulnessMetric:
    def __init__(self, score=0.92) -> None:
        self.score = score
        self.sample = None

    def single_turn_score(self, sample):
        self.sample = sample
        return self.score


class FailingFaithfulnessMetric:
    def single_turn_score(self, sample):
        raise RuntimeError("RAGAS faithfulness execution failed")


_UNSET = object()


def _evaluate(metric, retrieval=_UNSET, **kwargs):
    if retrieval is _UNSET:
        retrieval = RAGAS_RETRIEVAL
    return RAGASFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(
        RAGAS_INPUT,
        RAGAS_OUTPUT,
        retrieval,
        **kwargs,
    )


def test_arguments_map_to_response_and_retrieved_contexts():
    metric = RecordingFaithfulnessMetric()
    _evaluate(metric)
    assert metric.sample is not None
    assert metric.sample.user_input == RAGAS_INPUT
    assert metric.sample.response == RAGAS_OUTPUT
    assert metric.sample.retrieved_contexts == [
        "23 articles",
        "Selenium WebDriver Python",
    ]
    assert RAGAS_RETRIEVAL[0]["page_content"] == "23 articles"


def test_successful_ragas_score_is_preserved_exactly():
    metric = RecordingFaithfulnessMetric(score=0.92)
    results = _evaluate(metric)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert result.score == 0.92
    assert result.score is metric.score


def test_score_is_not_rounded_clamped_or_thresholded():
    metric = RecordingFaithfulnessMetric(score=0.823456789)
    result = _evaluate(metric)[0]
    assert result.score == 0.823456789
    assert "threshold" not in result.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__
    assert "severity" not in result.__dataclass_fields__
    assert "quality_policy" not in result.__dataclass_fields__
    assert "threshold" not in result.raw_result
    assert "passed" not in result.raw_result


def test_raw_result_retains_ragas_inputs_and_score():
    metric = RecordingFaithfulnessMetric(score=0.92)
    result = _evaluate(metric)[0]
    assert result.raw_result["score"] == 0.92
    assert result.raw_result["response"] == RAGAS_OUTPUT
    assert result.raw_result["retrieved_contexts"] == [
        "23 articles",
        "Selenium WebDriver Python",
    ]
    assert result.raw_result["user_input"] == RAGAS_INPUT
    assert result.reason == "RAGAS faithfulness score=0.92."


def test_evaluate_with_configuration_none():
    metric = RecordingFaithfulnessMetric(score=0.5)
    results = _evaluate(metric, configuration=None)
    assert len(results) == 1
    assert results[0].score == 0.5


def test_ragas_errors_propagate():
    evaluator = RAGASFaithfulnessEvaluator(
        faithfulness_metric=FailingFaithfulnessMetric()
    )
    try:
        evaluator.evaluate(RAGAS_INPUT, RAGAS_OUTPUT, RAGAS_RETRIEVAL)
    except RuntimeError as exc:
        assert "RAGAS faithfulness execution failed" in str(exc)
    else:
        raise AssertionError("expected RAGAS execution error to propagate")


def test_string_retrieval_is_passed_through():
    metric = RecordingFaithfulnessMetric()
    _evaluate(metric, retrieval=["ctx-1", "ctx-2"])
    assert metric.sample.retrieved_contexts == ["ctx-1", "ctx-2"]


def test_missing_or_empty_retrieval_becomes_an_empty_list():
    missing = RecordingFaithfulnessMetric()
    _evaluate(missing, retrieval=None)
    assert missing.sample.retrieved_contexts == []

    empty = RecordingFaithfulnessMetric()
    _evaluate(empty, retrieval=[])
    assert empty.sample.retrieved_contexts == []


def test_domain_package_does_not_import_ragas():
    domain_dir = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain"
    forbidden = {"ragas", "langchain", "langchain_openai", "openai"}
    for path in domain_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert forbidden.isdisjoint(imported), path.name


def test_adapter_does_not_import_evaluation_trace():
    adapter = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ai_qe_eval"
        / "evaluators"
        / "ragas.py"
    )
    tree = ast.parse(adapter.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "ai_qe_eval.domain.trace" not in imported
    assert not any(name.endswith(".trace") for name in imported)
