"""Focused tests for DeepEvalFaithfulnessEvaluator.

Injects a FaithfulnessMetric stub. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval import (
    DEEPEVAL_EVALUATOR_NAME,
    FAITHFULNESS_METRIC,
    DeepEvalFaithfulnessEvaluator,
)

FAITHFULNESS_INPUT = "What is 2 + 2?"
FAITHFULNESS_OUTPUT = "4"


class RecordingFaithfulnessMetric:
    def __init__(self, score=0.75, reason="The output is supported by the context.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.success = True
        self.threshold = 0.5

    def measure(self, test_case):
        self.test_case = test_case
        return self.score


def _evaluate(metric, retrieval, **kwargs):
    return DeepEvalFaithfulnessEvaluator(faithfulness_metric=metric).evaluate(
        FAITHFULNESS_INPUT,
        FAITHFULNESS_OUTPUT,
        retrieval,
        **kwargs,
    )


def test_string_retrieval_is_copied_to_retrieval_context():
    metric = RecordingFaithfulnessMetric()
    _evaluate(metric, ["context one", "context two"])
    assert metric.test_case.retrieval_context == ["context one", "context two"]


def test_page_content_retrieval_is_copied_to_retrieval_context():
    metric = RecordingFaithfulnessMetric()
    _evaluate(metric, [{"page_content": "context one"}])
    assert metric.test_case.retrieval_context == ["context one"]


def test_mixed_retrieval_keeps_strings_and_page_content():
    metric = RecordingFaithfulnessMetric()
    _evaluate(
        metric,
        [
            "context one",
            {"page_content": "context two", "file_name": "notes.docx"},
            "",
            {"file_name": "empty.docx"},
            {"page_content": ""},
            None,
        ],
    )
    assert metric.test_case.retrieval_context == ["context one", "context two"]


def test_missing_or_empty_retrieval_becomes_an_empty_list():
    missing = RecordingFaithfulnessMetric()
    _evaluate(missing, None)
    assert missing.test_case.retrieval_context == []

    empty = RecordingFaithfulnessMetric()
    _evaluate(empty, [])
    assert empty.test_case.retrieval_context == []


def test_expected_is_not_sent_on_the_deepeval_test_case():
    metric = RecordingFaithfulnessMetric()
    _evaluate(metric, ["2 + 2 = 4"])
    assert metric.test_case.expected_output is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        metric.test_case.actual_output,
        *metric.test_case.retrieval_context,
    )


def test_result_identity_is_faithfulness_from_deepeval():
    result = _evaluate(RecordingFaithfulnessMetric(), ["2 + 2 = 4"])[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == FAITHFULNESS_METRIC
    assert result.metric == "faithfulness"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"
    assert result.evaluator != "ragas"


def test_score_and_reason_are_copied_from_the_metric():
    metric = RecordingFaithfulnessMetric(score=0.75, reason="Supported by context.")
    result = _evaluate(metric, ["2 + 2 = 4"])[0]
    assert result.score == 0.75
    assert result.score is metric.score
    assert result.reason == "Supported by context."
    assert result.reason is metric.reason
    assert "success" not in result.raw_result
    assert "threshold" not in result.raw_result
    assert "passed" not in result.__dataclass_fields__


def test_adapter_source_does_not_import_ragas():
    adapter = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "evaluators" / "deepeval.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "ragas" not in imported
