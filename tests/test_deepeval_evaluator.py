"""Focused tests for P2-10 DeepEval GEval correctness evaluator.

Injects a GEval stub. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval import (
    CORRECTNESS_METRIC,
    DEEPEVAL_EVALUATOR_NAME,
    DeepEvalGEvalCorrectnessEvaluator,
)

GEVAL_INPUT = "How many articles are there in the Selenium webdriver python course?"
GEVAL_OUTPUT = "There are 23 articles."
GEVAL_EXPECTED = "23"


class RecordingGEvalMetric:
    def __init__(self, score=0.91, reason="The actual output matches the expected output.") -> None:
        self.score = score
        self.reason = reason
        self.name = "Correctness"
        self.criteria = "Determine whether the actual output is factually correct based on the expected output."
        self.evaluation_steps = None
        self.test_case = None
        self.threshold = 0.5
        self.success = True

    def measure(self, test_case):
        self.test_case = test_case
        return self.score


def _evaluate(metric=None, **kwargs):
    if metric is None:
        metric = RecordingGEvalMetric()
    return DeepEvalGEvalCorrectnessEvaluator(geval_metric=metric).evaluate(
        GEVAL_INPUT,
        GEVAL_OUTPUT,
        GEVAL_EXPECTED,
        **kwargs,
    )


def test_evaluate_returns_one_evaluation_result():
    results = _evaluate()
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], EvaluationResult)


def test_metric_and_evaluator_identity():
    result = _evaluate()[0]
    assert result.metric == CORRECTNESS_METRIC
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator != "geval"


def test_arguments_map_to_deepeval_llm_test_case_fields():
    metric = RecordingGEvalMetric()
    _evaluate(metric)
    assert metric.test_case is not None
    assert metric.test_case.input == GEVAL_INPUT
    assert metric.test_case.actual_output == GEVAL_OUTPUT
    assert metric.test_case.expected_output == GEVAL_EXPECTED


def test_score_is_preserved_exactly():
    metric = RecordingGEvalMetric(score=0.914)
    result = _evaluate(metric)[0]
    assert result.score == 0.914
    assert result.score is metric.score


def test_reason_is_preserved_from_deepeval():
    metric = RecordingGEvalMetric(reason="Matches the reference count of 23.")
    result = _evaluate(metric)[0]
    assert result.reason == "Matches the reference count of 23."


def test_raw_result_preserves_deepeval_information_without_policy_fields():
    metric = RecordingGEvalMetric(score=0.91, reason="Correct.")
    result = _evaluate(metric)[0]
    assert result.raw_result["score"] == 0.91
    assert result.raw_result["reason"] == "Correct."
    assert result.raw_result["name"] == "Correctness"
    assert result.raw_result["input"].startswith("How many articles")
    assert result.raw_result["actual_output"] == "There are 23 articles."
    assert result.raw_result["expected_output"] == "23"
    assert "threshold" not in result.raw_result
    assert "passed" not in result.raw_result
    assert "severity" not in result.raw_result
    assert "success" not in result.raw_result
    assert "threshold" not in result.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__
    assert "severity" not in result.__dataclass_fields__


def test_evaluate_accepts_optional_configuration():
    metric = RecordingGEvalMetric(score=0.7)
    evaluator = DeepEvalGEvalCorrectnessEvaluator(geval_metric=metric)
    without_config = evaluator.evaluate(GEVAL_INPUT, GEVAL_OUTPUT, GEVAL_EXPECTED)
    with_none = evaluator.evaluate(GEVAL_INPUT, GEVAL_OUTPUT, GEVAL_EXPECTED, None)
    with_ignored = evaluator.evaluate(
        GEVAL_INPUT, GEVAL_OUTPUT, GEVAL_EXPECTED, {"threshold": 0.99}
    )
    assert without_config[0].score == 0.7
    assert with_none[0].score == 0.7
    assert with_ignored[0].score == 0.7
    assert "threshold" not in with_ignored[0].__dataclass_fields__


def test_domain_package_does_not_import_deepeval():
    domain_dir = Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain"
    forbidden = {"deepeval", "ragas", "langchain", "langchain_openai", "openai"}
    for path in domain_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert forbidden.isdisjoint(imported), path.name
