"""Focused tests for DeepEvalAnswerRelevancyEvaluator.

Injects an AnswerRelevancyMetric stub. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval import (
    ANSWER_RELEVANCY_METRIC,
    DEEPEVAL_EVALUATOR_NAME,
    DeepEvalAnswerRelevancyEvaluator,
)

AR_INPUT = "What is 2 + 2?"
AR_OUTPUT = "4"


class RecordingAnswerRelevancyMetric:
    def __init__(self, score=0.82, reason="The output addresses the question.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.success = True
        self.threshold = 0.5

    def measure(self, test_case):
        self.test_case = test_case
        return self.score


def test_input_and_output_map_to_the_deepeval_test_case():
    metric = RecordingAnswerRelevancyMetric()
    DeepEvalAnswerRelevancyEvaluator(answer_relevancy_metric=metric).evaluate(
        AR_INPUT, AR_OUTPUT
    )
    assert metric.test_case.input == AR_INPUT
    assert metric.test_case.actual_output == AR_OUTPUT
    assert metric.test_case.expected_output is None
    assert metric.test_case.retrieval_context is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        metric.test_case.actual_output,
    )


def test_empty_output_is_passed_through():
    metric = RecordingAnswerRelevancyMetric()
    DeepEvalAnswerRelevancyEvaluator(answer_relevancy_metric=metric).evaluate(
        AR_INPUT, ""
    )
    assert metric.test_case.actual_output == ""


def test_result_identity_is_answer_relevancy_from_deepeval():
    result = DeepEvalAnswerRelevancyEvaluator(
        answer_relevancy_metric=RecordingAnswerRelevancyMetric()
    ).evaluate(AR_INPUT, AR_OUTPUT)[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == ANSWER_RELEVANCY_METRIC
    assert result.metric == "answer_relevancy"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"


def test_score_and_reason_are_copied_from_the_metric():
    metric = RecordingAnswerRelevancyMetric(
        score=0.82,
        reason="The output addresses the question.",
    )
    result = DeepEvalAnswerRelevancyEvaluator(answer_relevancy_metric=metric).evaluate(
        AR_INPUT, AR_OUTPUT
    )[0]
    assert result.score == 0.82
    assert result.score is metric.score
    assert result.reason == "The output addresses the question."
    assert result.reason is metric.reason
    assert "success" not in result.raw_result
    assert "threshold" not in result.raw_result
    assert "passed" not in result.__dataclass_fields__
