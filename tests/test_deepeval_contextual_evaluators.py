"""Focused tests for the DeepEval contextual RAG adapters.

Injects metric stubs. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval import (
    CONTEXTUAL_PRECISION_METRIC,
    CONTEXTUAL_RECALL_METRIC,
    CONTEXTUAL_RELEVANCY_METRIC,
    DEEPEVAL_EVALUATOR_NAME,
    DeepEvalContextualPrecisionEvaluator,
    DeepEvalContextualRecallEvaluator,
    DeepEvalContextualRelevancyEvaluator,
)

CTX_INPUT = "What is 2 + 2?"
CTX_EXPECTED = "4"
CTX_RETRIEVAL = ["2 + 2 = 4"]


class SentinelModel(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return "sentinel"

    def load_model(self):
        return self

    def generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("unit test must not call the model")

    async def a_generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("unit test must not call the model")


class RecordingContextualMetric:
    def __init__(self, score=0.8, reason="The context supports the case.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.success = True
        self.threshold = 0.5

    def measure(self, test_case):
        self.test_case = test_case
        return self.score


def _mixed_retrieval():
    return [
        "context one",
        {"page_content": "context two", "file_name": "notes.docx"},
        "",
        {"file_name": "empty.docx"},
        {"page_content": ""},
        None,
    ]


def _assert_result(result, metric_name, metric):
    assert isinstance(result, EvaluationResult)
    assert result.metric == metric_name
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"
    assert result.score == metric.score
    assert result.score is metric.score
    assert result.reason == metric.reason
    assert result.reason is metric.reason
    assert "success" not in result.raw_result
    assert "threshold" not in result.raw_result
    assert "passed" not in result.__dataclass_fields__


def test_contextual_relevancy_maps_input_and_retrieval_only():
    metric = RecordingContextualMetric()
    DeepEvalContextualRelevancyEvaluator(contextual_relevancy_metric=metric).evaluate(
        CTX_INPUT, _mixed_retrieval()
    )
    assert metric.test_case.input == CTX_INPUT
    assert metric.test_case.retrieval_context == ["context one", "context two"]
    assert metric.test_case.actual_output is None
    assert metric.test_case.expected_output is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        *metric.test_case.retrieval_context,
    )


def test_contextual_relevancy_missing_retrieval_is_an_empty_list():
    metric = RecordingContextualMetric()
    DeepEvalContextualRelevancyEvaluator(contextual_relevancy_metric=metric).evaluate(
        CTX_INPUT, None
    )
    assert metric.test_case.retrieval_context == []


def test_contextual_relevancy_copies_score_reason_and_identity():
    metric = RecordingContextualMetric(score=0.66, reason="Relevant statements.")
    result = DeepEvalContextualRelevancyEvaluator(
        contextual_relevancy_metric=metric
    ).evaluate(CTX_INPUT, CTX_RETRIEVAL)[0]
    _assert_result(result, CONTEXTUAL_RELEVANCY_METRIC, metric)
    assert result.metric == "contextual_relevancy"


def test_contextual_relevancy_passes_the_injected_model():
    model = SentinelModel()
    metric = DeepEvalContextualRelevancyEvaluator(model=model)._metric()
    assert metric.model is model


def test_contextual_precision_maps_input_expected_and_retrieval():
    metric = RecordingContextualMetric()
    DeepEvalContextualPrecisionEvaluator(contextual_precision_metric=metric).evaluate(
        CTX_INPUT, CTX_EXPECTED, _mixed_retrieval()
    )
    assert metric.test_case.input == CTX_INPUT
    assert metric.test_case.expected_output == CTX_EXPECTED
    assert metric.test_case.retrieval_context == ["context one", "context two"]
    assert metric.test_case.actual_output is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        metric.test_case.expected_output,
        *metric.test_case.retrieval_context,
    )


def test_contextual_precision_empty_expected_and_retrieval_are_passed_through():
    metric = RecordingContextualMetric()
    DeepEvalContextualPrecisionEvaluator(contextual_precision_metric=metric).evaluate(
        CTX_INPUT, "", []
    )
    assert metric.test_case.expected_output == ""
    assert metric.test_case.retrieval_context == []
    assert metric.test_case.actual_output is None


def test_contextual_precision_copies_score_reason_and_identity():
    metric = RecordingContextualMetric(score=0.91, reason="Useful context is ranked first.")
    result = DeepEvalContextualPrecisionEvaluator(
        contextual_precision_metric=metric
    ).evaluate(CTX_INPUT, CTX_EXPECTED, CTX_RETRIEVAL)[0]
    _assert_result(result, CONTEXTUAL_PRECISION_METRIC, metric)
    assert result.metric == "contextual_precision"


def test_contextual_precision_passes_the_injected_model():
    model = SentinelModel()
    metric = DeepEvalContextualPrecisionEvaluator(model=model)._metric()
    assert metric.model is model


def test_contextual_recall_maps_input_expected_and_retrieval():
    metric = RecordingContextualMetric()
    DeepEvalContextualRecallEvaluator(contextual_recall_metric=metric).evaluate(
        CTX_INPUT, CTX_EXPECTED, _mixed_retrieval()
    )
    assert metric.test_case.input == CTX_INPUT
    assert metric.test_case.expected_output == CTX_EXPECTED
    assert metric.test_case.retrieval_context == ["context one", "context two"]
    assert metric.test_case.actual_output is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        metric.test_case.expected_output,
        *metric.test_case.retrieval_context,
    )


def test_contextual_recall_empty_expected_and_missing_retrieval_are_passed_through():
    metric = RecordingContextualMetric()
    DeepEvalContextualRecallEvaluator(contextual_recall_metric=metric).evaluate(
        CTX_INPUT, "", None
    )
    assert metric.test_case.input == CTX_INPUT
    assert metric.test_case.expected_output == ""
    assert metric.test_case.retrieval_context == []
    assert metric.test_case.actual_output is None


def test_contextual_recall_copies_score_reason_and_identity():
    metric = RecordingContextualMetric(score=1.0, reason="The expected answer is in context.")
    result = DeepEvalContextualRecallEvaluator(
        contextual_recall_metric=metric
    ).evaluate(CTX_INPUT, CTX_EXPECTED, CTX_RETRIEVAL)[0]
    _assert_result(result, CONTEXTUAL_RECALL_METRIC, metric)
    assert result.metric == "contextual_recall"


def test_contextual_recall_passes_the_injected_model():
    model = SentinelModel()
    metric = DeepEvalContextualRecallEvaluator(model=model)._metric()
    assert metric.model is model
