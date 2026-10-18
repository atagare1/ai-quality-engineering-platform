"""Focused tests for DeepEvalHallucinationEvaluator.

Injects a HallucinationMetric stub. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval import (
    DEEPEVAL_EVALUATOR_NAME,
    HALLUCINATION_METRIC,
    DeepEvalHallucinationEvaluator,
)

HALL_INPUT = "What is the capital of France?"
HALL_OUTPUT = "Paris."
HALL_RETRIEVAL = ["Paris is the capital and largest city of France."]


class SentinelModel(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return "sentinel"

    def load_model(self):
        return self

    def generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("unit test must not call the model")

    async def a_generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("unit test must not call the model")


class RecordingHallucinationMetric:
    def __init__(self, score=1.0, reason="The output agrees with the context.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.success = True
        self.threshold = 0.5

    def measure(self, test_case):
        self.test_case = test_case
        return self.score


def test_input_output_and_retrieval_map_to_context():
    metric = RecordingHallucinationMetric()
    DeepEvalHallucinationEvaluator(hallucination_metric=metric).evaluate(
        HALL_INPUT,
        HALL_OUTPUT,
        [
            "context one",
            {"page_content": "context two", "file_name": "notes.docx"},
            "",
            {"file_name": "empty.docx"},
            {"page_content": ""},
            None,
        ],
    )
    assert metric.test_case.input == HALL_INPUT
    assert metric.test_case.actual_output == HALL_OUTPUT
    assert metric.test_case.context == ["context one", "context two"]
    assert metric.test_case.retrieval_context is None
    assert metric.test_case.expected_output is None
    assert "SHOULD_NOT_BE_SENT" not in (
        metric.test_case.input,
        metric.test_case.actual_output,
        *metric.test_case.context,
    )


def test_missing_retrieval_becomes_an_empty_context_list():
    metric = RecordingHallucinationMetric()
    DeepEvalHallucinationEvaluator(hallucination_metric=metric).evaluate(
        HALL_INPUT, HALL_OUTPUT, None
    )
    assert metric.test_case.context == []


def test_empty_output_is_passed_through():
    metric = RecordingHallucinationMetric()
    DeepEvalHallucinationEvaluator(hallucination_metric=metric).evaluate(
        HALL_INPUT, "", HALL_RETRIEVAL
    )
    assert metric.test_case.actual_output == ""


def test_result_identity_is_hallucination_from_deepeval():
    result = DeepEvalHallucinationEvaluator(
        hallucination_metric=RecordingHallucinationMetric()
    ).evaluate(HALL_INPUT, HALL_OUTPUT, HALL_RETRIEVAL)[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == HALLUCINATION_METRIC
    assert result.metric == "hallucination"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"


def test_score_and_reason_are_copied_from_the_metric():
    metric = RecordingHallucinationMetric(
        score=1.0,
        reason="The output agrees with the context.",
    )
    result = DeepEvalHallucinationEvaluator(hallucination_metric=metric).evaluate(
        HALL_INPUT, HALL_OUTPUT, HALL_RETRIEVAL
    )[0]
    assert result.score == 1.0
    assert result.score is metric.score
    assert result.reason == "The output agrees with the context."
    assert result.reason is metric.reason
    assert result.raw_result["context"] == HALL_RETRIEVAL
    assert "success" not in result.raw_result
    assert "threshold" not in result.raw_result
    assert "passed" not in result.__dataclass_fields__


def test_injected_model_is_passed_to_the_metric():
    model = SentinelModel()
    metric = DeepEvalHallucinationEvaluator(model=model)._metric()
    assert metric.model is model
