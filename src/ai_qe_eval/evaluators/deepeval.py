"""DeepEval evaluator adapters.

Each adapter maps caller-supplied evidence into DeepEval 4.2.6 LLMTestCase
fields and metric.measure. Does not read EvaluationTrace.

Evaluator identity is "deepeval". Does not apply framework thresholds or
PASS/FAIL. Does not auto-register.
"""

from __future__ import annotations

from typing import Any

from ai_qe_eval.domain.result import EvaluationResult

CORRECTNESS_METRIC = "correctness"
FAITHFULNESS_METRIC = "faithfulness"
ANSWER_RELEVANCY_METRIC = "answer_relevancy"
CONTEXTUAL_RELEVANCY_METRIC = "contextual_relevancy"
CONTEXTUAL_PRECISION_METRIC = "contextual_precision"
CONTEXTUAL_RECALL_METRIC = "contextual_recall"
HALLUCINATION_METRIC = "hallucination"
DEEPEVAL_EVALUATOR_NAME = "deepeval"
DEFAULT_GEVAL_NAME = "Correctness"
DEFAULT_CRITERIA = (
    "Determine whether the actual output is factually correct "
    "based on the expected output."
)

def _geval_llm_test_case(*, input: Any, output: Any, expected: Any) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        actual_output=output,
        expected_output=expected,
    )


def _default_geval_metric(
    *,
    model: Any | None,
    criteria: str | None,
    evaluation_steps: list[str] | None,
) -> Any:
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.test_case import SingleTurnParams

    kwargs: dict[str, Any] = {
        "name": DEFAULT_GEVAL_NAME,
        "evaluation_params": [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        "model": model,
        "async_mode": False,
    }
    if evaluation_steps is not None:
        kwargs["evaluation_steps"] = evaluation_steps
    else:
        kwargs["criteria"] = criteria or DEFAULT_CRITERIA
    return GEval(**kwargs)


class DeepEvalGEvalCorrectnessEvaluator:
    def __init__(
        self,
        *,
        geval_metric: Any | None = None,
        model: Any | None = None,
        criteria: str | None = None,
        evaluation_steps: list[str] | None = None,
    ) -> None:
        self._geval_metric = geval_metric
        self._model = model
        self._criteria = criteria
        self._evaluation_steps = evaluation_steps

    def _metric(self) -> Any:
        if self._geval_metric is not None:
            return self._geval_metric
        return _default_geval_metric(
            model=self._model,
            criteria=self._criteria,
            evaluation_steps=self._evaluation_steps,
        )

    def evaluate(
        self,
        input: Any,
        output: Any,
        expected: Any,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _geval_llm_test_case(
            input=input, output=output, expected=expected
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=CORRECTNESS_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "name": getattr(metric, "name", DEFAULT_GEVAL_NAME),
                    "criteria": getattr(metric, "criteria", self._criteria),
                    "evaluation_steps": getattr(
                        metric, "evaluation_steps", self._evaluation_steps
                    ),
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "expected_output": test_case.expected_output,
                },
            )
        ]


def _retrieval_context(retrieval: list[Any] | None) -> list[str]:
    if retrieval is None:
        return []
    contexts: list[str] = []
    for item in retrieval:
        if isinstance(item, str) and item:
            contexts.append(item)
        elif isinstance(item, dict):
            page_content = item.get("page_content")
            if isinstance(page_content, str) and page_content:
                contexts.append(page_content)
    return contexts


def _faithfulness_llm_test_case(
    *, input: Any, output: Any, retrieval: list[Any] | None
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        actual_output=output,
        retrieval_context=_retrieval_context(retrieval),
    )


def _default_faithfulness_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric

    return FaithfulnessMetric(model=model, async_mode=False)


class DeepEvalFaithfulnessEvaluator:
    """DeepEval Faithfulness adapter.

    Returns metric="faithfulness" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        faithfulness_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._faithfulness_metric = faithfulness_metric
        self._model = model

    def _metric(self) -> Any:
        if self._faithfulness_metric is not None:
            return self._faithfulness_metric
        return _default_faithfulness_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        output: Any,
        retrieval: list[Any] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _faithfulness_llm_test_case(
            input=input, output=output, retrieval=retrieval
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=FAITHFULNESS_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "retrieval_context": list(test_case.retrieval_context or []),
                },
            )
        ]


def _answer_relevancy_llm_test_case(*, input: Any, output: Any) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        actual_output=output,
    )


def _default_answer_relevancy_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric

    return AnswerRelevancyMetric(model=model, async_mode=False)


class DeepEvalAnswerRelevancyEvaluator:
    """DeepEval Answer Relevancy adapter.

    Returns metric="answer_relevancy" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        answer_relevancy_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._answer_relevancy_metric = answer_relevancy_metric
        self._model = model

    def _metric(self) -> Any:
        if self._answer_relevancy_metric is not None:
            return self._answer_relevancy_metric
        return _default_answer_relevancy_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        output: Any,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _answer_relevancy_llm_test_case(input=input, output=output)
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=ANSWER_RELEVANCY_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                },
            )
        ]


def _contextual_relevancy_llm_test_case(
    *, input: Any, retrieval: list[Any] | None
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        retrieval_context=_retrieval_context(retrieval),
    )


def _default_contextual_relevancy_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.contextual_relevancy.contextual_relevancy import (
        ContextualRelevancyMetric,
    )

    return ContextualRelevancyMetric(model=model, async_mode=False)


class DeepEvalContextualRelevancyEvaluator:
    """DeepEval Contextual Relevancy adapter.

    Returns metric="contextual_relevancy" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        contextual_relevancy_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._contextual_relevancy_metric = contextual_relevancy_metric
        self._model = model

    def _metric(self) -> Any:
        if self._contextual_relevancy_metric is not None:
            return self._contextual_relevancy_metric
        return _default_contextual_relevancy_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        retrieval: list[Any] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _contextual_relevancy_llm_test_case(
            input=input, retrieval=retrieval
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=CONTEXTUAL_RELEVANCY_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "retrieval_context": list(test_case.retrieval_context or []),
                },
            )
        ]


def _contextual_precision_llm_test_case(
    *, input: Any, expected: Any, retrieval: list[Any] | None
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        expected_output=expected,
        retrieval_context=_retrieval_context(retrieval),
    )


def _default_contextual_precision_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.contextual_precision.contextual_precision import (
        ContextualPrecisionMetric,
    )

    return ContextualPrecisionMetric(model=model, async_mode=False)


class DeepEvalContextualPrecisionEvaluator:
    """DeepEval Contextual Precision adapter.

    Returns metric="contextual_precision" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        contextual_precision_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._contextual_precision_metric = contextual_precision_metric
        self._model = model

    def _metric(self) -> Any:
        if self._contextual_precision_metric is not None:
            return self._contextual_precision_metric
        return _default_contextual_precision_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        expected: Any,
        retrieval: list[Any] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _contextual_precision_llm_test_case(
            input=input, expected=expected, retrieval=retrieval
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=CONTEXTUAL_PRECISION_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "expected_output": test_case.expected_output,
                    "retrieval_context": list(test_case.retrieval_context or []),
                },
            )
        ]


def _contextual_recall_llm_test_case(
    *, input: Any, expected: Any, retrieval: list[Any] | None
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        expected_output=expected,
        retrieval_context=_retrieval_context(retrieval),
    )


def _default_contextual_recall_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.contextual_recall.contextual_recall import (
        ContextualRecallMetric,
    )

    return ContextualRecallMetric(model=model, async_mode=False)


class DeepEvalContextualRecallEvaluator:
    """DeepEval Contextual Recall adapter.

    Returns metric="contextual_recall" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        contextual_recall_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._contextual_recall_metric = contextual_recall_metric
        self._model = model

    def _metric(self) -> Any:
        if self._contextual_recall_metric is not None:
            return self._contextual_recall_metric
        return _default_contextual_recall_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        expected: Any,
        retrieval: list[Any] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _contextual_recall_llm_test_case(
            input=input, expected=expected, retrieval=retrieval
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=CONTEXTUAL_RECALL_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "expected_output": test_case.expected_output,
                    "retrieval_context": list(test_case.retrieval_context or []),
                },
            )
        ]


def _hallucination_llm_test_case(
    *, input: Any, output: Any, retrieval: list[Any] | None
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        actual_output=output,
        context=_retrieval_context(retrieval),
    )


def _default_hallucination_metric(*, model: Any | None) -> Any:
    from deepeval.metrics.hallucination.hallucination import HallucinationMetric

    return HallucinationMetric(model=model, async_mode=False)


class DeepEvalHallucinationEvaluator:
    """DeepEval Hallucination adapter.

    Returns metric="hallucination" and evaluator="deepeval".
    DeepEval's score is the fraction of contexts the output agrees with.
    Higher means closer alignment, not a larger hallucination.
    Does not apply DeepEval's threshold or metric.success.
    Does not import RAGAS. Does not read EvaluationTrace.
    """

    def __init__(
        self,
        *,
        hallucination_metric: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._hallucination_metric = hallucination_metric
        self._model = model

    def _metric(self) -> Any:
        if self._hallucination_metric is not None:
            return self._hallucination_metric
        return _default_hallucination_metric(model=self._model)

    def evaluate(
        self,
        input: Any,
        output: Any,
        retrieval: list[Any] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        test_case = _hallucination_llm_test_case(
            input=input, output=output, retrieval=retrieval
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=HALLUCINATION_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                    "context": list(test_case.context or []),
                },
            )
        ]
