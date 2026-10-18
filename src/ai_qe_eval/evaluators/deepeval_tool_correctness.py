"""DeepEval Tool Correctness adapter.

Maps caller-supplied observed and expected tool calls into an LLMTestCase
and ToolCorrectnessMetric.measure, then returns one EvaluationResult.

Does not read EvaluationTrace. Observed and expected lists are independent.
Does not apply DeepEval's threshold or metric.success. Does not auto-register.
Does not import MCP SDK types.

Exact match compares arguments and results only when ToolCallParams
INPUT_PARAMETERS and OUTPUT are selected. Those parameters are enabled so
ToolInvocation.arguments and ToolInvocation.result participate in the score.
None and {} are passed through unchanged. DeepEval treats them as different.

DeepEval LLMTestCase still requires an input string. Pass input= to supply
one; otherwise a placeholder is used. That string is not scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.result import EvaluationResult

TOOL_CORRECTNESS_METRIC = "tool_correctness"
DEEPEVAL_EVALUATOR_NAME = "deepeval"
DEEPEVAL_INPUT_PLACEHOLDER = (
    "tool_correctness input placeholder; not EvaluationTrace.input"
)


def _as_tool_invocations(
    calls: Sequence[ToolInvocation] | None,
    *,
    field_name: str,
) -> list[ToolInvocation]:
    if calls is None:
        raise ValueError(
            f"DeepEvalToolCorrectnessEvaluator requires {field_name}"
        )
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise TypeError(
            f"DeepEvalToolCorrectnessEvaluator requires a sequence of "
            f"ToolInvocation for {field_name}, got {type(calls).__name__}"
        )
    materialized = list(calls)
    for index, call in enumerate(materialized):
        if not isinstance(call, ToolInvocation):
            raise TypeError(
                "DeepEvalToolCorrectnessEvaluator requires ToolInvocation "
                f"{field_name}, got {type(call).__name__} at index {index}"
            )
    return materialized


def _to_tool_call(invocation: ToolInvocation) -> Any:
    from deepeval.test_case import ToolCall

    return ToolCall(
        name=invocation.name,
        input_parameters=invocation.arguments,
        output=invocation.result,
    )


def _to_llm_test_case(
    *,
    observed: Sequence[ToolInvocation],
    expected: Sequence[ToolInvocation],
    input: Any,
) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input,
        tools_called=[_to_tool_call(call) for call in observed],
        expected_tools=[_to_tool_call(call) for call in expected],
    )


def _default_evaluation_params() -> list[Any]:
    from deepeval.test_case import ToolCallParams

    return [
        ToolCallParams.INPUT_PARAMETERS,
        ToolCallParams.OUTPUT,
    ]


def _default_tool_correctness_metric(
    *,
    model: Any | None,
    evaluation_params: list[Any] | None = None,
) -> Any:
    from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric

    return ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=(
            list(evaluation_params)
            if evaluation_params is not None
            else _default_evaluation_params()
        ),
        include_reason=True,
        async_mode=False,
        model=model,
    )


def _tool_record(invocation: ToolInvocation) -> dict[str, Any]:
    return {
        "name": invocation.name,
        "arguments": invocation.arguments,
        "result": invocation.result,
    }


class DeepEvalToolCorrectnessEvaluator:
    """DeepEval Tool Correctness adapter.

    Returns metric="tool_correctness" and evaluator="deepeval".
    Does not apply DeepEval's threshold or metric.success.

    Default evaluation_params compare INPUT_PARAMETERS and OUTPUT.
    Pass evaluation_params to override that default when constructing the
    metric. An injected tool_correctness_metric takes precedence.

    The default metric is built on first evaluate() and uses the
    caller-supplied model. model=None is passed through to DeepEval, which
    initializes its default provider and fails if that provider is not
    configured. Deterministic tests must inject a metric or model.
    """

    def __init__(
        self,
        *,
        tool_correctness_metric: Any | None = None,
        model: Any | None = None,
        evaluation_params: list[Any] | None = None,
    ) -> None:
        self._tool_correctness_metric = tool_correctness_metric
        self._model = model
        self._evaluation_params = (
            None if evaluation_params is None else list(evaluation_params)
        )

    def _metric(self) -> Any:
        if self._tool_correctness_metric is not None:
            return self._tool_correctness_metric
        return _default_tool_correctness_metric(
            model=self._model,
            evaluation_params=self._evaluation_params,
        )

    def evaluate(
        self,
        observed_tool_calls: Sequence[ToolInvocation],
        expected_tool_calls: Sequence[ToolInvocation],
        configuration: Any | None = None,
        *,
        input: Any | None = None,
    ) -> list[EvaluationResult]:
        actual = _as_tool_invocations(
            observed_tool_calls, field_name="observed_tool_calls"
        )
        expected = _as_tool_invocations(
            expected_tool_calls, field_name="expected_tool_calls"
        )
        test_case = _to_llm_test_case(
            observed=actual,
            expected=expected,
            input=DEEPEVAL_INPUT_PLACEHOLDER if input is None else input,
        )
        metric = self._metric()
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", None)
        return [
            EvaluationResult(
                metric=TOOL_CORRECTNESS_METRIC,
                evaluator=DEEPEVAL_EVALUATOR_NAME,
                score=score,
                reason=reason,
                raw_result={
                    "score": score,
                    "reason": reason,
                    "input": test_case.input,
                    "tools_called": [_tool_record(call) for call in actual],
                    "expected_tools": [_tool_record(call) for call in expected],
                },
            )
        ]
