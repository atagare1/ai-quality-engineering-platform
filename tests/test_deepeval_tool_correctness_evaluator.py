"""Focused tests for DeepEvalToolCorrectnessEvaluator.

Injects a ToolCorrectnessMetric stub. Does not call an external LLM.
Does not construct EvaluationTrace.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    DEEPEVAL_EVALUATOR_NAME,
    DEEPEVAL_INPUT_PLACEHOLDER,
    TOOL_CORRECTNESS_METRIC,
    DeepEvalToolCorrectnessEvaluator,
)
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy

_ADAPTER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "evaluators"
    / "deepeval_tool_correctness.py"
)

WEATHER_INPUT = "Find the weather for Pune."


class RecordingToolCorrectnessMetric:
    def __init__(self, score=1.0, reason="Exact match.") -> None:
        self.score = score
        self.reason = reason
        self.test_case = None
        self.threshold = 0.5
        self.success = True
        self.measure_calls = 0

    def measure(self, test_case):
        self.measure_calls += 1
        self.test_case = test_case
        return self.score


def _weather() -> ToolInvocation:
    return ToolInvocation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )


def _evaluate(metric, observed=None, expected=None, **kwargs):
    if observed is None:
        observed = [_weather()]
    if expected is None:
        expected = [_weather()]
    return DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=metric
    ).evaluate(observed, expected, **kwargs)


def test_single_tool_call_maps_name_arguments_and_result():
    metric = RecordingToolCorrectnessMetric()
    _evaluate(metric, input=WEATHER_INPUT)
    called = metric.test_case.tools_called
    expected = metric.test_case.expected_tools
    assert metric.test_case.input == WEATHER_INPUT
    assert type(metric.test_case).__name__ == "LLMTestCase"
    assert len(called) == 1
    assert called[0].name == "weather"
    assert called[0].input_parameters == {"location": "Pune"}
    assert called[0].output == "Temperature is 28 C and conditions are clear."
    assert len(expected) == 1
    assert expected[0].name == "weather"
    assert expected[0].input_parameters == {"location": "Pune"}
    assert expected[0].output == called[0].output


def test_input_defaults_to_placeholder_when_omitted():
    metric = RecordingToolCorrectnessMetric()
    _evaluate(metric)
    assert metric.test_case.input == DEEPEVAL_INPUT_PLACEHOLDER


def test_multiple_tool_calls_preserve_list_order():
    first = ToolInvocation(name="lookup_city", arguments={"q": "Pune"}, result="in")
    second = ToolInvocation(name="weather", arguments={"location": "Pune"}, result="28 C")
    metric = RecordingToolCorrectnessMetric()
    _evaluate(metric, observed=[first, second], expected=[first, second])
    assert [call.name for call in metric.test_case.tools_called] == ["lookup_city", "weather"]
    assert [call.name for call in metric.test_case.expected_tools] == ["lookup_city", "weather"]
    assert metric.test_case.tools_called[0].input_parameters == {"q": "Pune"}
    assert metric.test_case.tools_called[1].output == "28 C"


def test_none_arguments_are_not_rewritten_to_an_empty_dict():
    metric = RecordingToolCorrectnessMetric()
    _evaluate(
        metric,
        observed=[ToolInvocation(name="weather", arguments=None, result=None)],
        expected=[ToolInvocation(name="weather", arguments={}, result=None)],
    )
    actual = metric.test_case.tools_called[0]
    expected = metric.test_case.expected_tools[0]
    assert actual.input_parameters is None
    assert expected.input_parameters == {}
    assert actual != expected


def test_result_identity_score_and_reason():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Names and arguments match.")
    result = _evaluate(metric, input=WEATHER_INPUT)[0]
    assert isinstance(result, EvaluationResult)
    assert result.metric == TOOL_CORRECTNESS_METRIC
    assert result.metric == "tool_correctness"
    assert result.evaluator == DEEPEVAL_EVALUATOR_NAME
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.score is metric.score
    assert result.reason == "Names and arguments match."
    assert result.reason is metric.reason
    assert result.raw_result["tools_called"][0]["arguments"] == {"location": "Pune"}
    assert result.raw_result["input"] == WEATHER_INPUT
    assert "threshold" not in result.raw_result
    assert "success" not in result.raw_result
    assert "passed" not in result.raw_result


def test_mismatched_tools_are_passed_through_without_being_rewritten():
    metric = RecordingToolCorrectnessMetric(score=0.0, reason="Not an exact match.")
    _evaluate(
        metric,
        expected=[
            ToolInvocation(name="calendar", arguments={"location": "Pune"}, result="none")
        ],
    )
    assert metric.test_case.tools_called[0].name == "weather"
    assert metric.test_case.expected_tools[0].name == "calendar"


def test_empty_observed_list_maps_to_an_empty_called_list():
    metric = RecordingToolCorrectnessMetric()
    _evaluate(metric, observed=[], expected=[_weather()])
    assert metric.test_case.tools_called == []
    assert len(metric.test_case.expected_tools) == 1
    assert metric.measure_calls == 1


def test_empty_expected_tool_calls_are_not_fabricated():
    metric = RecordingToolCorrectnessMetric()
    _evaluate(metric, observed=[], expected=[])
    assert metric.test_case.tools_called == []
    assert metric.test_case.expected_tools == []
    assert metric.measure_calls == 1


def test_missing_observed_tool_calls_raise_before_the_metric_is_called():
    metric = RecordingToolCorrectnessMetric()
    with pytest.raises(ValueError, match="observed_tool_calls"):
        DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
            None, [_weather()]
        )
    assert metric.measure_calls == 0


def test_missing_expected_tool_calls_raise_before_the_metric_is_called():
    metric = RecordingToolCorrectnessMetric()
    with pytest.raises(ValueError, match="expected_tool_calls"):
        DeepEvalToolCorrectnessEvaluator(tool_correctness_metric=metric).evaluate(
            [_weather()], None
        )
    assert metric.measure_calls == 0


def test_default_metric_enables_exact_match_of_arguments_and_output():
    from deepeval.metrics.tool_correctness import tool_correctness as metric_module
    from deepeval.test_case import ToolCallParams

    captured: dict = {}

    class CapturingMetric:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.score = 1.0
            self.reason = "captured"

        def measure(self, test_case):
            self.test_case = test_case

    original = metric_module.ToolCorrectnessMetric
    metric_module.ToolCorrectnessMetric = CapturingMetric
    try:
        model = object()
        DeepEvalToolCorrectnessEvaluator(model=model).evaluate(
            [_weather()], [_weather()]
        )
    finally:
        metric_module.ToolCorrectnessMetric = original

    assert captured["should_exact_match"] is True
    assert captured["available_tools"] is None
    assert captured["include_reason"] is True
    assert captured["model"] is model
    assert ToolCallParams.INPUT_PARAMETERS in captured["evaluation_params"]
    assert ToolCallParams.OUTPUT in captured["evaluation_params"]


def test_evaluation_params_can_select_input_parameters_only():
    from deepeval.metrics.tool_correctness import tool_correctness as metric_module
    from deepeval.test_case import ToolCallParams

    captured: dict = {}

    class CapturingMetric:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.score = 1.0
            self.reason = "captured"

        def measure(self, test_case):
            self.test_case = test_case

    original = metric_module.ToolCorrectnessMetric
    metric_module.ToolCorrectnessMetric = CapturingMetric
    try:
        DeepEvalToolCorrectnessEvaluator(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ).evaluate([_weather()], [_weather()])
    finally:
        metric_module.ToolCorrectnessMetric = original

    assert captured["evaluation_params"] == [ToolCallParams.INPUT_PARAMETERS]
    assert captured["should_exact_match"] is True
    assert captured["model"] is None


def test_adapter_does_not_import_mcp_or_evaluation_trace():
    tree = ast.parse(_ADAPTER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "mcp" not in imported
    assert "ai_qe_eval.domain.trace" not in imported
    assert not any(name.endswith(".trace") for name in imported)


def test_policy_and_gate_apply_to_direct_tool_correctness_result():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Exact match.")
    result = _evaluate(metric)[0]
    decision = QualityPolicy(
        metric="tool_correctness", operator=">=", threshold=0.8
    ).apply(result)
    gate = QualityGate().evaluate([decision])

    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.reason == "Exact match."
    assert decision.passed is True
    assert decision.threshold == 0.8
    assert gate.passed is True
    assert metric.measure_calls == 1
    assert metric.test_case.expected_tools[0].name == "weather"
