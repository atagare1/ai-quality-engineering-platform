"""C.2 deterministic ToolCorrectness integration for MCP capture.

Proves:
capture tool invocations → DeepEvalToolCorrectnessEvaluator → EvaluationResult
→ QualityPolicy → QualityGate

Uses a mocked ToolCorrectnessMetric. No live provider. No capture-layer scoring.
Does not construct EvaluationTrace for ToolCorrectness.
"""

from __future__ import annotations

from ai_qe_eval.capture.mcp_trace import mcp_p0_request, tool_invocation_from_observation
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
    FinalStateEvaluator,
    MCPExecutionHealthEvaluator,
)
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

TOOL_CORRECTNESS_THRESHOLD = 0.80


class RecordingToolCorrectnessMetric:
    def __init__(self, score: float, reason: str) -> None:
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


def _captured_mcp_calls():
    actual = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    expected = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    return [actual], [expected]


def _evaluate(metric: RecordingToolCorrectnessMetric):
    observed, expected = _captured_mcp_calls()
    result = DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=metric
    ).evaluate(
        observed,
        expected,
        input="Find the weather for Pune.",
    )[0]
    policy_decision = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    ).apply(result)
    gate = QualityGate().evaluate([policy_decision])
    return result, policy_decision, gate


def test_mcp_capture_through_tool_correctness_passes_when_score_meets_policy():
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Exact match.")
    result, policy_decision, gate = _evaluate(metric)

    assert gate.passed is True
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.91
    assert result.reason == "Exact match."
    assert policy_decision.passed is True
    assert policy_decision.operator == ">="
    assert policy_decision.threshold == TOOL_CORRECTNESS_THRESHOLD
    assert metric.measure_calls == 1
    assert metric.test_case.tools_called[0].name == "weather"
    assert metric.test_case.expected_tools[0].name == "weather"


def test_mcp_p0_request_through_runner_policy_and_gate():
    observed, expected = _captured_mcp_calls()
    request = mcp_p0_request(
        observed_tool_calls=observed,
        expected_tool_calls=expected,
        final_state_ok=True,
    )
    metric = RecordingToolCorrectnessMetric(score=0.91, reason="Exact match.")
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(name="tool_correctness", evaluator="deepeval", category="agent")
    )
    registry.register(
        EvaluationCapability(
            name=FINAL_STATE_METRIC, evaluator="deterministic", category="agent"
        )
    )
    registry.register(
        EvaluationCapability(
            name=MCP_EXECUTION_HEALTH_METRIC,
            evaluator="deterministic",
            category="agent",
        )
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(
                tool_correctness_metric=metric
            ),
            FINAL_STATE_METRIC: FinalStateEvaluator(),
            MCP_EXECUTION_HEALTH_METRIC: MCPExecutionHealthEvaluator(),
        },
        policies={
            "tool_correctness": QualityPolicy(
                metric="tool_correctness",
                operator=">=",
                threshold=TOOL_CORRECTNESS_THRESHOLD,
            ),
            FINAL_STATE_METRIC: QualityPolicy(
                metric=FINAL_STATE_METRIC, operator="==", threshold=1.0
            ),
            MCP_EXECUTION_HEALTH_METRIC: QualityPolicy(
                metric=MCP_EXECUTION_HEALTH_METRIC, operator="==", threshold=1.0
            ),
        },
        gate=QualityGate(),
    )
    decision = runner.run(
        request,
        EvaluationConfig(
            evaluations=[
                "tool_correctness",
                FINAL_STATE_METRIC,
                MCP_EXECUTION_HEALTH_METRIC,
            ]
        ),
        run_id="mcp-p0-request",
    )
    assert runner.last_run is not None
    results = {result.metric: result for result in runner.last_run.results}
    assert decision.passed is True
    assert runner.last_run.requests[0] is request
    assert results["tool_correctness"].score == 0.91
    assert results[FINAL_STATE_METRIC].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    assert metric.measure_calls == 1


def test_mcp_capture_through_tool_correctness_fails_when_score_misses_policy():
    metric = RecordingToolCorrectnessMetric(score=0.40, reason="Not an exact match.")
    result, policy_decision, gate = _evaluate(metric)

    assert gate.passed is False
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert result.score == 0.40
    assert result.reason == "Not an exact match."
    assert policy_decision.passed is False
    assert policy_decision.threshold == TOOL_CORRECTNESS_THRESHOLD
    assert metric.measure_calls == 1
