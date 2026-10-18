"""Focused tests for first-class final-state and MCP execution-health results."""

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
    MCP_EXECUTION_HEALTH_SUCCEEDED_REASON,
    FinalStateEvaluator,
    MCPExecutionHealthEvaluator,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

_DETERMINISTIC_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "evaluators"
    / "deterministic.py"
)


def _tool_results(*, tool_error: bool, final_state_ok: bool = True) -> list[ToolInvocation]:
    return [
        ToolInvocation(
            name="browser_snapshot",
            arguments={},
            result={
                "isError": tool_error,
                "final_state_ok": final_state_ok,
            },
        )
    ]


@pytest.mark.parametrize(
    ("final_state_ok", "expected_score"),
    [(True, 1.0), (False, 0.0)],
)
def test_final_state_evaluator_emits_pass_and_fail_results(
    final_state_ok,
    expected_score,
):
    result = FinalStateEvaluator().evaluate(final_state_ok)[0]

    assert result.metric == FINAL_STATE_METRIC
    assert result.evaluator == "deterministic"
    assert result.score == expected_score
    assert result.raw_result == {"final_state_ok": final_state_ok}


def test_final_state_evaluator_rejects_non_bool():
    with pytest.raises(TypeError, match="final_state_ok"):
        FinalStateEvaluator().evaluate(1)


def test_final_state_class_does_not_read_evaluation_trace():
    tree = ast.parse(_DETERMINISTIC_SOURCE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "FinalStateEvaluator"
    )
    names = {node.id for node in ast.walk(class_node) if isinstance(node, ast.Name)}
    assert "EvaluationTrace" not in names


@pytest.mark.parametrize(
    ("tool_error", "expected_score", "failed_tools"),
    [(False, 1.0, []), (True, 0.0, ["browser_snapshot"])],
)
def test_mcp_execution_health_evaluator_emits_pass_and_fail_results(
    tool_error,
    expected_score,
    failed_tools,
):
    result = MCPExecutionHealthEvaluator().evaluate(
        _tool_results(tool_error=tool_error)
    )[0]

    assert result.metric == MCP_EXECUTION_HEALTH_METRIC
    assert result.evaluator == "deterministic"
    assert result.score == expected_score
    assert result.raw_result["execution_ok"] is (not tool_error)
    assert result.raw_result["failed_tools"] == failed_tools


@pytest.mark.parametrize("tool_results", [None, []])
def test_mcp_execution_health_empty_or_missing_tool_results_pass(tool_results):
    result = MCPExecutionHealthEvaluator().evaluate(tool_results)[0]

    assert result.metric == MCP_EXECUTION_HEALTH_METRIC
    assert result.evaluator == "deterministic"
    assert result.score == 1.0
    assert result.reason == MCP_EXECUTION_HEALTH_SUCCEEDED_REASON
    assert result.raw_result == {
        "execution_ok": True,
        "tool_call_count": 0,
        "failed_tools": [],
    }


def test_mcp_execution_health_class_does_not_read_evaluation_trace():
    tree = ast.parse(_DETERMINISTIC_SOURCE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MCPExecutionHealthEvaluator"
    )
    names = {node.id for node in ast.walk(class_node) if isinstance(node, ast.Name)}
    assert "EvaluationTrace" not in names


def _runner() -> EvaluationRunner:
    registry = EvaluationRegistry()
    for metric in (FINAL_STATE_METRIC, MCP_EXECUTION_HEALTH_METRIC):
        registry.register(
            EvaluationCapability(
                name=metric,
                evaluator="deterministic",
                category="agent",
            )
        )
    return EvaluationRunner(
        registry=registry,
        evaluators={
            FINAL_STATE_METRIC: FinalStateEvaluator(),
            MCP_EXECUTION_HEALTH_METRIC: MCPExecutionHealthEvaluator(),
        },
        policies={
            metric: QualityPolicy(metric=metric, operator="==", threshold=1.0)
            for metric in (FINAL_STATE_METRIC, MCP_EXECUTION_HEALTH_METRIC)
        },
    )


@pytest.mark.parametrize(
    ("final_state_ok", "tool_error", "expected_gate", "expected_decisions"),
    [
        (True, False, True, [True, True]),
        (False, False, False, [False, True]),
        (True, True, False, [True, False]),
    ],
)
def test_outcome_and_health_flow_independently_through_runner_policy_gate(
    final_state_ok,
    tool_error,
    expected_gate,
    expected_decisions,
):
    runner = _runner()

    decision = runner.run(
        {
            FINAL_STATE_METRIC: {"args": [final_state_ok], "kwargs": {}},
            MCP_EXECUTION_HEALTH_METRIC: {
                "args": [_tool_results(tool_error=tool_error)],
                "kwargs": {},
            },
        },
        EvaluationConfig(
            evaluations=[FINAL_STATE_METRIC, MCP_EXECUTION_HEALTH_METRIC]
        ),
        run_id="outcome-health",
    )

    assert decision.passed is expected_gate
    assert [item.passed for item in decision.decisions] == expected_decisions
    assert [item.metric for item in decision.decisions] == [
        FINAL_STATE_METRIC,
        MCP_EXECUTION_HEALTH_METRIC,
    ]
    assert runner.last_run is not None
    assert [result.metric for result in runner.last_run.results] == [
        FINAL_STATE_METRIC,
        MCP_EXECUTION_HEALTH_METRIC,
    ]
