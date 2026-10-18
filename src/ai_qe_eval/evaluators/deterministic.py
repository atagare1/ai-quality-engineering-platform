"""Deterministic evaluators.

Each evaluator produces an EvaluationResult and leaves PASS/FAIL to
QualityPolicy and QualityGate. None auto-register capabilities.
"""

from __future__ import annotations

from typing import Any, Sequence

from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.result import EvaluationResult

EXACT_MATCH_METRIC = "exact_match"
FINAL_STATE_METRIC = "final_state"
MCP_EXECUTION_HEALTH_METRIC = "mcp_execution_health"
DETERMINISTIC_EVALUATOR_NAME = "deterministic"
EXACT_MATCH_SUCCEEDED_REASON = "Exact match succeeded."
EXACT_MATCH_FAILED_REASON = "Exact match failed."
FINAL_STATE_SUCCEEDED_REASON = "Final-state check succeeded."
FINAL_STATE_FAILED_REASON = "Final-state check failed."
MCP_EXECUTION_HEALTH_SUCCEEDED_REASON = "No MCP tool execution errors were observed."
MCP_EXECUTION_HEALTH_FAILED_REASON = "One or more MCP tool execution errors were observed."


class DeterministicEvaluator:
    """Exact-match adapter.

    Compares caller-supplied output and expected with Python equality.
    Does not read EvaluationTrace.
    """

    def evaluate(
        self,
        output: Any,
        expected: Any,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        matched = output == expected
        score = 1.0 if matched else 0.0
        return [
            EvaluationResult(
                metric=EXACT_MATCH_METRIC,
                evaluator=DETERMINISTIC_EVALUATOR_NAME,
                score=score,
                reason=(
                    EXACT_MATCH_SUCCEEDED_REASON
                    if matched
                    else EXACT_MATCH_FAILED_REASON
                ),
                raw_result={
                    "output": output,
                    "expected": expected,
                    "matched": matched,
                },
            )
        ]


class FinalStateEvaluator:
    """Score a caller-supplied final-state boolean.

    Does not read EvaluationTrace, turns, expected, or events. Does not run
    scenario-specific extraction. Accepts only a bool.
    """

    def evaluate(
        self,
        final_state_ok: bool,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        if not isinstance(final_state_ok, bool):
            raise TypeError(
                "FinalStateEvaluator requires final_state_ok to be bool, "
                f"got {type(final_state_ok).__name__}"
            )
        return [
            EvaluationResult(
                metric=FINAL_STATE_METRIC,
                evaluator=DETERMINISTIC_EVALUATOR_NAME,
                score=1.0 if final_state_ok else 0.0,
                reason=(
                    FINAL_STATE_SUCCEEDED_REASON
                    if final_state_ok
                    else FINAL_STATE_FAILED_REASON
                ),
                raw_result={"final_state_ok": final_state_ok},
            )
        ]


class MCPExecutionHealthEvaluator:
    """Fail when a captured MCP result explicitly contains ``isError=True``.

    Accepts observed ToolInvocation items directly. Does not read
    EvaluationTrace, turns, expected, or events.
    """

    def evaluate(
        self,
        tool_results: Sequence[ToolInvocation] | None,
        configuration: Any | None = None,
    ) -> list[EvaluationResult]:
        calls = list(tool_results or [])
        failed_tools = [
            call.name
            for call in calls
            if isinstance(call.result, dict) and call.result.get("isError") is True
        ]
        execution_ok = not failed_tools
        return [
            EvaluationResult(
                metric=MCP_EXECUTION_HEALTH_METRIC,
                evaluator=DETERMINISTIC_EVALUATOR_NAME,
                score=1.0 if execution_ok else 0.0,
                reason=(
                    MCP_EXECUTION_HEALTH_SUCCEEDED_REASON
                    if execution_ok
                    else MCP_EXECUTION_HEALTH_FAILED_REASON
                ),
                raw_result={
                    "execution_ok": execution_ok,
                    "tool_call_count": len(calls),
                    "failed_tools": failed_tools,
                },
            )
        ]
