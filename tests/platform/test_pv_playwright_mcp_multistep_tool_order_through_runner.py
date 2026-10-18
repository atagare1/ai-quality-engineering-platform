"""PV P0 flagship e2e (PASS): NL task → Playwright MCP → request map → gate.

Natural-language task → real Playwright MCP multi-step TodoMVC flow →
request map → ordered ToolCorrectness through Runner/Policy/Gate →
deterministic final-state validation → scenario_task_succeeded=True.

Ordered tool names only:
browser_navigate → browser_snapshot → browser_click → browser_type
→ browser_click → browser_snapshot

Final browser_snapshot is checked for "Buy milk" (inline text or
[Snapshot](path) sidecar) by a separate deterministic evaluator.

Scenario success requires BOTH separate evidence signals:
1. ordered ToolCorrectness gate passes
2. final TodoMVC state contains "Buy milk"
Each signal remains a separate metric; no combined score is introduced.

Dynamic snapshot refs are used for live calls but are NOT copied into
expected_tool_calls. ToolCorrectness uses evaluation_params=[].
"""

from __future__ import annotations

import math
import re
import shutil

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval_live import live_deepeval_local_model
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ai_qe_eval.capture.mcp_trace import (
    mcp_p0_request,
    tool_invocation_from_observation,
)
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
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
    snapshot_body_from_serialized_result,
    snapshot_contains_list_item,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

TOOL_CORRECTNESS_THRESHOLD = 0.80
EXPECTED_TODO_TEXT = "Buy milk"
EXPECTED_TOOL_ORDER = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_click",
    "browser_snapshot",
]
P0_EVALUATIONS = [
    "tool_correctness",
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
]


def _snapshot_text(plain_result: dict) -> str:
    return snapshot_body_from_serialized_result(plain_result)


def _resolve_snapshot_body(plain_result: dict) -> str:
    """Compatibility helper returning evidence already captured by integration."""
    return snapshot_body_from_serialized_result(plain_result)


def _ref_for_label(snapshot_text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}.*?\[ref=(e\d+)\]"
    match = re.search(pattern, snapshot_text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"Could not find ref for label {label!r} in snapshot")
    return match.group(1)


def _final_state_contains_expected_todo(observed) -> bool:
    for call in reversed(observed or []):
        if call.name == "browser_snapshot" and isinstance(call.result, dict):
            return snapshot_contains_list_item(
                _resolve_snapshot_body(call.result),
                EXPECTED_TODO_TEXT,
            )
    return False


def _p0_runner(tool_correctness_metric) -> EvaluationRunner:
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    for metric_name in (FINAL_STATE_METRIC, MCP_EXECUTION_HEALTH_METRIC):
        registry.register(
            EvaluationCapability(
                name=metric_name,
                evaluator="deterministic",
                category="agent",
            )
        )
    return EvaluationRunner(
        registry=registry,
        evaluators={
            "tool_correctness": DeepEvalToolCorrectnessEvaluator(
                tool_correctness_metric=tool_correctness_metric,
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


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_flagship_todomvc_e2e_scenario_passes():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add 'Buy milk' to the Playwright TodoMVC demo."
    observed = []

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)

            async def _call(name: str, arguments: dict | None) -> dict:
                result = await session.call_tool(name, arguments)
                plain = serialize_call_tool_result(result)
                print("tool_call", name, "isError", plain["isError"])
                assert plain["isError"] is False, plain
                observed.append(
                    tool_invocation_from_observation(
                        name=name,
                        arguments=arguments,
                        result=plain,
                    )
                )
                return plain

            await _call("browser_navigate", {"url": TODO_MVC_URL})
            first_snapshot = await _call("browser_snapshot", {})
            snapshot_text = _snapshot_text(first_snapshot)
            textbox_ref = _ref_for_label(snapshot_text, 'textbox "What needs to be done?"')
            heading_ref = _ref_for_label(snapshot_text, 'heading "todos"')

            await _call(
                "browser_click",
                {
                    "element": "Todo input textbox",
                    "target": textbox_ref,
                },
            )
            await _call(
                "browser_type",
                {
                    "element": "Todo input textbox",
                    "target": textbox_ref,
                    "text": EXPECTED_TODO_TEXT,
                    "submit": True,
                },
            )
            # Second click uses a stable ref from the first snapshot so the
            # requested tool order remains executable without an extra snapshot.
            await _call(
                "browser_click",
                {
                    "element": "Todos heading",
                    "target": heading_ref,
                },
            )
            await _call("browser_snapshot", {})

    captured_names = [call.name for call in observed]
    print("qe_supplied_input", goal)
    print("captured_tool_names", captured_names)
    assert captured_names == EXPECTED_TOOL_ORDER

    # Ground truth is ordered tool names only. Do not copy dynamic args/refs.
    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    assert [call.name for call in observed] == EXPECTED_TOOL_ORDER
    assert all(call.arguments is None for call in expected_tool_calls)
    assert all(call.result is None for call in expected_tool_calls)
    request = mcp_p0_request(
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
        final_state_ok=_final_state_contains_expected_todo(observed),
    )

    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[],
        include_reason=True,
        async_mode=False,
        model=live_deepeval_local_model(),
    )
    runner = _p0_runner(metric)
    print("evaluation_params", [])
    print("policy_threshold", TOOL_CORRECTNESS_THRESHOLD)

    decision = runner.run(
        request,
        EvaluationConfig(evaluations=P0_EVALUATIONS),
        run_id="pv-playwright-mcp-flagship-todomvc-e2e",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.requests[0] is request
    results = {result.metric: result for result in runner.last_run.results}
    result = results["tool_correctness"]
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score == 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert all(item.passed for item in runner.last_run.decisions)
    assert results[FINAL_STATE_METRIC].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("final_state_score", results[FINAL_STATE_METRIC].score)
    print(
        "mcp_execution_health_score",
        results[MCP_EXECUTION_HEALTH_METRIC].score,
    )
    print("gate_passed", decision.passed)
