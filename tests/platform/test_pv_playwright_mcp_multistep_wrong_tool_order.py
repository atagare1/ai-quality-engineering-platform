"""PV P0.3b: final state ok but wrong tool order → scenario fails.

Negative counterpart to the flagship multi-step Playwright MCP scenario.

Proves the scenario AND:
- TodoMVC final state intentionally contains "Buy milk" (final_state_ok=True)
- observed tool order differs from expected → ToolCorrectness gate fails
- scenario_task_succeeded is therefore False

ToolCorrectness, final-state, and execution health remain separate metrics.
No combined score.
"""

from __future__ import annotations

import math
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
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
)
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)

from test_pv_playwright_mcp_multistep_tool_order_through_runner import (  # noqa: E402
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    P0_EVALUATIONS,
    TOOL_CORRECTNESS_THRESHOLD,
    _final_state_contains_expected_todo,
    _p0_runner,
    _ref_for_label,
    _resolve_snapshot_body,
    _snapshot_text,
)

# Same tools as flagship, but snapshot/click swapped after type.
WRONG_TOOL_ORDER = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_click",
]


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_multistep_wrong_tool_order_final_state_ok_scenario_fails():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add a todo item on the Playwright TodoMVC demo."
    observed = []

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("wrong_observed_order", WRONG_TOOL_ORDER)
            print("expected_tool_order", EXPECTED_TOOL_ORDER)

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
            # Wrong order vs flagship: snapshot before the second click.
            post_type_snapshot = await _call("browser_snapshot", {})
            await _call(
                "browser_click",
                {
                    "element": "Todos heading",
                    "target": heading_ref,
                },
            )

            # Confirm the application outcome is correct before pipeline evaluation.
            final_body = _resolve_snapshot_body(post_type_snapshot)
            print("final_state_ok", EXPECTED_TODO_TEXT in final_body)
            assert EXPECTED_TODO_TEXT in final_body, (
                f"Negative wrong-order case requires {EXPECTED_TODO_TEXT!r} present; "
                f"resolved snapshot body starts with: {final_body[:500]!r}"
            )

    captured_names = [call.name for call in observed]
    print("captured_tool_names", captured_names)
    assert captured_names == WRONG_TOOL_ORDER
    assert captured_names != EXPECTED_TOOL_ORDER

    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    print("qe_supplied_input", goal)
    request = mcp_p0_request(
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
        final_state_ok=_final_state_contains_expected_todo(observed),
    )
    assert [call.name for call in request["tool_correctness"]["args"][0]] == (
        WRONG_TOOL_ORDER
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

    decision = runner.run(
        request,
        EvaluationConfig(evaluations=P0_EVALUATIONS),
        run_id="pv-playwright-mcp-multistep-wrong-tool-order",
    )

    assert decision.passed is False
    assert runner.last_run is not None
    results = {result.metric: result for result in runner.last_run.results}
    result = results["tool_correctness"]
    assert result.metric == "tool_correctness"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score < TOOL_CORRECTNESS_THRESHOLD
    assert results[FINAL_STATE_METRIC].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    decisions = {item.metric: item for item in runner.last_run.decisions}
    assert decisions["tool_correctness"].passed is False
    assert decisions[FINAL_STATE_METRIC].passed is True
    assert decisions[MCP_EXECUTION_HEALTH_METRIC].passed is True
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("final_state_score", results[FINAL_STATE_METRIC].score)
    print("mcp_execution_health_score", results[MCP_EXECUTION_HEALTH_METRIC].score)
    print("gate_passed", decision.passed)
