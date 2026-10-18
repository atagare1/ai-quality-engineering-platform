"""PV P0.3c: MCP tool isError=True is preserved; scenario is not successful.

Focused live negative: force a real Playwright MCP tool failure, capture it
through serialize → ToolInvocation → request map, and prove the scenario
is not reported as successful.

Execution health and final state are separate deterministic metrics alongside
ToolCorrectness; no combined score or scenario abstraction.
"""

from __future__ import annotations

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
    P0_EVALUATIONS,
    _final_state_contains_expected_todo,
    _p0_runner,
    _resolve_snapshot_body,
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_execution_error_preserved_scenario_not_successful():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Add Buy milk on TodoMVC (execution-error negative path)."
    print("qe_supplied_input", goal)
    observed = []

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)

            async def _call_allow_error(name: str, arguments: dict | None) -> dict:
                result = await session.call_tool(name, arguments)
                plain = serialize_call_tool_result(result)
                print("tool_call", name, "isError", plain["isError"])
                observed.append(
                    tool_invocation_from_observation(
                        name=name,
                        arguments=arguments,
                        result=plain,
                    )
                )
                return plain

            nav = await _call_allow_error("browser_navigate", {"url": TODO_MVC_URL})
            assert nav["isError"] is False

            # Invalid target after a real page load → MCP tool-level error.
            err = await _call_allow_error(
                "browser_click",
                {
                    "element": "Nonexistent control",
                    "target": "e99999",
                },
            )
            assert err["isError"] is True

            # Optional post-error snapshot for final-state evidence only.
            snap = await _call_allow_error("browser_snapshot", {})
            final_body = (
                _resolve_snapshot_body(snap) if snap["isError"] is False else ""
            )
            print("final_state_ok", EXPECTED_TODO_TEXT in final_body)

    assert any(
        isinstance(call.result, dict) and call.result.get("isError") is True
        for call in observed
    )

    expected_tool_calls = [
        tool_invocation_from_observation(name=call.name, arguments=None, result=None)
        for call in observed
    ]
    request = mcp_p0_request(
        observed_tool_calls=observed,
        expected_tool_calls=expected_tool_calls,
        final_state_ok=_final_state_contains_expected_todo(observed),
    )

    # Failure preserved in captured tool evidence.
    error_calls = [
        call
        for call in observed
        if isinstance(call.result, dict) and call.result.get("isError") is True
    ]
    assert len(error_calls) >= 1
    assert error_calls[0].name == "browser_click"
    assert error_calls[0].result["isError"] is True
    assert error_calls[0].result["content"]
    print("preserved_error_tool", error_calls[0].name)
    print("preserved_is_error", error_calls[0].result["isError"])

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
        run_id="pv-playwright-mcp-execution-error",
    )

    assert decision.passed is False
    assert runner.last_run is not None
    results = {result.metric: result for result in runner.last_run.results}
    decisions = {item.metric: item for item in runner.last_run.decisions}
    assert results["tool_correctness"].score == 1.0
    assert results[FINAL_STATE_METRIC].score == 0.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 0.0
    assert decisions["tool_correctness"].passed is True
    assert decisions[FINAL_STATE_METRIC].passed is False
    assert decisions[MCP_EXECUTION_HEALTH_METRIC].passed is False
    print("tool_correctness_score", results["tool_correctness"].score)
    print("final_state_score", results[FINAL_STATE_METRIC].score)
    print("mcp_execution_health_score", results[MCP_EXECUTION_HEALTH_METRIC].score)
    print("gate_passed", decision.passed)
