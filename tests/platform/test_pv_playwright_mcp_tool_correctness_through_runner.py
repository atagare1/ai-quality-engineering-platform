"""PV: real Playwright MCP → capture → ToolCorrectness through Policy/Gate.

Uses INPUT_PARAMETERS-only exact match so dynamic browser_navigate output is
ignored. Compares tool name and URL arguments. Platform gate is an explicit
QualityPolicy of tool_correctness >= 0.80.
"""

from __future__ import annotations

import math
import shutil

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams
from deepeval_live import live_deepeval_local_model
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ai_qe_eval.capture.mcp_trace import tool_invocation_from_observation
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy

TOOL_CORRECTNESS_THRESHOLD = 0.80


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_tool_correctness_through_runner():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    goal = "Open the Playwright TodoMVC demo."
    arguments = {"url": TODO_MVC_URL}

    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_result = await session.call_tool("browser_navigate", arguments)
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("navigate_is_error", mcp_result.isError)
            assert mcp_result.isError is False

            observed = tool_invocation_from_observation(
                name="browser_navigate",
                arguments=arguments,
                result=serialize_call_tool_result(mcp_result),
            )
            expected_call = tool_invocation_from_observation(
                name="browser_navigate",
                arguments={"url": TODO_MVC_URL},
                result=None,
            )

    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        include_reason=True,
        async_mode=False,
        model=live_deepeval_local_model(),
    )
    policy = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    )
    print("policy_threshold", policy.threshold)
    print("evaluation_params", ["INPUT_PARAMETERS"])

    result = DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=metric,
    ).evaluate([observed], [expected_call], input=goal)[0]
    policy_decision = policy.apply(result)
    decision = QualityGate().evaluate([policy_decision])

    assert decision.passed is True
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert policy_decision.passed is True
    assert policy_decision.threshold == TOOL_CORRECTNESS_THRESHOLD
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", decision.passed)
