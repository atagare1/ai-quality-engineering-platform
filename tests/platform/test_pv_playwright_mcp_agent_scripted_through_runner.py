"""P1.2: scripted P1.1 executor through the existing P0 evaluation pipeline.

Deterministic: Fake MCP session returns CallToolResult values. No live browser.
Uses the existing P0 runner, policies, and gate. Does not modify P0 tests.
"""

from __future__ import annotations

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from mcp.types import CallToolResult, TextContent

from ai_qe_eval.capture.mcp_trace import (
    mcp_p0_request,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.integrations.playwright_mcp import TODO_MVC_URL
from ai_qe_eval.integrations.playwright_mcp_agent import run_playwright_mcp_agent
from ai_qe_eval.integrations.playwright_mcp_selector import SequenceToolSelector

from test_pv_playwright_mcp_multistep_tool_order_through_runner import (  # noqa: E402
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    P0_EVALUATIONS,
    _final_state_contains_expected_todo,
    _p0_runner,
)

GOAL = "Add 'Buy milk' to the Playwright TodoMVC demo."
P0_TOOL_SEQUENCE = [
    ("browser_navigate", {"url": TODO_MVC_URL}),
    ("browser_snapshot", {}),
    ("browser_click", {"element": "Todo input textbox", "target": "e5"}),
    (
        "browser_type",
        {
            "element": "Todo input textbox",
            "target": "e5",
            "text": EXPECTED_TODO_TEXT,
            "submit": True,
        },
    ),
    ("browser_click", {"element": "Todos heading", "target": "e3"}),
    ("browser_snapshot", {}),
]
_EMPTY_SNAPSHOT = """
- main:
  - textbox "What needs to be done?" [ref=e5]
  - heading "todos" [ref=e3]
""".strip()
_TODO_SNAPSHOT = f"""
- main:
  - textbox "What needs to be done?" [ref=e5]
  - heading "todos" [ref=e3]
  - list:
    - listitem [ref=e10]:
      - generic [ref=e12]: {EXPECTED_TODO_TEXT}
""".strip()


class UnusedToolCorrectnessJudge(DeepEvalBaseLLM):
    """Explicit unused judge so this test does not use DeepEval's default provider."""

    def get_model_name(self) -> str:
        return "unused-tool-correctness-judge"

    def load_model(self):
        return self

    def generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("scripted P0 name/order scoring must not call a judge")

    async def a_generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("scripted P0 name/order scoring must not call a judge")


class FakeSession:
    def __init__(self, results: list[CallToolResult]) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self._results = list(results)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._results.pop(0)


def _ok(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=None,
        isError=False,
    )


@pytest.mark.asyncio
async def test_scripted_executor_through_p0_runner_gate_passes():
    session = FakeSession(
        [
            _ok("navigated"),
            _ok(_EMPTY_SNAPSHOT),
            _ok("clicked"),
            _ok("typed"),
            _ok("clicked"),
            _ok(_TODO_SNAPSHOT),
        ]
    )

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=SequenceToolSelector(P0_TOOL_SEQUENCE),
        max_steps=8,
    )
    expected_tool_calls = [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]
    request = mcp_p0_request(
        observed_tool_calls=run.observed_tool_calls,
        expected_tool_calls=expected_tool_calls,
        final_state_ok=_final_state_contains_expected_todo(run.observed_tool_calls),
    )

    assert [call.name for call in run.observed_tool_calls] == EXPECTED_TOOL_ORDER
    assert all(call.arguments is None for call in expected_tool_calls)
    assert all(call.result is None for call in expected_tool_calls)
    assert request["tool_correctness"]["args"][0] == run.observed_tool_calls
    assert request["tool_correctness"]["args"][1] == expected_tool_calls
    assert request["tool_correctness"]["args"][1] is not run.observed_tool_calls
    assert run.goal == GOAL
    assert run.output

    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[],
        include_reason=True,
        async_mode=False,
        model=UnusedToolCorrectnessJudge(),
    )
    runner = _p0_runner(metric)
    decision = runner.run(
        request,
        EvaluationConfig(evaluations=P0_EVALUATIONS),
        run_id="p1-scripted-executor-through-p0-runner",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.requests[0] is request
    assert request["tool_correctness"]["args"][0] == run.observed_tool_calls
    results = {result.metric: result for result in runner.last_run.results}
    assert results["tool_correctness"].score == 1.0
    assert results["final_state"].score == 1.0
    assert results["mcp_execution_health"].score == 1.0
    assert [item["name"] for item in results["tool_correctness"].raw_result["tools_called"]] == (
        EXPECTED_TOOL_ORDER
    )
    assert all(item.passed for item in runner.last_run.decisions)
