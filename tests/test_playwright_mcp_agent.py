"""Deterministic tests for the Playwright MCP agent-execution boundary."""

from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent

from ai_qe_eval.capture.mcp_trace import (
    mcp_p0_request,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.integrations.playwright_mcp import TODO_MVC_URL
from ai_qe_eval.integrations.playwright_mcp_agent import (
    PlaywrightMcpAgentRun,
    run_playwright_mcp_agent,
)
from ai_qe_eval.integrations.playwright_mcp_selector import SequenceToolSelector

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
            "text": "Buy milk",
            "submit": True,
        },
    ),
    ("browser_click", {"element": "Todos heading", "target": "e3"}),
    ("browser_snapshot", {}),
]
EXPECTED_TOOL_ORDER = [name for name, _arguments in P0_TOOL_SEQUENCE]


class FakeSession:
    def __init__(self, results: list[CallToolResult] | None = None) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self._results = list(results or [])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._results:
            return self._results.pop(0)
        return CallToolResult(
            content=[TextContent(type="text", text=f"ok:{name}")],
            structuredContent=None,
            isError=False,
        )


def _error_result(name: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"error:{name}")],
        structuredContent=None,
        isError=True,
    )


@pytest.mark.asyncio
async def test_scripted_selector_preserves_p0_order_and_actual_payloads():
    session = FakeSession()
    selector = SequenceToolSelector(P0_TOOL_SEQUENCE)

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=selector,
        max_steps=8,
    )

    assert isinstance(run, PlaywrightMcpAgentRun)
    assert run.goal == GOAL
    assert [call.name for call in run.observed_tool_calls] == EXPECTED_TOOL_ORDER
    assert session.calls == P0_TOOL_SEQUENCE
    assert all(call.arguments is not None for call in run.observed_tool_calls)
    assert run.observed_tool_calls[0].arguments == {"url": TODO_MVC_URL}
    assert run.observed_tool_calls[3].arguments["text"] == "Buy milk"
    assert all(isinstance(call.result, dict) for call in run.observed_tool_calls)
    assert all(call.result["content"] for call in run.observed_tool_calls)
    assert selector.index == len(EXPECTED_TOOL_ORDER)


@pytest.mark.asyncio
async def test_expected_tool_calls_are_supplied_independently_by_the_test():
    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=FakeSession(),
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
        final_state_ok=True,
    )
    observed_args, expected_args = request["tool_correctness"]["args"]

    assert [call.name for call in run.observed_tool_calls] == EXPECTED_TOOL_ORDER
    assert [call.name for call in expected_tool_calls] == EXPECTED_TOOL_ORDER
    assert all(call.arguments is None for call in expected_tool_calls)
    assert all(call.result is None for call in expected_tool_calls)
    assert all(call.arguments is not None for call in run.observed_tool_calls)
    assert all(call.result is not None for call in run.observed_tool_calls)
    assert run.observed_tool_calls is not expected_tool_calls
    assert run.observed_tool_calls != expected_tool_calls
    assert expected_args is not run.observed_tool_calls
    assert observed_args == run.observed_tool_calls
    assert expected_args == expected_tool_calls


@pytest.mark.asyncio
async def test_selector_none_stops_without_further_tool_calls():
    session = FakeSession()
    selector = SequenceToolSelector(P0_TOOL_SEQUENCE[:2])

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=selector,
        max_steps=8,
    )

    assert [call.name for call in run.observed_tool_calls] == [
        "browser_navigate",
        "browser_snapshot",
    ]
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_max_steps_stops_even_when_selector_would_continue():
    session = FakeSession()
    selector = SequenceToolSelector(P0_TOOL_SEQUENCE)

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=selector,
        max_steps=3,
    )

    assert [call.name for call in run.observed_tool_calls] == EXPECTED_TOOL_ORDER[:3]
    assert len(session.calls) == 3
    assert selector.index == 3


@pytest.mark.asyncio
async def test_mcp_is_error_is_preserved_and_does_not_raise():
    session = FakeSession(results=[_error_result("browser_click")])
    selector = SequenceToolSelector(
        [("browser_click", {"element": "Nonexistent", "target": "e99999"})]
    )

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=selector,
        max_steps=2,
    )

    assert len(run.observed_tool_calls) == 1
    assert run.observed_tool_calls[0].result["isError"] is True
    assert run.observed_tool_calls[0].result["content"]


@pytest.mark.asyncio
async def test_on_tool_call_receives_captured_invocations_in_execution_order():
    session = FakeSession()
    seen: list[tuple[int, ToolInvocation]] = []

    def _on_tool_call(invocation: ToolInvocation, *, order: int) -> None:
        seen.append((order, invocation))

    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=SequenceToolSelector(P0_TOOL_SEQUENCE),
        max_steps=8,
        on_tool_call=_on_tool_call,
    )

    assert [order for order, _invocation in seen] == list(
        range(1, len(EXPECTED_TOOL_ORDER) + 1)
    )
    assert [invocation.name for _order, invocation in seen] == EXPECTED_TOOL_ORDER
    assert [invocation for _order, invocation in seen] == run.observed_tool_calls
    assert all(isinstance(invocation, ToolInvocation) for _order, invocation in seen)
    assert seen[0][1].arguments == {"url": TODO_MVC_URL}
    assert seen[3][1].arguments["text"] == "Buy milk"
    assert all(isinstance(invocation.result, dict) for _order, invocation in seen)


@pytest.mark.asyncio
async def test_agent_does_not_dump_mcp_results_to_stdout(capsys):
    session = FakeSession()
    await run_playwright_mcp_agent(
        goal=GOAL,
        session=session,
        selector=SequenceToolSelector(P0_TOOL_SEQUENCE),
        max_steps=8,
    )
    captured = capsys.readouterr()
    assert "MCP_DIAG" not in captured.out
    assert "MCP_DIAG" not in captured.err
    assert "CallToolResult" not in captured.out
