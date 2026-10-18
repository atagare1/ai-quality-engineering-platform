"""Scripted flagship MCP P0 demo.

Reuses run_playwright_mcp_agent, SequenceToolSelector, capture helpers, and
mcp_p0_request. Does not start a live Playwright MCP server or evaluate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from ai_qe_eval.capture.mcp_trace import mcp_p0_request, tool_invocation_from_observation
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.integrations.playwright_mcp import (
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    snapshot_body_from_serialized_result,
    snapshot_contains_list_item,
)
from ai_qe_eval.integrations.playwright_mcp_agent import run_playwright_mcp_agent
from ai_qe_eval.integrations.playwright_mcp_selector import SequenceToolSelector

GOAL = "Add 'Buy milk' to the Playwright TodoMVC demo."
EXPECTED_TODO_TEXT = "Buy milk"
# Existing P0.3a negative scenario: same tools, wrong application outcome.
TYPED_TODO_TEXT_FAIL = "Buy bread"
EXPECTED_TOOL_ORDER = (
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_click",
    "browser_snapshot",
)
_EMPTY_SNAPSHOT = """
- main:
  - textbox "What needs to be done?" [ref=e5]
  - heading "todos" [ref=e3]
""".strip()


class ScriptedMcpSession:
    """Deterministic MCP session fixture. Not a live transport."""

    def __init__(self, results: Sequence[CallToolResult]) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self._results = list(results)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if not self._results:
            raise RuntimeError(f"Scripted MCP session has no result for {name!r}")
        return self._results.pop(0)


def p0_tool_sequence(*, todo_text: str) -> list[tuple[str, dict]]:
    return [
        ("browser_navigate", {"url": TODO_MVC_URL}),
        ("browser_snapshot", {}),
        ("browser_click", {"element": "Todo input textbox", "target": "e5"}),
        (
            "browser_type",
            {
                "element": "Todo input textbox",
                "target": "e5",
                "text": todo_text,
                "submit": True,
            },
        ),
        ("browser_click", {"element": "Todos heading", "target": "e3"}),
        ("browser_snapshot", {}),
    ]


def expected_p0_tool_calls() -> list[ToolInvocation]:
    return [
        tool_invocation_from_observation(name=name, arguments=None, result=None)
        for name in EXPECTED_TOOL_ORDER
    ]


def final_state_contains_expected_todo(observed: Sequence[ToolInvocation]) -> bool:
    for call in reversed(list(observed or [])):
        if call.name == "browser_snapshot" and isinstance(call.result, dict):
            return snapshot_contains_list_item(
                snapshot_body_from_serialized_result(call.result),
                EXPECTED_TODO_TEXT,
            )
    return False


def _ok(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=None,
        isError=False,
    )


def _todo_snapshot(todo_text: str) -> str:
    return f"""
- main:
  - textbox "What needs to be done?" [ref=e5]
  - heading "todos" [ref=e3]
  - list:
    - listitem [ref=e10]:
      - generic [ref=e12]: {todo_text}
""".strip()


def scripted_session(scenario: str) -> ScriptedMcpSession:
    todo_text = EXPECTED_TODO_TEXT if scenario == "pass" else TYPED_TODO_TEXT_FAIL
    return ScriptedMcpSession(
        [
            _ok("navigated"),
            _ok(_EMPTY_SNAPSHOT),
            _ok("clicked"),
            _ok("typed"),
            _ok("clicked"),
            _ok(_todo_snapshot(todo_text)),
        ]
    )


def _ref_for_label(snapshot_text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}.*?\[ref=(e\d+)\]"
    match = re.search(pattern, snapshot_text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not find ref for label {label!r} in snapshot")
    return match.group(1)


class TodoMvcLiveSelector:
    """P0 TodoMVC steps; snapshot refs come from the live page, not fixtures."""

    def __init__(self, *, todo_text: str) -> None:
        self._todo_text = todo_text
        self._step = 0
        self._textbox_ref: str | None = None
        self._heading_ref: str | None = None

    def select_next(self, goal, observed, last_plain_result):
        if self._step == 0:
            self._step = 1
            return ("browser_navigate", {"url": TODO_MVC_URL})
        if self._step == 1:
            self._step = 2
            return ("browser_snapshot", {})
        if self._step == 2:
            if not isinstance(last_plain_result, dict):
                raise RuntimeError("browser_snapshot did not return a serialized result")
            body = snapshot_body_from_serialized_result(last_plain_result)
            self._textbox_ref = _ref_for_label(
                body, 'textbox "What needs to be done?"'
            )
            self._heading_ref = _ref_for_label(body, 'heading "todos"')
            self._step = 3
            return (
                "browser_click",
                {"element": "Todo input textbox", "target": self._textbox_ref},
            )
        if self._step == 3:
            self._step = 4
            return (
                "browser_type",
                {
                    "element": "Todo input textbox",
                    "target": self._textbox_ref,
                    "text": self._todo_text,
                    "submit": True,
                },
            )
        if self._step == 4:
            self._step = 5
            return (
                "browser_click",
                {"element": "Todos heading", "target": self._heading_ref},
            )
        if self._step == 5:
            self._step = 6
            return ("browser_snapshot", {})
        return None


async def live_mcp_p0_request(
    scenario: str = "pass",
    *,
    on_tool_call: Callable[..., Any] | None = None,
) -> dict[str, dict[str, list]]:
    """Run the P0 flow on a real Playwright MCP stdio ClientSession."""
    if scenario not in {"pass", "fail"}:
        raise ValueError(f"Unsupported MCP P0 demo scenario: {scenario!r}")
    todo_text = EXPECTED_TODO_TEXT if scenario == "pass" else TYPED_TODO_TEXT_FAIL
    async with stdio_client(playwright_mcp_stdio_parameters()) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            run = await run_playwright_mcp_agent(
                goal=GOAL,
                session=session,
                selector=TodoMvcLiveSelector(todo_text=todo_text),
                max_steps=8,
                on_tool_call=on_tool_call,
            )
    return mcp_p0_request(
        observed_tool_calls=run.observed_tool_calls,
        expected_tool_calls=expected_p0_tool_calls(),
        final_state_ok=final_state_contains_expected_todo(run.observed_tool_calls),
    )


async def scripted_mcp_p0_request(
    scenario: str = "pass",
    *,
    on_tool_call: Callable[..., Any] | None = None,
) -> dict[str, dict[str, list]]:
    """Execute the scripted P0 flow and return a Design A request map."""
    if scenario not in {"pass", "fail"}:
        raise ValueError(f"Unsupported MCP P0 demo scenario: {scenario!r}")
    todo_text = EXPECTED_TODO_TEXT if scenario == "pass" else TYPED_TODO_TEXT_FAIL
    run = await run_playwright_mcp_agent(
        goal=GOAL,
        session=scripted_session(scenario),
        selector=SequenceToolSelector(p0_tool_sequence(todo_text=todo_text)),
        max_steps=8,
        on_tool_call=on_tool_call,
    )
    return mcp_p0_request(
        observed_tool_calls=run.observed_tool_calls,
        expected_tool_calls=expected_p0_tool_calls(),
        final_state_ok=final_state_contains_expected_todo(run.observed_tool_calls),
    )
