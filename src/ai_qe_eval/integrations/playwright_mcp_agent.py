"""Thin Playwright MCP agent-execution boundary.

Runs an injected selector against an existing MCP ClientSession and captures
ordered ToolInvocation observations. Does not choose tools itself, build
expected_tool_calls, or run evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_qe_eval.capture.mcp_trace import tool_invocation_from_observation
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.integrations.playwright_mcp import serialize_call_tool_result
from ai_qe_eval.integrations.playwright_mcp_selector import ToolSelector


@dataclass(frozen=True)
class PlaywrightMcpAgentRun:
    goal: str
    observed_tool_calls: list[ToolInvocation]
    output: str


def _next_selection(
    selector: ToolSelector,
    goal: str,
    observed: list[ToolInvocation],
    last_plain_result: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None] | None:
    selected = selector.select_next(goal, list(observed), last_plain_result)
    if selected is None:
        return None
    if not isinstance(selected, tuple) or len(selected) != 2:
        raise TypeError(
            "select_next must return (name, arguments) or None, "
            f"got {type(selected).__name__}"
        )
    name, arguments = selected
    if not isinstance(name, str) or not name:
        raise TypeError("select_next name must be a non-empty str")
    if arguments is not None and not isinstance(arguments, dict):
        raise TypeError(
            "select_next arguments must be a dict or None, "
            f"got {type(arguments).__name__}"
        )
    return name, arguments


async def run_playwright_mcp_agent(
    *,
    goal: str,
    session: Any,
    selector: ToolSelector,
    max_steps: int,
    on_tool_call: Callable[..., Any] | None = None,
) -> PlaywrightMcpAgentRun:
    """Execute selector-chosen Playwright MCP tools and capture observations."""
    if not isinstance(goal, str) or not goal:
        raise ValueError("goal must be a non-empty str")
    if not hasattr(selector, "select_next") or not callable(selector.select_next):
        raise TypeError("selector must provide a select_next method")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be an int >= 1")
    if not hasattr(session, "call_tool"):
        raise TypeError("session must provide an async call_tool method")

    observed: list[ToolInvocation] = []
    last_plain_result: dict[str, Any] | None = None
    for _ in range(max_steps):
        selected = _next_selection(selector, goal, observed, last_plain_result)
        if selected is None:
            break
        name, arguments = selected
        mcp_result = await session.call_tool(name, arguments)
        plain = serialize_call_tool_result(mcp_result)
        invocation = tool_invocation_from_observation(
            name=name,
            arguments=arguments,
            result=plain,
        )
        observed.append(invocation)
        last_plain_result = plain
        if on_tool_call is not None:
            on_tool_call(invocation, order=len(observed))

    return PlaywrightMcpAgentRun(
        goal=goal,
        observed_tool_calls=observed,
        output=f"Executed {len(observed)} Playwright MCP tool call(s).",
    )
