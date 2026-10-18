"""Deterministic MCP observation capture.

Converts plain observed tool data into ToolInvocation values and Design A
request maps. Does not execute MCP transports, evaluate metrics, or apply
quality gates.

Does not import the MCP SDK or DeepEval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ai_qe_eval.domain.conversation import ToolInvocation


def tool_invocation_from_observation(
    *,
    name: str,
    arguments: dict[str, Any] | None = None,
    result: Any | None = None,
) -> ToolInvocation:
    """Map one observed tool call into a domain ToolInvocation.

    Values are preserved exactly. None stays None. {} stays {}.
    """
    return ToolInvocation(name=name, arguments=arguments, result=result)


def capture_executed_tool(
    *,
    name: str,
    arguments: dict[str, Any] | None = None,
    execute: Callable[[], Any],
) -> ToolInvocation:
    """Execute a tool call and capture the result.

    If execute raises, the exception propagates. No ToolInvocation is returned
    and the failure is not rewritten as an empty tool-call list.
    """
    if not callable(execute):
        raise TypeError(
            "capture_executed_tool requires a callable execute, "
            f"got {type(execute).__name__}"
        )
    result = execute()
    return tool_invocation_from_observation(
        name=name,
        arguments=arguments,
        result=result,
    )


def _as_tool_invocations(
    calls: Sequence[ToolInvocation],
    *,
    field_name: str,
) -> list[ToolInvocation]:
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a sequence of ToolInvocation, "
            f"got {type(calls).__name__}"
        )
    materialized = list(calls)
    for index, call in enumerate(materialized):
        if not isinstance(call, ToolInvocation):
            raise TypeError(
                f"{field_name} items must be ToolInvocation, "
                f"got {type(call).__name__} at index {index}"
            )
    return materialized


def mcp_p0_request(
    *,
    observed_tool_calls: Sequence[ToolInvocation],
    expected_tool_calls: Sequence[ToolInvocation],
    final_state_ok: bool,
) -> dict[str, dict[str, list]]:
    """Build a Design A P0 request from captured MCP tool calls.

    Observed calls and expected calls are independent sequences. Expected
    calls are a QE specification input and are never derived from telemetry.
    final_state_ok is a caller-supplied bool.
    """
    if not isinstance(final_state_ok, bool):
        raise TypeError(
            "mcp_p0_request requires final_state_ok to be bool, "
            f"got {type(final_state_ok).__name__}"
        )
    observed = _as_tool_invocations(
        observed_tool_calls,
        field_name="observed_tool_calls",
    )
    expected = _as_tool_invocations(
        expected_tool_calls,
        field_name="expected_tool_calls",
    )
    return {
        "tool_correctness": {"args": [observed, expected]},
        "mcp_execution_health": {"args": [observed]},
        "final_state": {"args": [final_state_ok]},
    }
