"""Focused tests for deterministic MCP observation capture.

Does not import the MCP SDK, DeepEval, Runner, Policy, or Gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.capture.mcp_trace import (
    capture_executed_tool,
    mcp_p0_request,
    tool_invocation_from_observation,
)
from ai_qe_eval.domain.conversation import ToolInvocation

_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "capture"
    / "mcp_trace.py"
)


def test_single_capture_preserves_name_arguments_and_result():
    call = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )
    assert isinstance(call, ToolInvocation)
    assert call.name == "weather"
    assert call.arguments == {"location": "Pune"}
    assert call.result == "Temperature is 28 C and conditions are clear."


def test_multiple_captures_preserve_order():
    first = tool_invocation_from_observation(
        name="lookup_city",
        arguments={"q": "Pune"},
        result="in",
    )
    second = tool_invocation_from_observation(
        name="weather",
        arguments={"location": "Pune"},
        result="28 C",
    )
    observed = [first, second]
    assert [call.name for call in observed] == ["lookup_city", "weather"]
    assert observed[0].arguments == {"q": "Pune"}
    assert observed[1].result == "28 C"


def test_none_arguments_remain_none_and_empty_dict_remains_empty():
    none_args = tool_invocation_from_observation(
        name="weather",
        arguments=None,
        result=None,
    )
    empty_args = tool_invocation_from_observation(
        name="weather",
        arguments={},
        result=None,
    )
    assert none_args.arguments is None
    assert empty_args.arguments == {}
    assert none_args.arguments is not empty_args.arguments
    assert none_args != empty_args


def test_result_is_preserved_exactly():
    nested = {"status": "ok", "rows": [{"id": 1}]}
    call = tool_invocation_from_observation(
        name="query",
        arguments={"sql": "select 1"},
        result=nested,
    )
    assert call.result is nested
    assert call.result["rows"][0]["id"] == 1


def test_capture_executed_tool_propagates_failure_without_fabricating_calls():
    def boom():
        raise RuntimeError("tool transport failed")

    with pytest.raises(RuntimeError, match="tool transport failed"):
        capture_executed_tool(
            name="weather",
            arguments={"location": "Pune"},
            execute=boom,
        )


def test_capture_executed_tool_records_successful_result():
    call = capture_executed_tool(
        name="weather",
        arguments={"location": "Pune"},
        execute=lambda: "Temperature is 28 C and conditions are clear.",
    )
    assert call.name == "weather"
    assert call.arguments == {"location": "Pune"}
    assert call.result == "Temperature is 28 C and conditions are clear."


def test_mcp_p0_request_keeps_expected_independent_of_observed():
    observed = [
        tool_invocation_from_observation(name="a", arguments={"n": 1}, result=1),
        tool_invocation_from_observation(name="b", arguments={"n": 2}, result=2),
    ]
    expected_calls = [
        tool_invocation_from_observation(name="b", arguments=None, result=None),
        tool_invocation_from_observation(name="a", arguments=None, result=None),
    ]
    request = mcp_p0_request(
        observed_tool_calls=observed,
        expected_tool_calls=expected_calls,
        final_state_ok=True,
    )
    assert set(request) == {
        "tool_correctness",
        "mcp_execution_health",
        "final_state",
    }
    observed_args, expected_args = request["tool_correctness"]["args"]
    assert [call.name for call in observed_args] == ["a", "b"]
    assert [call.name for call in expected_args] == ["b", "a"]
    assert expected_args is not observed_args
    assert request["mcp_execution_health"]["args"][0] is observed_args
    assert request["final_state"]["args"] == [True]
    assert "kwargs" not in request["tool_correctness"]


def test_mcp_p0_request_rejects_non_bool_final_state():
    observed = [tool_invocation_from_observation(name="a")]
    expected_calls = [tool_invocation_from_observation(name="a")]
    with pytest.raises(TypeError, match="final_state_ok"):
        mcp_p0_request(
            observed_tool_calls=observed,
            expected_tool_calls=expected_calls,
            final_state_ok=1,  # type: ignore[arg-type]
        )


def test_capture_module_does_not_import_mcp_or_deepeval():
    tree = ast.parse(_CAPTURE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "mcp" not in imported
    assert "deepeval" not in imported
