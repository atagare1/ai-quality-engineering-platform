"""Focused tests for Langfuse observation capture.

Uses recorded observation-shaped dicts. Does not call Langfuse, OpenAI, or
the Agents SDK. Does not import those packages.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_qe_eval.capture.langfuse_trace import (
    langfuse_chat_correctness_requests,
    langfuse_chat_turn_relevancy_request,
    langfuse_correctness_request,
    langfuse_tool_correctness_request,
    observed_assistant_output,
    observed_chat_turns,
    observed_tool_invocations,
    observed_user_input,
    parse_observation_io,
    tool_invocation_from_langfuse_observation,
)
from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation

_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "capture"
    / "langfuse_trace.py"
)


def _tool_row(
    *,
    obs_id: str,
    name: str,
    start: str,
    trace_id: str = "trace-1",
    input_value: object = "{}",
    output_value: object = '{"ok": true}',
    level: str = "DEFAULT",
    status_message: str | None = None,
    parent: str = "parent-1",
) -> dict:
    return {
        "id": obs_id,
        "type": "TOOL",
        "name": name,
        "trace_id": trace_id,
        "parent_observation_id": parent,
        "start_time": start,
        "input": input_value,
        "output": output_value,
        "level": level,
        "status_message": status_message,
    }


def test_parse_observation_io_loads_json_object_strings():
    assert parse_observation_io('{"iana_timezone": "UTC"}') == {"iana_timezone": "UTC"}
    assert parse_observation_io({}) == {}
    assert parse_observation_io("not-json") == "not-json"


def test_tool_invocation_parses_string_input_as_arguments_dict():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="1",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            input_value='{"iana_timezone": "UTC"}',
            output_value='{"http_status": 200}',
        )
    )
    assert call.name == "fetch_timezone_clock"
    assert call.arguments == {"iana_timezone": "UTC"}
    assert call.result == {"http_status": 200}


def test_object_input_maps_iana_timezone_argument():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="1b",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            input_value={"iana_timezone": "UTC"},
            output_value={"http_status": 200},
        )
    )
    assert call.name == "fetch_timezone_clock"
    assert call.arguments == {"iana_timezone": "UTC"}


def test_httpbin_tool_result_preserves_slideshow_title_in_body():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="json-1",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:16Z",
            input_value={},
            output_value={
                "http_status": 200,
                "source_url": "https://httpbin.org/json",
                "body": '{"slideshow": {"title": "Sample Slide Show"}}',
            },
        )
    )
    assert call.name == "fetch_httpbin_json"
    assert call.result["http_status"] == 200
    assert "Sample Slide Show" in call.result["body"]


def test_empty_object_arguments_remain_empty_dict():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="2",
            name="fetch_public_uuid",
            start="2026-10-02T15:12:13Z",
            input_value="{}",
        )
    )
    assert call.arguments == {}


def test_error_observation_preserves_status_on_result():
    call = tool_invocation_from_langfuse_observation(
        _tool_row(
            obs_id="3",
            name="fetch_github_zen",
            start="2026-10-02T15:12:15Z",
            output_value=None,
            level="ERROR",
            status_message="Client error '415 Unsupported Media Type'",
        )
    )
    assert call.name == "fetch_github_zen"
    assert isinstance(call.result, dict)
    assert call.result["level"] == "ERROR"
    assert "415" in call.result["status_message"]
    assert call.result["output"] is None


def test_retries_remain_separate_ordered_entries():
    rows = [
        _tool_row(
            obs_id="c",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:16Z",
            parent="t3",
        ),
        _tool_row(
            obs_id="a",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
            parent="t1",
            input_value='{"iana_timezone": "UTC"}',
        ),
        _tool_row(
            obs_id="b",
            name="fetch_public_uuid",
            start="2026-10-02T15:12:13Z",
            parent="t2",
        ),
        _tool_row(
            obs_id="c2",
            name="fetch_httpbin_json",
            start="2026-10-02T15:12:19Z",
            parent="t4",
            level="ERROR",
            status_message="retry",
            output_value=None,
        ),
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == [
        "fetch_timezone_clock",
        "fetch_public_uuid",
        "fetch_httpbin_json",
        "fetch_httpbin_json",
    ]
    assert calls[3].result["status_message"] == "retry"


def test_generation_rows_are_not_treated_as_tool_calls():
    rows = [
        {
            "id": "g",
            "type": "GENERATION",
            "name": "generation",
            "trace_id": "trace-1",
            "start_time": "2026-10-02T15:12:00Z",
            "input": "[]",
            "output": '{"tool_calls": [{"function": {"name": "ignored"}}]}',
        },
        _tool_row(
            obs_id="a",
            name="fetch_timezone_clock",
            start="2026-10-02T15:12:10Z",
        ),
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == ["fetch_timezone_clock"]


def test_foreign_trace_id_is_rejected():
    rows = [
        _tool_row(obs_id="a", name="a", start="2026-10-02T15:12:10Z", trace_id="other"),
    ]
    with pytest.raises(ValueError, match="trace_id"):
        observed_tool_invocations(rows, trace_id="trace-1")


def test_tool_correctness_request_keeps_expected_independent_of_observed():
    observed_rows = [
        _tool_row(obs_id="a", name="fetch_timezone_clock", start="2026-10-02T15:12:10Z"),
        _tool_row(obs_id="b", name="fetch_public_uuid", start="2026-10-02T15:12:13Z"),
        _tool_row(obs_id="c", name="fetch_httpbin_json", start="2026-10-02T15:12:16Z"),
        {
            "id": "g",
            "type": "GENERATION",
            "name": "generation",
            "trace_id": "trace-1",
            "start_time": "2026-10-02T15:12:00Z",
        },
    ]
    expected = [
        ToolInvocation(name="fetch_httpbin_json"),
        ToolInvocation(name="fetch_timezone_clock"),
        ToolInvocation(name="fetch_public_uuid"),
    ]
    request = langfuse_tool_correctness_request(
        observations=observed_rows,
        trace_id="trace-1",
        expected_tool_calls=expected,
    )
    observed, expected_args = request["tool_correctness"]["args"]
    assert [call.name for call in observed] == [
        "fetch_timezone_clock",
        "fetch_public_uuid",
        "fetch_httpbin_json",
    ]
    assert [call.name for call in expected_args] == [
        "fetch_httpbin_json",
        "fetch_timezone_clock",
        "fetch_public_uuid",
    ]
    assert expected_args is not observed
    assert "kwargs" not in request["tool_correctness"]


def test_datetime_start_times_sort_with_strings():
    rows = [
        _tool_row(
            obs_id="b",
            name="second",
            start="2026-10-02T15:12:13.000000+00:00",
        ),
        {
            **_tool_row(obs_id="a", name="first", start="ignored"),
            "start_time": datetime(2026, 10, 2, 15, 12, 10, tzinfo=timezone.utc),
        },
    ]
    calls = observed_tool_invocations(rows, trace_id="trace-1")
    assert [call.name for call in calls] == ["first", "second"]


USER_PROMPT = (
    "Collect UTC time, a public UUID, and httpbin json, then summarize all three."
)
FINAL_ANSWER = (
    "I collected all three values successfully: the UTC clock API returned "
    "the datetime and the slideshow title \"Sample Slide Show\"."
)
QE_EXPECTED = "Quote the UTC datetime, UUID, and Sample Slide Show from the tools."


def _generation_row(
    *,
    obs_id: str,
    start: str,
    trace_id: str = "trace-1",
    input_value: object,
    output_value: object,
) -> dict:
    return {
        "id": obs_id,
        "type": "GENERATION",
        "name": "generation",
        "trace_id": trace_id,
        "start_time": start,
        "input": input_value,
        "output": output_value,
    }


def _tool_call_generation(*, obs_id: str = "g1", start: str = "2026-10-03T10:21:48Z") -> dict:
    return _generation_row(
        obs_id=obs_id,
        start=start,
        input_value=[
            {"role": "system", "content": "Call tools in order."},
            {"role": "user", "content": USER_PROMPT},
        ],
        output_value=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-clock",
                        "type": "function",
                        "function": {
                            "name": "fetch_timezone_clock",
                            "arguments": '{"iana_timezone":"UTC"}',
                        },
                    }
                ],
            }
        ],
    )


def _final_generation(*, obs_id: str = "g2", start: str = "2026-10-03T10:22:08Z") -> dict:
    return _generation_row(
        obs_id=obs_id,
        start=start,
        input_value=[
            {"role": "system", "content": "Call tools in order."},
            {"role": "user", "content": USER_PROMPT},
        ],
        output_value=[
            {
                "role": "assistant",
                "content": FINAL_ANSWER,
                "tool_calls": None,
            }
        ],
    )


def test_correctness_extracts_first_user_and_last_assistant_text():
    rows = [
        _final_generation(),
        _tool_call_generation(),
        _tool_row(
            obs_id="clock",
            name="fetch_timezone_clock",
            start="2026-10-03T10:22:06Z",
        ),
    ]
    request = langfuse_correctness_request(
        observations=rows,
        trace_id="trace-1",
        expected=QE_EXPECTED,
    )
    assert request["correctness"]["args"] == [USER_PROMPT, FINAL_ANSWER, QE_EXPECTED]
    assert "kwargs" not in request["correctness"]
    assert observed_user_input(rows, trace_id="trace-1") == USER_PROMPT
    assert observed_assistant_output(rows, trace_id="trace-1") == FINAL_ANSWER


def test_correctness_uses_earliest_user_and_latest_assistant_by_start_time():
    rows = [
        _generation_row(
            obs_id="later",
            start="2026-10-03T10:22:10Z",
            input_value=[{"role": "user", "content": "later question"}],
            output_value=[{"role": "assistant", "content": "later answer"}],
        ),
        _generation_row(
            obs_id="earlier",
            start="2026-10-03T10:21:48Z",
            input_value=[{"role": "user", "content": "earlier question"}],
            output_value=[{"role": "assistant", "content": "earlier answer"}],
        ),
    ]
    request = langfuse_correctness_request(
        observations=rows,
        trace_id="trace-1",
        expected=QE_EXPECTED,
    )
    assert request["correctness"]["args"] == [
        "earlier question",
        "later answer",
        QE_EXPECTED,
    ]


def test_tool_call_only_generation_is_not_assistant_text():
    rows = [_tool_call_generation()]
    assert observed_user_input(rows, trace_id="trace-1") == USER_PROMPT
    assert observed_assistant_output(rows, trace_id="trace-1") is None
    with pytest.raises(ValueError, match="final non-empty role=assistant"):
        langfuse_correctness_request(
            observations=rows,
            trace_id="trace-1",
            expected=QE_EXPECTED,
        )
    content_with_tools = _generation_row(
        obs_id="g-tools-text",
        start="2026-10-03T10:22:09Z",
        input_value=[{"role": "user", "content": USER_PROMPT}],
        output_value=[
            {
                "role": "assistant",
                "content": "I will call tools now.",
                "tool_calls": [{"id": "call-2", "type": "function"}],
            }
        ],
    )
    assert observed_assistant_output(
        [_tool_call_generation(), content_with_tools],
        trace_id="trace-1",
    ) is None


def test_missing_final_output_raises_even_when_user_input_exists():
    rows = [
        _generation_row(
            obs_id="empty-out",
            start="2026-10-03T10:22:08Z",
            input_value=[{"role": "user", "content": USER_PROMPT}],
            output_value=[{"role": "assistant", "content": "   ", "tool_calls": None}],
        )
    ]
    with pytest.raises(ValueError, match="final non-empty role=assistant"):
        langfuse_correctness_request(
            observations=rows,
            trace_id="trace-1",
            expected=QE_EXPECTED,
        )


def test_correctness_expected_stays_qe_supplied():
    rows = [_tool_call_generation(), _final_generation()]
    request = langfuse_correctness_request(
        observations=rows,
        trace_id="trace-1",
        expected=QE_EXPECTED,
    )
    assert request["correctness"]["args"][2] is QE_EXPECTED
    assert FINAL_ANSWER not in QE_EXPECTED


def _chat_root(
    *,
    obs_id: str,
    start: str,
    user_text: str,
    assistant_text: str | None,
    session_id: str = "session-1",
    name: str = "handle-chat-message",
    is_root: bool = True,
    extra: dict | None = None,
) -> dict:
    row = {
        "id": obs_id,
        "type": "SPAN",
        "name": name,
        "session_id": session_id,
        "start_time": start,
        "is_root_observation": is_root,
        "input": user_text,
        "output": assistant_text,
    }
    if extra:
        row.update(extra)
    return row


def test_chat_roots_map_two_turns_in_start_time_order():
    rows = [
        _chat_root(
            obs_id="later",
            start="2026-10-04T17:07:26Z",
            user_text="How do I group several of those messages into one session?",
            assistant_text="Use a session identifier.",
        ),
        {
            "id": "generation",
            "type": "GENERATION",
            "name": "chat openai/gpt-4o-mini",
            "session_id": "session-1",
            "start_time": "2026-10-04T17:07:24Z",
            "is_root_observation": False,
            "input": [{"role": "user", "parts": [{"type": "text", "content": "ignored"}]}],
            "output": [{"role": "assistant", "parts": [{"type": "text", "content": "ignored"}]}],
        },
        _chat_root(
            obs_id="earlier",
            start="2026-10-04T17:07:24Z",
            user_text="What is Langfuse in one sentence?",
            assistant_text="Langfuse is an observability platform.",
        ),
    ]
    turns = observed_chat_turns(rows, session_id="session-1")
    assert [(turn.role, turn.content) for turn in turns] == [
        ("user", "What is Langfuse in one sentence?"),
        ("assistant", "Langfuse is an observability platform."),
        ("user", "How do I group several of those messages into one session?"),
        ("assistant", "Use a session identifier."),
    ]


def test_chat_roots_ignore_non_root_and_other_names():
    rows = [
        _chat_root(
            obs_id="nested-same-name",
            start="2026-10-04T17:07:24Z",
            user_text="nested",
            assistant_text="should not count",
            is_root=False,
        ),
        _chat_root(
            obs_id="other-name",
            start="2026-10-04T17:07:25Z",
            user_text="other",
            assistant_text="should not count",
            name="invoke_agent openai/gpt-4o-mini",
        ),
        _chat_root(
            obs_id="root",
            start="2026-10-04T17:07:26Z",
            user_text="What is Langfuse in one sentence?",
            assistant_text="Langfuse is an observability platform.",
        ),
    ]
    turns = observed_chat_turns(rows, session_id="session-1")
    assert [turn.content for turn in turns] == [
        "What is Langfuse in one sentence?",
        "Langfuse is an observability platform.",
    ]


def test_chat_root_missing_output_raises():
    rows = [
        _chat_root(
            obs_id="incomplete",
            start="2026-10-04T17:07:24Z",
            user_text="What is Langfuse in one sentence?",
            assistant_text=None,
        )
    ]
    with pytest.raises(ValueError, match="missing a non-empty output"):
        observed_chat_turns(rows, session_id="session-1")


def test_chat_request_shapes_for_turn_relevancy_and_per_turn_geval():
    rows = [
        _chat_root(
            obs_id="t1",
            start="2026-10-04T17:07:24Z",
            user_text="What is Langfuse in one sentence?",
            assistant_text="Langfuse is an observability platform.",
        ),
        _chat_root(
            obs_id="t2",
            start="2026-10-04T17:07:26Z",
            user_text="How do I group several of those messages into one session?",
            assistant_text="Use a session identifier.",
        ),
    ]
    turn_request = langfuse_chat_turn_relevancy_request(
        observations=rows,
        session_id="session-1",
    )
    turns = turn_request["turn_relevancy"]["args"][0]
    assert "kwargs" not in turn_request["turn_relevancy"]
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert all(isinstance(turn, ConversationTurn) for turn in turns)

    correctness_requests = langfuse_chat_correctness_requests(
        observations=rows,
        session_id="session-1",
        expected=QE_EXPECTED,
    )
    assert [request["correctness"]["args"] for request in correctness_requests] == [
        [
            "What is Langfuse in one sentence?",
            "Langfuse is an observability platform.",
            QE_EXPECTED,
        ],
        [
            "How do I group several of those messages into one session?",
            "Use a session identifier.",
            QE_EXPECTED,
        ],
    ]
    assert correctness_requests[0]["correctness"]["args"][2] is QE_EXPECTED


def test_capture_module_does_not_import_vendor_sdks():
    tree = ast.parse(_CAPTURE_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert "langfuse" not in imported
    assert "agents" not in imported
    assert "openai" not in imported
    assert "deepeval" not in imported
