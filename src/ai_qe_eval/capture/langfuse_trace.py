"""Langfuse observation capture.

Converts retrieved Langfuse observation dicts into ToolInvocation values and
Design A request maps. Does not call the Langfuse API, run an agent, or
evaluate metrics.

Does not import the Langfuse SDK, OpenAI Agents SDK, or DeepEval.

Observed tool calls come only from TOOL observations. OpenAI Agents
user/assistant text comes from GENERATION observations. User-feedback
chat turns come from root handle-chat-message SPAN input/output.
Expected tool calls and expected answers are supplied by the caller
and are never derived from telemetry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation

CHAT_ROOT_NAME = "handle-chat-message"


def parse_observation_io(value: Any) -> Any:
    """Parse Observations API v2 I/O strings. Non-JSON values are unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _row_type(row: Mapping[str, Any]) -> str:
    return str(row.get("type") or "").upper()


def _row_trace_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("trace_id")
    if value is None:
        value = row.get("traceId")
    return str(value) if value is not None else None


def _row_session_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("session_id")
    if value is None:
        value = row.get("sessionId")
    return str(value) if value is not None else None


def _is_root_observation(row: Mapping[str, Any]) -> bool:
    value = row.get("is_root_observation")
    if value is None:
        value = row.get("isRootObservation")
    return value is True


def _row_start_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    start = row.get("start_time")
    if start is None:
        start = row.get("startTime")
    if hasattr(start, "isoformat"):
        start_key = start.isoformat()
    else:
        start_key = str(start or "")
    return (start_key, str(row.get("id") or ""))


def _status_message(row: Mapping[str, Any]) -> Any:
    if row.get("status_message") is not None:
        return row.get("status_message")
    return row.get("statusMessage")


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


def tool_invocation_from_langfuse_observation(
    row: Mapping[str, Any],
) -> ToolInvocation:
    """Map one Langfuse TOOL observation into a domain ToolInvocation.

    Retries stay as separate rows. Errors are preserved on result when Langfuse
    reports level=ERROR or a status_message; ToolInvocation has no error field.
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            "Langfuse observation must be a mapping, "
            f"got {type(row).__name__}"
        )
    name = row.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Langfuse TOOL observation is missing a string name")
    parsed_input = parse_observation_io(row.get("input"))
    arguments: dict[str, Any] | None
    if parsed_input is None:
        arguments = None
    elif isinstance(parsed_input, dict):
        nested = parsed_input.get("arguments")
        arguments = nested if isinstance(nested, dict) else parsed_input
    else:
        arguments = None
    parsed_output = parse_observation_io(row.get("output"))
    level = row.get("level")
    status = _status_message(row)
    if str(level or "").upper() == "ERROR" or status:
        result: Any = {
            "output": parsed_output,
            "level": level,
            "status_message": status,
        }
    else:
        result = parsed_output
    return ToolInvocation(name=name, arguments=arguments, result=result)


def _as_messages(value: Any) -> list[Mapping[str, Any]]:
    parsed = parse_observation_io(value)
    if isinstance(parsed, Mapping) and not isinstance(parsed, (str, bytes)):
        return [parsed]
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return [
            item
            for item in parsed
            if isinstance(item, Mapping) and not isinstance(item, (str, bytes))
        ]
    return []


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    return value


def _has_tool_calls(message: Mapping[str, Any]) -> bool:
    calls = message.get("tool_calls")
    if calls is None:
        return False
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        return len(calls) > 0
    return bool(calls)


def _user_text(value: Any) -> str | None:
    for message in _as_messages(value):
        if str(message.get("role") or "") != "user":
            continue
        text = _nonempty_text(message.get("content"))
        if text is not None:
            return text
    return None


def _assistant_text(value: Any) -> str | None:
    last: str | None = None
    for message in _as_messages(value):
        if str(message.get("role") or "") != "assistant":
            continue
        if _has_tool_calls(message):
            continue
        text = _nonempty_text(message.get("content"))
        if text is not None:
            last = text
    return last


def _typed_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str | None,
    observation_type: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError(
            "observations must be a sequence of mappings, "
            f"got {type(observations).__name__}"
        )
    selected: list[Mapping[str, Any]] = []
    for index, row in enumerate(observations):
        if not isinstance(row, Mapping):
            raise TypeError(
                "observations items must be mappings, "
                f"got {type(row).__name__} at index {index}"
            )
        row_trace = _row_trace_id(row)
        if trace_id is not None and row_trace not in (None, trace_id):
            raise ValueError(
                f"observation {row.get('id')!r} has trace_id {row_trace!r}, "
                f"expected {trace_id!r}"
            )
        if _row_type(row) != observation_type:
            continue
        if trace_id is not None and row_trace is None:
            raise ValueError(
                f"{observation_type} observation {row.get('id')!r} is missing "
                "trace_id"
            )
        selected.append(row)
    selected.sort(key=_row_start_sort_key)
    return selected


def observed_user_input(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str | None = None,
) -> str | None:
    """Return the first non-empty GENERATION role=user content."""
    for row in _typed_rows(
        observations, trace_id=trace_id, observation_type="GENERATION"
    ):
        text = _user_text(row.get("input"))
        if text is not None:
            return text
    return None


def observed_assistant_output(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str | None = None,
) -> str | None:
    """Return the last non-empty GENERATION assistant text.

    Assistant messages that include tool_calls are not treated as text.
    """
    last: str | None = None
    for row in _typed_rows(
        observations, trace_id=trace_id, observation_type="GENERATION"
    ):
        text = _assistant_text(row.get("output"))
        if text is not None:
            last = text
    return last


def langfuse_correctness_request(
    *,
    observations: Sequence[Mapping[str, Any]],
    trace_id: str,
    expected: Any,
) -> dict[str, dict[str, list[Any]]]:
    """Build a Design A G-Eval Correctness request from GENERATION rows.

    input is the first non-empty user content. output is the last settled
    assistant text. expected remains a QE specification and is never read
    from Langfuse.
    """
    user_input = observed_user_input(observations, trace_id=trace_id)
    output = observed_assistant_output(observations, trace_id=trace_id)
    if user_input is None:
        raise ValueError(
            "Langfuse GENERATION observations are missing a non-empty "
            "role=user input"
        )
    if output is None:
        raise ValueError(
            "Langfuse GENERATION observations are missing a final "
            "non-empty role=assistant output"
        )
    return {
        "correctness": {
            "args": [user_input, output, expected],
        }
    }


def observed_chat_roots(
    observations: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
) -> list[Mapping[str, Any]]:
    """Return root handle-chat-message rows for one session, by start_time."""
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError(
            "observations must be a sequence of mappings, "
            f"got {type(observations).__name__}"
        )
    selected: list[Mapping[str, Any]] = []
    for index, row in enumerate(observations):
        if not isinstance(row, Mapping):
            raise TypeError(
                "observations items must be mappings, "
                f"got {type(row).__name__} at index {index}"
            )
        row_session = _row_session_id(row)
        if row_session not in (None, session_id):
            raise ValueError(
                f"observation {row.get('id')!r} has session_id {row_session!r}, "
                f"expected {session_id!r}"
            )
        if row.get("name") != CHAT_ROOT_NAME:
            continue
        if not _is_root_observation(row):
            continue
        if row_session is None:
            raise ValueError(
                f"root {CHAT_ROOT_NAME} observation {row.get('id')!r} is "
                "missing session_id"
            )
        selected.append(row)
    selected.sort(key=_row_start_sort_key)
    return selected


def observed_chat_turns(
    observations: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
) -> list[ConversationTurn]:
    """Map each complete root SPAN into user then assistant ConversationTurns."""
    turns: list[ConversationTurn] = []
    for row in observed_chat_roots(observations, session_id=session_id):
        user_text = _nonempty_text(parse_observation_io(row.get("input")))
        assistant_text = _nonempty_text(parse_observation_io(row.get("output")))
        if user_text is None:
            raise ValueError(
                f"root {CHAT_ROOT_NAME} observation {row.get('id')!r} is "
                "missing a non-empty input"
            )
        if assistant_text is None:
            raise ValueError(
                f"root {CHAT_ROOT_NAME} observation {row.get('id')!r} is "
                "missing a non-empty output"
            )
        turns.append(ConversationTurn(role="user", content=user_text))
        turns.append(ConversationTurn(role="assistant", content=assistant_text))
    if not turns:
        raise ValueError(
            "Langfuse session has no complete root handle-chat-message turns"
        )
    return turns


def langfuse_chat_turn_relevancy_request(
    *,
    observations: Sequence[Mapping[str, Any]],
    session_id: str,
) -> dict[str, dict[str, list[list[ConversationTurn]]]]:
    """Build a Design A Turn Relevancy request from root chat SPANs."""
    return {
        "turn_relevancy": {
            "args": [observed_chat_turns(observations, session_id=session_id)],
        }
    }


def langfuse_chat_correctness_requests(
    *,
    observations: Sequence[Mapping[str, Any]],
    session_id: str,
    expected: Any,
) -> list[dict[str, dict[str, list[Any]]]]:
    """Build one G-Eval Correctness request per complete root chat turn.

    expected remains a QE specification and is never read from Langfuse.
    """
    turns = observed_chat_turns(observations, session_id=session_id)
    requests: list[dict[str, dict[str, list[Any]]]] = []
    for index in range(0, len(turns), 2):
        requests.append(
            {
                "correctness": {
                    "args": [turns[index].content, turns[index + 1].content, expected],
                }
            }
        )
    return requests


def langfuse_tool_correctness_request(
    *,
    observations: Sequence[Mapping[str, Any]],
    trace_id: str,
    expected_tool_calls: Sequence[ToolInvocation],
) -> dict[str, dict[str, list[ToolInvocation]]]:
    """Build a Design A ToolCorrectness request from Langfuse observations.

    Observed calls come only from TOOL observations. expected_tool_calls is
    a QE specification input and is never derived from telemetry.
    """
    observed = observed_tool_invocations(observations, trace_id=trace_id)
    expected = _as_tool_invocations(
        expected_tool_calls,
        field_name="expected_tool_calls",
    )
    return {
        "tool_correctness": {
            "args": [observed, expected],
        }
    }


def observed_tool_invocations(
    observations: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
) -> list[ToolInvocation]:
    """Return TOOL observations for one trace, ordered by start_time then id."""
    tool_rows = _typed_rows(
        observations, trace_id=trace_id, observation_type="TOOL"
    )
    return [tool_invocation_from_langfuse_observation(row) for row in tool_rows]
