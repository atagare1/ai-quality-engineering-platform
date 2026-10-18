"""Langfuse Observations API v2 retrieval boundary.

Paginates langfuse.api.observations.get_many for one trace_id or
session_id and returns plain observation mappings. ingest_langfuse_trace
settles, then builds a ToolCorrectness request map.
ingest_langfuse_correctness settles until a final assistant GENERATION
is present, then builds a G-Eval Correctness request map.

Does not import the OpenAI Agents SDK, Langfuse SDK, evaluators, Runner,
Policy, or Gate. The client is injected. Does not use langfuse.trace() or
api.trace.list.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_qe_eval.capture.langfuse_trace import (
    langfuse_correctness_request,
    langfuse_tool_correctness_request,
    observed_assistant_output,
)
from ai_qe_eval.domain.conversation import ToolInvocation

DEFAULT_PAGE_LIMIT = 50
DEFAULT_SETTLE_TIMEOUT_S = 45.0
DEFAULT_POLL_INTERVAL_S = 0.5
OBSERVATION_FIELDS = (
    "core,basic,io,metadata,time,trace_context,model,usage,prompt,metrics"
)

SleepFn = Callable[[float], None]
MonotonicFn = Callable[[], float]
SettlePredicate = Callable[[list[dict[str, Any]]], bool]


class LangfuseTraceIncompleteError(TimeoutError):
    """Raised when paginated retrieval never meets the settle condition."""

    def __init__(
        self,
        message: str,
        *,
        trace_id: str,
        observations: list[dict[str, Any]],
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.observations = observations
        self.attempts = attempts


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump())
        except TypeError:
            return _to_plain(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _to_plain(value.dict())
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _observation_rows(payload: Any) -> list[dict[str, Any]]:
    plain = _to_plain(payload)
    if isinstance(plain, dict):
        data = plain.get("data", plain)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return [plain]
    if isinstance(plain, list):
        return [item for item in plain if isinstance(item, dict)]
    return []


def _payload_cursor(payload: Any) -> str | None:
    plain = _to_plain(payload)
    if not isinstance(plain, dict):
        return None
    meta = plain.get("meta")
    if not isinstance(meta, dict):
        return None
    cursor = meta.get("cursor")
    if cursor is None or cursor == "":
        return None
    return str(cursor)


def _is_tool_row(row: dict[str, Any]) -> bool:
    return str(row.get("type") or "").upper() == "TOOL"


def min_tool_observations(min_count: int) -> SettlePredicate:
    """Settle when at least min_count TOOL observations are queryable."""
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    def is_complete(rows: list[dict[str, Any]]) -> bool:
        return sum(1 for row in rows if _is_tool_row(row)) >= min_count

    return is_complete


def has_final_assistant_generation(trace_id: str | None = None) -> SettlePredicate:
    """Settle when a GENERATION row has non-empty assistant text.

    Tool-call-only assistant outputs do not satisfy this predicate.
    """

    def is_complete(rows: list[dict[str, Any]]) -> bool:
        try:
            return observed_assistant_output(rows, trace_id=trace_id) is not None
        except (TypeError, ValueError):
            return False

    return is_complete


def flush_langfuse_client(client: Any) -> bool:
    """Flush if the injected client exposes flush(). Does not import Langfuse."""
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return False
    flush()
    return True


def fetch_observation_pages(
    client: Any,
    *,
    trace_id: str,
    from_start_time: datetime,
    to_start_time: datetime,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    fields: str = OBSERVATION_FIELDS,
) -> list[dict[str, Any]]:
    """Retrieve every observation page for one trace_id."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _ in range(50):
        payload = client.api.observations.get_many(
            trace_id=trace_id,
            fields=fields,
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            limit=page_limit,
            cursor=cursor,
        )
        page_rows = _observation_rows(payload)
        rows.extend(page_rows)
        next_cursor = _payload_cursor(payload)
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        if not page_rows:
            break
    return rows


def fetch_session_observation_pages(
    client: Any,
    *,
    session_id: str,
    from_start_time: datetime,
    to_start_time: datetime,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    fields: str = OBSERVATION_FIELDS,
) -> list[dict[str, Any]]:
    """Retrieve every observation page for one session_id."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _ in range(50):
        payload = client.api.observations.get_many(
            session_id=session_id,
            fields=fields,
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            limit=page_limit,
            cursor=cursor,
        )
        page_rows = _observation_rows(payload)
        rows.extend(page_rows)
        next_cursor = _payload_cursor(payload)
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        if not page_rows:
            break
    return rows


def wait_for_settled_observations(
    client: Any,
    trace_id: str,
    *,
    from_start_time: datetime,
    is_complete: SettlePredicate,
    to_start_time: datetime | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
) -> list[dict[str, Any]]:
    """Paginate until is_complete(rows) or a bounded timeout.

    Poll interval is only used between incomplete retrievals. The first
    retrieval happens immediately.
    """
    if not callable(is_complete):
        raise TypeError("is_complete must be callable")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if poll_interval_s < 0:
        raise ValueError("poll_interval_s must be >= 0")

    deadline = monotonic() + timeout_s
    last_rows: list[dict[str, Any]] = []
    attempts = 0
    while True:
        attempts += 1
        end = to_start_time or (datetime.now(timezone.utc) + timedelta(minutes=5))
        last_rows = fetch_observation_pages(
            client,
            trace_id=trace_id,
            from_start_time=from_start_time,
            to_start_time=end,
            page_limit=page_limit,
        )
        if is_complete(last_rows):
            return last_rows
        remaining = deadline - monotonic()
        if remaining <= 0:
            tool_count = sum(1 for row in last_rows if _is_tool_row(row))
            raise LangfuseTraceIncompleteError(
                "Langfuse trace did not settle before timeout: "
                f"trace_id={trace_id!r} attempts={attempts} "
                f"observations={len(last_rows)} tools={tool_count} "
                f"timeout_s={timeout_s}",
                trace_id=trace_id,
                observations=last_rows,
                attempts=attempts,
            )
        sleep(poll_interval_s if poll_interval_s <= remaining else remaining)


def get_observations_for_trace(
    client: Any,
    trace_id: str,
    *,
    from_start_time: datetime,
    to_start_time: datetime | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    is_complete: SettlePredicate | None = None,
    timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
) -> list[dict[str, Any]]:
    """Paginate get_many. When is_complete is set, poll until settled."""
    if is_complete is None:
        end = to_start_time or (datetime.now(timezone.utc) + timedelta(minutes=5))
        return fetch_observation_pages(
            client,
            trace_id=trace_id,
            from_start_time=from_start_time,
            to_start_time=end,
            page_limit=page_limit,
        )
    return wait_for_settled_observations(
        client,
        trace_id,
        from_start_time=from_start_time,
        to_start_time=to_start_time,
        page_limit=page_limit,
        is_complete=is_complete,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        sleep=sleep,
        monotonic=monotonic,
    )


def ingest_langfuse_trace(
    *,
    client: Any,
    trace_id: str,
    input: Any,
    output: Any,
    expected: Any,
    expected_tool_calls: Sequence[ToolInvocation],
    from_start_time: datetime,
    is_complete: SettlePredicate,
    to_start_time: datetime | None = None,
    application_id: str | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
) -> dict[str, dict[str, list[ToolInvocation]]]:
    """Settle one Langfuse trace, then build a ToolCorrectness request map.

    input, output, expected, and expected_tool_calls remain QE-supplied and
    are never inferred from Langfuse. expected_tool_calls is the request
    expected list. input, output, expected, and application_id are retained
    as caller-supplied values and are not copied onto a new evidence object.
    """
    observations = wait_for_settled_observations(
        client,
        trace_id,
        from_start_time=from_start_time,
        to_start_time=to_start_time,
        page_limit=page_limit,
        is_complete=is_complete,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        sleep=sleep,
        monotonic=monotonic,
    )
    return langfuse_tool_correctness_request(
        observations=observations,
        trace_id=trace_id,
        expected_tool_calls=expected_tool_calls,
    )


def ingest_langfuse_correctness(
    *,
    client: Any,
    trace_id: str,
    expected: Any,
    from_start_time: datetime,
    is_complete: SettlePredicate | None = None,
    to_start_time: datetime | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: SleepFn = time.sleep,
    monotonic: MonotonicFn = time.monotonic,
) -> dict[str, dict[str, list[Any]]]:
    """Settle until final assistant text exists, then build a Correctness map.

    expected remains QE-supplied and is never inferred from Langfuse.
    """
    observations = wait_for_settled_observations(
        client,
        trace_id,
        from_start_time=from_start_time,
        to_start_time=to_start_time,
        page_limit=page_limit,
        is_complete=is_complete or has_final_assistant_generation(trace_id),
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        sleep=sleep,
        monotonic=monotonic,
    )
    return langfuse_correctness_request(
        observations=observations,
        trace_id=trace_id,
        expected=expected,
    )
