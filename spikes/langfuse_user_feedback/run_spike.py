"""Read-only evidence spike for the official Langfuse user-feedback chatbot.

Does not import ai_qe_eval. Does not evaluate metrics. Assumes the example
app is already running at CHAT_BASE_URL (default http://127.0.0.1:3000).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SPIKE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SPIKE_DIR / "artifacts"
PARENT_ENV_PATH = SPIKE_DIR.parent.parent / ".env"
OBSERVATION_FIELDS = (
    "core,basic,io,metadata,time,trace_context,model,usage,prompt,metrics"
)
PAGE_LIMIT = 50
SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|authorization|password|credential)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{8,}|pk-lf-[A-Za-z0-9_\-]{8,}|sk-lf-[A-Za-z0-9_\-]{8,}"
    r"|sk-or-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9\.\-_]{8,})",
    re.IGNORECASE,
)
TURN1_TEXT = "What is Langfuse in one sentence?"
TURN2_TEXT = "How do I group several of those messages into one session?"


def _load_env() -> None:
    from dotenv import load_dotenv

    if PARENT_ENV_PATH.exists():
        load_dotenv(PARENT_ENV_PATH, override=False)
    print(f"Loaded parent env present={PARENT_ENV_PATH.exists()}")


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if SECRET_KEY_RE.search(str(key)) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_RE.sub("[redacted]", value)
    return value


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
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
    return str(value)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _user_message(message_id: str, text: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "role": "user",
        "parts": [{"type": "text", "text": text}],
    }


def _assistant_message(message_id: str, text: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "role": "assistant",
        "parts": [{"type": "text", "text": text}],
    }


def _parse_stream(raw: str) -> dict[str, Any]:
    message_ids: list[str] = []
    texts: list[str] = []
    events: list[Any] = []
    for line in raw.splitlines():
        payload = line
        if payload.startswith("data:"):
            payload = payload[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        if payload[0] in "{[":
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events.append(parsed)
            if isinstance(parsed, dict):
                for key in ("messageId", "id"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value:
                        message_ids.append(value)
                event_type = str(parsed.get("type") or "")
                if event_type in {"text-delta", "text-delta-start"}:
                    delta = parsed.get("delta") or parsed.get("textDelta") or ""
                    if isinstance(delta, str) and delta:
                        texts.append(delta)
                if event_type == "text" and isinstance(parsed.get("text"), str):
                    texts.append(parsed["text"])
                if isinstance(parsed.get("textDelta"), str):
                    texts.append(parsed["textDelta"])
        elif payload.startswith("0:"):
            try:
                texts.append(json.loads(payload[2:]))
            except json.JSONDecodeError:
                texts.append(payload[2:])
    return {
        "raw_chars": len(raw),
        "raw_preview": raw[:2000],
        "parsed_event_count": len(events),
        "message_ids": list(dict.fromkeys(message_ids)),
        "assistant_text": "".join(texts),
        "event_types": [
            item.get("type")
            for item in events
            if isinstance(item, dict) and item.get("type") is not None
        ],
    }


def _post_chat(
    *,
    base_url: str,
    chat_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    body = json.dumps(
        {
            "chatId": chat_id,
            "messages": messages,
            "model": os.environ.get("CHAT_MODEL", "openai/gpt-4o-mini"),
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json, */*",
        },
        method="POST",
    )
    started = time.monotonic()
    with urlopen(request, timeout=120) as response:
        raw = response.read().decode("utf-8", errors="replace")
        status = response.status
        headers = {str(key): str(value) for key, value in response.headers.items()}
    parsed = _parse_stream(raw)
    parsed.update(
        {
            "http_status": status,
            "duration_s": round(time.monotonic() - started, 3),
            "response_headers": _sanitize(headers),
        }
    )
    return parsed


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


def _row_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or "").upper()


def _row_session_id(row: dict[str, Any]) -> str | None:
    value = row.get("session_id")
    if value is None:
        value = row.get("sessionId")
    return str(value) if value is not None else None


def _row_trace_id(row: dict[str, Any]) -> str | None:
    value = row.get("trace_id")
    if value is None:
        value = row.get("traceId")
    return str(value) if value is not None else None


def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
    start = row.get("start_time") or row.get("startTime")
    if hasattr(start, "isoformat"):
        start_key = start.isoformat()
    else:
        start_key = str(start or "")
    return (start_key, str(row.get("id") or ""))


def _summarize_observation(row: dict[str, Any]) -> dict[str, Any]:
    return _sanitize(
        {
            "id": row.get("id"),
            "type": row.get("type"),
            "name": row.get("name"),
            "trace_id": _row_trace_id(row),
            "session_id": _row_session_id(row),
            "parent_observation_id": row.get("parent_observation_id")
            or row.get("parentObservationId"),
            "start_time": _to_plain(row.get("start_time") or row.get("startTime")),
            "end_time": _to_plain(row.get("end_time") or row.get("endTime")),
            "level": row.get("level"),
            "is_root_observation": row.get("is_root_observation")
            or row.get("isRootObservation"),
            "input": _maybe_json(row.get("input")),
            "output": _maybe_json(row.get("output")),
            "metadata": row.get("metadata"),
            "raw_keys": sorted(row.keys()),
        }
    )


def _fetch_session_pages(
    langfuse: Any,
    *,
    session_id: str,
    from_start: datetime,
    to_start: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    query_mode = "session_id_kwarg"
    for page_num in range(1, 51):
        kwargs: dict[str, Any] = {
            "fields": OBSERVATION_FIELDS,
            "from_start_time": from_start,
            "to_start_time": to_start,
            "limit": PAGE_LIMIT,
            "cursor": cursor,
        }
        try:
            payload = langfuse.api.observations.get_many(
                session_id=session_id, **kwargs
            )
        except TypeError:
            query_mode = "filter_sessionId"
            payload = langfuse.api.observations.get_many(
                filter=json.dumps(
                    [
                        {
                            "type": "string",
                            "column": "sessionId",
                            "operator": "=",
                            "value": session_id,
                        }
                    ]
                ),
                **kwargs,
            )
        page_rows = _observation_rows(payload)
        pages.append(
            {
                "page": page_num,
                "row_count": len(page_rows),
                "supplied_cursor": cursor is not None,
                "next_cursor_present": bool(_payload_cursor(payload)),
                "ids": [row.get("id") for row in page_rows],
            }
        )
        rows.extend(page_rows)
        next_cursor = _payload_cursor(payload)
        if not next_cursor:
            break
        cursor = next_cursor
        if not page_rows:
            break
    return rows, pages, query_mode


def _retrieve_until_settled(
    langfuse: Any,
    *,
    session_id: str,
    started_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    deadline = time.monotonic() + 90.0
    last_rows: list[dict[str, Any]] = []
    last_pages: list[dict[str, Any]] = []
    query_mode = "unset"
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        rows, pages, query_mode = _fetch_session_pages(
            langfuse,
            session_id=session_id,
            from_start=started_at - timedelta(minutes=5),
            to_start=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        last_rows, last_pages = rows, pages
        types = sorted({_row_type(row) for row in rows})
        trace_ids = {_row_trace_id(row) for row in rows if _row_trace_id(row)}
        print(
            f"session get_many attempt {attempt}: rows={len(rows)} "
            f"traces={len(trace_ids)} types={types} mode={query_mode}"
        )
        if len(trace_ids) >= 2:
            time.sleep(5)
            extra_rows, extra_pages, query_mode = _fetch_session_pages(
                langfuse,
                session_id=session_id,
                from_start=started_at - timedelta(minutes=5),
                to_start=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            print(
                f"completeness re-query: rows={len(extra_rows)} "
                f"delta={len(extra_rows) - len(rows)}"
            )
            return extra_rows, extra_pages, query_mode
        time.sleep(3)
    return last_rows, last_pages, query_mode


def _analyze(rows: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    ordered = sorted(rows, key=_sort_key)
    generations = [row for row in ordered if _row_type(row) == "GENERATION"]
    roots = [
        row
        for row in ordered
        if row.get("is_root_observation") or row.get("isRootObservation")
    ]
    return {
        "observation_count": len(ordered),
        "types": sorted({_row_type(row) for row in ordered}),
        "trace_ids": list(
            dict.fromkeys(_row_trace_id(row) for row in ordered if _row_trace_id(row))
        ),
        "session_ids": list(
            dict.fromkeys(
                _row_session_id(row) for row in ordered if _row_session_id(row)
            )
        ),
        "requested_session_id": session_id,
        "root_count": len(roots),
        "generation_count": len(generations),
        "timeline": [
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "name": row.get("name"),
                "trace_id": _row_trace_id(row),
                "session_id": _row_session_id(row),
                "parent_observation_id": row.get("parent_observation_id")
                or row.get("parentObservationId"),
                "start_time": _to_plain(row.get("start_time") or row.get("startTime")),
                "end_time": _to_plain(row.get("end_time") or row.get("endTime")),
                "is_root_observation": row.get("is_root_observation")
                or row.get("isRootObservation"),
            }
            for row in ordered
        ],
        "root_io": [
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "name": row.get("name"),
                "trace_id": _row_trace_id(row),
                "input_python_type": type(row.get("input")).__name__,
                "output_python_type": type(row.get("output")).__name__,
                "input": _maybe_json(row.get("input")),
                "output": _maybe_json(row.get("output")),
            }
            for row in roots
        ],
        "generation_io": [
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "trace_id": _row_trace_id(row),
                "input_python_type": type(row.get("input")).__name__,
                "output_python_type": type(row.get("output")).__name__,
                "input": _maybe_json(row.get("input")),
                "output": _maybe_json(row.get("output")),
            }
            for row in generations
        ],
    }


def main() -> int:
    _load_env()
    missing = [
        name
        for name in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
        )
        if not os.environ.get(name)
    ]
    print("Credential presence (values not printed):")
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ):
        print(f"  {name}={'set' if os.environ.get(name) else 'unset'}")
    host = urlparse(os.environ.get("LANGFUSE_BASE_URL", "")).netloc
    print(f"  LANGFUSE_BASE_URL host={host or 'unset'}")
    if missing:
        print("Missing Langfuse credentials: " + ", ".join(missing), file=sys.stderr)
        return 2

    from langfuse import get_client

    langfuse = get_client()
    if not langfuse.auth_check():
        print("Langfuse auth_check() failed", file=sys.stderr)
        return 2

    base_url = os.environ.get("CHAT_BASE_URL", "http://127.0.0.1:3000")
    chat_id = os.environ.get("CHAT_SESSION_ID") or uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    print(f"chat_base_url={base_url}")
    print(f"chat_id={chat_id}")

    turn1_error: str | None = None
    turn2_error: str | None = None
    turn1: dict[str, Any] = {}
    turn2: dict[str, Any] = {}
    try:
        turn1 = _post_chat(
            base_url=base_url,
            chat_id=chat_id,
            messages=[_user_message("user-turn-1", TURN1_TEXT)],
        )
        print(
            f"turn1 status={turn1.get('http_status')} "
            f"chars={turn1.get('raw_chars')} "
            f"assistant_len={len(str(turn1.get('assistant_text') or ''))}"
        )
    except Exception as exc:  # noqa: BLE001
        turn1_error = f"{type(exc).__name__}: {exc}"
        print(f"turn1 error: {turn1_error}", file=sys.stderr)

    assistant_id = None
    assistant_text = ""
    if turn1:
        ids = turn1.get("message_ids") or []
        assistant_id = ids[-1] if ids else "assistant-turn-1"
        assistant_text = str(turn1.get("assistant_text") or "")

    if not turn1_error:
        try:
            turn2 = _post_chat(
                base_url=base_url,
                chat_id=chat_id,
                messages=[
                    _user_message("user-turn-1", TURN1_TEXT),
                    _assistant_message(str(assistant_id), assistant_text),
                    _user_message("user-turn-2", TURN2_TEXT),
                ],
            )
            print(
                f"turn2 status={turn2.get('http_status')} "
                f"chars={turn2.get('raw_chars')} "
                f"assistant_len={len(str(turn2.get('assistant_text') or ''))}"
            )
        except Exception as exc:  # noqa: BLE001
            turn2_error = f"{type(exc).__name__}: {exc}"
            print(f"turn2 error: {turn2_error}", file=sys.stderr)

    flushed = False
    flush = getattr(langfuse, "flush", None)
    if callable(flush):
        flush()
        flushed = True
    print(f"langfuse.flush() completed={flushed}")

    query_error: str | None = None
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    query_mode = "unset"
    try:
        rows, pages, query_mode = _retrieve_until_settled(
            langfuse, session_id=chat_id, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        query_error = f"{type(exc).__name__}: {exc}"
        print(f"Observation query error: {query_error}", file=sys.stderr)

    rows_sorted = sorted(rows, key=_sort_key)
    analysis = _analyze(rows_sorted, chat_id)
    inspection = {
        "spike": "langfuse_user_feedback_two_turn",
        "official_example": (
            "https://github.com/langfuse/langfuse-examples/tree/main/"
            "applications/user-feedback"
        ),
        "retrieval_api": "langfuse.api.observations.get_many",
        "retrieval_query_mode": query_mode,
        "pagination": {
            "page_limit": PAGE_LIMIT,
            "page_count": len(pages),
            "pages": pages,
        },
        "deprecated_apis_used": [],
        "session_id": chat_id,
        "chat_base_url": base_url,
        "posted_turns": [
            {
                "turn": 1,
                "user_text": TURN1_TEXT,
                "http": _sanitize(turn1),
                "error": turn1_error,
            },
            {
                "turn": 2,
                "user_text": TURN2_TEXT,
                "http": _sanitize(turn2),
                "error": turn2_error,
            },
        ],
        "query_error": query_error,
        "observation_count": len(rows_sorted),
        "observations": [_summarize_observation(row) for row in rows_sorted],
        "analysis": _sanitize(analysis),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "last_session_inspection.json"
    serialized = json.dumps(inspection, indent=2, default=str)
    out_path.write_text(serialized, encoding="utf-8")
    print(f"Wrote sanitized inspection to {out_path}")
    print(
        json.dumps(
            {
                "session_id": chat_id,
                "observation_count": len(rows_sorted),
                "types": analysis.get("types"),
                "trace_ids": analysis.get("trace_ids"),
                "generation_count": analysis.get("generation_count"),
                "root_count": analysis.get("root_count"),
                "turn1_error": turn1_error,
                "turn2_error": turn2_error,
                "query_error": query_error,
            },
            indent=2,
        )
    )
    if turn1_error or turn2_error or query_error or not rows_sorted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
