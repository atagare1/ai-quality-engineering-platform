"""Standalone Langfuse + OpenAI Agents SDK ingestion spike.

Official integration pattern:
https://langfuse.com/integrations/frameworks/openai-agents

Retrieval uses current Langfuse Python SDK v4 Observations API v2:
  langfuse.api.observations.get_many(...)
Does not call langfuse.trace() or langfuse.api.trace.list().

This script is outside src/ai_qe_eval. It does not import the evaluation platform.
It does not construct ToolInvocation or EvaluationTrace objects.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SPIKE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SPIKE_DIR / "artifacts"
ENV_PATH = SPIKE_DIR / ".env"
PARENT_ENV_PATH = SPIKE_DIR.parent.parent / ".env"
FAILED_ARTIFACT = ARTIFACT_DIR / "inspection_failed_mixtral.json"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "openrouter/free"
PAGE_LIMIT = 2
MIN_TOOL_OBSERVATIONS = 3
TIMEZONE_RE = re.compile(r"^[A-Za-z0-9_+\-/]+$")

REQUIRED_LANGFUSE = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
)
TOGETHER_HOST_MARKERS = ("together.xyz", "together.ai")
OBSERVATION_FIELDS = (
    "core,basic,io,metadata,time,trace_context,model,usage,prompt,metrics"
)
CLOCK_URLS = (
    "https://worldtimeapi.org/api/timezone/{tz}",
    "https://timeapi.io/api/Time/current/zone?timeZone={tz}",
)
SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|authorization|password|credential)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{8,}|pk-lf-[A-Za-z0-9_\-]{8,}|sk-lf-[A-Za-z0-9_\-]{8,}"
    r"|sk-or-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9\.\-_]{8,})",
    re.IGNORECASE,
)


def _load_spike_env() -> None:
    from dotenv import load_dotenv

    loaded: list[str] = []
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)
        loaded.append(str(ENV_PATH))
    if PARENT_ENV_PATH.exists():
        load_dotenv(PARENT_ENV_PATH, override=False)
        loaded.append(str(PARENT_ENV_PATH))
    print("Loaded env files (later files fill unset keys only):")
    if loaded:
        for path in loaded:
            print(f"  {path}")
    else:
        print("  (none)")
    _configure_openrouter()


def _env_presence(name: str) -> str:
    value = os.environ.get(name)
    return "set" if value else "unset"


def _openai_base_host() -> str:
    from urllib.parse import urlparse

    base = os.environ.get("OPENAI_BASE_URL", "")
    if not base:
        return ""
    return urlparse(base).netloc.lower()


def _configure_openrouter() -> None:
    """Use OpenRouter for inference. Keep LANGFUSE_* unchanged. Ignore Together."""
    host = _openai_base_host()
    if any(marker in host for marker in TOGETHER_HOST_MARKERS):
        print(f"Ignoring Together OPENAI_BASE_URL host={host}")
        os.environ.pop("OPENAI_API_BASE", None)
    if os.environ.get("RAGAS_LLM_MODEL"):
        print("Ignoring RAGAS_LLM_MODEL for this spike")
        os.environ.pop("RAGAS_LLM_MODEL", None)
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    router_key = os.environ.get("OPENROUTER_API_KEY")
    if router_key:
        os.environ["OPENAI_API_KEY"] = router_key
    os.environ["OPENAI_MODEL"] = DEFAULT_OPENAI_MODEL
    print(f"Configured OpenRouter host={_openai_base_host()} model={DEFAULT_OPENAI_MODEL}")


def _resolve_model() -> tuple[str, str]:
    return DEFAULT_OPENAI_MODEL, "spike OPENAI_MODEL=openrouter/free"


def _require_credentials() -> None:
    missing = [name for name in REQUIRED_LANGFUSE if not os.environ.get(name)]
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    print("Credential presence (values not printed):")
    for name in (*REQUIRED_LANGFUSE, "OPENROUTER_API_KEY", "OPENAI_MODEL"):
        print(f"  {name}={_env_presence(name)}")
    print(f"  OPENAI_BASE_URL={_env_presence('OPENAI_BASE_URL')}")
    host = _openai_base_host()
    if host:
        print(f"  OPENAI_BASE_URL host={host}")
    if missing:
        print(
            "Missing required environment variables: " + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(2)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                out[key] = "[redacted]"
            else:
                out[key] = _sanitize(item)
        return out
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
    if not text:
        return value
    if text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def fetch_timezone_clock(iana_timezone: str) -> str:
    """GET the current time for an IANA timezone from a public HTTP clock API."""
    import httpx

    tz = (iana_timezone or "").strip()
    if not tz or not TIMEZONE_RE.fullmatch(tz) or ".." in tz:
        raise ValueError(f"Rejected iana_timezone={tz!r}")
    errors: list[str] = []
    for template in CLOCK_URLS:
        url = template.format(tz=tz)
        try:
            response = httpx.get(url, timeout=20.0, follow_redirects=True)
            response.raise_for_status()
            return json.dumps(
                {
                    "iana_timezone": tz,
                    "source_url": url,
                    "http_status": response.status_code,
                    "body": response.text[:2000],
                },
                sort_keys=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface the real transport error
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All real clock HTTP requests failed: " + " | ".join(errors))


def fetch_public_uuid() -> str:
    """GET a UUID from a public HTTP API."""
    import httpx

    url = "https://httpbin.org/uuid"
    response = httpx.get(url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    return json.dumps(
        {
            "source_url": url,
            "http_status": response.status_code,
            "body": response.text[:500],
        },
        sort_keys=True,
    )


def fetch_httpbin_json() -> str:
    """GET a public JSON payload from httpbin."""
    import httpx

    url = "https://httpbin.org/json"
    response = httpx.get(url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    return json.dumps(
        {
            "source_url": url,
            "http_status": response.status_code,
            "body": response.text[:1000],
        },
        sort_keys=True,
    )


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


def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
    start = str(row.get("start_time") or row.get("startTime") or "")
    obs_id = str(row.get("id") or "")
    return (start, obs_id)


def _row_trace_id(row: dict[str, Any]) -> str | None:
    value = row.get("trace_id") if row.get("trace_id") is not None else row.get("traceId")
    return str(value) if value is not None else None


def _row_parent_id(row: dict[str, Any]) -> str | None:
    value = row.get("parent_observation_id")
    if value is None:
        value = row.get("parentObservationId")
    return str(value) if value is not None else None


def _row_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or "").upper()


def _is_tool_like(row: dict[str, Any]) -> bool:
    if _row_type(row) == "TOOL":
        return True
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    kind = str(
        metadata.get("openinference.span.kind")
        or metadata.get("attributes.openinference.span.kind")
        or ""
    ).upper()
    return kind == "TOOL"


def _summarize_observation(row: dict[str, Any]) -> dict[str, Any]:
    return _sanitize(
        {
            "id": row.get("id"),
            "type": row.get("type"),
            "name": row.get("name"),
            "trace_id": _row_trace_id(row),
            "parent_observation_id": _row_parent_id(row),
            "start_time": _to_plain(row.get("start_time") or row.get("startTime")),
            "end_time": _to_plain(row.get("end_time") or row.get("endTime")),
            "level": row.get("level"),
            "status_message": row.get("status_message") or row.get("statusMessage"),
            "input": _maybe_json(row.get("input")),
            "output": _maybe_json(row.get("output")),
            "metadata": row.get("metadata"),
            "raw_keys": sorted(row.keys()),
        }
    )


def _extract_tool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for row in rows:
        if not _is_tool_like(row):
            continue
        raw_input = _maybe_json(row.get("input"))
        raw_output = _maybe_json(row.get("output"))
        arguments: Any = None
        if isinstance(raw_input, dict):
            if isinstance(raw_input.get("arguments"), dict):
                arguments = raw_input.get("arguments")
            else:
                arguments = raw_input
        tools.append(
            {
                "observation_id": row.get("id"),
                "observation_type": row.get("type"),
                "name": row.get("name"),
                "trace_id": _row_trace_id(row),
                "parent_observation_id": _row_parent_id(row),
                "start_time": _to_plain(row.get("start_time") or row.get("startTime")),
                "end_time": _to_plain(row.get("end_time") or row.get("endTime")),
                "level": row.get("level"),
                "status_message": row.get("status_message") or row.get("statusMessage"),
                "input_python_type": type(row.get("input")).__name__,
                "output_python_type": type(row.get("output")).__name__,
                "arguments": arguments,
                "result": raw_output,
            }
        )
    return tools


def _analyze_trajectory(trace_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("id")): row for row in rows if row.get("id") is not None}
    tools = sorted(
        [row for row in rows if _is_tool_like(row)],
        key=_sort_key,
    )
    generations = sorted(
        [row for row in rows if _row_type(row) == "GENERATION"],
        key=_sort_key,
    )
    missing_parents = []
    for row in rows:
        parent_id = _row_parent_id(row)
        if parent_id and parent_id not in by_id:
            missing_parents.append(
                {
                    "child_id": row.get("id"),
                    "child_type": row.get("type"),
                    "child_name": row.get("name"),
                    "missing_parent_id": parent_id,
                }
            )
    foreign_trace_rows = [
        {
            "id": row.get("id"),
            "type": row.get("type"),
            "name": row.get("name"),
            "trace_id": _row_trace_id(row),
        }
        for row in rows
        if _row_trace_id(row) not in (None, trace_id)
    ]
    generation_to_tool = []
    for tool in tools:
        parent_id = _row_parent_id(tool)
        tool_start = str(tool.get("start_time") or tool.get("startTime") or "")
        prior_same_parent = []
        for gen in generations:
            if _row_parent_id(gen) != parent_id:
                continue
            gen_start = str(gen.get("start_time") or gen.get("startTime") or "")
            if gen_start <= tool_start:
                prior_same_parent.append(
                    {
                        "generation_id": gen.get("id"),
                        "generation_name": gen.get("name"),
                        "generation_start_time": _to_plain(
                            gen.get("start_time") or gen.get("startTime")
                        ),
                    }
                )
        generation_to_tool.append(
            {
                "tool_id": tool.get("id"),
                "tool_name": tool.get("name"),
                "tool_start_time": _to_plain(tool.get("start_time") or tool.get("startTime")),
                "parent_observation_id": parent_id,
                "prior_generations_same_parent": prior_same_parent,
            }
        )
    tool_names_by_start = [row.get("name") for row in tools]
    timestamps_distinct = len({_sort_key(row)[0] for row in tools}) == len(tools)
    parent_ids = [_row_parent_id(row) for row in tools]
    parents_alone_order = (
        "INSUFFICIENT — multiple TOOL rows share a parent, so parent_id does not order them"
        if len(set(parent_ids)) < len(tools)
        else "Each TOOL has a distinct parent; parent links locate the node but timestamps still order siblings"
    )
    return {
        "observation_ids_in_start_time_order": [
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "name": row.get("name"),
                "parent_observation_id": _row_parent_id(row),
                "start_time": _to_plain(row.get("start_time") or row.get("startTime")),
                "end_time": _to_plain(row.get("end_time") or row.get("endTime")),
                "level": row.get("level"),
                "status_message": row.get("status_message") or row.get("statusMessage"),
            }
            for row in sorted(rows, key=_sort_key)
        ],
        "tool_execution_order_by_start_time": tool_names_by_start,
        "tool_count": len(tools),
        "generation_count": len(generations),
        "timestamps_distinct_for_tools": timestamps_distinct,
        "parent_relationships_order_tools": parents_alone_order,
        "timestamps_sufficient_to_order_tools": bool(tools) and timestamps_distinct,
        "missing_parent_ids_in_retrieved_set": missing_parents,
        "foreign_trace_rows_in_query_result": foreign_trace_rows,
        "unrelated_observations_can_enter_if_filtering_by_trace_id": bool(
            foreign_trace_rows
        ),
        "generation_to_tool": generation_to_tool,
        "limitations": [
            "Parent ID locates a TOOL under a CHAIN/AGENT; it does not encode sibling order.",
            "A first-page get_many is not the full trace; cursor pagination is required.",
            "Ingestion lag can omit wrapper SPAN/AGENT rows on the first complete page set.",
            "Missing parent IDs mean the retrieved set is not a closed tree.",
        ],
    }


def _assert_retrieved_payload(
    *,
    trace_id: str,
    rows: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[str]:
    """Deterministic checks on the real retrieved payload. Does not invent rows."""
    failures: list[str] = []
    if not rows:
        failures.append("no observations retrieved")
        return failures
    if any(_row_trace_id(row) not in (None, trace_id) for row in rows):
        failures.append("retrieved rows include a trace_id other than the spike trace")
    if len(tools) < MIN_TOOL_OBSERVATIONS:
        failures.append(
            f"retrieved TOOL observations={len(tools)}, need at least {MIN_TOOL_OBSERVATIONS}"
        )
    if len(rows) > PAGE_LIMIT and len(pages) < 2:
        failures.append(
            f"row_count={len(rows)} exceeds PAGE_LIMIT={PAGE_LIMIT} but pagination used {len(pages)} page(s)"
        )
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate observation ids across paginated pages")
    return failures


def _compare_with_failed(current: dict[str, Any]) -> dict[str, Any] | None:
    if not FAILED_ARTIFACT.exists():
        return None
    previous = json.loads(FAILED_ARTIFACT.read_text(encoding="utf-8"))
    prev_obs = previous.get("observations") or []
    curr_obs = current.get("observations") or []
    prev_types = sorted({str(item.get("type")) for item in prev_obs})
    curr_types = sorted({str(item.get("type")) for item in curr_obs})
    return {
        "previous_artifact": str(FAILED_ARTIFACT),
        "previous_trace_id": previous.get("trace_id"),
        "current_trace_id": current.get("trace_id"),
        "previous_agent_error_present": previous.get("agent_error") is not None,
        "current_agent_error_present": current.get("agent_error") is not None,
        "previous_observation_count": previous.get("observation_count"),
        "current_observation_count": current.get("observation_count"),
        "previous_observation_types": prev_types,
        "current_observation_types": curr_types,
        "types_only_in_current": sorted(set(curr_types) - set(prev_types)),
        "tool_observations_appeared": bool(current.get("tool_rows")),
    }


async def _run_agent(model: str) -> Any:
    from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        default_headers={
            "HTTP-Referer": "https://github.com/atagare1/llm-rag-evaluation-ragas",
            "X-Title": "langfuse-openai-agents-spike",
        },
    )
    timezone_tool = function_tool(fetch_timezone_clock)
    uuid_tool = function_tool(fetch_public_uuid)
    json_tool = function_tool(fetch_httpbin_json)
    agent = Agent(
        name="Multi-step clock uuid json agent",
        instructions=(
            "You must call three tools in this exact order, waiting for each result:\n"
            "1. fetch_timezone_clock with iana_timezone exactly 'UTC'\n"
            "2. fetch_public_uuid with no arguments\n"
            "3. fetch_httpbin_json with no arguments\n"
            "After all three succeed, answer in one paragraph that quotes the UTC "
            "datetime, the uuid, and a slideshow title from the JSON tool result. "
            "Do not invent values. Do not skip a tool. Do not retry a successful tool."
        ),
        tools=[timezone_tool, uuid_tool, json_tool],
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
    )
    return await Runner.run(
        agent,
        "Collect UTC time, a public UUID, and httpbin json, then summarize all three.",
    )


def _fetch_pages(
    langfuse: Any,
    *,
    trace_id: str,
    from_start: datetime,
    to_start: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    page_num = 0
    seen_cursors: set[str] = set()
    while page_num < 50:
        page_num += 1
        payload = langfuse.api.observations.get_many(
            trace_id=trace_id,
            fields=OBSERVATION_FIELDS,
            from_start_time=from_start,
            to_start_time=to_start,
            limit=PAGE_LIMIT,
            cursor=cursor,
        )
        page_rows = _observation_rows(payload)
        next_cursor = _payload_cursor(payload)
        pages.append(
            {
                "page": page_num,
                "limit": PAGE_LIMIT,
                "row_count": len(page_rows),
                "supplied_cursor": cursor is not None,
                "next_cursor_present": next_cursor is not None,
                "ids": [row.get("id") for row in page_rows],
            }
        )
        rows.extend(page_rows)
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            pages.append({"page": page_num, "error": "cursor repeated; stopping"})
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        if not page_rows:
            break
    return rows, pages


def _retrieve_all_observations(
    langfuse: Any, trace_id: str, started_at: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from_start = started_at - timedelta(minutes=5)
    last_error: Exception | None = None
    deadline = time.monotonic() + 90.0
    attempt = 0
    last_rows: list[dict[str, Any]] = []
    last_pages: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        attempt += 1
        to_start = datetime.now(timezone.utc) + timedelta(minutes=5)
        try:
            rows, pages = _fetch_pages(
                langfuse,
                trace_id=trace_id,
                from_start=from_start,
                to_start=to_start,
            )
            tool_count = sum(1 for row in rows if _is_tool_like(row))
            print(
                f"paginated get_many attempt {attempt}: "
                f"rows={len(rows)} pages={len(pages)} tools={tool_count}"
            )
            last_rows, last_pages = rows, pages
            if tool_count >= MIN_TOOL_OBSERVATIONS:
                time.sleep(8)
                extra_rows, extra_pages = _fetch_pages(
                    langfuse,
                    trace_id=trace_id,
                    from_start=from_start,
                    to_start=datetime.now(timezone.utc) + timedelta(minutes=5),
                )
                print(
                    f"completeness re-query: rows={len(extra_rows)} "
                    f"pages={len(extra_pages)} "
                    f"delta={len(extra_rows) - len(rows)}"
                )
                if len(extra_rows) >= len(rows):
                    extra_pages = [
                        {**page, "completeness_requery": True} for page in extra_pages
                    ]
                    return extra_rows, extra_pages
                return rows, pages
            print("waiting for ingestion of TOOL observations")
        except Exception as exc:  # noqa: BLE001 — retry real query errors
            last_error = exc
            print(
                f"paginated get_many attempt {attempt} raised "
                f"{type(exc).__name__}: {exc}"
            )
        time.sleep(5)
    if last_error is not None and not last_rows:
        raise last_error
    return last_rows, last_pages


def main() -> int:
    _load_spike_env()
    _require_credentials()

    from langfuse import get_client
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    OpenAIAgentsInstrumentor().instrument()
    langfuse = get_client()
    if not langfuse.auth_check():
        print(
            "Langfuse auth_check() failed. Check LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.",
            file=sys.stderr,
        )
        return 2

    model, model_source = _resolve_model()
    trace_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    print(f"model={model} (source={model_source})")
    print(f"preallocated_trace_id={trace_id}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "last_trace_id.txt").write_text(trace_id, encoding="utf-8")
    print(f"pagination PAGE_LIMIT={PAGE_LIMIT}")

    async def _execute() -> Any:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="langfuse-openai-agents-multistep-spike",
            trace_context={"trace_id": trace_id},
        ):
            return await _run_agent(model)

    agent_error: str | None = None
    agent_result: Any = None
    try:
        agent_result = asyncio.run(_execute())
    except Exception as exc:  # noqa: BLE001 — keep going so we still query Langfuse
        agent_error = f"{type(exc).__name__}: {exc}"
        print(f"Agent run error: {agent_error}", file=sys.stderr)

    langfuse.flush()
    print("langfuse.flush() completed; paginating observations.get_many")

    query_error: str | None = None
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    try:
        rows, pages = _retrieve_all_observations(langfuse, trace_id, started_at)
    except Exception as exc:  # noqa: BLE001
        query_error = f"{type(exc).__name__}: {exc}"
        print(f"Observation query error: {query_error}", file=sys.stderr)

    rows_sorted = sorted(rows, key=_sort_key)
    tool_rows = _extract_tool_rows(rows_sorted)
    final_output = getattr(agent_result, "final_output", None) if agent_result else None
    trajectory = _analyze_trajectory(trace_id, rows_sorted)
    assertion_failures = _assert_retrieved_payload(
        trace_id=trace_id,
        rows=rows_sorted,
        pages=pages,
        tools=tool_rows,
    )

    inspection = {
        "spike": "langfuse_openai_agents_multistep",
        "official_example": "https://langfuse.com/integrations/frameworks/openai-agents",
        "retrieval_api": "langfuse.api.observations.get_many",
        "pagination": {
            "page_limit": PAGE_LIMIT,
            "page_count": len(pages),
            "pages": pages,
        },
        "deprecated_apis_used": [],
        "trace_id": trace_id,
        "agent_final_output": _sanitize(_to_plain(final_output)),
        "agent_error": agent_error,
        "query_error": query_error,
        "observation_count": len(rows_sorted),
        "observations": [_summarize_observation(row) for row in rows_sorted],
        "tool_rows": _sanitize(tool_rows),
        "trajectory": _sanitize(trajectory),
        "payload_assertions": {
            "failures": assertion_failures,
            "passed": not assertion_failures,
        },
        "domain_objects_constructed": [],
    }
    inspection["comparison_with_failed_mixtral_run"] = _compare_with_failed(inspection)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "last_inspection.json"
    multistep_path = ARTIFACT_DIR / "last_multistep_inspection.json"
    serialized = json.dumps(inspection, indent=2, default=str)
    out_path.write_text(serialized, encoding="utf-8")
    multistep_path.write_text(serialized, encoding="utf-8")
    print(serialized)
    print(f"Wrote sanitized inspection to {out_path}")
    print(f"Wrote sanitized inspection to {multistep_path}")

    if agent_error or query_error or not rows_sorted:
        return 1
    if assertion_failures:
        print("Payload assertion failures: " + " | ".join(assertion_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
