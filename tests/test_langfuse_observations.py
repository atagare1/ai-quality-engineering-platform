"""Pagination and settle tests for Langfuse observation retrieval.

Uses an in-process fake client that implements get_many. Does not claim to
be a live Langfuse run.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.integrations.langfuse_observations import (
    LangfuseTraceIncompleteError,
    fetch_observation_pages,
    fetch_session_observation_pages,
    flush_langfuse_client,
    get_observations_for_trace,
    has_final_assistant_generation,
    ingest_langfuse_correctness,
    ingest_langfuse_trace,
    min_tool_observations,
    wait_for_settled_observations,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

_OBSERVATIONS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "integrations"
    / "langfuse_observations.py"
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeObservations:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.calls: list[dict] = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        cursor = kwargs.get("cursor")
        index = 0 if cursor is None else int(cursor)
        data = self.pages[index] if index < len(self.pages) else []
        next_cursor = str(index + 1) if index + 1 < len(self.pages) else None
        return {"data": data, "meta": {"cursor": next_cursor}}


class _FakeClient:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.api = type("API", (), {"observations": _FakeObservations(pages)})()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


class _GrowingClient:
    """First paginated pass is incomplete; later passes grow TOOL rows."""

    def __init__(self) -> None:
        self.calls = 0
        self.api = type("API", (), {})()
        self.api.observations = self

    def get_many(self, **kwargs):
        self.calls += 1
        cursor = kwargs.get("cursor")
        if self.calls <= 2:
            if cursor is None:
                return {
                    "data": [{"id": "1", "type": "TOOL", "trace_id": "t", "name": "a"}],
                    "meta": {"cursor": "1"},
                }
            return {
                "data": [{"id": "2", "type": "GENERATION", "trace_id": "t"}],
                "meta": {"cursor": None},
            }
        if cursor is None:
            return {
                "data": [{"id": "1", "type": "TOOL", "trace_id": "t", "name": "a"}],
                "meta": {"cursor": "1"},
            }
        return {
            "data": [
                {"id": "2", "type": "GENERATION", "trace_id": "t"},
                {"id": "3", "type": "TOOL", "trace_id": "t", "name": "b"},
            ],
            "meta": {"cursor": None},
        }


def test_fetch_observation_pages_walks_cursor_until_exhausted():
    pages = [
        [{"id": "1", "type": "TOOL", "trace_id": "t"}],
        [{"id": "2", "type": "TOOL", "trace_id": "t"}],
        [{"id": "3", "type": "GENERATION", "trace_id": "t"}],
    ]
    client = _FakeClient(pages)
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    rows = fetch_observation_pages(
        client,
        trace_id="t",
        from_start_time=start,
        to_start_time=start,
        page_limit=1,
    )
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert len(client.api.observations.calls) == 3
    assert client.api.observations.calls[0]["cursor"] is None
    assert client.api.observations.calls[1]["cursor"] == "1"
    assert client.api.observations.calls[2]["limit"] == 1


def test_fetch_session_observation_pages_uses_session_id_and_io_fields():
    pages = [
        [
            {
                "id": "root-1",
                "type": "SPAN",
                "name": "handle-chat-message",
                "session_id": "session-1",
                "is_root_observation": True,
            }
        ],
        [
            {
                "id": "gen-1",
                "type": "GENERATION",
                "name": "chat openai/gpt-4o-mini",
                "session_id": "session-1",
            }
        ],
    ]
    client = _FakeClient(pages)
    start = datetime(2026, 10, 4, tzinfo=timezone.utc)
    rows = fetch_session_observation_pages(
        client,
        session_id="session-1",
        from_start_time=start,
        to_start_time=start,
        page_limit=1,
    )
    assert [row["id"] for row in rows] == ["root-1", "gen-1"]
    assert client.api.observations.calls[0]["session_id"] == "session-1"
    assert "trace_id" not in client.api.observations.calls[0]
    assert "io" in client.api.observations.calls[0]["fields"].split(",")


def test_wait_for_settled_observations_polls_until_complete():
    client = _GrowingClient()
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    rows = wait_for_settled_observations(
        client,
        "t",
        from_start_time=start,
        to_start_time=start,
        page_limit=1,
        is_complete=min_tool_observations(2),
        timeout_s=5.0,
        poll_interval_s=0.2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert sum(1 for row in rows if row["type"] == "TOOL") == 2
    assert clock.sleeps == [0.2]
    assert client.calls > 2


def test_wait_for_settled_observations_times_out_when_incomplete():
    client = _FakeClient(
        [[{"id": "1", "type": "TOOL", "trace_id": "t", "name": "a"}]]
    )
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    with pytest.raises(LangfuseTraceIncompleteError) as raised:
        wait_for_settled_observations(
            client,
            "t",
            from_start_time=start,
            to_start_time=start,
            is_complete=min_tool_observations(3),
            timeout_s=0.5,
            poll_interval_s=0.2,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    error = raised.value
    assert error.trace_id == "t"
    assert error.attempts >= 2
    assert [row["id"] for row in error.observations] == ["1"]
    assert clock.sleeps


def test_get_observations_for_trace_without_predicate_does_not_sleep():
    client = _FakeClient(
        [[{"id": "1", "type": "TOOL", "trace_id": "t"}]]
    )
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    rows = get_observations_for_trace(
        client,
        "t",
        from_start_time=start,
        to_start_time=start,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert [row["id"] for row in rows] == ["1"]
    assert clock.sleeps == []


def test_ingest_langfuse_trace_builds_request_only_after_settle():
    client = _GrowingClient()
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    expected = [
        ToolInvocation(name="spec-a", arguments=None, result=None),
        ToolInvocation(name="spec-b", arguments=None, result=None),
    ]
    request = ingest_langfuse_trace(
        client=client,
        trace_id="t",
        input="user",
        output="qe-spec-output",
        expected="qe-spec-output",
        expected_tool_calls=expected,
        from_start_time=start,
        to_start_time=start,
        page_limit=1,
        is_complete=min_tool_observations(2),
        timeout_s=5.0,
        poll_interval_s=0.2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert set(request) == {"tool_correctness"}
    observed, expected_args = request["tool_correctness"]["args"]
    assert [call.name for call in observed] == ["a", "b"]
    assert [call.name for call in expected_args] == ["spec-a", "spec-b"]
    assert expected_args is not observed
    assert clock.sleeps == [0.2]


def test_ingest_langfuse_request_through_runner_policy_and_gate():
    client = _GrowingClient()
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    expected = [
        ToolInvocation(name="spec-a", arguments=None, result=None),
        ToolInvocation(name="spec-b", arguments=None, result=None),
    ]
    request = ingest_langfuse_trace(
        client=client,
        trace_id="t",
        input="user",
        output="qe-spec-output",
        expected="qe-spec-output",
        expected_tool_calls=expected,
        from_start_time=start,
        to_start_time=start,
        page_limit=1,
        is_complete=min_tool_observations(2),
        timeout_s=5.0,
        poll_interval_s=0.2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    class RecordingEvaluator:
        def __init__(self) -> None:
            self.received_args = None

        def evaluate(self, *args, **kwargs):
            self.received_args = args
            return [
                EvaluationResult(
                    metric="tool_correctness",
                    evaluator="fake",
                    score=0.91,
                    reason="synthetic langfuse request",
                )
            ]

    evaluator = RecordingEvaluator()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="fake",
            category="agent",
        )
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"tool_correctness": evaluator},
        policies={
            "tool_correctness": QualityPolicy(
                metric="tool_correctness",
                operator=">=",
                threshold=0.80,
            )
        },
        gate=QualityGate(),
    )
    decision = runner.run(
        request,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="langfuse-ingest-request",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.requests[0] is request
    assert [call.name for call in evaluator.received_args[0]] == ["a", "b"]
    assert [call.name for call in evaluator.received_args[1]] == ["spec-a", "spec-b"]
    assert runner.last_run.results[0].metric == "tool_correctness"
    assert runner.last_run.results[0].score == 0.91
    assert runner.last_run.decisions[0].passed is True
    assert clock.sleeps == [0.2]


def _tool_call_generation_row() -> dict:
    return {
        "id": "g1",
        "type": "GENERATION",
        "trace_id": "t",
        "name": "generation",
        "start_time": "2026-10-03T10:21:48Z",
        "input": [{"role": "user", "content": "Collect UTC time."}],
        "output": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            }
        ],
    }


def _final_generation_row() -> dict:
    return {
        "id": "g2",
        "type": "GENERATION",
        "trace_id": "t",
        "name": "generation",
        "start_time": "2026-10-03T10:22:10Z",
        "input": [{"role": "user", "content": "Collect UTC time."}],
        "output": [
            {
                "role": "assistant",
                "content": "UTC time collected.",
                "tool_calls": None,
            }
        ],
    }


class _GrowingCorrectnessClient:
    """First pass has tools and a tool-call GENERATION; later adds final text."""

    def __init__(self) -> None:
        self.calls = 0
        self.api = type("API", (), {})()
        self.api.observations = self

    def get_many(self, **kwargs):
        self.calls += 1
        cursor = kwargs.get("cursor")
        first_page = [
            {"id": "1", "type": "TOOL", "trace_id": "t", "name": "a"},
            _tool_call_generation_row(),
        ]
        if self.calls <= 1:
            if cursor is None:
                return {"data": first_page, "meta": {"cursor": None}}
            return {"data": [], "meta": {"cursor": None}}
        if cursor is None:
            return {
                "data": first_page + [_final_generation_row()],
                "meta": {"cursor": None},
            }
        return {"data": [], "meta": {"cursor": None}}


def test_has_final_assistant_generation_ignores_tool_call_only_rows():
    predicate = has_final_assistant_generation("t")
    assert predicate([_tool_call_generation_row()]) is False
    assert predicate([_tool_call_generation_row(), _final_generation_row()]) is True


def test_ingest_langfuse_correctness_waits_for_final_assistant_text():
    client = _GrowingCorrectnessClient()
    clock = _FakeClock()
    start = datetime(2026, 10, 2, tzinfo=timezone.utc)
    request = ingest_langfuse_correctness(
        client=client,
        trace_id="t",
        expected="QE expected remains caller-supplied.",
        from_start_time=start,
        to_start_time=start,
        timeout_s=5.0,
        poll_interval_s=0.2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert set(request) == {"correctness"}
    assert request["correctness"]["args"] == [
        "Collect UTC time.",
        "UTC time collected.",
        "QE expected remains caller-supplied.",
    ]
    assert clock.sleeps == [0.2]
    assert client.calls == 2


def test_langfuse_observations_module_does_not_import_evaluation_trace():
    tree = ast.parse(_OBSERVATIONS_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "ai_qe_eval.domain.trace" not in imported
    assert not any(name.endswith(".trace") for name in imported)


def test_flush_langfuse_client_calls_flush_when_present():
    client = _FakeClient([])
    assert flush_langfuse_client(client) is True
    assert client.flush_calls == 1
    assert flush_langfuse_client(object()) is False
