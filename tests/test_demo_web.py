"""Focused tests for the live Playwright MCP web demo.

Proves the demo opens playwright_mcp_stdio_parameters → stdio_client →
ClientSession (not ScriptedMcpSession), streams on_tool_call events, and
uses EvaluationRunner. Transport doubles keep the focused suite offline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

from ai_qe_eval.cli import P0_EVALUATIONS
from ai_qe_eval.demo import mcp_p0 as mcp_p0_mod
from ai_qe_eval.demo.mcp_p0 import (
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    TYPED_TODO_TEXT_FAIL,
    ScriptedMcpSession,
    _ok,
    _todo_snapshot,
    _EMPTY_SNAPSHOT,
)
from ai_qe_eval.demo.web import (
    DemoHandler,
    _client_error_message,
    _final_snapshot_body,
    build_demo_payload,
    execute_demo,
    tool_timeline_row,
)
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_FAILED_REASON,
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
)
from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    snapshot_body_from_serialized_result,
)
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from tool_correctness_test_doubles import UnusedToolCorrectnessJudge


class _RecordingClientSession:
    def __init__(self, read_stream, write_stream, results) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False
        self.calls: list[tuple[str, dict | None]] = []
        self._results = list(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        self.initialized = True

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if not self._results:
            raise RuntimeError(f"no canned result for {name!r}")
        return self._results.pop(0)


def _install_stdio_double(monkeypatch, *, todo_text: str) -> dict:
    recorded: dict = {"params": None, "sessions": []}
    results = [
        _ok("navigated"),
        _ok(_EMPTY_SNAPSHOT),
        _ok("clicked"),
        _ok("typed"),
        _ok("clicked"),
        _ok(_todo_snapshot(todo_text)),
    ]

    @asynccontextmanager
    async def fake_stdio_client(params):
        recorded["params"] = params
        yield ("read", "write")

    def fake_client_session(read_stream, write_stream):
        session = _RecordingClientSession(read_stream, write_stream, results)
        recorded["sessions"].append(session)
        return session

    monkeypatch.setattr(mcp_p0_mod, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp_p0_mod, "ClientSession", fake_client_session)
    return recorded


def test_execute_demo_pass_uses_stdio_client_session_not_scripted(monkeypatch):
    recorded = _install_stdio_double(monkeypatch, todo_text=EXPECTED_TODO_TEXT)
    seen: list[tuple[int, ToolInvocation]] = []
    runners: list[EvaluationRunner] = []
    original_run = EvaluationRunner.run

    def _recording_run(self, request, configuration, *, run_id=""):
        runners.append(self)
        return original_run(self, request, configuration, run_id=run_id)

    monkeypatch.setattr(EvaluationRunner, "run", _recording_run)

    def on_tool_call(invocation: ToolInvocation, *, order: int) -> None:
        seen.append((order, invocation))

    run = execute_demo(
        "pass",
        on_tool_call=on_tool_call,
        pace_s=0,
        model=UnusedToolCorrectnessJudge(),
    )
    payload = build_demo_payload(run, "pass")
    session = recorded["sessions"][0]
    params = recorded["params"]

    assert runners, "demo must call EvaluationRunner.run"
    assert params is not None
    assert params.command == "npx"
    assert PLAYWRIGHT_MCP_PACKAGE in params.args
    assert "--headless" in params.args
    assert "--isolated" in params.args
    assert session.initialized is True
    assert not isinstance(session, ScriptedMcpSession)
    assert session.calls[0] == ("browser_navigate", {"url": TODO_MVC_URL})
    assert session.calls[0][1]["url"] == "https://demo.playwright.dev/todomvc"
    assert [name for name, _args in session.calls] == list(EXPECTED_TOOL_ORDER)
    assert session.calls[3][1]["text"] == EXPECTED_TODO_TEXT
    assert [order for order, _inv in seen] == list(range(1, 7))
    assert [inv.name for _order, inv in seen] == list(EXPECTED_TOOL_ORDER)
    assert [inv for _order, inv in seen] == run.requests[0]["tool_correctness"]["args"][0]
    assert payload["typed_todo"] == EXPECTED_TODO_TEXT
    assert payload["todos"] == [EXPECTED_TODO_TEXT]
    assert payload["final_snapshot"] == snapshot_body_from_serialized_result(
        run.requests[0]["tool_correctness"]["args"][0][-1].result
    )
    assert "listitem" in payload["final_snapshot"]
    assert EXPECTED_TODO_TEXT in payload["final_snapshot"]
    assert payload["final_snapshot"] != _EMPTY_SNAPSHOT
    assert payload["gate"]["result"] == "PASS"
    assert [row["evaluator"] for row in payload["evaluators"]] == [
        "Tool Correctness",
        "MCP Execution Health",
        "Final State",
    ]
    assert all(row["result"] == "PASS" for row in payload["evaluators"])
    assert [result.metric for result in run.results] == list(P0_EVALUATIONS)
    assert all(tool_timeline_row(inv, order=order)["ok"] for order, inv in seen)


def test_execute_demo_fail_types_buy_bread_on_live_session(monkeypatch):
    recorded = _install_stdio_double(monkeypatch, todo_text=TYPED_TODO_TEXT_FAIL)
    seen: list[str] = []

    def on_tool_call(invocation: ToolInvocation, *, order: int) -> None:
        seen.append(invocation.name)

    run = execute_demo(
        "fail",
        on_tool_call=on_tool_call,
        pace_s=0,
        model=UnusedToolCorrectnessJudge(),
    )
    payload = build_demo_payload(run, "fail")
    session = recorded["sessions"][0]
    results = {result.metric: result for result in run.results}

    assert session.initialized is True
    assert session.calls[0] == ("browser_navigate", {"url": TODO_MVC_URL})
    assert session.calls[3][1]["text"] == TYPED_TODO_TEXT_FAIL
    assert seen == list(EXPECTED_TOOL_ORDER)
    assert payload["typed_todo"] == TYPED_TODO_TEXT_FAIL
    assert payload["todos"] == [TYPED_TODO_TEXT_FAIL]
    assert payload["expected_todo"] == EXPECTED_TODO_TEXT
    assert payload["final_snapshot"] == snapshot_body_from_serialized_result(
        run.requests[0]["tool_correctness"]["args"][0][-1].result
    )
    assert TYPED_TODO_TEXT_FAIL in payload["final_snapshot"]
    assert "listitem" in payload["final_snapshot"]
    assert EXPECTED_TODO_TEXT not in payload["final_snapshot"]
    assert results["tool_correctness"].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    assert results[FINAL_STATE_METRIC].score == 0.0
    assert payload["gate"]["result"] == "FAIL"
    assert payload["evaluators"][2]["reason"] == FINAL_STATE_FAILED_REASON


def test_demo_index_page_is_served():
    captured: dict[str, bytes] = {}

    class _Probe(DemoHandler):
        def __init__(self) -> None:
            self.wfile = BytesIO()
            self.path = "/"

        def send_response(self, code, message=None):
            captured["code"] = str(code).encode()

        def send_header(self, keyword, value):
            captured[keyword] = value.encode() if isinstance(value, str) else value

        def end_headers(self):
            captured["ended"] = b"1"

    handler = _Probe()
    handler.do_GET()
    body = handler.wfile.getvalue().decode("utf-8")
    assert captured.get("code") == b"200"
    assert "https://demo.playwright.dev/todomvc" in body
    assert "Run PASS — Buy milk" in body
    assert "Run FAIL — Buy bread" in body
    assert "EventSource" in body
    assert "/api/run" in body
    assert "Live MCP Evidence" in body
    assert "final_snapshot" in body
    assert "browser_screenshot" not in body
    assert "browser_take_screenshot" not in body


def test_final_snapshot_body_uses_last_browser_snapshot_only():
    first = ToolInvocation(
        name="browser_snapshot",
        arguments={},
        result={"isError": False, "content": [{"type": "text", "text": _EMPTY_SNAPSHOT}]},
    )
    last = ToolInvocation(
        name="browser_snapshot",
        arguments={},
        result={
            "isError": False,
            "content": [{"type": "text", "text": _todo_snapshot(EXPECTED_TODO_TEXT)}],
        },
    )
    typed = ToolInvocation(
        name="browser_type",
        arguments={"text": TYPED_TODO_TEXT_FAIL},
        result={"isError": False, "content": []},
    )
    body = _final_snapshot_body([first, typed, last])
    assert body == _todo_snapshot(EXPECTED_TODO_TEXT)
    assert TYPED_TODO_TEXT_FAIL not in body
    assert _final_snapshot_body([]) == ""


def test_emit_swallows_client_disconnect():
    class _Broken:
        def write(self, _data):
            raise ConnectionAbortedError(10053, "aborted")

        def flush(self):
            return None

    class _Probe(DemoHandler):
        def __init__(self) -> None:
            self.wfile = _Broken()

    _Probe()._emit("tool", {"order": 6, "name": "browser_snapshot"})


def test_client_error_message_unwraps_task_group():
    inner = ConnectionAbortedError(10053, "aborted")
    grouped = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    message = _client_error_message(grouped)
    assert message.startswith("ConnectionAbortedError:")
    assert "aborted" in message
