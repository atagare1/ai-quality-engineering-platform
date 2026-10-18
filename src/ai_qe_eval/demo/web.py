"""Minimal browser demo for the scripted Playwright MCP P0 path.

Streams real on_tool_call events, then runs the existing EvaluationRunner.
Does not duplicate the MCP loop or add a new evidence type.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_qe_eval.cli import (
    P0_EVALUATIONS,
    _evaluator_label,
    build_p0_runner,
)
from ai_qe_eval.demo.mcp_p0 import (
    EXPECTED_TODO_TEXT,
    GOAL,
    live_mcp_p0_request,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.integrations.playwright_mcp import (
    snapshot_body_from_serialized_result,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_DEFAULT_PACE_S = 0.35


def tool_timeline_row(invocation: ToolInvocation, *, order: int) -> dict[str, Any]:
    """JSON view of an existing ToolInvocation. Not a new domain type."""
    result = invocation.result if isinstance(invocation.result, dict) else {}
    return {
        "order": order,
        "name": invocation.name,
        "arguments": invocation.arguments,
        "ok": result.get("isError") is not True,
    }


def _typed_todo(observed: list[ToolInvocation]) -> str | None:
    for call in observed:
        if call.name == "browser_type" and isinstance(call.arguments, dict):
            text = call.arguments.get("text")
            if isinstance(text, str):
                return text
    return None


def _final_snapshot_body(observed: list[ToolInvocation]) -> str:
    """Last browser_snapshot body. Same text Final State inspects."""
    for call in reversed(observed):
        if call.name == "browser_snapshot" and isinstance(call.result, dict):
            return snapshot_body_from_serialized_result(call.result)
    return ""


def build_demo_payload(run: EvaluationRun, scenario: str) -> dict[str, Any]:
    request = run.requests[0]
    observed = list(request["tool_correctness"]["args"][0])
    typed = _typed_todo(observed)
    evaluators = []
    for result, decision in zip(run.results, run.decisions):
        evaluators.append(
            {
                "metric": result.metric,
                "evaluator": _evaluator_label(result.metric),
                "score": f"{float(result.score):.2f}",
                "passed": decision.passed,
                "result": "PASS" if decision.passed else "FAIL",
                "reason": result.reason if (not decision.passed and result.reason) else None,
            }
        )
    gate_passed = run.gate_decision is not None and run.gate_decision.passed
    return {
        "scenario": scenario,
        "goal": GOAL,
        "expected_todo": EXPECTED_TODO_TEXT,
        "typed_todo": typed,
        "todos": [typed] if typed else [],
        "final_snapshot": _final_snapshot_body(observed),
        "evaluators": evaluators,
        "gate": {
            "passed": gate_passed,
            "result": "PASS" if gate_passed else "FAIL",
        },
    }


def execute_demo(
    scenario: str,
    *,
    on_tool_call: Callable[..., Any] | None = None,
    pace_s: float = 0.0,
    tool_correctness_metric: Any | None = None,
    model: Any | None = None,
) -> EvaluationRun:
    """Run live Playwright MCP + existing Runner. Timeline rows come from on_tool_call."""
    if scenario not in {"pass", "fail"}:
        raise ValueError(f"Unsupported demo scenario: {scenario!r}")

    def _hook(invocation: ToolInvocation, *, order: int) -> None:
        if on_tool_call is not None:
            on_tool_call(invocation, order=order)
        if pace_s > 0:
            time.sleep(pace_s)

    hook = _hook if on_tool_call is not None or pace_s > 0 else None
    request = asyncio.run(live_mcp_p0_request(scenario, on_tool_call=hook))
    runner = build_p0_runner(
        tool_correctness_metric=tool_correctness_metric,
        model=model,
    )
    runner.run(
        request,
        EvaluationConfig(evaluations=list(P0_EVALUATIONS)),
        run_id=f"web-mcp-p0-{scenario}",
    )
    if runner.last_run is None:
        raise RuntimeError("EvaluationRunner did not record last_run")
    return runner.last_run


def _client_error_message(exc: BaseException) -> str:
    """Return the innermost exception. ExceptionGroup str() hides the cause."""
    current = exc
    while isinstance(current, BaseExceptionGroup) and current.exceptions:
        current = current.exceptions[0]
    return f"{type(current).__name__}: {current}"


class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AIQEDemo/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_static("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            scenario = (query.get("scenario") or ["pass"])[0]
            self._stream_run(scenario)
            return
        self.send_error(404, "Not found")

    def _send_static(self, name: str, content_type: str) -> None:
        path = _STATIC_DIR / name
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _stream_run(self, scenario: str) -> None:
        if scenario not in {"pass", "fail"}:
            self.send_error(400, "scenario must be pass or fail")
            return
        self.close_connection = False
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def on_tool_call(invocation: ToolInvocation, *, order: int) -> None:
            self._emit("tool", tool_timeline_row(invocation, order=order))

        try:
            run = execute_demo(
                scenario,
                on_tool_call=on_tool_call,
                pace_s=_DEFAULT_PACE_S,
            )
            self._emit("evaluation", build_demo_payload(run, scenario))
            self._emit("done", {})
        except Exception as exc:
            self._emit("error", {"message": _client_error_message(exc)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai_qe_eval.demo.web",
        description="Interactive live Playwright MCP evaluation demo.",
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"AI-QE demo: http://{args.host}:{args.port}/")
    print("Live Playwright MCP stdio — https://demo.playwright.dev/todomvc")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
