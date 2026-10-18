"""Focused tests for the MVP-03 scripted MCP P0 CLI demo.

Proves the CLI reaches EvaluationRunner on the existing P0 path.
No live Playwright MCP server. No Langfuse.
"""

from __future__ import annotations

from ai_qe_eval import cli as cli_mod
from ai_qe_eval.cli import P0_EVALUATIONS, main, run_demo
from tool_correctness_test_doubles import (
    UnusedToolCorrectnessJudge,
    install_explicit_p0_tool_correctness_stub,
)
from ai_qe_eval.demo.mcp_p0 import (
    EXPECTED_TODO_TEXT,
    EXPECTED_TOOL_ORDER,
    TYPED_TODO_TEXT_FAIL,
)
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_FAILED_REASON,
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
)
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def _typed_text(observed) -> str:
    for call in observed:
        if call.name == "browser_type" and isinstance(call.arguments, dict):
            return call.arguments["text"]
    raise AssertionError("browser_type was not captured")


def test_cli_pass_demo_reaches_evaluation_runner(monkeypatch):
    seen: list[EvaluationRunner] = []
    original_run = EvaluationRunner.run

    def _recording_run(self, request, configuration, *, run_id=""):
        seen.append(self)
        return original_run(self, request, configuration, run_id=run_id)

    monkeypatch.setattr(EvaluationRunner, "run", _recording_run)

    run = run_demo("pass", model=UnusedToolCorrectnessJudge())
    request = run.requests[0]
    observed = request["tool_correctness"]["args"][0]
    expected = request["tool_correctness"]["args"][1]
    results = {result.metric: result for result in run.results}

    assert seen, "CLI demo must call EvaluationRunner.run"
    assert isinstance(run, EvaluationRun)
    assert isinstance(run.gate_decision, GateDecision)
    assert run.gate_decision.passed is True
    assert [call.name for call in observed] == list(EXPECTED_TOOL_ORDER)
    assert [call.name for call in expected] == list(EXPECTED_TOOL_ORDER)
    assert all(call.arguments is None for call in expected)
    assert all(call.result is None for call in expected)
    assert all(isinstance(call.result, dict) for call in observed)
    assert all(call.result.get("isError") is False for call in observed)
    assert _typed_text(observed) == EXPECTED_TODO_TEXT
    assert results["tool_correctness"].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    assert results[FINAL_STATE_METRIC].score == 1.0
    assert [result.metric for result in run.results] == list(P0_EVALUATIONS)


def test_cli_fail_demo_reuses_final_state_mismatch_and_reaches_runner(monkeypatch):
    seen: list[EvaluationRunner] = []
    original_run = EvaluationRunner.run

    def _recording_run(self, request, configuration, *, run_id=""):
        seen.append(self)
        return original_run(self, request, configuration, run_id=run_id)

    monkeypatch.setattr(EvaluationRunner, "run", _recording_run)

    run = run_demo("fail", model=UnusedToolCorrectnessJudge())
    request = run.requests[0]
    observed = request["tool_correctness"]["args"][0]
    results = {result.metric: result for result in run.results}
    decisions = {item.metric: item for item in run.decisions}

    assert seen, "CLI demo must call EvaluationRunner.run"
    assert run.gate_decision is not None
    assert run.gate_decision.passed is False
    assert [call.name for call in observed] == list(EXPECTED_TOOL_ORDER)
    assert _typed_text(observed) == TYPED_TODO_TEXT_FAIL
    assert results["tool_correctness"].score == 1.0
    assert results[MCP_EXECUTION_HEALTH_METRIC].score == 1.0
    assert results[FINAL_STATE_METRIC].score == 0.0
    assert results[FINAL_STATE_METRIC].reason == FINAL_STATE_FAILED_REASON
    assert decisions["tool_correctness"].passed is True
    assert decisions[MCP_EXECUTION_HEALTH_METRIC].passed is True
    assert decisions[FINAL_STATE_METRIC].passed is False


def test_cli_pass_and_fail_exit_codes(capsys, monkeypatch):
    install_explicit_p0_tool_correctness_stub(monkeypatch, cli_mod)
    assert main(["--scenario", "pass"]) == 0
    pass_output = capsys.readouterr().out
    assert "Quality Gate                         PASS" in pass_output
    assert "Tool Correctness" in pass_output

    assert main(["--scenario", "fail"]) == 1
    fail_output = capsys.readouterr().out
    assert "Quality Gate                         FAIL" in fail_output
    assert FINAL_STATE_FAILED_REASON in fail_output
