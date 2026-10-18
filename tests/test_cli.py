"""Focused tests for the CLI evaluation report.

Uses the deterministic demo fixture. No live MCP or Langfuse.
"""

from __future__ import annotations

from ai_qe_eval import cli as cli_mod
from ai_qe_eval.cli import P0_EVALUATIONS, format_run, main, run_demo
from ai_qe_eval.domain.result import EvaluationResult
from tool_correctness_test_doubles import (
    UnusedToolCorrectnessJudge,
    install_explicit_p0_tool_correctness_stub,
)
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.evaluators.deterministic import FINAL_STATE_FAILED_REASON
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.policy.quality_policy import PolicyDecision

_PASS_REPORT = """\
AI-QE Evaluation
────────────────────────────────
Scenario: MCP P0 Demo

Evaluator                  Score    Result
Tool Correctness            1.00     PASS
MCP Execution Health        1.00     PASS
Final State                 1.00     PASS

Quality Gate                         PASS
────────────────────────────────"""

_FAIL_REPORT = """\
AI-QE Evaluation
────────────────────────────────
Scenario: MCP P0 Demo

Evaluator                  Score    Result
Tool Correctness            1.00     PASS
MCP Execution Health        1.00     PASS
Final State                 0.00     FAIL
  Final-state check failed.

Quality Gate                         FAIL
────────────────────────────────"""


def _result(metric: str, score: float, reason: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        evaluator="test",
        score=score,
        reason=reason,
    )


def _decision(metric: str, score: float, passed: bool) -> PolicyDecision:
    return PolicyDecision(
        metric=metric,
        score=score,
        operator=">=",
        threshold=1.0,
        passed=passed,
    )


def _run(
    scores: dict[str, float],
    *,
    passed: dict[str, bool],
    reasons: dict[str, str] | None = None,
    gate_passed: bool,
) -> EvaluationRun:
    reasons = reasons or {}
    results = [
        _result(metric, scores[metric], reasons.get(metric))
        for metric in P0_EVALUATIONS
    ]
    decisions = [
        _decision(metric, scores[metric], passed[metric]) for metric in P0_EVALUATIONS
    ]
    return EvaluationRun(
        run_id="cli-format",
        results=results,
        decisions=decisions,
        gate_decision=GateDecision(
            passed=gate_passed,
            decisions=decisions,
            reason=None,
        ),
    )


def test_format_run_pass_matches_report_layout():
    run = _run(
        {
            "tool_correctness": 1.0,
            "mcp_execution_health": 1.0,
            "final_state": 1.0,
        },
        passed={
            "tool_correctness": True,
            "mcp_execution_health": True,
            "final_state": True,
        },
        gate_passed=True,
    )
    assert format_run(run) == _PASS_REPORT


def test_format_run_fail_shows_fail_and_evaluator_reason():
    run = _run(
        {
            "tool_correctness": 1.0,
            "mcp_execution_health": 1.0,
            "final_state": 0.0,
        },
        passed={
            "tool_correctness": True,
            "mcp_execution_health": True,
            "final_state": False,
        },
        reasons={"final_state": FINAL_STATE_FAILED_REASON},
        gate_passed=False,
    )
    report = format_run(run)
    assert report == _FAIL_REPORT
    assert "FAIL" in report
    assert FINAL_STATE_FAILED_REASON in report


def test_cli_demo_pass_prints_report_and_exits_zero(capsys, monkeypatch):
    install_explicit_p0_tool_correctness_stub(monkeypatch, cli_mod)
    exit_code = main(["--scenario", "pass"])
    output = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert output == _PASS_REPORT
    assert "Quality Gate                         FAIL" not in output


def test_cli_demo_fail_prints_report_and_exits_nonzero(capsys, monkeypatch):
    install_explicit_p0_tool_correctness_stub(monkeypatch, cli_mod)
    exit_code = main(["--scenario", "fail"])
    output = capsys.readouterr().out.strip()

    assert exit_code == 1
    assert output == _FAIL_REPORT
    assert "Quality Gate                         PASS" not in output


def test_cli_run_demo_uses_existing_runner_and_does_not_aggregate():
    run = run_demo("fail", model=UnusedToolCorrectnessJudge())
    report = format_run(run)

    assert run.gate_decision is not None
    assert run.gate_decision.passed is False
    assert [result.metric for result in run.results] == list(P0_EVALUATIONS)
    assert [decision.metric for decision in run.decisions] == list(P0_EVALUATIONS)
    assert report.count("Quality Gate") == 1
    assert "average" not in report.lower()
    assert "pass rate" not in report.lower()
