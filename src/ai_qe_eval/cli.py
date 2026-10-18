"""Minimal CLI for the existing evaluation engine.

Runs the scripted Playwright MCP P0 demo through capture, mcp_p0_request,
and EvaluationRunner. Does not add a new execution abstraction or Langfuse.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import Any

from ai_qe_eval.demo.mcp_p0 import scripted_mcp_p0_request
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    TOOL_CORRECTNESS_METRIC,
    DeepEvalToolCorrectnessEvaluator,
)
from ai_qe_eval.evaluators.deterministic import (
    FINAL_STATE_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
    FinalStateEvaluator,
    MCPExecutionHealthEvaluator,
)
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

P0_EVALUATIONS = (
    TOOL_CORRECTNESS_METRIC,
    MCP_EXECUTION_HEALTH_METRIC,
    FINAL_STATE_METRIC,
)
TOOL_CORRECTNESS_THRESHOLD = 0.80
_CLI_SCENARIOS = ("pass", "fail")
_REPORT_RULE = "─" * 32
_REPORT_NAME_WIDTH = 24
_REPORT_SCORE_WIDTH = 8
_EVALUATOR_LABELS = {
    TOOL_CORRECTNESS_METRIC: "Tool Correctness",
    MCP_EXECUTION_HEALTH_METRIC: "MCP Execution Health",
    FINAL_STATE_METRIC: "Final State",
}


def demo_request(scenario: str = "pass") -> dict[str, dict[str, list]]:
    """Build the flagship P0 request from the scripted MCP path."""
    if scenario not in _CLI_SCENARIOS:
        raise ValueError(f"Unsupported CLI scenario: {scenario!r}")
    return asyncio.run(scripted_mcp_p0_request(scenario))


def build_p0_runner(
    *,
    tool_correctness_metric: Any | None = None,
    model: Any | None = None,
) -> EvaluationRunner:
    """Wire the existing P0 evaluators, policies, and gate. Not a new runner.

    model and tool_correctness_metric are passed through unchanged. Omitting
    both uses DeepEval's default provider when the metric is constructed.
    """
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name=TOOL_CORRECTNESS_METRIC,
            evaluator="deepeval",
            category="agent",
        )
    )
    registry.register(
        EvaluationCapability(
            name=MCP_EXECUTION_HEALTH_METRIC,
            evaluator="deterministic",
            category="agent",
        )
    )
    registry.register(
        EvaluationCapability(
            name=FINAL_STATE_METRIC,
            evaluator="deterministic",
            category="agent",
        )
    )
    return EvaluationRunner(
        registry=registry,
        evaluators={
            TOOL_CORRECTNESS_METRIC: DeepEvalToolCorrectnessEvaluator(
                evaluation_params=[],
                tool_correctness_metric=tool_correctness_metric,
                model=model,
            ),
            MCP_EXECUTION_HEALTH_METRIC: MCPExecutionHealthEvaluator(),
            FINAL_STATE_METRIC: FinalStateEvaluator(),
        },
        policies={
            TOOL_CORRECTNESS_METRIC: QualityPolicy(
                metric=TOOL_CORRECTNESS_METRIC,
                operator=">=",
                threshold=TOOL_CORRECTNESS_THRESHOLD,
            ),
            MCP_EXECUTION_HEALTH_METRIC: QualityPolicy(
                metric=MCP_EXECUTION_HEALTH_METRIC,
                operator="==",
                threshold=1.0,
            ),
            FINAL_STATE_METRIC: QualityPolicy(
                metric=FINAL_STATE_METRIC,
                operator="==",
                threshold=1.0,
            ),
        },
        gate=QualityGate(),
    )


def _evaluator_label(metric: str) -> str:
    return _EVALUATOR_LABELS.get(metric, metric.replace("_", " ").title())


def format_run(run: EvaluationRun, *, scenario: str = "MCP P0 Demo") -> str:
    """Format EvaluationRun as a concise report. Does not aggregate scores."""
    lines = [
        "AI-QE Evaluation",
        _REPORT_RULE,
        f"Scenario: {scenario}",
        "",
        (
            f"{'Evaluator':<{_REPORT_NAME_WIDTH}}"
            f"{'Score':>{_REPORT_SCORE_WIDTH}}    Result"
        ),
    ]
    for result, decision in zip(run.results, run.decisions):
        status = "PASS" if decision.passed else "FAIL"
        score = f"{float(result.score):.2f}"
        lines.append(
            f"{_evaluator_label(result.metric):<{_REPORT_NAME_WIDTH}}"
            f"{score:>{_REPORT_SCORE_WIDTH}}     {status}"
        )
        if not decision.passed and result.reason:
            lines.append(f"  {result.reason}")
    gate_passed = run.gate_decision is not None and run.gate_decision.passed
    gate_status = "PASS" if gate_passed else "FAIL"
    lines.append("")
    lines.append(
        f"{'Quality Gate':<{_REPORT_NAME_WIDTH}}"
        f"{'':>{_REPORT_SCORE_WIDTH}}     {gate_status}"
    )
    lines.append(_REPORT_RULE)
    return "\n".join(lines)


def run_demo(
    scenario: str = "pass",
    *,
    tool_correctness_metric: Any | None = None,
    model: Any | None = None,
) -> EvaluationRun:
    runner = build_p0_runner(
        tool_correctness_metric=tool_correctness_metric,
        model=model,
    )
    runner.run(
        demo_request(scenario),
        EvaluationConfig(evaluations=list(P0_EVALUATIONS)),
        run_id=f"cli-mcp-p0-{scenario}",
    )
    if runner.last_run is None:
        raise RuntimeError("EvaluationRunner did not record last_run")
    return runner.last_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai_qe_eval",
        description=(
            "Run the scripted Playwright MCP P0 demo through EvaluationRunner."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=_CLI_SCENARIOS,
        default="pass",
        help="pass (Buy milk) or fail (final_state mismatch: Buy bread).",
    )
    args = parser.parse_args(argv)
    run = run_demo(args.scenario)
    print(format_run(run))
    if run.gate_decision is None or not run.gate_decision.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
