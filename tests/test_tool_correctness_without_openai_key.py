"""Regression: deterministic Tool Correctness injects a stub, not a fake key.

DeepEval ToolCorrectnessMetric.__init__ calls initialize_model(). Passing
model=None builds the default provider client. These tests inject an
unused judge explicitly and delete OPENAI_API_KEY after conftest
load_dotenv() so they match GitHub Actions (no .env, no secrets).
"""

from __future__ import annotations

import os

from ai_qe_eval.cli import P0_EVALUATIONS, run_demo
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    DeepEvalToolCorrectnessEvaluator,
)
from tool_correctness_test_doubles import (
    UnusedToolCorrectnessJudge,
    unused_name_order_metric,
)


def _matching_navigate_and_snapshot():
    observed = [
        ToolInvocation(name="browser_navigate", arguments=None, result=None),
        ToolInvocation(name="browser_snapshot", arguments=None, result=None),
    ]
    expected = [
        ToolInvocation(name="browser_navigate", arguments=None, result=None),
        ToolInvocation(name="browser_snapshot", arguments=None, result=None),
    ]
    return observed, expected


def test_explicit_stub_model_name_order_without_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert os.getenv("OPENAI_API_KEY") is None

    observed, expected = _matching_navigate_and_snapshot()
    result = DeepEvalToolCorrectnessEvaluator(
        evaluation_params=[],
        model=UnusedToolCorrectnessJudge(),
    ).evaluate(observed, expected, input="Add a todo.")[0]

    assert result.score == 1.0
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"


def test_explicit_stub_metric_without_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert os.getenv("OPENAI_API_KEY") is None

    observed, expected = _matching_navigate_and_snapshot()
    result = DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=unused_name_order_metric()
    ).evaluate(observed, expected, input="Add a todo.")[0]

    assert result.score == 1.0


def test_scripted_cli_demo_with_explicit_stub_without_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert os.getenv("OPENAI_API_KEY") is None

    run = run_demo("pass", model=UnusedToolCorrectnessJudge())

    assert run.gate_decision is not None
    assert run.gate_decision.passed is True
    assert [result.metric for result in run.results] == list(P0_EVALUATIONS)
    scores = {result.metric: result.score for result in run.results}
    assert scores["tool_correctness"] == 1.0
