"""Focused tests for P2-05 Evaluator contract.

Uses in-process fakes only. No RAGAS, DeepEval, or Phase 1 adapters.
Does not construct EvaluationTrace. Evidence arguments are evaluator-specific.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.result import EvaluationResult

_EVALUATOR_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "ai_qe_eval" / "domain" / "evaluator.py"
)


class FakeEvaluator:
    def __init__(self) -> None:
        self.received_args = None
        self.received_kwargs = None
        self.received_configuration = None

    def evaluate(self, *args, configuration=None, **kwargs):
        self.received_args = args
        self.received_kwargs = kwargs
        self.received_configuration = configuration
        return [
            EvaluationResult(metric="fake_metric", evaluator="fake", score=1.0),
        ]


class MultiMetricFakeEvaluator:
    def evaluate(self, *args, configuration=None, **kwargs):
        return [
            EvaluationResult(metric="context_precision", evaluator="fake", score=0.95),
            EvaluationResult(metric="faithfulness", evaluator="fake", score=0.91),
        ]


class DeterministicStyleEvaluator:
    def evaluate(self, output, expected, configuration=None):
        score = 1.0 if output == expected else 0.0
        return [
            EvaluationResult(metric="exact_match", evaluator="deterministic", score=score),
        ]


class SemanticStyleEvaluator:
    def evaluate(self, input, output, configuration=None):
        return [
            EvaluationResult(metric="correctness", evaluator="semantic", score=0.91),
        ]


def test_fake_evaluator_conforms_to_evaluator_protocol():
    evaluator = FakeEvaluator()
    assert isinstance(evaluator, Evaluator)


def test_evaluator_receives_evaluator_specific_evidence():
    evaluator = FakeEvaluator()
    evaluator.evaluate("question-1", "answer-1")
    assert evaluator.received_args == ("question-1", "answer-1")
    assert evaluator.received_kwargs == {}


def test_configuration_is_passed_through_unchanged():
    evaluator = FakeEvaluator()
    configuration = {"mode": "strict", "custom_value": 123}
    evaluator.evaluate("question-1", configuration=configuration)
    assert evaluator.received_configuration is configuration
    assert evaluator.received_configuration["mode"] == "strict"
    assert evaluator.received_configuration["custom_value"] == 123


def test_evaluate_returns_a_list_of_evaluation_results():
    results = FakeEvaluator().evaluate("question-1")
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], EvaluationResult)
    assert results[0].metric == "fake_metric"
    assert results[0].evaluator == "fake"
    assert results[0].score == 1.0


def test_evaluator_can_return_multiple_evaluation_results():
    results = MultiMetricFakeEvaluator().evaluate("question-1")
    assert isinstance(MultiMetricFakeEvaluator(), Evaluator)
    assert [item.metric for item in results] == ["context_precision", "faithfulness"]
    assert [item.score for item in results] == [0.95, 0.91]


def test_deterministic_style_evaluator_conforms_to_the_same_contract():
    evaluator = DeterministicStyleEvaluator()
    assert isinstance(evaluator, Evaluator)
    results = evaluator.evaluate("answer-1", "expected-1")
    assert results[0].metric == "exact_match"
    assert results[0].evaluator == "deterministic"
    assert results[0].score == 0.0


def test_semantic_style_evaluator_conforms_to_the_same_contract():
    evaluator = SemanticStyleEvaluator()
    assert isinstance(evaluator, Evaluator)
    results = evaluator.evaluate("question-1", "answer-1")
    assert results[0].metric == "correctness"
    assert results[0].evaluator == "semantic"
    assert results[0].score == 0.91


def test_contract_does_not_require_threshold_passed_or_severity():
    results = FakeEvaluator().evaluate("question-1")
    fields = results[0].__dataclass_fields__
    assert "threshold" not in fields
    assert "passed" not in fields
    assert "severity" not in fields


def test_evaluate_may_be_called_without_configuration():
    evaluator = FakeEvaluator()
    evaluator.evaluate("question-1")
    assert evaluator.received_configuration is None


def test_evaluator_module_does_not_import_vendor_or_transport_packages():
    tree = ast.parse(_EVALUATOR_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            imported.update(names)
            imported_roots.update(name.split(".", 1)[0] for name in names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "ragas",
        "deepeval",
        "langchain",
        "langchain_openai",
        "openai",
        "requests",
        "httpx",
        "pytest",
        "mcp",
        "langfuse",
        "langgraph",
        "opentelemetry",
        "abc",
    }
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.domain.trace" not in imported
    assert not any(name.endswith(".trace") for name in imported)
