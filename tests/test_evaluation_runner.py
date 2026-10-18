"""Focused tests for P2-14 Thin Evaluation Runner.

Orchestration only. No live RAGAS or DeepEval providers.
Uses capability-keyed request maps. Does not construct EvaluationTrace.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.normalization.result_normalizer import normalize_many
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

_RUNNER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "runner"
    / "evaluation_runner.py"
)


def _payload(*args, **kwargs) -> dict:
    return {"args": list(args), "kwargs": dict(kwargs)}


def _request(*names: str) -> dict[str, dict]:
    return {name: _payload("question") for name in names}


def _result(metric: str, score: float, evaluator: str = "fake") -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        evaluator=evaluator,
        score=score,
        reason="synthetic test result",
    )


class FakeEvaluator:
    def __init__(self, results: list[EvaluationResult] | None = None, error: Exception | None = None):
        self.results = results if results is not None else [_result("correctness", 0.9)]
        self.error = error
        self.received_args = None
        self.received_kwargs = None
        self.call_count = 0

    def evaluate(self, *args, **kwargs) -> list[EvaluationResult]:
        self.received_args = args
        self.received_kwargs = kwargs
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return list(self.results)


class FakeNormalizer:
    def __init__(self):
        self.received = None

    def normalize_many(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        self.received = results
        return normalize_many(results)


class FakePolicy:
    def __init__(self, decision: PolicyDecision | None = None, error: Exception | None = None):
        self.decision = decision
        self.error = error
        self.received = None

    def apply(self, result: EvaluationResult) -> PolicyDecision:
        self.received = result
        if self.error is not None:
            raise self.error
        if self.decision is not None:
            return self.decision
        return PolicyDecision(
            metric=result.metric,
            score=result.score,
            operator=">=",
            threshold=0.8,
            passed=True,
        )


class FakeGate:
    def __init__(self, decision: GateDecision | None = None):
        self.received = None
        self.decision = decision

    def evaluate(self, decisions: list[PolicyDecision]) -> GateDecision:
        self.received = list(decisions)
        if self.decision is not None:
            return self.decision
        return QualityGate().evaluate(decisions)


def _registry(*names: str) -> EvaluationRegistry:
    registry = EvaluationRegistry()
    for name in names:
        registry.register(
            EvaluationCapability(name=name, evaluator="fake", category="test")
        )
    return registry


def _runner(
    *,
    names: tuple[str, ...] = ("correctness",),
    evaluators: dict | None = None,
    policies: dict | None = None,
    normalizer=None,
    gate=None,
) -> EvaluationRunner:
    evaluator_map = evaluators if evaluators is not None else {
        name: FakeEvaluator([_result(name, 0.9)]) for name in names
    }
    policy_map = policies if policies is not None else {
        name: QualityPolicy(metric=name, operator=">=", threshold=0.8) for name in names
    }
    return EvaluationRunner(
        registry=_registry(*names),
        evaluators=evaluator_map,
        policies=policy_map,
        normalizer=normalizer,
        gate=gate,
    )


def test_single_evaluator_end_to_end_pass():
    evaluator = FakeEvaluator([_result("correctness", 0.9)])
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": evaluator},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    decision = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert isinstance(decision, GateDecision)
    assert decision.passed is True
    assert evaluator.received_args == ("question",)


def test_evaluator_receives_the_exact_request_arguments():
    evaluator = FakeEvaluator()
    request = {"correctness": _payload("Paris", "Paris")}
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": evaluator},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    runner.run(request, EvaluationConfig(evaluations=["correctness"]))
    assert evaluator.received_args == ("Paris", "Paris")
    assert evaluator.received_kwargs == {}


def test_evaluation_config_controls_which_capability_runs():
    first = FakeEvaluator([_result("faithfulness", 0.9)])
    second = FakeEvaluator([_result("correctness", 0.9)])
    runner = EvaluationRunner(
        registry=_registry("faithfulness", "correctness"),
        evaluators={"faithfulness": first, "correctness": second},
        policies={
            "faithfulness": QualityPolicy(metric="faithfulness", operator=">=", threshold=0.8),
            "correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8),
        },
    )
    runner.run(
        _request("faithfulness", "correctness"),
        EvaluationConfig(evaluations=["faithfulness"]),
    )
    assert first.call_count == 1
    assert second.call_count == 0


def test_multiple_evaluators_execute_in_configuration_order():
    order: list[str] = []

    class OrderedEvaluator(FakeEvaluator):
        def __init__(self, name: str, results):
            super().__init__(results)
            self.name = name

        def evaluate(self, *args, **kwargs):
            order.append(self.name)
            return super().evaluate(*args, **kwargs)

    first = OrderedEvaluator("faithfulness", [_result("faithfulness", 0.95)])
    second = OrderedEvaluator("correctness", [_result("correctness", 0.88)])
    third = OrderedEvaluator("exact_match", [_result("exact_match", 1.0)])
    runner = EvaluationRunner(
        registry=_registry("faithfulness", "correctness", "exact_match"),
        evaluators={
            "faithfulness": first,
            "correctness": second,
            "exact_match": third,
        },
        policies={
            "faithfulness": QualityPolicy(metric="faithfulness", operator=">=", threshold=0.8),
            "correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8),
            "exact_match": QualityPolicy(metric="exact_match", operator=">=", threshold=1.0),
        },
    )
    decision = runner.run(
        _request("faithfulness", "correctness", "exact_match"),
        EvaluationConfig(evaluations=["correctness", "faithfulness", "exact_match"]),
    )
    assert order == ["correctness", "faithfulness", "exact_match"]
    assert [item.metric for item in decision.decisions] == [
        "correctness",
        "faithfulness",
        "exact_match",
    ]


def test_multiple_results_from_one_evaluator_are_preserved():
    evaluator = FakeEvaluator(
        [
            _result("faithfulness", 0.95),
            _result("answer_relevancy", 0.91),
        ]
    )
    runner = EvaluationRunner(
        registry=_registry("rag_bundle"),
        evaluators={"rag_bundle": evaluator},
        policies={
            "faithfulness": QualityPolicy(metric="faithfulness", operator=">=", threshold=0.8),
            "answer_relevancy": QualityPolicy(metric="answer_relevancy", operator=">=", threshold=0.8),
        },
    )
    decision = runner.run(_request("rag_bundle"), EvaluationConfig(evaluations=["rag_bundle"]))
    assert [item.metric for item in decision.decisions] == [
        "faithfulness",
        "answer_relevancy",
    ]
    assert len(decision.decisions) == 2


def test_results_are_passed_through_normalizer():
    raw = _result("correctness", 0.9)
    evaluator = FakeEvaluator([raw])
    normalizer = FakeNormalizer()
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": evaluator},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
        normalizer=normalizer,
    )
    runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert normalizer.received is not None
    assert normalizer.received[0] is raw


def test_policy_receives_normalized_result():
    raw = _result("correctness", 0.9)
    policy = FakePolicy()
    normalizer = FakeNormalizer()
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator([raw])},
        policies={"correctness": policy},
        normalizer=normalizer,
    )
    runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert policy.received is not None
    assert policy.received is not raw
    assert policy.received.score == 0.9
    assert policy.received.metric == "correctness"


def test_quality_gate_receives_complete_policy_decisions():
    gate = FakeGate()
    runner = EvaluationRunner(
        registry=_registry("faithfulness", "correctness"),
        evaluators={
            "faithfulness": FakeEvaluator([_result("faithfulness", 0.95)]),
            "correctness": FakeEvaluator([_result("correctness", 0.88)]),
        },
        policies={
            "faithfulness": QualityPolicy(metric="faithfulness", operator=">=", threshold=0.8),
            "correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8),
        },
        gate=gate,
    )
    returned = runner.run(
        _request("faithfulness", "correctness"),
        EvaluationConfig(evaluations=["faithfulness", "correctness"]),
    )
    assert gate.received is not None
    assert [item.metric for item in gate.received] == ["faithfulness", "correctness"]
    assert all(isinstance(item, PolicyDecision) for item in gate.received)
    assert returned.passed is True


def test_missing_evaluator_capability_in_registry_raises_key_error():
    runner = EvaluationRunner(
        registry=_registry("faithfulness"),
        evaluators={"faithfulness": FakeEvaluator()},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    with pytest.raises(KeyError, match="Unknown evaluation capability"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_missing_wired_evaluator_instance_raises_key_error():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    with pytest.raises(KeyError, match="evaluator instance"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_missing_request_payload_raises_key_error():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator()},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    with pytest.raises(KeyError, match="No request payload for capability"):
        runner.run({}, EvaluationConfig(evaluations=["correctness"]))
    assert runner.last_run is None


def test_missing_policy_raises_key_error():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator([_result("correctness", 0.9)])},
        policies={},
    )
    with pytest.raises(KeyError, match="quality policy"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_evaluator_exception_propagates():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator(error=RuntimeError("evaluator failed"))},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    with pytest.raises(RuntimeError, match="evaluator failed"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_policy_exception_propagates():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator([_result("correctness", 0.9)])},
        policies={"correctness": FakePolicy(error=RuntimeError("policy failed"))},
    )
    with pytest.raises(RuntimeError, match="policy failed"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_no_score_aggregation_or_threshold_logic_in_runner():
    source = _RUNNER_SOURCE.read_text(encoding="utf-8")
    forbidden_snippets = (
        "average",
        "weighted",
        "overall_score",
        "score >=",
        "score >",
        "score <",
        "eval(",
    )
    for snippet in forbidden_snippets:
        assert snippet not in source
    decision = _runner().run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    for name in ("average_score", "overall_score", "weighted_score"):
        assert not hasattr(decision, name)


def test_request_is_not_mutated():
    request = {"correctness": _payload("original")}
    snapshot = deepcopy(request)
    _runner().run(request, EvaluationConfig(evaluations=["correctness"]))
    assert request == snapshot


def test_evaluation_result_is_not_mutated():
    raw = _result("correctness", 0.9)
    snapshot = replace(raw)
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator([raw])},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert raw == snapshot
    assert raw.score == 0.9


def test_empty_config_fails_before_evaluator_or_gate_execution():
    evaluator = FakeEvaluator()
    gate = FakeGate()
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": evaluator},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
        gate=gate,
    )
    with pytest.raises(ValueError, match="at least one configured evaluation"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=[]))

    assert evaluator.call_count == 0
    assert gate.received is None
    assert runner.last_run is None


def test_end_to_end_failure_case():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": FakeEvaluator([_result("correctness", 0.7)])},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    decision = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert decision.passed is False
    assert decision.decisions[0].score == 0.7
    assert decision.decisions[0].passed is False


def test_spy_delegation_does_not_implement_component_logic():
    evaluator = FakeEvaluator([_result("correctness", 0.9)])
    normalizer = FakeNormalizer()
    policy = FakePolicy(
        PolicyDecision(
            metric="correctness",
            score=0.9,
            operator=">=",
            threshold=0.8,
            passed=True,
        )
    )
    expected = GateDecision(passed=True, decisions=[], reason="spy")
    gate = FakeGate(expected)
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": evaluator},
        policies={"correctness": policy},
        normalizer=normalizer,
        gate=gate,
    )
    returned = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert evaluator.received_args is not None
    assert normalizer.received is not None
    assert policy.received is not None
    assert gate.received is not None
    assert returned is expected


def test_runner_module_has_no_vendor_or_concrete_evaluator_imports():
    tree = ast.parse(_RUNNER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            imported.update(names)
            imported_roots.update(name.split(".", 1)[0] for name in names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported_roots.add(node.module.split(".", 1)[0])
            imported_modules.add(node.module)
    forbidden = {"ragas", "deepeval", "langchain", "openai", "pytest"}
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.evaluators" not in imported_modules
    assert "ai_qe_eval.evaluators.ragas" not in imported_modules
    assert "ai_qe_eval.evaluators.deepeval" not in imported_modules
    assert "ai_qe_eval.evaluators.deterministic" not in imported_modules
    assert "ai_qe_eval.domain.trace" not in imported
