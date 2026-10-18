"""P2-15 end-to-end validation of the frozen Phase 2 pipeline.

Deterministic test doubles only for the primary scenarios.
No Together, RAGAS, or DeepEval live provider calls.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.normalization.result_normalizer import normalize_many
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def _request(*names: str, output: str = "Paris", expected: str = "Paris") -> dict:
    return {name: {"args": [output, expected], "kwargs": {}} for name in names}


def _result(
    metric: str,
    score: float,
    evaluator: str = "fake",
    reason: str = "synthetic test result",
    raw_result=None,
) -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        evaluator=evaluator,
        score=score,
        reason=reason,
        raw_result={"synthetic": True} if raw_result is None else raw_result,
    )


class RecordingEvaluator:
    def __init__(self, results: list[EvaluationResult]):
        self.results = results
        self.received_args = None
        self.call_count = 0

    def evaluate(self, *args, **kwargs) -> list[EvaluationResult]:
        self.received_args = args
        self.call_count += 1
        return list(self.results)


class RecordingNormalizer:
    def __init__(self):
        self.received = None
        self.returned = None

    def normalize_many(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        self.received = list(results)
        self.returned = normalize_many(results)
        return self.returned


class RecordingPolicy:
    def __init__(self, policy: QualityPolicy):
        self.policy = policy
        self.received = None
        self.returned = None

    def apply(self, result: EvaluationResult) -> PolicyDecision:
        self.received = result
        self.returned = self.policy.apply(result)
        return self.returned


class RecordingGate:
    def __init__(self):
        self.received = None
        self.returned = None
        self.inner = QualityGate()

    def evaluate(self, decisions: list[PolicyDecision]) -> GateDecision:
        self.received = list(decisions)
        self.returned = self.inner.evaluate(decisions)
        return self.returned


def _registry(*names: str) -> EvaluationRegistry:
    registry = EvaluationRegistry()
    for name in names:
        registry.register(
            EvaluationCapability(name=name, evaluator="fake", category="e2e")
        )
    return registry


def _wired_runner(
    evaluators: dict,
    policies: dict,
    *,
    names: tuple[str, ...] | None = None,
    normalizer=None,
    gate=None,
) -> EvaluationRunner:
    capability_names = names if names is not None else tuple(evaluators.keys())
    return EvaluationRunner(
        registry=_registry(*capability_names),
        evaluators=evaluators,
        policies=policies,
        normalizer=normalizer,
        gate=gate,
    )


def test_e2e_success_path_two_capabilities():
    exact_results = [
        _result(
            "exact_match",
            1.0,
            evaluator="deterministic",
            reason="Exact match succeeded.",
            raw_result={"matched": True},
        )
    ]
    correctness_results = [
        _result(
            "correctness",
            0.90,
            evaluator="fake",
            reason="synthetic correctness",
            raw_result={"score": 0.90},
        )
    ]
    exact_eval = RecordingEvaluator(exact_results)
    correctness_eval = RecordingEvaluator(correctness_results)
    normalizer = RecordingNormalizer()
    exact_policy = RecordingPolicy(QualityPolicy(metric="exact_match", operator=">=", threshold=1.0))
    correctness_policy = RecordingPolicy(
        QualityPolicy(metric="correctness", operator=">=", threshold=0.80)
    )
    gate = RecordingGate()
    runner = _wired_runner(
        {"exact_match": exact_eval, "correctness": correctness_eval},
        {"exact_match": exact_policy, "correctness": correctness_policy},
        normalizer=normalizer,
        gate=gate,
    )
    request = _request("exact_match", "correctness")
    snapshot = {"exact_match": dict(request["exact_match"]), "correctness": dict(request["correctness"])}
    decision = runner.run(
        request,
        EvaluationConfig(evaluations=["exact_match", "correctness"]),
    )

    assert exact_eval.call_count == 1
    assert correctness_eval.call_count == 1
    assert exact_eval.received_args == ("Paris", "Paris")
    assert correctness_eval.received_args == ("Paris", "Paris")
    assert request["exact_match"]["args"] == snapshot["exact_match"]["args"]
    assert request["correctness"]["args"] == snapshot["correctness"]["args"]

    assert normalizer.received is not None
    assert [item.metric for item in normalizer.received] == ["exact_match", "correctness"]
    assert normalizer.received[0] is exact_results[0]
    assert normalizer.returned is not None
    assert normalizer.returned[0] is not exact_results[0]

    assert exact_policy.received is normalizer.returned[0]
    assert correctness_policy.received is normalizer.returned[1]
    assert exact_policy.returned.passed is True
    assert correctness_policy.returned.passed is True
    assert exact_policy.received.score == 1.0
    assert correctness_policy.received.score == 0.90

    assert gate.received is not None
    assert [item.metric for item in gate.received] == ["exact_match", "correctness"]
    assert all(item.passed for item in gate.received)
    assert isinstance(decision, GateDecision)
    assert decision is gate.returned
    assert decision.passed is True
    assert [item.score for item in decision.decisions] == [1.0, 0.90]


def test_e2e_failure_path_one_policy_fails():
    exact_eval = RecordingEvaluator([_result("exact_match", 1.0, evaluator="deterministic")])
    correctness_eval = RecordingEvaluator([_result("correctness", 0.70, evaluator="fake")])
    gate = RecordingGate()
    runner = _wired_runner(
        {"exact_match": exact_eval, "correctness": correctness_eval},
        {
            "exact_match": QualityPolicy(metric="exact_match", operator=">=", threshold=1.0),
            "correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.80),
        },
        gate=gate,
    )
    decision = runner.run(
        _request("exact_match", "correctness"),
        EvaluationConfig(evaluations=["exact_match", "correctness"]),
    )
    assert decision.passed is False
    assert gate.received[0].passed is True
    assert gate.received[1].passed is False
    assert gate.received[1].score == 0.70
    assert gate.received[1].threshold == 0.80
    assert [item.passed for item in decision.decisions] == [True, False]


def test_e2e_multi_result_evaluator_preserves_order_without_aggregation():
    bundle = RecordingEvaluator(
        [
            _result("metric_a", 0.91, reason="a"),
            _result("metric_b", 0.42, reason="b"),
        ]
    )
    normalizer = RecordingNormalizer()
    gate = RecordingGate()
    runner = _wired_runner(
        {"bundle": bundle},
        {
            "metric_a": QualityPolicy(metric="metric_a", operator=">=", threshold=0.80),
            "metric_b": QualityPolicy(metric="metric_b", operator=">=", threshold=0.80),
        },
        normalizer=normalizer,
        gate=gate,
    )
    decision = runner.run(_request("bundle"), EvaluationConfig(evaluations=["bundle"]))
    assert [item.metric for item in normalizer.received] == ["metric_a", "metric_b"]
    assert [item.metric for item in gate.received] == ["metric_a", "metric_b"]
    assert [item.passed for item in decision.decisions] == [True, False]
    assert decision.passed is False
    for name in ("average_score", "overall_score", "weighted_score"):
        assert not hasattr(decision, name)


def test_e2e_empty_configuration_fails_closed_before_gate():
    evaluator = RecordingEvaluator([_result("correctness", 0.9)])
    gate = RecordingGate()
    runner = _wired_runner(
        {"correctness": evaluator},
        {"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.80)},
        gate=gate,
    )
    with pytest.raises(ValueError, match="at least one configured evaluation"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=[]))

    assert evaluator.call_count == 0
    assert gate.received is None
    assert runner.last_run is None


def test_e2e_missing_evaluator_capability_fails_clearly():
    runner = EvaluationRunner(
        registry=_registry("exact_match"),
        evaluators={"exact_match": RecordingEvaluator([_result("exact_match", 1.0)])},
        policies={"exact_match": QualityPolicy(metric="exact_match", operator=">=", threshold=1.0)},
    )
    with pytest.raises(KeyError, match="Unknown evaluation capability"):
        runner.run(_request("unknown_metric"), EvaluationConfig(evaluations=["unknown_metric"]))


def test_e2e_missing_policy_fails_clearly():
    runner = _wired_runner(
        {"correctness": RecordingEvaluator([_result("correctness", 0.9)])},
        {},
        names=("correctness",),
    )
    with pytest.raises(KeyError, match="No quality policy configured for metric"):
        runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))


def test_e2e_request_identity_and_immutability():
    evaluator = RecordingEvaluator([_result("correctness", 0.9)])
    request = _request("correctness")
    before = {"correctness": {"args": list(request["correctness"]["args"]), "kwargs": {}}}
    runner = _wired_runner(
        {"correctness": evaluator},
        {"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.80)},
    )
    runner.run(request, EvaluationConfig(evaluations=["correctness"]))
    assert evaluator.received_args == ("Paris", "Paris")
    assert request == before


def test_e2e_result_integrity_through_normalization_and_policy():
    raw = _result(
        "correctness",
        0.90,
        evaluator="fake",
        reason="synthetic integrity",
        raw_result={"original": True, "nested": {"k": 1}},
    )
    snapshot = replace(raw, raw_result={"original": True, "nested": {"k": 1}})
    policy = RecordingPolicy(QualityPolicy(metric="correctness", operator=">=", threshold=0.80))
    runner = _wired_runner(
        {"correctness": RecordingEvaluator([raw])},
        {"correctness": policy},
        normalizer=RecordingNormalizer(),
    )
    decision = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert raw == snapshot
    assert raw.score == 0.90
    normalized = policy.received
    assert normalized is not raw
    assert normalized.metric == "correctness"
    assert normalized.evaluator == "fake"
    assert normalized.score == 0.90
    assert normalized.reason == "synthetic integrity"
    assert normalized.raw_result == {"original": True, "nested": {"k": 1}}
    assert decision.decisions[0].score == 0.90
    assert decision.decisions[0].passed is True


def test_e2e_observable_pipeline_sequence():
    sequence: list[str] = []

    class SequencedEvaluator(RecordingEvaluator):
        def evaluate(self, *args, **kwargs):
            sequence.append("evaluator")
            return super().evaluate(*args, **kwargs)

    class SequencedNormalizer(RecordingNormalizer):
        def normalize_many(self, results):
            sequence.append("normalizer")
            return super().normalize_many(results)

    class SequencedPolicy(RecordingPolicy):
        def apply(self, result):
            sequence.append("policy")
            return super().apply(result)

    class SequencedGate(RecordingGate):
        def evaluate(self, decisions):
            sequence.append("gate")
            return super().evaluate(decisions)

    runner = _wired_runner(
        {
            "exact_match": SequencedEvaluator([_result("exact_match", 1.0)]),
            "correctness": SequencedEvaluator([_result("correctness", 0.90)]),
        },
        {
            "exact_match": SequencedPolicy(
                QualityPolicy(metric="exact_match", operator=">=", threshold=1.0)
            ),
            "correctness": SequencedPolicy(
                QualityPolicy(metric="correctness", operator=">=", threshold=0.80)
            ),
        },
        normalizer=SequencedNormalizer(),
        gate=SequencedGate(),
    )
    decision = runner.run(
        _request("exact_match", "correctness"),
        EvaluationConfig(evaluations=["exact_match", "correctness"]),
    )
    assert sequence == [
        "evaluator",
        "evaluator",
        "normalizer",
        "policy",
        "policy",
        "gate",
    ]
    assert decision.passed is True


def test_e2e_gate_consumes_policy_passed_not_recalculated_scores():
    class AuthoritativePolicy:
        def apply(self, result: EvaluationResult) -> PolicyDecision:
            return PolicyDecision(
                metric=result.metric,
                score=result.score,
                operator=">=",
                threshold=0.80,
                passed=False,
            )

    gate = RecordingGate()
    runner = _wired_runner(
        {"correctness": RecordingEvaluator([_result("correctness", 0.95)])},
        {"correctness": AuthoritativePolicy()},
        gate=gate,
    )
    decision = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert gate.received[0].score == 0.95
    assert gate.received[0].threshold == 0.80
    assert gate.received[0].passed is False
    assert decision.passed is False


def test_e2e_real_deterministic_evaluator_smoke_without_live_providers():
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="exact_match",
            evaluator="deterministic",
            category="deterministic",
        )
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"exact_match": DeterministicEvaluator()},
        policies={"exact_match": QualityPolicy(metric="exact_match", operator=">=", threshold=1.0)},
        gate=QualityGate(),
    )
    passing = runner.run(
        _request("exact_match"),
        EvaluationConfig(evaluations=["exact_match"]),
    )
    assert passing.passed is True
    assert passing.decisions[0].metric == "exact_match"
    assert passing.decisions[0].score == 1.0

    failing = runner.run(
        _request("exact_match", output="Lyon", expected="Paris"),
        EvaluationConfig(evaluations=["exact_match"]),
    )
    assert failing.passed is False
    assert failing.decisions[0].score == 0.0
