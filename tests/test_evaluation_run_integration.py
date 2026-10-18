"""P3-02 EvaluationRun integration.

Runner still returns GateDecision. It records one EvaluationRun.
No live RAGAS, DeepEval, or Together calls.
"""

from __future__ import annotations

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def _request(*names: str) -> dict:
    return {name: {"args": ["question"], "kwargs": {}} for name in names}


def _result(metric: str, score: float, evaluator: str) -> EvaluationResult:
    return EvaluationResult(
        metric=metric,
        evaluator=evaluator,
        score=score,
        reason="synthetic",
    )


class StubEvaluator:
    def __init__(self, results: list[EvaluationResult], error: Exception | None = None):
        self.results = results
        self.error = error
        self.call_count = 0

    def evaluate(self, *args, **kwargs) -> list[EvaluationResult]:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return list(self.results)


def _registry(*names: str) -> EvaluationRegistry:
    registry = EvaluationRegistry()
    for name in names:
        registry.register(EvaluationCapability(name=name, evaluator="stub", category="test"))
    return registry


def _policies(*metrics: str) -> dict[str, QualityPolicy]:
    return {
        metric: QualityPolicy(metric=metric, operator=">=", threshold=0.8)
        for metric in metrics
    }


def test_runner_records_an_evaluation_run():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": StubEvaluator([_result("correctness", 0.9, "fake")])},
        policies=_policies("correctness"),
    )
    request = _request("correctness")
    decision = runner.run(
        request,
        EvaluationConfig(evaluations=["correctness"]),
        run_id="run-001",
    )
    recorded = runner.last_run
    assert isinstance(recorded, EvaluationRun)
    assert recorded.run_id == "run-001"
    assert recorded.requests == [request]
    assert recorded.gate_decision is decision
    assert decision.passed is True


def test_single_evaluator_result_is_recorded():
    raw = _result("correctness", 0.91, "fake")
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": StubEvaluator([raw])},
        policies=_policies("correctness"),
    )
    runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]), run_id="run-single")
    recorded = runner.last_run
    assert recorded is not None
    assert len(recorded.results) == 1
    assert recorded.results[0] is not raw
    assert recorded.results[0].metric == "correctness"
    assert recorded.results[0].evaluator == "fake"
    assert recorded.results[0].score == 0.91
    assert raw.score == 0.91


def test_multiple_evaluators_share_one_run():
    runner = EvaluationRunner(
        registry=_registry("faithfulness", "correctness", "exact_match"),
        evaluators={
            "faithfulness": StubEvaluator([_result("faithfulness", 0.95, "ragas")]),
            "correctness": StubEvaluator([_result("correctness", 0.88, "deepeval")]),
            "exact_match": StubEvaluator([_result("exact_match", 1.0, "deterministic")]),
        },
        policies=_policies("faithfulness", "correctness", "exact_match"),
    )
    runner.run(
        _request("faithfulness", "correctness", "exact_match"),
        EvaluationConfig(evaluations=["faithfulness", "correctness", "exact_match"]),
        run_id="run-multi",
    )
    recorded = runner.last_run
    assert recorded is not None
    assert len(recorded.requests) == 1
    assert [item.evaluator for item in recorded.results] == [
        "ragas",
        "deepeval",
        "deterministic",
    ]
    assert [item.metric for item in recorded.decisions] == [
        "faithfulness",
        "correctness",
        "exact_match",
    ]


def test_run_keeps_policy_decisions_without_reapplying_policy():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": StubEvaluator([_result("correctness", 0.7, "fake")])},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]), run_id="run-policy")
    recorded = runner.last_run
    assert recorded is not None
    assert len(recorded.decisions) == 1
    decision = recorded.decisions[0]
    assert isinstance(decision, PolicyDecision)
    assert decision.passed is False
    assert decision.score == 0.7
    assert decision.threshold == 0.8
    assert not hasattr(recorded, "apply")


def test_run_exposes_the_gate_decision():
    runner = EvaluationRunner(
        registry=_registry("exact_match", "correctness"),
        evaluators={
            "exact_match": StubEvaluator([_result("exact_match", 1.0, "deterministic")]),
            "correctness": StubEvaluator([_result("correctness", 0.7, "fake")]),
        },
        policies={
            "exact_match": QualityPolicy(metric="exact_match", operator=">=", threshold=1.0),
            "correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8),
        },
    )
    returned = runner.run(
        _request("exact_match", "correctness"),
        EvaluationConfig(evaluations=["exact_match", "correctness"]),
        run_id="run-gate",
    )
    recorded = runner.last_run
    assert recorded is not None
    assert isinstance(recorded.gate_decision, GateDecision)
    assert recorded.gate_decision is returned
    assert recorded.gate_decision.passed is False
    assert recorded.decisions is not recorded.gate_decision.decisions
    assert [item.passed for item in recorded.gate_decision.decisions] == [True, False]


def test_existing_run_return_value_remains_a_gate_decision():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": StubEvaluator([_result("correctness", 0.9, "fake")])},
        policies=_policies("correctness"),
    )
    decision = runner.run(_request("correctness"), EvaluationConfig(evaluations=["correctness"]))
    assert isinstance(decision, GateDecision)
    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.run_id == ""
    assert runner.last_run.gate_decision is decision


def test_evaluator_failure_does_not_record_a_run():
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={
            "correctness": StubEvaluator(
                [_result("correctness", 0.9, "fake")],
                error=RuntimeError("evaluator failed"),
            )
        },
        policies=_policies("correctness"),
    )
    with pytest.raises(RuntimeError, match="evaluator failed"):
        runner.run(
            _request("correctness"),
            EvaluationConfig(evaluations=["correctness"]),
            run_id="run-fail",
        )
    assert runner.last_run is None


def test_recorded_run_round_trip():
    request = _request("correctness")
    runner = EvaluationRunner(
        registry=_registry("correctness"),
        evaluators={"correctness": StubEvaluator([_result("correctness", 0.9, "fake")])},
        policies=_policies("correctness"),
    )
    runner.run(request, EvaluationConfig(evaluations=["correctness"]), run_id="run-serial")
    restored = EvaluationRun.from_dict(runner.last_run.to_dict())
    assert restored.run_id == "run-serial"
    assert restored.requests[0]["correctness"]["args"] == ["question"]
    assert restored.results[0].score == 0.9
    assert restored.decisions[0].passed is True
    assert restored.gate_decision is not None
    assert restored.gate_decision.passed is True
