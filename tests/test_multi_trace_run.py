"""P3-03 multi-request EvaluationRun.

run() remains one request map. run_many() keeps per-request results and decisions.
No live providers. No score aggregation.
"""

from __future__ import annotations

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.run import EvaluationRun
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def _request(subject_id: str, output: str) -> dict:
    return {
        "correctness": {"args": [subject_id, output], "kwargs": {}},
        "faithfulness": {"args": [subject_id, output], "kwargs": {}},
    }


class RecordingEvaluator:
    def __init__(self, metric: str, score_for_output: dict[str, float]):
        self.metric = metric
        self.score_for_output = score_for_output
        self.received: list[str] = []

    def evaluate(self, subject_id, output, configuration=None) -> list[EvaluationResult]:
        self.received.append(subject_id)
        return [
            EvaluationResult(
                metric=self.metric,
                evaluator="stub",
                score=self.score_for_output[output],
                reason=subject_id,
            )
        ]


def _runner(evaluators: dict) -> EvaluationRunner:
    registry = EvaluationRegistry()
    policies = {}
    for name in evaluators:
        registry.register(EvaluationCapability(name=name, evaluator="stub", category="test"))
        policies[name] = QualityPolicy(metric=name, operator=">=", threshold=0.8)
    return EvaluationRunner(registry=registry, evaluators=evaluators, policies=policies)


def test_single_request_run_still_returns_gate_decision():
    evaluator = RecordingEvaluator("correctness", {"ok": 0.9})
    runner = _runner({"correctness": evaluator})
    request = _request("t1", "ok")
    decision = runner.run(request, EvaluationConfig(evaluations=["correctness"]), run_id="one")
    assert isinstance(decision, GateDecision)
    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.requests == [request]
    assert len(runner.last_run.trace_evaluations) == 1
    assert runner.last_run.trace_evaluations[0].results[0].score == 0.9


def test_multiple_requests_stay_associated():
    evaluator = RecordingEvaluator("correctness", {"high": 0.95, "low": 0.2})
    runner = _runner({"correctness": evaluator})
    first = _request("trace-a", "high")
    second = _request("trace-b", "low")
    decision = runner.run_many(
        [first, second],
        EvaluationConfig(evaluations=["correctness"]),
        run_id="multi",
    )
    recorded = runner.last_run
    assert isinstance(recorded, EvaluationRun)
    assert recorded.requests[0] is first
    assert recorded.requests[1] is second
    assert evaluator.received == ["trace-a", "trace-b"]
    assert recorded.trace_evaluations[0].results[0].reason == "trace-a"
    assert recorded.trace_evaluations[0].results[0].score == 0.95
    assert recorded.trace_evaluations[1].results[0].reason == "trace-b"
    assert recorded.trace_evaluations[1].results[0].score == 0.2
    assert recorded.trace_evaluations[0].decisions[0].passed is True
    assert recorded.trace_evaluations[1].decisions[0].passed is False
    assert decision.passed is False
    assert decision is recorded.gate_decision


def test_two_requests_times_two_evaluators_do_not_mix():
    faithfulness = RecordingEvaluator("faithfulness", {"a": 0.91, "b": 0.7})
    correctness = RecordingEvaluator("correctness", {"a": 0.88, "b": 0.99})
    runner = _runner({"faithfulness": faithfulness, "correctness": correctness})
    requests = [_request("trace-a", "a"), _request("trace-b", "b")]
    runner.run_many(
        requests,
        EvaluationConfig(evaluations=["faithfulness", "correctness"]),
        run_id="grid",
    )
    recorded = runner.last_run
    assert recorded is not None
    assert len(recorded.results) == 4
    assert [item.reason for item in recorded.trace_evaluations[0].results] == [
        "trace-a",
        "trace-a",
    ]
    assert [item.metric for item in recorded.trace_evaluations[0].results] == [
        "faithfulness",
        "correctness",
    ]
    assert [item.score for item in recorded.trace_evaluations[0].results] == [0.91, 0.88]
    assert [item.metric for item in recorded.trace_evaluations[1].results] == [
        "faithfulness",
        "correctness",
    ]
    assert [item.score for item in recorded.trace_evaluations[1].results] == [0.7, 0.99]
    assert [item.passed for item in recorded.trace_evaluations[0].decisions] == [True, True]
    assert [item.passed for item in recorded.trace_evaluations[1].decisions] == [False, True]
    assert [item.passed for item in recorded.decisions] == [True, True, False, True]


def test_per_request_policy_decisions_match_that_requests_results():
    evaluator = RecordingEvaluator("correctness", {"high": 0.9, "low": 0.1})
    runner = _runner({"correctness": evaluator})
    runner.run_many(
        [_request("trace-a", "high"), _request("trace-b", "low")],
        EvaluationConfig(evaluations=["correctness"]),
        run_id="policy",
    )
    recorded = runner.last_run
    assert recorded is not None
    for group in recorded.trace_evaluations:
        assert len(group.results) == len(group.decisions) == 1
        assert group.decisions[0].metric == group.results[0].metric
        assert group.decisions[0].score == group.results[0].score


def test_empty_request_list_is_rejected():
    runner = _runner({"correctness": RecordingEvaluator("correctness", {})})
    with pytest.raises(ValueError, match="at least one"):
        runner.run_many([], EvaluationConfig(evaluations=["correctness"]))
    assert runner.last_run is None


def test_evaluator_failure_does_not_keep_a_partial_run():
    class Failing:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("boom")

    registry = EvaluationRegistry()
    registry.register(EvaluationCapability(name="correctness", evaluator="stub", category="test"))
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"correctness": Failing()},
        policies={"correctness": QualityPolicy(metric="correctness", operator=">=", threshold=0.8)},
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner.run_many(
            [_request("trace-a", "a"), _request("trace-b", "b")],
            EvaluationConfig(evaluations=["correctness"]),
        )
    assert runner.last_run is None


def test_multi_request_record_round_trip_keeps_groups():
    evaluator = RecordingEvaluator("correctness", {"high": 0.9, "low": 0.1})
    runner = _runner({"correctness": evaluator})
    runner.run_many(
        [_request("trace-a", "high"), _request("trace-b", "low")],
        EvaluationConfig(evaluations=["correctness"]),
        run_id="serial",
    )
    restored = EvaluationRun.from_dict(runner.last_run.to_dict())
    assert [item["correctness"]["args"][0] for item in restored.requests] == [
        "trace-a",
        "trace-b",
    ]
    assert restored.trace_evaluations[0].results[0].score == 0.9
    assert restored.trace_evaluations[1].decisions[0].passed is False
    assert restored.gate_decision is not None
    assert restored.gate_decision.passed is False
