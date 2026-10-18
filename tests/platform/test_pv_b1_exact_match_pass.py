"""PV-B1: one exact_match PASS through the real evaluation runner.

Wires real domain, evaluator, policy, gate, and runner objects.
Does not use evaluator, policy, or gate doubles.
"""

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_b1_deterministic_exact_match_pass_through_runner():
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
        policies={
            "exact_match": QualityPolicy(
                metric="exact_match",
                operator=">=",
                threshold=1.0,
            )
        },
        gate=QualityGate(),
    )
    request = {"exact_match": {"args": ["4", "4"], "kwargs": {}}}
    config = EvaluationConfig(evaluations=["exact_match"])

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.results[0].score == 1.0
    assert runner.last_run.results[0].metric == "exact_match"
