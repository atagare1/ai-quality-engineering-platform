"""PV-C2: two requests, both exact_match PASS, through run_many.

Same wiring as PV-B3. The run gate passes when every policy decision passes.
TraceEvaluation has no passed field; PASS is PolicyDecision.passed.
"""

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_c2_multi_trace_all_pass_through_runner():
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
    request1 = {"exact_match": {"args": ["4", "4"], "kwargs": {}}}
    request2 = {"exact_match": {"args": ["4", "4"], "kwargs": {}}}
    config = EvaluationConfig(evaluations=["exact_match"])

    decision = runner.run_many([request1, request2], config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 2
    assert runner.last_run.results[0].score == 1.0
    assert runner.last_run.results[1].score == 1.0
    assert runner.last_run.trace_evaluations[0].decisions[0].passed is True
    assert runner.last_run.trace_evaluations[1].decisions[0].passed is True
