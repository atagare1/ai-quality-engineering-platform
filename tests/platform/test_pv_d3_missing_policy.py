"""PV-D3: a registered capability with no quality policy fails before a run is recorded.

Uses the real EvaluationRunner, registry, and DeterministicEvaluator.
No QualityPolicy is wired for the evaluator metric.
"""

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


def test_pv_d3_missing_policy_raises_and_does_not_record_run():
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
        policies={},
        gate=QualityGate(),
    )
    request = {"exact_match": {"args": ["4", "4"], "kwargs": {}}}
    config = EvaluationConfig(evaluations=["exact_match"])

    with pytest.raises(KeyError, match="No quality policy configured for metric"):
        runner.run(request, config)

    assert runner.last_run is None
