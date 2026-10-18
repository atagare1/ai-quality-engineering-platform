"""PV-D4: an evaluator RuntimeError propagates and does not record a run.

The failing evaluator exists only in this test. The runner, registry, and gate are real.
"""

import pytest

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner


class _FailingEvaluator:
    def evaluate(self, *args, **kwargs):
        raise RuntimeError("synthetic evaluator failure")


def test_pv_d4_evaluator_runtime_failure_does_not_record_run():
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="synthetic",
            evaluator="synthetic",
            category="test",
        )
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"synthetic": _FailingEvaluator()},
        policies={
            "synthetic": QualityPolicy(
                metric="synthetic",
                operator=">=",
                threshold=1.0,
            )
        },
        gate=QualityGate(),
    )
    request = {"synthetic": {"args": ["4", "4"], "kwargs": {}}}
    config = EvaluationConfig(evaluations=["synthetic"])

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        runner.run(request, config)

    assert runner.last_run is None
