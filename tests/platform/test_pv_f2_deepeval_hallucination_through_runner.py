"""PV-F2: one live DeepEval hallucination result through the real EvaluationRunner.

Uses the known-good Paris trace from the hallucination smoke and the same
OpenRouter LocalModel. DeepEval 4.2.6 scores alignment, so higher is better.
The policy is hallucination >= metric_threshold("hallucination"), default 0.8.
DeepEval's own threshold is not the platform gate.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval import DeepEvalHallucinationEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from utils import metric_threshold

@pytest.mark.live
def test_pv_f2_deepeval_hallucination_through_runner():

    request = {
        "hallucination": {
            "args": [
                "What is the capital of France?",
                "Paris.",
                ["Paris is the capital and largest city of France."],
            ],
            "kwargs": {},
        }
    }
    model = live_deepeval_local_model()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="hallucination",
            evaluator="deepeval",
            category="rag",
        )
    )
    policy = QualityPolicy(
        metric="hallucination",
        operator=">=",
        threshold=metric_threshold("hallucination"),
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "hallucination": DeepEvalHallucinationEvaluator(model=model)
        },
        policies={"hallucination": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["hallucination"])
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)
    print("policy_operator", policy.operator)

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "hallucination"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= policy.threshold
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].operator == ">="
    assert runner.last_run.decisions[0].threshold == policy.threshold
    print("hallucination_score", result.score)
    print("hallucination_reason", result.reason)
    print("gate_passed", decision.passed)
