"""PV-F2: one live DeepEval contextual relevancy result through EvaluationRunner.

Uses the same OpenRouter LocalModel and simple RAG trace as the smoke test.
The policy is contextual_relevancy >= metric_threshold("contextual_relevancy").
DeepEval's own threshold is not the platform gate.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval import DeepEvalContextualRelevancyEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from utils import metric_threshold

@pytest.mark.live
def test_pv_f2_deepeval_contextual_relevancy_through_runner():

    request = {
        "contextual_relevancy": {
            "args": ["What is 2 + 2?", ["2 + 2 = 4"]],
            "kwargs": {},
        }
    }
    model = live_deepeval_local_model()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="contextual_relevancy",
            evaluator="deepeval",
            category="rag",
        )
    )
    policy = QualityPolicy(
        metric="contextual_relevancy",
        operator=">=",
        threshold=metric_threshold("contextual_relevancy"),
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "contextual_relevancy": DeepEvalContextualRelevancyEvaluator(model=model)
        },
        policies={"contextual_relevancy": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["contextual_relevancy"])
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)
    print("policy_threshold_source", "metric_threshold('contextual_relevancy')")

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "contextual_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= policy.threshold
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == policy.threshold
    print("contextual_relevancy_score", result.score)
    print("contextual_relevancy_reason", result.reason)
    print("gate_passed", decision.passed)
