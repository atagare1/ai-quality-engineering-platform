"""PV-F2: one live DeepEval answer relevancy result through the real EvaluationRunner.

Uses the known-good trace from the DeepEval Answer Relevancy smoke and the same
OpenRouter LocalModel. The policy is the existing answer relevancy rule:
answer_relevancy >= metric_threshold("answer_relevancy"), default 0.8.
DeepEval's own threshold is not the platform gate.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval import DeepEvalAnswerRelevancyEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from utils import metric_threshold

@pytest.mark.live
def test_pv_f2_deepeval_answer_relevancy_through_runner():

    request = {
        "answer_relevancy": {"args": ["What is 2 + 2?", "4"], "kwargs": {}},
    }
    model = live_deepeval_local_model()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="answer_relevancy",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(
        metric="answer_relevancy",
        operator=">=",
        threshold=metric_threshold("answer_relevancy"),
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "answer_relevancy": DeepEvalAnswerRelevancyEvaluator(model=model)
        },
        policies={"answer_relevancy": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["answer_relevancy"])
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "answer_relevancy"
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
    print("answer_relevancy_score", result.score)
    print("answer_relevancy_reason", result.reason)
    print("gate_passed", decision.passed)
