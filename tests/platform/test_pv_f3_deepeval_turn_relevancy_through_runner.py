"""PV-F3: one live turn relevancy result through the real EvaluationRunner.

Uses the same OpenRouter LocalModel as the turn relevancy smoke. The platform
gate is an explicit QualityPolicy of turn_relevancy >= 0.80, matching the
numeric convention used for correctness and answer relevancy. DeepEval's own
threshold is not the gate. This key is not added to the historical RAG
threshold map.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

TURN_RELEVANCY_THRESHOLD = 0.80

@pytest.mark.live
def test_pv_f3_deepeval_turn_relevancy_through_runner():

    turns = [
        ConversationTurn(role="user", content="What is 2 + 2?"),
        ConversationTurn(role="assistant", content="2 + 2 is 4."),
        ConversationTurn(role="user", content="What is 3 + 3?"),
        ConversationTurn(role="assistant", content="3 + 3 is 6."),
    ]
    request = {
        "turn_relevancy": {
            "args": [turns],
            "kwargs": {},
        }
    }
    model = live_deepeval_local_model()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="turn_relevancy",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(
        metric="turn_relevancy",
        operator=">=",
        threshold=TURN_RELEVANCY_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "turn_relevancy": DeepEvalTurnRelevancyEvaluator(model=model)
        },
        policies={"turn_relevancy": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["turn_relevancy"])
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)

    decision = runner.run(request, config, run_id="pv-f3-turn-relevancy")

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TURN_RELEVANCY_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == TURN_RELEVANCY_THRESHOLD
    print("turn_relevancy_score", result.score)
    print("turn_relevancy_reason", result.reason)
    print("gate_passed", decision.passed)
