"""PV-F4: one live tool correctness result through QualityPolicy and QualityGate.

Uses the same OpenRouter LocalModel as the tool correctness smoke. The platform
gate is an explicit QualityPolicy of tool_correctness >= 0.80. DeepEval's own
threshold is not the gate. tool_correctness is not added to the historical
RAG threshold map. EvaluationRunner is not used because ToolCorrectness no
longer accepts EvaluationTrace.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy

TOOL_CORRECTNESS_THRESHOLD = 0.80

def _weather() -> ToolInvocation:
    return ToolInvocation(
        name="weather",
        arguments={"location": "Pune"},
        result="Temperature is 28 C and conditions are clear.",
    )

@pytest.mark.live
def test_pv_f4_deepeval_tool_correctness_through_runner():

    model = live_deepeval_local_model()
    policy = QualityPolicy(
        metric="tool_correctness",
        operator=">=",
        threshold=TOOL_CORRECTNESS_THRESHOLD,
    )
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)

    result = DeepEvalToolCorrectnessEvaluator(model=model).evaluate(
        [_weather()],
        [_weather()],
        input="Find the weather for Pune.",
    )[0]
    policy_decision = policy.apply(result)
    decision = QualityGate().evaluate([policy_decision])

    assert decision.passed is True
    assert result.metric == "tool_correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    assert policy_decision.passed is True
    assert policy_decision.threshold == TOOL_CORRECTNESS_THRESHOLD
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", decision.passed)
