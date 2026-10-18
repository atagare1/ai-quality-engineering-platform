"""Live DeepEval Contextual Recall smoke.

Uses the same OpenRouter LocalModel as the other DeepEval smokes. Calls
DeepEvalContextualRecallEvaluator directly. Does not apply a quality threshold.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.evaluators.deepeval import DeepEvalContextualRecallEvaluator

@pytest.mark.live
def test_pv_f_deepeval_contextual_recall_smoke_returns_numeric_score():

    model = live_deepeval_local_model()
    print("provider_model", deepeval_judge_model_name())

    results = DeepEvalContextualRecallEvaluator(model=model).evaluate(
        "What is 2 + 2?",
        "4",
        ["2 + 2 = 4"],
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == "contextual_recall"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("contextual_recall_score", result.score)
    print("contextual_recall_reason", result.reason)
