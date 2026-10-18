"""Live DeepEval Hallucination smoke.

Uses the same OpenRouter LocalModel as the other DeepEval smokes. Calls
DeepEvalHallucinationEvaluator directly. Does not apply a quality threshold.

DeepEval 4.2.6 scores alignment: 1 means the output agrees with the context.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.evaluators.deepeval import DeepEvalHallucinationEvaluator

@pytest.mark.live
def test_pv_f_deepeval_hallucination_smoke_returns_numeric_score():

    model = live_deepeval_local_model()
    print("provider_model", deepeval_judge_model_name())

    results = DeepEvalHallucinationEvaluator(model=model).evaluate(
        "What is the capital of France?",
        "Paris.",
        ["Paris is the capital and largest city of France."],
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == "hallucination"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("hallucination_score", result.score)
    print("hallucination_reason", result.reason)
