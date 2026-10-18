"""PV-F: one live DeepEval G-Eval correctness call.

Uses the installed DeepEval 4.2.6 LocalModel. The evaluator's model=
argument receives that object. Credentials come from OPENROUTER_API_KEY
and OPENAI_BASE_URL. The judge model is DEEPEVAL_JUDGE_MODEL or the
OpenRouter default.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.evaluators.deepeval import DeepEvalGEvalCorrectnessEvaluator

@pytest.mark.live
def test_pv_f_deepeval_geval_smoke_returns_numeric_score():

    model = live_deepeval_local_model()
    print("provider_model", deepeval_judge_model_name())

    results = DeepEvalGEvalCorrectnessEvaluator(model=model).evaluate(
        "What is 2 + 2?",
        "4",
        "4",
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == "correctness"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("geval_score", result.score)
    print("geval_reason", result.reason)
