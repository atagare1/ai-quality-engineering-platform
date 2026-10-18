"""PV-F: one live DeepEval turn relevancy call.

Uses the installed DeepEval LocalModel and the shared OpenRouter judge.
Credentials come from OPENROUTER_API_KEY and OPENAI_BASE_URL.
The conversation ends on an assistant turn so DeepEval's unit grouping keeps it.
"""

import math

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model
from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator

@pytest.mark.live
def test_pv_f_deepeval_turn_relevancy_smoke_returns_numeric_score():

    turns = [
        ConversationTurn(role="user", content="What is 2 + 2?"),
        ConversationTurn(role="assistant", content="2 + 2 is 4."),
        ConversationTurn(role="user", content="What is 3 + 3?"),
        ConversationTurn(role="assistant", content="3 + 3 is 6."),
    ]
    model = live_deepeval_local_model()
    print("provider_model", deepeval_judge_model_name())

    results = DeepEvalTurnRelevancyEvaluator(model=model).evaluate(turns)

    assert len(results) == 1
    result = results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.reason is not None
    assert isinstance(result.reason, str) and result.reason.strip() != ""
    print("turn_relevancy_score", result.score)
    print("turn_relevancy_reason", result.reason)
