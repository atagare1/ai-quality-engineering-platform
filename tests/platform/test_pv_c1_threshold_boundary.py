"""PV-C1: score equal to the configured threshold is a policy PASS.

Uses the real QualityPolicy operator already covered by the policy tests:
faithfulness >= 0.80. Equality is included by >=.
"""

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.policy.quality_policy import QualityPolicy


def test_pv_c1_score_equal_to_threshold_is_policy_pass():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    result = EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.80)

    decision = policy.apply(result)

    assert result.score == policy.threshold
    assert decision.passed is True
    assert decision.score == policy.threshold
