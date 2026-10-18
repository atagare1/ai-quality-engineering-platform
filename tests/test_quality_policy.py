"""Focused tests for P2-12 Quality Policy.

One EvaluationResult + one QualityPolicy → PolicyDecision.
No live evaluators, RAGAS, or DeepEval.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy
from utils import metric_threshold

_POLICY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "policy"
    / "quality_policy.py"
)


def _result(metric: str, score, evaluator: str = "ragas") -> EvaluationResult:
    return EvaluationResult(metric=metric, evaluator=evaluator, score=score)


def test_policy_construction():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    assert policy.metric == "faithfulness"
    assert policy.operator == ">="
    assert policy.threshold == 0.80


@pytest.mark.parametrize(
    ("operator", "score", "threshold", "expected"),
    [
        (">", 0.91, 0.80, True),
        (">=", 0.80, 0.80, True),
        ("<", 0.70, 0.80, True),
        ("<=", 0.80, 0.80, True),
        ("==", 0.80, 0.80, True),
        ("!=", 0.75, 0.80, True),
        (">", 0.80, 0.80, False),
        ("<", 0.80, 0.80, False),
        ("!=", 0.80, 0.80, False),
    ],
)
def test_supported_operators(operator, score, threshold, expected):
    policy = QualityPolicy(metric="faithfulness", operator=operator, threshold=threshold)
    decision = policy.apply(_result("faithfulness", score))
    assert decision.passed is expected
    assert decision.score == score
    assert decision.threshold == threshold


def test_invalid_operator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        QualityPolicy(metric="faithfulness", operator="approximately", threshold=0.8)


def test_empty_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        QualityPolicy(metric="", operator=">=", threshold=0.8)
    with pytest.raises(ValueError, match="metric"):
        QualityPolicy(metric="   ", operator=">=", threshold=0.8)


def test_non_numeric_threshold_is_rejected():
    with pytest.raises(TypeError, match="threshold"):
        QualityPolicy(metric="faithfulness", operator=">=", threshold="0.80")
    with pytest.raises(TypeError, match="threshold"):
        QualityPolicy(metric="faithfulness", operator=">=", threshold=True)


def test_score_greater_than_or_equal_boundary_passes():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    decision = policy.apply(_result("faithfulness", 0.80))
    assert decision.passed is True
    assert decision.score == 0.80


def test_hallucination_policy_treats_higher_alignment_as_pass(monkeypatch):
    monkeypatch.delenv("RAGAS_THRESHOLD_HALLUCINATION", raising=False)
    threshold = metric_threshold("hallucination")
    policy = QualityPolicy(
        metric="hallucination",
        operator=">=",
        threshold=threshold,
    )
    aligned = policy.apply(_result("hallucination", threshold, evaluator="deepeval"))
    less_aligned = policy.apply(
        _result("hallucination", threshold - 0.01, evaluator="deepeval")
    )
    assert aligned.passed is True
    assert less_aligned.passed is False
    assert less_aligned.score < threshold


def test_contextual_precision_policy_uses_configured_threshold(monkeypatch):
    monkeypatch.delenv("RAGAS_THRESHOLD_CONTEXTUAL_PRECISION", raising=False)
    threshold = metric_threshold("contextual_precision")
    policy = QualityPolicy(
        metric="contextual_precision",
        operator=">=",
        threshold=threshold,
    )
    at_threshold = policy.apply(
        _result("contextual_precision", threshold, evaluator="deepeval")
    )
    below_threshold = policy.apply(
        _result("contextual_precision", threshold - 0.01, evaluator="deepeval")
    )
    assert at_threshold.passed is True
    assert below_threshold.passed is False
    assert below_threshold.score < threshold


def test_contextual_recall_policy_uses_configured_threshold(monkeypatch):
    monkeypatch.delenv("RAGAS_THRESHOLD_CONTEXTUAL_RECALL", raising=False)
    threshold = metric_threshold("contextual_recall")
    policy = QualityPolicy(
        metric="contextual_recall",
        operator=">=",
        threshold=threshold,
    )
    at_threshold = policy.apply(
        _result("contextual_recall", threshold, evaluator="deepeval")
    )
    below_threshold = policy.apply(
        _result("contextual_recall", threshold - 0.01, evaluator="deepeval")
    )
    assert at_threshold.passed is True
    assert below_threshold.passed is False
    assert below_threshold.score < threshold


def test_contextual_relevancy_policy_uses_configured_threshold(monkeypatch):
    monkeypatch.delenv("RAGAS_THRESHOLD_CONTEXTUAL_RELEVANCY", raising=False)
    threshold = metric_threshold("contextual_relevancy")
    policy = QualityPolicy(
        metric="contextual_relevancy",
        operator=">=",
        threshold=threshold,
    )
    at_threshold = policy.apply(
        _result("contextual_relevancy", threshold, evaluator="deepeval")
    )
    below_threshold = policy.apply(
        _result("contextual_relevancy", threshold - 0.01, evaluator="deepeval")
    )
    assert at_threshold.passed is True
    assert at_threshold.threshold == threshold
    assert below_threshold.passed is False
    assert below_threshold.score < threshold


def test_score_below_threshold_fails():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    decision = policy.apply(_result("faithfulness", 0.79))
    assert decision.passed is False
    assert decision.score == 0.79


def test_metric_mismatch_raises():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    with pytest.raises(ValueError, match="does not match"):
        policy.apply(_result("correctness", 0.91, evaluator="deepeval"))


def test_same_metric_from_different_evaluator_is_allowed():
    policy = QualityPolicy(metric="correctness", operator=">=", threshold=0.80)
    decision = policy.apply(_result("correctness", 0.85, evaluator="deepeval"))
    assert decision.passed is True
    assert decision.metric == "correctness"


def test_score_is_not_modified_or_rounded():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    result = _result("faithfulness", 0.799)
    decision = policy.apply(result)
    assert result.score == 0.799
    assert decision.score == 0.799
    assert decision.score != 0.80
    assert decision.passed is False


def test_invalid_score_type_is_rejected():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    with pytest.raises(TypeError, match="score"):
        policy.apply(_result("faithfulness", "0.85"))
    with pytest.raises(TypeError, match="score"):
        policy.apply(_result("faithfulness", True))


def test_policy_is_deterministic():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    result = _result("faithfulness", 0.91)
    first = policy.apply(result)
    second = policy.apply(result)
    assert first == second


def test_policy_decision_contains_pass_fail_not_on_evaluation_result():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80)
    result = _result("faithfulness", 0.91)
    decision = policy.apply(result)
    assert isinstance(decision, PolicyDecision)
    assert decision.passed is True
    assert "passed" in decision.__dataclass_fields__
    assert "threshold" in decision.__dataclass_fields__
    assert "passed" not in result.__dataclass_fields__
    assert "threshold" not in result.__dataclass_fields__
    assert "severity" not in result.__dataclass_fields__


def test_serialization_round_trip():
    policy = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.8)
    restored_policy = QualityPolicy.from_dict(policy.to_dict())
    assert restored_policy == policy
    decision = policy.apply(_result("faithfulness", 0.91))
    restored_decision = PolicyDecision.from_dict(decision.to_dict())
    assert restored_decision.metric == "faithfulness"
    assert restored_decision.score == 0.91
    assert restored_decision.operator == ">="
    assert restored_decision.threshold == 0.8
    assert restored_decision.passed is True


def test_cross_evaluator_family_examples():
    exact = QualityPolicy(metric="exact_match", operator=">=", threshold=1.0).apply(
        EvaluationResult(metric="exact_match", evaluator="deterministic", score=1.0)
    )
    faithfulness = QualityPolicy(metric="faithfulness", operator=">=", threshold=0.80).apply(
        EvaluationResult(metric="faithfulness", evaluator="ragas", score=0.91)
    )
    correctness = QualityPolicy(metric="correctness", operator=">=", threshold=0.80).apply(
        EvaluationResult(metric="correctness", evaluator="deepeval", score=0.75)
    )
    assert exact.passed is True
    assert faithfulness.passed is True
    assert correctness.passed is False


def test_policy_module_has_no_vendor_or_evaluator_imports():
    tree = ast.parse(_POLICY_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
            imported_modules.add(node.module)
    forbidden = {"ragas", "deepeval", "langchain", "openai", "pytest"}
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.evaluators" not in imported_modules
    assert "ai_qe_eval.normalization" not in imported_modules
