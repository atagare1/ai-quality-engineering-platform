"""Focused tests for P2-11 Result Normalizer.

Normalizes EvaluationResult structure only. No live evaluators or vendor SDKs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.evaluators.deterministic import DeterministicEvaluator
from ai_qe_eval.normalization.result_normalizer import normalize, normalize_many

_NORMALIZER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_qe_eval"
    / "normalization"
    / "result_normalizer.py"
)


def _ragas_fixture() -> EvaluationResult:
    return EvaluationResult(
        metric="faithfulness",
        evaluator="ragas",
        score=0.95,
        reason="RAGAS faithfulness score=0.95.",
        raw_result={"score": 0.95, "provider_metric": "faithfulness"},
    )


def _deepeval_fixture() -> EvaluationResult:
    return EvaluationResult(
        metric="correctness",
        evaluator="deepeval",
        score=0.82,
        reason="The actual output is consistent with the expected output.",
        raw_result={
            "score": 0.82,
            "reason": "The actual output is consistent with the expected output.",
            "name": "correctness",
        },
    )


def _deterministic_fixture() -> EvaluationResult:
    return EvaluationResult(
        metric="exact_match",
        evaluator="deterministic",
        score=1.0,
        reason="Exact match succeeded.",
        raw_result={"output": "23", "expected": "23", "matched": True},
    )


def test_canonical_result_is_semantically_unchanged():
    original = EvaluationResult(
        metric="faithfulness",
        evaluator="ragas",
        score=0.91,
        reason="ok",
        raw_result={"score": 0.91},
    )
    normalized = normalize(original)
    assert normalized.metric == original.metric
    assert normalized.evaluator == original.evaluator
    assert normalized.score == original.score
    assert normalized.reason == original.reason
    assert normalized.raw_result == original.raw_result


def test_ragas_style_fixture_is_unchanged():
    original = _ragas_fixture()
    normalized = normalize(original)
    assert normalized.metric == "faithfulness"
    assert normalized.evaluator == "ragas"
    assert normalized.score == 0.95
    assert normalized.raw_result["provider_metric"] == "faithfulness"


def test_deepeval_style_fixture_is_unchanged():
    original = _deepeval_fixture()
    normalized = normalize(original)
    assert normalized.metric == "correctness"
    assert normalized.evaluator == "deepeval"
    assert normalized.score == 0.82
    assert normalized.reason == original.reason
    assert normalized.raw_result["name"] == "correctness"


def test_deterministic_style_fixture_is_unchanged():
    original = _deterministic_fixture()
    normalized = normalize(original)
    assert normalized.metric == "exact_match"
    assert normalized.evaluator == "deterministic"
    assert normalized.score == 1.0
    assert normalized.raw_result["matched"] is True


def test_int_and_float_scores_are_preserved_without_transformation():
    int_result = normalize(EvaluationResult(metric="m", evaluator="e", score=0))
    float_result = normalize(EvaluationResult(metric="m", evaluator="e", score=0.929))
    assert int_result.score == 0
    assert isinstance(int_result.score, int)
    assert float_result.score == 0.929
    assert float_result.score != 0.93
    assert float_result.score != 92.9


def test_reason_and_raw_result_are_preserved():
    original = _deepeval_fixture()
    normalized = normalize(original)
    assert normalized.reason == original.reason
    assert normalized.raw_result == original.raw_result
    assert normalized.raw_result is not original.raw_result


def test_no_policy_fields_or_score_recalculation():
    normalized = normalize(_ragas_fixture())
    assert "threshold" not in normalized.__dataclass_fields__
    assert "passed" not in normalized.__dataclass_fields__
    assert "severity" not in normalized.__dataclass_fields__
    payload = normalized.to_dict()
    assert "threshold" not in payload
    assert "passed" not in payload
    assert normalized.score == 0.95


def test_normalize_many_preserves_count_order_and_meaning():
    originals = [_ragas_fixture(), _deepeval_fixture(), _deterministic_fixture()]
    normalized = normalize_many(originals)
    assert len(normalized) == 3
    assert [item.metric for item in normalized] == [
        "faithfulness",
        "correctness",
        "exact_match",
    ]
    assert [item.evaluator for item in normalized] == [
        "ragas",
        "deepeval",
        "deterministic",
    ]
    assert [item.score for item in normalized] == [0.95, 0.82, 1.0]


def test_normalization_is_idempotent():
    original = _deepeval_fixture()
    once = normalize(original)
    twice = normalize(once)
    assert twice.metric == once.metric
    assert twice.evaluator == once.evaluator
    assert twice.score == once.score
    assert twice.reason == once.reason
    assert twice.raw_result == once.raw_result
    assert twice.raw_result is not once.raw_result


def test_invalid_input_is_rejected():
    with pytest.raises(TypeError, match="EvaluationResult"):
        normalize(None)
    with pytest.raises(TypeError, match="EvaluationResult"):
        normalize({"metric": "faithfulness"})
    with pytest.raises(TypeError, match="list"):
        normalize_many(None)


def test_original_result_and_raw_result_are_not_mutated():
    raw = {"score": 0.95, "extra": {"n": 1}}
    original = EvaluationResult(
        metric="faithfulness",
        evaluator="ragas",
        score=0.95,
        raw_result=raw,
    )
    normalized = normalize(original)
    normalized.raw_result["score"] = 0
    normalized.raw_result["extra"]["n"] = 99
    assert original.raw_result["score"] == 0.95
    assert original.raw_result["extra"]["n"] == 1
    assert raw["score"] == 0.95
    assert raw["extra"]["n"] == 1


def test_evaluator_to_normalizer_contract_does_not_change_meaning():
    before = DeterministicEvaluator().evaluate("23", "23")[0]
    after = normalize(before)
    assert after.metric == before.metric
    assert after.evaluator == before.evaluator
    assert after.score == before.score
    assert after.reason == before.reason
    assert after.raw_result == before.raw_result
    assert after.raw_result is not before.raw_result


def test_normalizer_module_has_no_evaluator_or_vendor_imports():
    tree = ast.parse(_NORMALIZER_SOURCE.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
            imported_modules.add(node.module)
    forbidden = {
        "ragas",
        "deepeval",
        "langchain",
        "langchain_openai",
        "openai",
        "pytest",
    }
    assert forbidden.isdisjoint(imported_roots)
    assert "ai_qe_eval.evaluators" not in imported_modules
    assert "ai_qe_eval.evaluators.ragas" not in imported_modules
    assert "ai_qe_eval.evaluators.deepeval" not in imported_modules
