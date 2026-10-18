"""External RAG demo capture.

Calls the existing Phase 1 RAG HTTP client and maps the live answer plus
retrieved contexts into Design A request maps. Does not evaluate metrics
or apply quality gates.

Does not change Phase 1 tests, fixtures, or thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from utils import get_api_response, map_rag_response

# Capability names match currently wired Runner evaluators.
# Args follow those evaluate() signatures.
_ALWAYS_AVAILABLE = (
    "faithfulness",
    "answer_relevancy",
    "contextual_relevancy",
    "hallucination",
)
_REFERENCE_REQUIRED = (
    "contextual_precision",
    "contextual_recall",
    "correctness",
)
SUPPORTED_RAG_DEMO_EVALUATIONS = _ALWAYS_AVAILABLE + _REFERENCE_REQUIRED


def _payload(args: list[Any]) -> dict[str, list[Any]]:
    return {"args": args}


def _require_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("live_rag_demo_request requires a non-empty question")
    return question


def _selected_evaluations(
    evaluations: Sequence[str] | None,
    *,
    reference: Any,
) -> list[str]:
    if evaluations is None:
        selected = list(_ALWAYS_AVAILABLE)
        if reference is not None:
            selected.extend(_REFERENCE_REQUIRED)
        return selected
    if isinstance(evaluations, (str, bytes)) or not isinstance(evaluations, Sequence):
        raise TypeError(
            "evaluations must be a sequence of capability names, "
            f"got {type(evaluations).__name__}"
        )
    selected = list(evaluations)
    unknown = [name for name in selected if name not in SUPPORTED_RAG_DEMO_EVALUATIONS]
    if unknown:
        raise ValueError(
            "Unsupported RAG demo evaluation(s): "
            + ", ".join(repr(name) for name in unknown)
        )
    missing_reference = [
        name for name in selected if name in _REFERENCE_REQUIRED and reference is None
    ]
    if missing_reference:
        raise ValueError(
            "RAG demo evaluations require reference: "
            + ", ".join(repr(name) for name in missing_reference)
        )
    return selected


def rag_demo_request(
    *,
    user_input: Any,
    response: Any,
    retrieved_contexts: list[Any] | None,
    reference: Any | None = None,
    evaluations: Sequence[str] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Build a Design A request map from one RAG demo turn.

    Retrieval is the extracted context list. Expected/reference is a
    caller-supplied gold answer and is never read from the RAG API.
    """
    contexts = list(retrieved_contexts or [])
    selected = _selected_evaluations(evaluations, reference=reference)
    builders: dict[str, dict[str, list[Any]]] = {
        "faithfulness": _payload([user_input, response, contexts]),
        "answer_relevancy": _payload([user_input, response]),
        "contextual_relevancy": _payload([user_input, contexts]),
        "hallucination": _payload([user_input, response, contexts]),
        "contextual_precision": _payload([user_input, reference, contexts]),
        "contextual_recall": _payload([user_input, reference, contexts]),
        "correctness": _payload([user_input, response, reference]),
    }
    return {name: builders[name] for name in selected}


def live_rag_demo_request(
    question: str,
    *,
    reference: Any | None = None,
    evaluations: Sequence[str] | None = None,
    response_data: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Call the external RAG demo (unless response_data is injected) and map it."""
    passed_data = {"question": _require_question(question), "reference": reference}
    data = (
        dict(response_data)
        if response_data is not None
        else get_api_response(passed_data)
    )
    mapped = map_rag_response(passed_data, data)
    return rag_demo_request(
        user_input=mapped["user_input"],
        response=mapped["response"],
        retrieved_contexts=mapped["retrieved_contexts"],
        reference=mapped.get("reference"),
        evaluations=evaluations,
    )
