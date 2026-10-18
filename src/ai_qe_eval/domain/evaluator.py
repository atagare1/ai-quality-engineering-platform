"""Evaluator contract — P2-05.

Synchronous, evaluator-agnostic invocation:

    evaluate(*args, **kwargs) -> list[EvaluationResult]

Evidence arguments are evaluator-specific. configuration=None remains an
optional convention; this protocol does not require it. The protocol only
guarantees evaluate(...) -> list[EvaluationResult].

Thresholds, PASS/FAIL, registry, adapters, Runner invocation shape, and
concrete evaluators are deferred.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ai_qe_eval.domain.result import EvaluationResult


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, *args: Any, **kwargs: Any) -> list[EvaluationResult]:
        ...
