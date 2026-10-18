"""P2-14 Thin Evaluation Runner.

Coordinates frozen evaluation components. Does not evaluate, normalize,
apply thresholds, or decide the gate outcome itself.

A request is a capability-keyed map:

    {"capability_name": {"args": [...], "kwargs": {...}}}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.run import EvaluationRun, TraceEvaluation
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.normalization.result_normalizer import normalize_many
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy


def _as_request_list(requests: Any) -> list[Mapping[str, Any]]:
    if not isinstance(requests, list):
        raise TypeError(
            "EvaluationRunner.run_many() requires a list of request maps, "
            f"got {type(requests).__name__}"
        )
    if not requests:
        raise ValueError("EvaluationRunner.run_many() requires at least one request")
    materialized: list[Mapping[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping) or isinstance(request, (str, bytes)):
            raise TypeError(
                "EvaluationRunner.run_many() requires request map items, "
                f"got {type(request).__name__} at index {index}"
            )
        materialized.append(request)
    return materialized


def _call_payload(request: Mapping[str, Any], capability_name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if capability_name not in request:
        raise KeyError(
            f"No request payload for capability {capability_name!r}"
        )
    payload = request[capability_name]
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        raise TypeError(
            "EvaluationRunner request payload must be a mapping with args/kwargs, "
            f"got {type(payload).__name__} for capability {capability_name!r}"
        )
    args = payload.get("args", ())
    kwargs = payload.get("kwargs", {})
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise TypeError(
            "EvaluationRunner request args must be a sequence, "
            f"got {type(args).__name__} for capability {capability_name!r}"
        )
    if not isinstance(kwargs, Mapping) or isinstance(kwargs, (str, bytes)):
        raise TypeError(
            "EvaluationRunner request kwargs must be a mapping, "
            f"got {type(kwargs).__name__} for capability {capability_name!r}"
        )
    return tuple(args), dict(kwargs)


class EvaluationRunner:
    """Synchronous orchestrator.

    run() evaluates one request map. run_many() evaluates several request
    maps in order and records one EvaluationRun. Dependencies are injected.
    """

    def __init__(
        self,
        registry: EvaluationRegistry,
        evaluators: Mapping[str, Any],
        policies: Mapping[str, QualityPolicy],
        normalizer: Any | None = None,
        gate: QualityGate | None = None,
    ) -> None:
        self._registry = registry
        self._evaluators = dict(evaluators)
        self._policies = dict(policies)
        self._normalizer = normalizer
        self._gate = gate if gate is not None else QualityGate()
        self.last_run: EvaluationRun | None = None

    def run(
        self,
        request: Mapping[str, Any],
        configuration: EvaluationConfig,
        *,
        run_id: str = "",
    ) -> GateDecision:
        return self.run_many([request], configuration, run_id=run_id)

    def run_many(
        self,
        requests: list[Mapping[str, Any]],
        configuration: EvaluationConfig,
        *,
        run_id: str = "",
    ) -> GateDecision:
        subjects = _as_request_list(requests)

        self.last_run = None
        if not configuration.evaluations:
            raise ValueError(
                "EvaluationRunner requires at least one configured evaluation"
            )
        groups: list[TraceEvaluation] = []
        flat_results: list[EvaluationResult] = []
        flat_decisions: list[PolicyDecision] = []
        for request in subjects:
            normalized, decisions = self._evaluate_request(request, configuration)
            groups.append(TraceEvaluation(results=normalized, decisions=decisions))
            flat_results.extend(normalized)
            flat_decisions.extend(decisions)

        gate_decision = self._gate.evaluate(flat_decisions)
        self.last_run = EvaluationRun(
            run_id=run_id,
            requests=list(subjects),
            results=flat_results,
            decisions=flat_decisions,
            gate_decision=gate_decision,
            trace_evaluations=groups,
        )
        return gate_decision

    def _evaluate_request(
        self,
        request: Mapping[str, Any],
        configuration: EvaluationConfig,
    ) -> tuple[list[EvaluationResult], list[PolicyDecision]]:
        collected: list[EvaluationResult] = []
        for capability_name in configuration.evaluations:
            self._registry.get(capability_name)
            evaluator = self._resolve_evaluator(capability_name)
            args, kwargs = _call_payload(request, capability_name)
            collected.extend(evaluator.evaluate(*args, **kwargs))
        normalized = self._normalize(collected)
        decisions = [
            self._resolve_policy(result.metric).apply(result) for result in normalized
        ]
        return normalized, decisions

    def _resolve_evaluator(self, capability_name: str) -> Any:
        try:
            return self._evaluators[capability_name]
        except KeyError:
            raise KeyError(
                f"No evaluator instance wired for capability {capability_name!r}"
            ) from None

    def _resolve_policy(self, metric: str) -> QualityPolicy:
        try:
            return self._policies[metric]
        except KeyError:
            raise KeyError(
                f"No quality policy configured for metric {metric!r}"
            ) from None

    def _normalize(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        if self._normalizer is None:
            return normalize_many(results)
        return self._normalizer.normalize_many(results)
