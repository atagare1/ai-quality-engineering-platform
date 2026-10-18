"""EvaluationRun — P2-03 domain model; P3-02 execution record; P3-03 multi-request.

Container for one logical evaluation execution of one or more request maps.
Does not evaluate, apply thresholds, decide the gate, or aggregate scores.

Association is index-aligned:
    requests[i] <-> trace_evaluations[i]

EvaluationResult has no request id. TraceEvaluation keeps that request's
normalized results and policy decisions together.

results and decisions on the run are the same objects in request order,
then evaluator/result order. They are a flat view, not a second scoring model.

The gate decision is whatever QualityGate returned for that flat decision
list. EvaluationRun does not invent a separate multi-request rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.gate.quality_gate import GateDecision
from ai_qe_eval.policy.quality_policy import PolicyDecision


def _copy_request(request: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name, payload in request.items():
        if isinstance(payload, Mapping) and not isinstance(payload, (str, bytes)):
            copied[str(name)] = {
                "args": list(payload.get("args") or ()),
                "kwargs": dict(payload.get("kwargs") or {}),
            }
        else:
            copied[str(name)] = payload
    return copied


@dataclass
class TraceEvaluation:
    """Normalized results and policy decisions for one request. Not an executor."""

    results: list[EvaluationResult] = field(default_factory=list)
    decisions: list[PolicyDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvaluation:
        raw_results = data.get("results") or []
        raw_decisions = data.get("decisions") or []
        return cls(
            results=[EvaluationResult.from_dict(item) for item in raw_results],
            decisions=[PolicyDecision.from_dict(item) for item in raw_decisions],
        )


@dataclass
class EvaluationRun:
    run_id: str
    requests: list[Mapping[str, Any]] = field(default_factory=list)
    results: list[EvaluationResult] = field(default_factory=list)
    decisions: list[PolicyDecision] = field(default_factory=list)
    gate_decision: GateDecision | None = None
    trace_evaluations: list[TraceEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requests": [_copy_request(request) for request in self.requests],
            "results": [result.to_dict() for result in self.results],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "gate_decision": (
                None if self.gate_decision is None else self.gate_decision.to_dict()
            ),
            "trace_evaluations": [item.to_dict() for item in self.trace_evaluations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        raw_requests = data.get("requests") or []
        raw_results = data.get("results") or []
        raw_decisions = data.get("decisions") or []
        raw_gate = data.get("gate_decision")
        raw_groups = data.get("trace_evaluations") or []
        return cls(
            run_id=data["run_id"],
            requests=[_copy_request(item) for item in raw_requests],
            results=[EvaluationResult.from_dict(item) for item in raw_results],
            decisions=[PolicyDecision.from_dict(item) for item in raw_decisions],
            gate_decision=None if raw_gate is None else GateDecision.from_dict(raw_gate),
            trace_evaluations=[TraceEvaluation.from_dict(item) for item in raw_groups],
        )
