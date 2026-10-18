"""Evaluator-agnostic evaluation domain.

P2-02: dictionary trace events. P2-03: EvaluationRun.
P2-04: EvaluationResult. P2-05: Evaluator contract. P2-06: Evaluation Registry.
P2-07: EvaluationConfig. P2-08: DeterministicEvaluator (exact_match).
P2-09: RAGASFaithfulnessEvaluator. P2-10: DeepEvalGEvalCorrectnessEvaluator. P2-11: Result Normalizer. P2-12: Quality Policy. P2-13: Quality Gate. P2-14: Thin Evaluation Runner.
Downstream Phase 2 packages are not created here.
"""

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn, ToolInvocation
from ai_qe_eval.domain.evaluator import Evaluator
from ai_qe_eval.domain.events import EVENT_TYPE_KEY, TraceEvent, make_trace_event
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.domain.result import EvaluationResult
from ai_qe_eval.domain.run import EvaluationRun, TraceEvaluation
from ai_qe_eval.evaluators.deepeval import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalContextualPrecisionEvaluator,
    DeepEvalContextualRecallEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalHallucinationEvaluator,
    DeepEvalGEvalCorrectnessEvaluator,
)
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    DeepEvalToolCorrectnessEvaluator,
)
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator
from ai_qe_eval.evaluators.deterministic import (
    DeterministicEvaluator,
    FinalStateEvaluator,
    MCPExecutionHealthEvaluator,
)
from ai_qe_eval.evaluators.ragas import RAGASFaithfulnessEvaluator
from ai_qe_eval.normalization.result_normalizer import normalize, normalize_many
from ai_qe_eval.gate.quality_gate import GateDecision, QualityGate
from ai_qe_eval.policy.quality_policy import PolicyDecision, QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

__all__ = [
    "ConversationTurn",
    "DeepEvalAnswerRelevancyEvaluator",
    "DeepEvalContextualPrecisionEvaluator",
    "DeepEvalContextualRecallEvaluator",
    "DeepEvalContextualRelevancyEvaluator",
    "DeepEvalFaithfulnessEvaluator",
    "DeepEvalHallucinationEvaluator",
    "DeepEvalGEvalCorrectnessEvaluator",
    "DeepEvalToolCorrectnessEvaluator",
    "DeepEvalTurnRelevancyEvaluator",
    "DeterministicEvaluator",
    "FinalStateEvaluator",
    "EVENT_TYPE_KEY",
    "RAGASFaithfulnessEvaluator",
    "EvaluationCapability",
    "EvaluationConfig",
    "EvaluationRegistry",
    "EvaluationResult",
    "EvaluationRun",
    "TraceEvaluation",
    "EvaluationRunner",
    "Evaluator",
    "GateDecision",
    "MCPExecutionHealthEvaluator",
    "normalize",
    "normalize_many",
    "PolicyDecision",
    "QualityGate",
    "QualityPolicy",
    "ToolInvocation",
    "TraceEvent",
    "make_trace_event",
]