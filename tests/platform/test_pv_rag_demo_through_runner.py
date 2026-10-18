"""Live RAG demo SUT through EvaluationRunner.

Calls the external RAG API via live_rag_demo_request, then evaluates
faithfulness through the real Runner. Does not use Phase 1 root tests or
change their fixtures/thresholds.
"""

import math

import pytest
from ragas.llms import LangchainLLMWrapper

from ai_qe_eval.capture.rag_demo import live_rag_demo_request
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.ragas import RAGASFaithfulnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from ragas_live import live_ragas_llama_chat, ragas_llama_judge_model_name
from utils import metric_threshold, read_test_data


@pytest.mark.live
def test_pv_rag_demo_faithfulness_through_runner():
    payload = read_test_data("rag_test_data_faithfulness.json")
    request = live_rag_demo_request(
        payload["question"],
        reference=payload.get("reference"),
        evaluations=["faithfulness"],
    )
    faithfulness_args = request["faithfulness"]["args"]
    assert faithfulness_args[0] == payload["question"]
    assert isinstance(faithfulness_args[1], str) and faithfulness_args[1].strip()
    assert isinstance(faithfulness_args[2], list) and faithfulness_args[2]
    assert all(isinstance(item, str) and item.strip() for item in faithfulness_args[2])
    print("sut_answer", faithfulness_args[1])
    print("retrieved_context_count", len(faithfulness_args[2]))

    llm = live_ragas_llama_chat()
    wrapper = LangchainLLMWrapper(llm)
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="faithfulness",
            evaluator="ragas",
            category="rag",
        )
    )
    policy = QualityPolicy(
        metric="faithfulness",
        operator=">=",
        threshold=metric_threshold("faithfulness"),
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={"faithfulness": RAGASFaithfulnessEvaluator(llm=wrapper)},
        policies={"faithfulness": policy},
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["faithfulness"])
    print("provider_model", ragas_llama_judge_model_name())
    print("policy_threshold", policy.threshold)

    decision = runner.run(request, config, run_id="pv-rag-demo-faithfulness")

    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert runner.last_run.requests == [request]
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "faithfulness"
    assert result.evaluator == "ragas"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].metric == "faithfulness"
    assert runner.last_run.decisions[0].threshold == policy.threshold
    print("faithfulness_score", result.score)
    print("gate_passed", decision.passed)
