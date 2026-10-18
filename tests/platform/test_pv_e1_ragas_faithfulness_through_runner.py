"""PV-E1: one live RAGAS faithfulness result through the real EvaluationRunner.

Reuses the Llama smoke judge and Phase 1 faithfulness test data.
The judge is RAGAS_LLM_MODEL or the OpenRouter Llama default.
The policy is the existing experimental faithfulness threshold.
It is not a new production quality bar.
"""

import math

import pytest
from ragas.llms import LangchainLLMWrapper

from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.ragas import RAGASFaithfulnessEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner
from ragas_live import live_ragas_llama_chat, ragas_llama_judge_model_name
from utils import metric_threshold, read_test_data


@pytest.mark.live
@pytest.mark.parametrize(
    "get_test_data",
    [read_test_data("rag_test_data_faithfulness.json")],
    indirect=True,
)
def test_pv_e1_ragas_faithfulness_through_runner(get_test_data):
    sample = get_test_data
    request = {
        "faithfulness": {
            "args": [
                sample.user_input,
                sample.response,
                list(sample.retrieved_contexts or []),
            ],
            "kwargs": {},
        }
    }
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
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "faithfulness": RAGASFaithfulnessEvaluator(llm=wrapper)
        },
        policies={
            "faithfulness": QualityPolicy(
                metric="faithfulness",
                operator=">=",
                threshold=metric_threshold("faithfulness"),
            )
        },
        gate=QualityGate(),
    )
    config = EvaluationConfig(evaluations=["faithfulness"])
    print("provider_model", ragas_llama_judge_model_name())

    decision = runner.run(request, config)

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "faithfulness"
    assert result.evaluator == "ragas"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    print("faithfulness_score", result.score)
