"""Focused checks that platform-owned DeepEval defaults are synchronous."""

import importlib

import pytest

from ai_qe_eval.evaluators.deepeval import (
    _default_answer_relevancy_metric,
    _default_contextual_precision_metric,
    _default_contextual_recall_metric,
    _default_contextual_relevancy_metric,
    _default_faithfulness_metric,
    _default_geval_metric,
    _default_hallucination_metric,
)
from ai_qe_eval.evaluators.deepeval_tool_correctness import (
    _default_tool_correctness_metric,
)
from ai_qe_eval.evaluators.deepeval_turn_relevancy import (
    _default_turn_relevancy_metric,
)


DEFAULT_BUILDERS = [
    (
        _default_geval_metric,
        {
            "model": None,
            "criteria": None,
            "evaluation_steps": None,
        },
        "deepeval.metrics.g_eval.g_eval",
        "GEval",
    ),
    (
        _default_faithfulness_metric,
        {"model": None},
        "deepeval.metrics.faithfulness.faithfulness",
        "FaithfulnessMetric",
    ),
    (
        _default_answer_relevancy_metric,
        {"model": None},
        "deepeval.metrics.answer_relevancy.answer_relevancy",
        "AnswerRelevancyMetric",
    ),
    (
        _default_contextual_relevancy_metric,
        {"model": None},
        "deepeval.metrics.contextual_relevancy.contextual_relevancy",
        "ContextualRelevancyMetric",
    ),
    (
        _default_contextual_precision_metric,
        {"model": None},
        "deepeval.metrics.contextual_precision.contextual_precision",
        "ContextualPrecisionMetric",
    ),
    (
        _default_contextual_recall_metric,
        {"model": None},
        "deepeval.metrics.contextual_recall.contextual_recall",
        "ContextualRecallMetric",
    ),
    (
        _default_hallucination_metric,
        {"model": None},
        "deepeval.metrics.hallucination.hallucination",
        "HallucinationMetric",
    ),
    (
        _default_tool_correctness_metric,
        {"model": None, "evaluation_params": []},
        "deepeval.metrics.tool_correctness.tool_correctness",
        "ToolCorrectnessMetric",
    ),
    (
        _default_turn_relevancy_metric,
        {"model": None},
        "deepeval.metrics.turn_relevancy.turn_relevancy",
        "TurnRelevancyMetric",
    ),
]


@pytest.mark.parametrize(
    ("builder", "builder_kwargs", "module_name", "class_name"),
    DEFAULT_BUILDERS,
    ids=[
        "geval",
        "faithfulness",
        "answer-relevancy",
        "contextual-relevancy",
        "contextual-precision",
        "contextual-recall",
        "hallucination",
        "tool-correctness",
        "turn-relevancy",
    ],
)
def test_default_deepeval_metric_construction_disables_async_mode(
    monkeypatch,
    builder,
    builder_kwargs,
    module_name,
    class_name,
):
    captured = {}

    class CapturingMetric:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, class_name, CapturingMetric)

    builder(**builder_kwargs)

    assert captured["async_mode"] is False
