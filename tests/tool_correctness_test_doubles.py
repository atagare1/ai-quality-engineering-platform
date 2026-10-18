"""Explicit Tool Correctness doubles for deterministic tests.

Not imported by production code. DeepEval ToolCorrectnessMetric still
requires a model object at construction; these tests supply one that
must never be called.
"""

from __future__ import annotations

from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM


class UnusedToolCorrectnessJudge(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return "unused-tool-correctness-judge"

    def load_model(self):
        return self

    def generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("deterministic Tool Correctness must not call a judge")

    async def a_generate(self, prompt: str, *args, **kwargs):
        raise AssertionError("deterministic Tool Correctness must not call a judge")


def unused_name_order_metric() -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[],
        include_reason=True,
        async_mode=False,
        model=UnusedToolCorrectnessJudge(),
    )


def install_explicit_p0_tool_correctness_stub(monkeypatch, *modules) -> None:
    """Point P0 builders at an explicit unused judge.

    Used where tests call main() and cannot pass model= through argparse.
    """
    from ai_qe_eval.cli import build_p0_runner

    def _build(**kwargs):
        if kwargs.get("tool_correctness_metric") is None and kwargs.get("model") is None:
            kwargs["model"] = UnusedToolCorrectnessJudge()
        return build_p0_runner(**kwargs)

    for module in modules:
        monkeypatch.setattr(module, "build_p0_runner", _build)
