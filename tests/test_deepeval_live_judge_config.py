"""Unit tests for the live DeepEval OpenRouter judge helper.

Does not call a provider.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from deepeval.models.llms.local_model import LocalModel

from deepeval_live import (
    DEFAULT_DEEPEVAL_JUDGE_MODEL,
    deepeval_judge_model_name,
    live_deepeval_local_model,
)

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
AGENT_LIVE_TEST = (
    PLATFORM_ROOT
    / "tests"
    / "platform"
    / "test_pv_langfuse_tool_correctness_through_runner.py"
)
HELPER_SOURCE = PLATFORM_ROOT / "deepeval_live.py"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_default_deepeval_judge_model_is_openrouter_llama(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert DEFAULT_DEEPEVAL_JUDGE_MODEL == "meta-llama/llama-3.3-70b-instruct"
    assert deepeval_judge_model_name() == DEFAULT_DEEPEVAL_JUDGE_MODEL


def test_deepeval_judge_model_env_override(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_JUDGE_MODEL", "  example/model  ")
    assert deepeval_judge_model_name() == "example/model"


def test_live_deepeval_local_model_uses_openrouter_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    model = live_deepeval_local_model()

    assert isinstance(model, LocalModel)
    assert DEFAULT_DEEPEVAL_JUDGE_MODEL in model.get_model_name()


def test_live_deepeval_local_model_skips_without_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(pytest.skip.Exception, match="OPENROUTER_API_KEY"):
        live_deepeval_local_model()


def test_live_deepeval_helper_does_not_map_openrouter_to_openai_api_key():
    text = HELPER_SOURCE.read_text(encoding="utf-8")
    assert 'os.getenv("OPENROUTER_API_KEY")' in text
    assert 'os.getenv("OPENAI_API_KEY")' not in text
    assert "OPENAI_API_KEY" not in text


def test_langfuse_agent_tool_correctness_injects_openrouter_judge():
    source = AGENT_LIVE_TEST.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "deepeval_live":
            imported.update(alias.name for alias in node.names)
    assert "live_deepeval_local_model" in imported

    metric_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node.func) == "ToolCorrectnessMetric"
    ]
    assert len(metric_calls) == 2
    for call in metric_calls:
        model_kw = next(kw for kw in call.keywords if kw.arg == "model")
        assert isinstance(model_kw.value, ast.Call)
        assert _call_name(model_kw.value.func) == "live_deepeval_local_model"
        assert not model_kw.value.args
        assert not model_kw.value.keywords

    assert "model=None" not in source
