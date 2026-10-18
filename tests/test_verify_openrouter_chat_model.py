"""Offline checks for chatbot OpenRouter model verification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "verify_openrouter_chat_model.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_openrouter_chat_model",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_chat_model_is_listed_and_separate_from_judge(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    module = _load_script()
    assert module.chat_model_name() == "openai/gpt-4o-mini"
    assert module.judge_model_name() == "meta-llama/llama-3.3-70b-instruct"
    module.assert_chat_model_separate_from_judge(
        module.chat_model_name(),
        module.judge_model_name(),
    )


def test_chat_model_must_differ_from_judge(monkeypatch):
    module = _load_script()
    monkeypatch.setenv("CHAT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    with pytest.raises(ValueError, match="must differ"):
        module.assert_chat_model_separate_from_judge(
            module.chat_model_name(),
            module.judge_model_name(),
        )


def test_catalog_requires_configured_model_id():
    module = _load_script()
    ids = module.model_ids_from_catalog(
        {"data": [{"id": "openai/gpt-4o-mini"}, {"id": "other/model"}]}
    )
    module.assert_model_listed("openai/gpt-4o-mini", ids)
    with pytest.raises(ValueError, match="does not list"):
        module.assert_model_listed("missing/model", ids)


def test_catalog_rejects_empty_payload():
    module = _load_script()
    with pytest.raises(ValueError, match="missing data"):
        module.model_ids_from_catalog({})
