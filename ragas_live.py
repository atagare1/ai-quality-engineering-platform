"""Live RAGAS judge construction (test-only).

Provider is explicit via RAGAS_LLM_PROVIDER=openai|openrouter|together.
OpenAI uses OPENAI_API_KEY + api.openai.com.
OpenRouter uses OPENROUTER_API_KEY + openrouter.ai.
Together uses TOGETHER_API_KEY + api.together.xyz.
Credentials are never sent to another provider.
Does not change Phase 1 defaults, embeddings, or production evaluators.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DEFAULT_RAGAS_OPENAI_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_RAGAS_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_RAGAS_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RAGAS_TOGETHER_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_RAGAS_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_RAGAS_LLAMA_JUDGE_MODEL = DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL
PROVIDER_ENV = "RAGAS_LLM_PROVIDER"
SUPPORTED_RAGAS_LLM_PROVIDERS = frozenset({"openai", "openrouter", "together"})
# Official OpenAI gpt-4o-mini supports this. Together Turbo uses the same
# JSON-mode request. OpenRouter Llama 3.3 70B returned finish_reason=error
# and a truncated "{" when it was sent, so OpenRouter stays off.
RAGAS_JUDGE_JSON_RESPONSE_FORMAT = {"type": "json_object"}
_OFFICIAL_OPENAI_HOSTS = frozenset({"api.openai.com"})
_OPENROUTER_HOSTS = frozenset({"openrouter.ai"})
_TOGETHER_HOSTS = frozenset({"api.together.xyz", "api.together.ai"})
_TOGETHER_HOST_MARKERS = ("together.xyz", "together.ai")


def load_ragas_live_dotenv(dotenv_path: str | Path | None = None) -> bool:
    """Load .env without overriding process environment (conftest convention)."""
    if dotenv_path is None:
        return load_dotenv(override=False)
    return load_dotenv(dotenv_path, override=False)


load_ragas_live_dotenv()


def ragas_llm_provider() -> str:
    configured = os.getenv(PROVIDER_ENV)
    if configured is None or configured.strip() == "":
        pytest.fail(
            "RAGAS_LLM_PROVIDER must be set to "
            "'openai', 'openrouter', or 'together'"
        )
    provider = configured.strip().lower()
    if provider not in SUPPORTED_RAGAS_LLM_PROVIDERS:
        pytest.fail(
            f"RAGAS_LLM_PROVIDER {provider!r} is not supported; "
            "use 'openai', 'openrouter', or 'together'"
        )
    return provider


def ragas_llama_judge_model_name() -> str:
    configured = os.getenv("RAGAS_LLM_MODEL")
    if configured is not None and configured.strip() != "":
        return configured.strip()
    provider = ragas_llm_provider()
    if provider == "openrouter":
        return DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL
    if provider == "together":
        return DEFAULT_RAGAS_TOGETHER_JUDGE_MODEL
    return DEFAULT_RAGAS_OPENAI_JUDGE_MODEL


def _base_url_host(base_url: str) -> str:
    return (urlparse(base_url).hostname or "").lower()


def _is_together_host(host: str) -> bool:
    return host in _TOGETHER_HOSTS or any(m in host for m in _TOGETHER_HOST_MARKERS)


def _configured_openai_base_url() -> str | None:
    configured = os.getenv("OPENAI_BASE_URL")
    if configured is None or configured.strip() == "":
        return None
    return configured.strip()


def ragas_live_env_issues() -> list[str]:
    """Return missing or conflicting variable names. Never include values."""
    configured = os.getenv(PROVIDER_ENV)
    if configured is None or configured.strip() == "":
        return [PROVIDER_ENV]
    provider = configured.strip().lower()
    if provider not in SUPPORTED_RAGAS_LLM_PROVIDERS:
        return [PROVIDER_ENV]
    issues: list[str] = []
    if provider == "together":
        together_key = os.getenv("TOGETHER_API_KEY")
        if together_key is None or together_key.strip() == "":
            issues.append("TOGETHER_API_KEY")
        openai_url = _configured_openai_base_url()
        if openai_url is not None:
            host = _base_url_host(openai_url)
            if host in _OFFICIAL_OPENAI_HOSTS or host in _OPENROUTER_HOSTS:
                issues.append("OPENAI_BASE_URL")
            elif not _is_together_host(host):
                issues.append("OPENAI_BASE_URL")
    elif provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key is None or openai_key.strip() == "":
            issues.append("OPENAI_API_KEY")
        openai_url = _configured_openai_base_url()
        if openai_url is not None:
            host = _base_url_host(openai_url)
            if host not in _OFFICIAL_OPENAI_HOSTS:
                issues.append("OPENAI_BASE_URL")
    else:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key is None or openrouter_key.strip() == "":
            issues.append("OPENROUTER_API_KEY")
        openai_url = _configured_openai_base_url()
        if openai_url is not None:
            host = _base_url_host(openai_url)
            if host not in _OPENROUTER_HOSTS:
                issues.append("OPENAI_BASE_URL")
    return issues


def live_ragas_llama_chat() -> ChatOpenAI:
    provider = ragas_llm_provider()
    if provider == "openai":
        return _openai_chat()
    if provider == "together":
        return _together_chat()
    return _openrouter_chat()


def _openai_chat() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or api_key.strip() == "":
        pytest.skip(
            "OPENAI_API_KEY is not set; official OpenAI is required "
            "when RAGAS_LLM_PROVIDER=openai"
        )
    configured = _configured_openai_base_url()
    base_url = configured or DEFAULT_RAGAS_OPENAI_BASE_URL
    host = _base_url_host(base_url)
    if host in _OPENROUTER_HOSTS or any(m in host for m in _TOGETHER_HOST_MARKERS):
        pytest.fail(
            "RAGAS_LLM_PROVIDER=openai requires official OpenAI "
            f"({DEFAULT_RAGAS_OPENAI_BASE_URL}); "
            f"OPENAI_BASE_URL host {host!r} is not compatible"
        )
    if host not in _OFFICIAL_OPENAI_HOSTS:
        pytest.fail(
            "RAGAS_LLM_PROVIDER=openai requires host api.openai.com; "
            f"OPENAI_BASE_URL host {host!r} is not compatible"
        )
    return _judge_chat(
        api_key=api_key.strip(),
        base_url=base_url,
        json_mode=True,
    )


def _openrouter_chat() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None or api_key.strip() == "":
        pytest.skip(
            "OPENROUTER_API_KEY is not set; OpenRouter is required "
            "when RAGAS_LLM_PROVIDER=openrouter"
        )
    configured = _configured_openai_base_url()
    if configured is not None:
        host = _base_url_host(configured)
        if host in _OFFICIAL_OPENAI_HOSTS or any(
            m in host for m in _TOGETHER_HOST_MARKERS
        ):
            pytest.fail(
                "RAGAS_LLM_PROVIDER=openrouter does not use official OpenAI "
                "or Together; unset OPENAI_BASE_URL or set it to "
                f"{DEFAULT_RAGAS_OPENROUTER_BASE_URL}"
            )
        if host not in _OPENROUTER_HOSTS:
            pytest.fail(
                "RAGAS_LLM_PROVIDER=openrouter requires host openrouter.ai; "
                f"OPENAI_BASE_URL host {host!r} is not compatible"
            )
    return _judge_chat(
        api_key=api_key.strip(),
        base_url=DEFAULT_RAGAS_OPENROUTER_BASE_URL,
        json_mode=False,
    )


def _together_chat() -> ChatOpenAI:
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key is None or api_key.strip() == "":
        pytest.skip(
            "TOGETHER_API_KEY is not set; Together is required "
            "when RAGAS_LLM_PROVIDER=together"
        )
    configured = _configured_openai_base_url()
    if configured is not None:
        host = _base_url_host(configured)
        if host in _OFFICIAL_OPENAI_HOSTS or host in _OPENROUTER_HOSTS:
            pytest.fail(
                "RAGAS_LLM_PROVIDER=together does not use official OpenAI "
                "or OpenRouter; unset OPENAI_BASE_URL or set it to "
                f"{DEFAULT_RAGAS_TOGETHER_BASE_URL}"
            )
        if not _is_together_host(host):
            pytest.fail(
                "RAGAS_LLM_PROVIDER=together requires a Together host; "
                f"OPENAI_BASE_URL host {host!r} is not compatible"
            )
    return _judge_chat(
        api_key=api_key.strip(),
        base_url=DEFAULT_RAGAS_TOGETHER_BASE_URL,
        json_mode=True,
    )


def _judge_chat(*, api_key: str, base_url: str, json_mode: bool) -> ChatOpenAI:
    kwargs: dict = {
        "model": ragas_llama_judge_model_name(),
        "temperature": 0,
        "api_key": api_key,
        "base_url": base_url,
    }
    if json_mode:
        kwargs["model_kwargs"] = {
            "response_format": RAGAS_JUDGE_JSON_RESPONSE_FORMAT
        }
    return ChatOpenAI(**kwargs)
