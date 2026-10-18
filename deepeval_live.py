"""Live DeepEval judge construction (OpenRouter).

Test-only helper. Not used by RAGAS, embeddings, or production evaluators.
"""

from __future__ import annotations

import os

import pytest
from deepeval.models.llms.local_model import LocalModel

DEFAULT_DEEPEVAL_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEEPEVAL_JUDGE_MODEL_ENV = "DEEPEVAL_JUDGE_MODEL"


def deepeval_judge_model_name() -> str:
    configured = os.getenv(DEEPEVAL_JUDGE_MODEL_ENV)
    if configured is None or configured.strip() == "":
        return DEFAULT_DEEPEVAL_JUDGE_MODEL
    return configured.strip()


def live_deepeval_local_model() -> LocalModel:
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        pytest.skip("OPENROUTER_API_KEY or OPENAI_BASE_URL is not set")
    return LocalModel(
        model=deepeval_judge_model_name(),
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
