"""Confirm the chatbot OpenRouter model is listed and distinct from the judge.

Uses the public OpenRouter models catalog. Does not print secrets or map
OPENROUTER_API_KEY to OPENAI_API_KEY. A free-tier model is not selected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from adapt_langfuse_chatbot_openrouter import (
    DEFAULT_CHAT_OPENROUTER_MODEL,
    DEFAULT_DEEPEVAL_JUDGE_MODEL,
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_model_name() -> str:
    configured = os.getenv("CHAT_MODEL")
    if configured is None or configured.strip() == "":
        return DEFAULT_CHAT_OPENROUTER_MODEL
    return configured.strip()


def judge_model_name() -> str:
    configured = os.getenv("DEEPEVAL_JUDGE_MODEL")
    if configured is None or configured.strip() == "":
        return DEFAULT_DEEPEVAL_JUDGE_MODEL
    return configured.strip()


def assert_chat_model_separate_from_judge(chat_model: str, judge_model: str) -> None:
    if chat_model == judge_model:
        raise ValueError(
            f"CHAT_MODEL {chat_model} must differ from DeepEval judge {judge_model}"
        )


def model_ids_from_catalog(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter models response is not an object")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("OpenRouter models response is missing data")
    ids: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            ids.append(row["id"])
    if not ids:
        raise ValueError("OpenRouter models catalog is empty")
    return ids


def assert_model_listed(chat_model: str, model_ids: list[str]) -> None:
    if chat_model not in model_ids:
        raise ValueError(f"OpenRouter catalog does not list {chat_model}")


def _authorized_request(url: str, api_key: str, body: bytes | None = None) -> object:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
        status = response.status
    if status >= 400:
        raise ValueError(f"OpenRouter returned HTTP {status}")
    return json.loads(raw)


def fetch_catalog(api_key: str) -> object:
    return _authorized_request(OPENROUTER_MODELS_URL, api_key)


def probe_chat_completion(api_key: str, chat_model: str) -> None:
    body = json.dumps(
        {
            "model": chat_model,
            "messages": [{"role": "user", "content": "Reply with the single word pong."}],
            "max_tokens": 8,
        }
    ).encode("utf-8")
    payload = _authorized_request(OPENROUTER_CHAT_URL, api_key, body=body)
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter chat response is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter chat response has no choices")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-completion", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is required to verify the chatbot model")
        return 1

    chat_model = chat_model_name()
    judge_model = judge_model_name()
    try:
        assert_chat_model_separate_from_judge(chat_model, judge_model)
        catalog = fetch_catalog(api_key)
        assert_model_listed(chat_model, model_ids_from_catalog(catalog))
        if args.probe_completion:
            probe_chat_completion(api_key, chat_model)
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"ERROR: OpenRouter chatbot model check failed: {exc}")
        return 1

    print(f"OK: OpenRouter catalog lists chatbot model {chat_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
