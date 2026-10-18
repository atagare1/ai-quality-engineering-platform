"""PV: official Langfuse user-feedback chatbot through Turn Relevancy.

Lifecycle: POST two related turns on one chatId → consume both streams →
FLUSH → retrieve session via the existing adapter → EvaluationRunner →
Turn Relevancy → explicit QualityPolicy >= 0.80 → QualityGate.

External System Under Test: official applications/user-feedback chatbot
already used by spikes/langfuse_user_feedback. This test does not start
that app, parse GENERATION observations, or add retrieval retries.
"""

from __future__ import annotations

import importlib.util
import math
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
from deepeval_live import deepeval_judge_model_name, live_deepeval_local_model

from ai_qe_eval.capture.langfuse_trace import (
    langfuse_chat_turn_relevancy_request,
    observed_chat_roots,
    observed_chat_turns,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ConversationTurn
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_turn_relevancy import DeepEvalTurnRelevancyEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.integrations.langfuse_observations import (
    fetch_session_observation_pages,
    flush_langfuse_client,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
SPIKE_SCRIPT = (
    PLATFORM_ROOT / "spikes" / "langfuse_user_feedback" / "run_spike.py"
)
TURN_RELEVANCY_THRESHOLD = 0.80


def _load_spike():
    spec = importlib.util.spec_from_file_location(
        "langfuse_user_feedback_spike",
        SPIKE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"user-feedback spike not found: {SPIKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_live_env() -> None:
    required = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip("Missing live credentials: " + ", ".join(missing))


def _require_chatbot(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3000
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        pytest.skip(
            "Official user-feedback chatbot is not reachable at "
            f"{base_url}: {exc}"
        )


def _start_key(row: dict) -> str:
    start = row.get("start_time")
    if start is None:
        start = row.get("startTime")
    if hasattr(start, "isoformat"):
        return start.isoformat()
    return str(start or "")


@pytest.mark.live
def test_pv_langfuse_user_feedback_turn_relevancy_through_runner():
    _require_live_env()
    try:
        from langfuse import get_client
    except ImportError:
        pytest.skip("langfuse is not installed in this environment")
    if not SPIKE_SCRIPT.exists():
        pytest.skip(f"user-feedback spike not found: {SPIKE_SCRIPT}")

    spike = _load_spike()
    base_url = os.environ.get("CHAT_BASE_URL", "http://127.0.0.1:3000")
    _require_chatbot(base_url)
    chat_id = os.environ.get("CHAT_SESSION_ID") or uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)

    turn1 = spike._post_chat(
        base_url=base_url,
        chat_id=chat_id,
        messages=[spike._user_message("user-turn-1", spike.TURN1_TEXT)],
    )
    assert turn1.get("http_status") == 200
    assert str(turn1.get("assistant_text") or "").strip()
    print("turn1_status", turn1.get("http_status"))
    print("turn1_assistant_len", len(str(turn1.get("assistant_text") or "")))

    assistant_ids = turn1.get("message_ids") or []
    assistant_id = assistant_ids[-1] if assistant_ids else "assistant-turn-1"
    turn2 = spike._post_chat(
        base_url=base_url,
        chat_id=chat_id,
        messages=[
            spike._user_message("user-turn-1", spike.TURN1_TEXT),
            spike._assistant_message(
                str(assistant_id),
                str(turn1.get("assistant_text") or ""),
            ),
            spike._user_message("user-turn-2", spike.TURN2_TEXT),
        ],
    )
    assert turn2.get("http_status") == 200
    assert str(turn2.get("assistant_text") or "").strip()
    print("turn2_status", turn2.get("http_status"))
    print("turn2_assistant_len", len(str(turn2.get("assistant_text") or "")))

    client = get_client()
    if not client.auth_check():
        pytest.fail("Langfuse auth_check() failed")
    flushed = flush_langfuse_client(client)
    print("lifecycle FLUSH", flushed)
    print("session_id", chat_id)
    time.sleep(30)

    observations = fetch_session_observation_pages(
        client,
        session_id=chat_id,
        from_start_time=started_at - timedelta(minutes=5),
        to_start_time=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    print("observation_count", len(observations))

    roots = observed_chat_roots(observations, session_id=chat_id)
    assert len(roots) == 2
    assert _start_key(roots[0]) <= _start_key(roots[1])
    assert str(roots[0].get("input") or "") == spike.TURN1_TEXT
    assert str(roots[1].get("input") or "") == spike.TURN2_TEXT

    turns = observed_chat_turns(observations, session_id=chat_id)
    assert len(turns) == 4
    assert [turn.role for turn in turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert all(isinstance(turn, ConversationTurn) for turn in turns)
    assert turns[0].content == spike.TURN1_TEXT
    assert turns[2].content == spike.TURN2_TEXT
    assert turns[1].content.strip()
    assert turns[3].content.strip()

    request = langfuse_chat_turn_relevancy_request(
        observations=observations,
        session_id=chat_id,
    )
    assert request["turn_relevancy"]["args"][0] == turns

    model = live_deepeval_local_model()
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="turn_relevancy",
            evaluator="deepeval",
            category="semantic",
        )
    )
    policy = QualityPolicy(
        metric="turn_relevancy",
        operator=">=",
        threshold=TURN_RELEVANCY_THRESHOLD,
    )
    runner = EvaluationRunner(
        registry=registry,
        evaluators={
            "turn_relevancy": DeepEvalTurnRelevancyEvaluator(model=model)
        },
        policies={"turn_relevancy": policy},
        gate=QualityGate(),
    )
    print("provider_model", deepeval_judge_model_name())
    print("policy_threshold", policy.threshold)

    decision = runner.run(
        request,
        EvaluationConfig(evaluations=["turn_relevancy"]),
        run_id="pv-langfuse-user-feedback-turn-relevancy",
    )

    assert decision.passed is True
    assert runner.last_run is not None
    assert runner.last_run.gate_decision is decision
    assert len(runner.last_run.results) == 1
    result = runner.last_run.results[0]
    assert result.metric == "turn_relevancy"
    assert result.evaluator == "deepeval"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= TURN_RELEVANCY_THRESHOLD
    assert len(runner.last_run.decisions) == 1
    assert runner.last_run.decisions[0].passed is True
    assert runner.last_run.decisions[0].threshold == TURN_RELEVANCY_THRESHOLD
    print("turn_relevancy_score", result.score)
    print("turn_relevancy_reason", result.reason)
    print("gate_passed", decision.passed)
