"""Offline checks for the CI-side OpenRouter chat-route adaptation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "adapt_langfuse_chatbot_openrouter.py"
)

OFFICIAL_ROUTE = """import { openai } from "@ai-sdk/openai";
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { isSpanContextValid } from "@opentelemetry/api";
import { convertToModelMessages, generateId, streamText, type UIMessage } from "ai";
import { after } from "next/server";

import { langfuseSpanProcessor } from "@/instrumentation";

export async function POST(req: Request) {
  const {
    messages,
    chatId,
    model = "gpt-4o-mini",
  }: { messages: UIMessage[]; chatId: string; model?: string } = await req.json();

  return startActiveObservation(
    "handle-chat-message",
    async (span) => {
      return propagateAttributes(
        { traceName: "langfuse-chatbot", sessionId: chatId },
        async () => {
          const result = streamText({
            model: openai(model),
            messages: await convertToModelMessages(messages),
          });
          return result.toUIMessageStreamResponse({
            generateMessageId: () => generateId(),
          });
        },
      );
    },
    { endOnExit: false },
  );
}
"""


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "adapt_langfuse_chatbot_openrouter",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapt_replaces_openai_provider_without_mapping_keys():
    module = _load_script()
    adapted = module.adapt_route_source(OFFICIAL_ROUTE)
    assert '@openrouter/ai-sdk-provider' in adapted
    assert "createOpenRouter" in adapted
    assert "process.env.OPENROUTER_API_KEY" in adapted
    assert "openrouter(model)" in adapted
    assert "openai/gpt-4o-mini" in adapted
    assert "@ai-sdk/openai" not in adapted
    assert "openai(model)" not in adapted
    assert "OPENAI_API_KEY" not in adapted
    assert "startActiveObservation" in adapted
    assert "handle-chat-message" in adapted
    assert "toUIMessageStreamResponse" in adapted
    assert "chatId" in adapted
    assert "messages" in adapted


def test_adapt_refuses_missing_official_contract(tmp_path):
    module = _load_script()
    with pytest.raises(ValueError, match="@ai-sdk/openai"):
        module.adapt_route_source("export async function POST() {}")


def test_adapt_writes_route_file(tmp_path):
    module = _load_script()
    route = tmp_path / "route.ts"
    route.write_text(OFFICIAL_ROUTE, encoding="utf-8")
    module.adapt_route_file(route)
    text = route.read_text(encoding="utf-8")
    assert "createOpenRouter" in text
    assert "OPENAI_API_KEY" not in text
