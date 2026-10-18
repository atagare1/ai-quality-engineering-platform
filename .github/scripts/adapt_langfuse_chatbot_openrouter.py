"""CI-side OpenRouter adaptation for the official user-feedback chat route.

Patches a cloned Langfuse example ``app/api/chat/route.ts`` so inference uses
``@openrouter/ai-sdk-provider`` and ``OPENROUTER_API_KEY``. Does not map that
key to ``OPENAI_API_KEY``. Leaves the /api/chat body, UI stream contract, and
Langfuse tracing calls unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OPENROUTER_AI_SDK_PROVIDER_SPEC = "@openrouter/ai-sdk-provider@3.1.0"
DEFAULT_CHAT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_DEEPEVAL_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"

_OPENAI_IMPORT_RE = re.compile(
    r'import\s+\{\s*openai\s*\}\s+from\s+["\']@ai-sdk/openai["\'];'
)
_OPENAI_MODEL_RE = re.compile(r"openai\(\s*model\s*\)")
_DEFAULT_MODEL_RE = re.compile(r'model\s*=\s*"gpt-4o-mini"')

_REQUIRED_MARKERS = (
    "startActiveObservation",
    "handle-chat-message",
    "toUIMessageStreamResponse",
    "chatId",
    "messages",
)

_OPENROUTER_IMPORT = (
    'import { createOpenRouter } from "@openrouter/ai-sdk-provider";'
)
_OPENROUTER_CLIENT = (
    "const openrouter = createOpenRouter({\n"
    "  apiKey: process.env.OPENROUTER_API_KEY,\n"
    "});"
)


def adapt_route_source(source: str) -> str:
    if "OPENAI_API_KEY" in source:
        raise ValueError("refusing to adapt a route that already mentions OPENAI_API_KEY")
    if _OPENAI_IMPORT_RE.search(source) is None:
        raise ValueError("official @ai-sdk/openai import not found")
    if _OPENAI_MODEL_RE.search(source) is None:
        raise ValueError("official openai(model) call not found")
    missing = [name for name in _REQUIRED_MARKERS if name not in source]
    if missing:
        raise ValueError("official /api/chat or Langfuse markers missing: " + ", ".join(missing))

    adapted = _OPENAI_IMPORT_RE.sub(_OPENROUTER_IMPORT, source, count=1)
    if _OPENROUTER_CLIENT not in adapted:
        adapted = adapted.replace(
            _OPENROUTER_IMPORT + "\n",
            _OPENROUTER_IMPORT + "\n\n" + _OPENROUTER_CLIENT + "\n",
            1,
        )
    adapted = _OPENAI_MODEL_RE.sub("openrouter(model)", adapted)
    adapted = _DEFAULT_MODEL_RE.sub(
        f'model = process.env.CHAT_MODEL || "{DEFAULT_CHAT_OPENROUTER_MODEL}"',
        adapted,
        count=1,
    )

    if "@ai-sdk/openai" in adapted or _OPENAI_MODEL_RE.search(adapted):
        raise ValueError("OpenAI provider usage remained after adaptation")
    if "OPENAI_API_KEY" in adapted:
        raise ValueError("adaptation introduced OPENAI_API_KEY")
    if "createOpenRouter" not in adapted or "OPENROUTER_API_KEY" not in adapted:
        raise ValueError("OpenRouter provider was not wired")
    for name in _REQUIRED_MARKERS:
        if name not in adapted:
            raise ValueError(f"adaptation dropped official marker {name}")
    return adapted


def adapt_route_file(route_path: Path) -> None:
    source = route_path.read_text(encoding="utf-8")
    route_path.write_text(adapt_route_source(source), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route_ts", type=Path)
    args = parser.parse_args()
    if not args.route_ts.is_file():
        print(f"ERROR: chat route not found: {args.route_ts}")
        return 1
    try:
        adapt_route_file(args.route_ts)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot adapt official chat route: {exc}")
        return 1
    print(f"OK: adapted {args.route_ts} to {OPENROUTER_AI_SDK_PROVIDER_SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
