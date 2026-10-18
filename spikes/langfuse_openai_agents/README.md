# Langfuse + OpenAI Agents SDK ingestion spike

Standalone spike. It is **not** part of `src/ai_qe_eval`. It does not build a Langfuse adapter.

Goal: run a real OpenAI Agents SDK agent with a real HTTP tool, ingest the run into Langfuse via the official OpenInference instrumentor, then retrieve the actual observations and inspect whether they contain enough fields to construct `ToolInvocation(name, arguments, result)` and `EvaluationTrace`.

Official example this follows:

- https://langfuse.com/integrations/frameworks/openai-agents

Current retrieval API (Langfuse Python SDK v4 / Observations API v2):

- `langfuse.api.observations.get_many(...)`

Do **not** use deprecated `langfuse.trace()` or `langfuse.api.trace.list()`.

## Prerequisites

Real credentials in the process environment, a local `.env` next to this README, or the parent project `.env` at `llm-rag-evaluation-ragas/.env` (never commit secrets). The spike loads the local file first, then fills unset keys from the parent file.

| Variable | Purpose |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `LANGFUSE_BASE_URL` | e.g. `https://cloud.langfuse.com` |
| `OPENROUTER_API_KEY` | OpenRouter API key used for the real agent run |

This spike forces:

- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_MODEL=openrouter/free`

Together `OPENAI_BASE_URL` and `RAGAS_LLM_MODEL` are ignored. Langfuse Cloud remains the trace backend.

The OpenAI Agents SDK talks to OpenRouter through `OpenAIChatCompletionsModel` plus the official OpenInference Langfuse instrumentor.

## Install (spike-only venv)

Do not merge these packages into production `requirements.txt`.

```powershell
cd llm-rag-evaluation-ragas\llm-rag-evaluation-ragas\spikes\langfuse_openai_agents
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --default-timeout=120 -r requirements.txt
```

On Windows, a long OneDrive path can fail with `WinError 206` (filename too long). If that happens, use a short venv path:

```powershell
python -m venv C:\Users\aarya\.venvs\lf-oa-spike
C:\Users\aarya\.venvs\lf-oa-spike\Scripts\python.exe -m pip install --default-timeout=120 -r requirements.txt
```

## Run

```powershell
cd llm-rag-evaluation-ragas\llm-rag-evaluation-ragas\spikes\langfuse_openai_agents
C:\Users\aarya\.venvs\lf-oa-spike\Scripts\python.exe run_spike.py
```

The script:

1. Authenticates to Langfuse with `get_client()` / `auth_check()`.
2. Instruments the official OpenAI Agents SDK with `OpenAIAgentsInstrumentor().instrument()`.
3. Runs one agent that must call three real HTTP tools in order: `fetch_timezone_clock("UTC")`, `fetch_public_uuid`, `fetch_httpbin_json`.
4. Flushes Langfuse, then **paginates** `api.observations.get_many` (`limit=2`) until the cursor is exhausted, then re-queries for late-arriving rows.
5. Prints a sanitized inspection and writes `artifacts/last_inspection.json` and `artifacts/last_multistep_inspection.json`.

Secrets are never printed.

This spike does **not** construct `ToolInvocation` or `EvaluationTrace`.

## What this spike does not do

- It does not modify `src/ai_qe_eval`, `EvaluationTrace`, evaluators, Runner, or existing tests.
- It does not implement a Langfuse adapter.
- It does not mock HTTP, tool results, or Langfuse payloads.
