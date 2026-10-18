# AI Quality Engineering Evaluation Platform

Conventional functional tests answer “did the request succeed?” AI systems fail in other ways. A RAG answer can be unfaithful after retrieval succeeds. An agent can choose the wrong tool. A browser workflow can execute every tool correctly and still reach the wrong application state. The evaluator or judge can fail independently of the system under test.

This repository is a provider-agnostic **evaluation core** for those failures. Capture helpers turn observed evidence into a request map. `EvaluationRunner` executes injected evaluator adapters and normalizes results. `QualityPolicy` applies a threshold to each `EvaluationResult`. `QualityGate` makes a fail-closed decision **for that path**. `EvaluationResult.score` is not pass/fail. There is **no** combined cross-metric or cross-suite score.

Four **representative** SUT paths exercise the same core. They are demonstration targets, not production applications, and they do **not** share the same validation coverage (see [Validation status](#validation-status-and-known-limitations)).

| Representative path | Role |
|---|---|
| External RAG | Live answer + retrieved contexts → faithfulness |
| Langfuse OpenAI Agent | Tool traces → tool correctness and final state |
| Playwright MCP | Serialized tools + snapshot state → tool correctness, execution health, and final state |
| Langfuse chatbot | Conversation turns → turn relevancy and per-turn G-Eval |

The differentiator is isolation: evaluate the SUT separately from the judge, and never average RAG, agent, MCP, and chatbot outcomes into one headline number.

Playwright MCP is the interactive demonstration (scripted CLI and live `demo.web`). The other three representative paths use the same runner through pytest and opt-in live jobs, not CLI scenarios.

Package: [`src/ai_qe_eval`](src/ai_qe_eval). License: [Apache License 2.0](LICENSE). Validated CI Python: **3.11** ([`.github/workflows/tests.yml`](.github/workflows/tests.yml), [`.github/workflows/live-tests.yml`](.github/workflows/live-tests.yml)).

---

## Architecture

SUT-specific capture sits **outside** the core. The core is provider-agnostic: domain, runner, normalizer, policy, and gate do not import RAGAS or DeepEval. Adapters do.

```text
SUT → evidence capture → request map → EvaluationRunner
      → evaluator adapters → EvaluationResult[]
      → normalize_many → QualityPolicy.apply → QualityGate.evaluate (per path)
```

| Representative path | Capture | Evaluated through the runner |
|---|---|---|
| External RAG | `live_rag_demo_request` | Faithfulness (other RAG maps are adapters / unit-tested) |
| Langfuse Agent | `get_many(trace_id)` TOOL rows | Tool correctness; final state. GENERATION G-Eval is unit-tested only |
| Playwright MCP | `mcp_p0_request` | Tool correctness; MCP execution health; final state |
| Langfuse chatbot | `get_many(session_id)` root turns | Turn relevancy; per-turn G-Eval |

`QualityGate` is an in-process any-fail decision for one run. It is not a CI plugin and not a combined score across SUTs.

Diagrams: [overview source](docs/architecture-overview.mmd) · [architecture notes](docs/architecture.md) · [validation strategy](docs/platform-validation-strategy.md)

---

## Quick start and offline validation

Clone the current repository. The first reproducible walkthrough is the **offline** suite. `pytest.ini` applies `-m "not live"` and `pythonpath = src`. Push/PR CI runs `pytest tests -q` only. Offline tests do not require Together, OpenRouter, or Langfuse credentials and do **not** establish semantic quality of model judgments.

```bash
git clone https://github.com/atagare1/ai-quality-engineering-platform.git
cd ai-quality-engineering-platform
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

pytest tests -q
pytest tests/test_cli.py tests/test_cli_mcp_p0_demo.py -q
```

The CLI pytest files inject a Tool Correctness stub and assert the scripted MCP report through `EvaluationRunner`. They are the verified offline demo of “tools pass, final state fails.”

`python -m pytest` also collects unmarked Phase 1 root tests. Those are not `EvaluationRunner` tests and are not the offline walkthrough.

---

## Playwright MCP demonstration

Goal: add `Buy milk` on [Playwright TodoMVC](https://demo.playwright.dev/todomvc).

**Scripted CLI** (`python -m ai_qe_eval`) uses a predetermined tool session. It does not start `@playwright/mcp` or a browser. `cli.main()` does not inject the pytest stub. Omitting `model` uses DeepEval’s default provider. Do not treat the CLI as credential-free or as live browser execution.

**Live browser demo** (`python -m ai_qe_eval.demo.web`) drives real Playwright MCP stdio against TodoMVC and needs Node/`npx`.

**Live GitHub Actions / `pytest -m live`** start the real MCP server. They need provider credentials and external services.

Scripted tool order (also the QE-supplied expected names):

`browser_navigate` → `browser_snapshot` → `browser_click` → `browser_type` → `browser_click` → `browser_snapshot`

Final State checks whether the last snapshot contains a list item for `Buy milk`. The fail scenario keeps the same six tools and `isError=false` results, types `Buy bread`, and the gate fails on Final State only.

```text
PASS — tools, health, and expected state
  Tool Correctness 1.00  MCP Execution Health 1.00  Final State 1.00  → Quality Gate PASS

FAIL — same tools, wrong outcome
  Tool Correctness 1.00  MCP Execution Health 1.00  Final State 0.00  → Quality Gate FAIL
```

CLI exit codes, when the CLI is run: QualityGate PASS = `0`, FAIL = `1`. Expected tool names are QE-supplied (`evaluation_params=[]` name/order on the default CLI path). They are not inferred from telemetry.

---

## Implemented evaluators and integrations

Adapters return `EvaluationResult`. They do not apply `QualityPolicy` or `QualityGate`. The default CLI runs only `tool_correctness`, `mcp_execution_health`, and `final_state`.

**Deterministic** (`evaluators/deterministic.py`)

| Evaluator | Metric | Input |
|---|---|---|
| `DeterministicEvaluator` | `exact_match` | `output`, `expected` |
| `MCPExecutionHealthEvaluator` | `mcp_execution_health` | observed `ToolInvocation` results; fails on `isError=true` |
| `FinalStateEvaluator` | `final_state` | caller-supplied `bool` |

**RAGAS** (`evaluators/ragas.py`) — `RAGASFaithfulnessEvaluator` / `faithfulness` only (RAGAS 0.2.15). Other Phase 1 RAGAS metrics are not this adapter.

**DeepEval** (`evaluators/deepeval.py`, `deepeval_tool_correctness.py`, `deepeval_turn_relevancy.py`)

| Evaluator | Metric |
|---|---|
| `DeepEvalGEvalCorrectnessEvaluator` | `correctness` |
| `DeepEvalFaithfulnessEvaluator` | `faithfulness` |
| `DeepEvalAnswerRelevancyEvaluator` | `answer_relevancy` |
| `DeepEvalContextualRelevancyEvaluator` | `contextual_relevancy` |
| `DeepEvalContextualPrecisionEvaluator` | `contextual_precision` |
| `DeepEvalContextualRecallEvaluator` | `contextual_recall` |
| `DeepEvalHallucinationEvaluator` | `hallucination` |
| `DeepEvalToolCorrectnessEvaluator` | `tool_correctness` |
| `DeepEvalTurnRelevancyEvaluator` | `turn_relevancy` |

**Playwright MCP** (`integrations/playwright_mcp.py`, `playwright_mcp_agent.py`, `playwright_mcp_selector.py`) — `@playwright/mcp@0.0.82` over Python `mcp` stdio. Serializes `CallToolResult`. `mcp_p0_request()` builds the P0 map.

**Langfuse** (`integrations/langfuse_observations.py`, `capture/langfuse_trace.py`) — Observations API v2 `get_many` only. The CLI does not call Langfuse.

- Agent (`spikes/langfuse_openai_agents`): `trace_id` + TOOL rows → `langfuse_tool_correctness_request()`. Live Tool Correctness judge is OpenRouter (`live_deepeval_local_model()`). GENERATION G-Eval is unit-tested, not live-validated through Langfuse.
- Chatbot (official Langfuse `applications/user-feedback` example, not in this repo): `session_id` + root `handle-chat-message` spans → `ConversationTurn` maps. Local live tests expect an app at `CHAT_BASE_URL` (default `http://127.0.0.1:3000`). CI clones a pinned example, adapts chat to OpenRouter, starts the app, evaluates, and stops it.

---

## Live GitHub Actions suites

Live evaluation is **manual**. [`.github/workflows/live-tests.yml`](.github/workflows/live-tests.yml) is `workflow_dispatch` only and does not run on push or pull request. Five independent checkboxes default to `false`. Selecting none fails `select-jobs`. An unselected job does **not** run and is **not** a pass.

Jobs do not share a combined score. Each job needs its own secrets and external services.

| Checkbox | Job | Selected tests | Judge / SUT | JUnit minimum |
|---|---|---|---|---|
| `run_ragas` | RAGAS Together | 3 | Together AI judge + external RAG demo | `--min-tests 3` |
| `run_deepeval` | DeepEval OpenRouter | 18 | OpenRouter `LocalModel` judge | `--min-tests 18` |
| `run_playwright_mcp` | Playwright MCP | 6 | Live `@playwright/mcp@0.0.82`; OpenRouter Tool Correctness | `--min-tests 6` |
| `run_langfuse_agent` | Langfuse agent | 5 | Spike traces to Langfuse; OpenRouter Tool Correctness; deterministic Final State | `--min-tests 5` |
| `run_langfuse_chatbot` | Langfuse chatbot | 2 | Official example, CI-adapted to OpenRouter; OpenRouter DeepEval judge | `--min-tests 2` |

Dispatch inventory if every box is checked: **34** selected tests (`3+18+6+5+2`). That is a checkbox inventory, not a claim that the workflow passed.

Chatbot inference (`CHAT_MODEL`, default `openai/gpt-4o-mini`) is separate from the DeepEval judge. CI does **not** map `OPENROUTER_API_KEY` to `OPENAI_API_KEY`.

### Secrets and environment (no values)

Copy [`.env.example`](.env.example) for local placeholders. Never commit `.env`.

| Suite | Repository secrets | Job-set environment (not secrets) |
|---|---|---|
| RAGAS | `TOGETHER_API_KEY` | `RAGAS_LLM_PROVIDER=together`, Together `OPENAI_BASE_URL`. The job sets `OPENAI_API_KEY` from the Together secret for the OpenAI-compatible RAGAS client. That is not an OpenRouter mapping. |
| DeepEval | `OPENROUTER_API_KEY` | `OPENAI_BASE_URL=https://openrouter.ai/api/v1`. Optional `DEEPEVAL_JUDGE_MODEL`. |
| Playwright MCP | `OPENROUTER_API_KEY` | Same OpenRouter base URL. Requires `npx` / Node 20 in CI. |
| Langfuse Agent | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `OPENROUTER_API_KEY` | OpenRouter base URL. CI installs `spikes/langfuse_openai_agents/requirements.txt`. |
| Langfuse Chatbot | same Langfuse + `OPENROUTER_API_KEY` | `CHAT_BASE_URL=http://127.0.0.1:3000`, `CHAT_MODEL=openai/gpt-4o-mini`, pinned `LANGFUSE_EXAMPLES_SHA=77c0d4406ec7cffc211a21d98e2f43e0046a29ae`. The official app starts with `OPENAI_API_KEY` and `OPENAI_BASE_URL` unset. |

### How to run a live suite

1. Open **Actions → live-tests → Run workflow**.
2. Check only the suite(s) to run. Leave the others `false`.
3. Confirm the matching repository secrets exist.
4. Do not treat an unselected job as passed.

Local live tests, when credentials and services are configured:

```bash
pytest -m live --override-ini="addopts=-p no:deepeval"
```

### JUnit safeguards

Each live job writes JUnit XML and runs [`.github/scripts/require_live_junit.py`](.github/scripts/require_live_junit.py) with that suite’s `--min-tests`. The helper fails if the report is unreadable, collected tests are below the minimum, any required test was skipped, or executed tests are below the minimum.

These checks prove required tests **executed**. They do not establish semantic quality of LLM judgments. A real assertion failure (for example chatbot G-Eval below threshold) still fails pytest.

---

## Validation status and known limitations

| Class | What it means |
|---|---|
| Implemented | Core, capture helpers, listed adapters, scripted MCP CLI, `demo.web`, five opt-in live jobs, JUnit collection/skip helper |
| Offline command | `pytest tests -q` is the offline suite (`pytest.ini` applies `-m "not live"`; push/PR CI runs that command). It is not a freshly measured pass count in this README. It covers adapter stubs, CLI pytest with a Tool Correctness stub, and unit-tested Agent GENERATION G-Eval / non-faithfulness RAG maps. It does not prove live semantic quality. |
| Live-validated on `main` | Playwright MCP live job — **user-reported** successful run. No Actions-run URL is in this repository, so none is linked. This is not a claim that the other four live jobs passed on `main`. |
| Historical reported | Other live-suite outcomes below. Not re-executed for this README and **not** claimed for current `main` |
| Not yet validated / known fail | Live Agent GENERATION G-Eval; remaining Phase 1 RAGAS metrics as Phase 2 adapters; historical chatbot G-Eval **0.6 vs 0.8** |

The workflow is configured with a minimum of six Playwright MCP tests (`--min-tests 6`), including expected-negative cases for tool order, final state, and execution errors. This describes the workflow configuration, not verified results from the user-reported `main` run.

**Historical live outcomes (prior runs; not current-`main` evidence)**

| Representative suite | Selected tests | Reported outcome |
|---|---|---|
| RAGAS (Together AI) | 3 | Reported passed |
| DeepEval (OpenRouter) | 18 | Reported passed |
| Langfuse Agent | 5 | Reported passed, including expected-negative tool-order and wrong-expected-title cases |
| Langfuse Chatbot | 2 | Both tests executed (0 skipped). Chatbot HTTP 200. Langfuse observations retrieved. Turn Relevancy passed. G-Eval **failed** at **0.6** against threshold **0.8**. Connectivity and tracing succeeded; the semantic quality gate failed. The threshold was not lowered. Historical only — not a workflow-wide pass and not claimed for current `main`. |

**Historical judge-block (not the Together RAGAS suite).** An earlier OpenRouter RAGAS Faithfulness attempt was **JUDGE_BLOCKED / NOT SCORED** after the RAG API returned HTTP 200. The SUT succeeded; that judge did not. Record that as a judge failure, not FAIL `0.0` on the application.

Limitations: no persistence, lineage, or dashboard; `QualityGate` is not a CI plugin; CLI/web do not run RAG or Langfuse; empty `QualityGate.evaluate([])` still PASSes while `EvaluationRunner` raises if `evaluations` is empty.

---

## Roadmap

**Implemented** — Design A request maps; thin `EvaluationRunner`; registry, config, normalizer, `QualityPolicy`, `QualityGate`; listed adapters; Playwright MCP capture, scripted CLI, and `demo.web`; Langfuse agent and chatbot capture; external RAG capture; five opt-in live jobs with JUnit collection/skip gates; concise CLI report with no score aggregation.

**Not implemented / future** — mapping `GateDecision.passed` into a dedicated CI plugin; persistence or a dashboard; distributed execution; security or adversarial evaluation; CLI coverage beyond the MCP P0 demo; remaining Phase 1 RAGAS metrics as Phase 2 adapters; default-CLI Langfuse or RAG execution; live Langfuse G-Eval on the agent GENERATION path.

---

## Historical RAGAS POC

The original RAGAS proof of concept (Together AI + external RAG demo, root `test_*.py`) is retained for reference. It is **not** the current architecture. Current evaluation is `EvaluationRunner`, evaluator adapters, `QualityPolicy`, and `QualityGate`. See [Architecture](#architecture) and [validation strategy](docs/platform-validation-strategy.md).

---

## Technical reference

Domain, policy, gate, normalizer, and runner do not import RAGAS or DeepEval. Adapters do.

| Component | Module | Responsibility |
|---|---|---|
| Request map | Runner input | `{"name": {"args": [...], "kwargs": {...}}}`. The runner does not rebuild the map. |
| `EvaluationRun` / `TraceEvaluation` | `domain/run.py` | Execution record. No score aggregation. |
| `EvaluationResult` | `domain/result.py` | `metric`, `evaluator`, `score`, optional `reason`. No pass/fail. |
| `Evaluator` | `domain/evaluator.py` | `evaluate(*args, **kwargs) -> list[EvaluationResult]` |
| `EvaluationRegistry` | `domain/registry.py` | Name catalog only. |
| `EvaluationConfig` | `domain/config.py` | Ordered capability names. |
| Result normalizer | `normalization/result_normalizer.py` | Copies results. Does not change scores. |
| `QualityPolicy` | `policy/quality_policy.py` | Threshold operators. `passed` lives here. |
| `QualityGate` | `gate/quality_gate.py` | Any `passed is False` fails that path. Empty decision list passes. |
| `EvaluationRunner` | `runner/evaluation_runner.py` | `run` / `run_many`. Calls injected evaluators, normalizes, applies policies, gates once. |

`EvaluationResult.score` is not pass/fail. `QualityPolicy` is not a release decision. `QualityGate` is not GitHub Actions or a deployment block.

| Suite | Role |
|---|---|
| `tests/test_evaluation_*.py`, `test_evaluator_contract.py` | Domain contracts |
| `tests/test_deterministic_evaluator.py` | Exact match, final state, MCP health |
| `tests/test_ragas_evaluator.py`, `tests/test_deepeval_*.py` | Adapter mapping with stubs. No live LLM. |
| `tests/test_evaluation_runner.py`, `test_quality_policy.py`, `test_quality_gate.py` | Runner, policy, gate |
| `tests/test_cli.py`, `tests/test_cli_mcp_p0_demo.py` | Scripted CLI report through `EvaluationRunner` |
| `tests/platform/test_pv_*.py` | Platform validation, including `@pytest.mark.live` |
| `tests/test_live_tests_workflow.py`, `test_require_live_junit.py`, `test_deepeval_live_judge_config.py` | Offline contracts for live-job checkboxes, JUnit helper, OpenRouter judge wiring |
| Root `test_*.py` | Historical Phase 1 RAGAS. Collected by `python -m pytest`, not by `pytest tests -q` |

End-to-end deterministic tests prove `evaluator` → `normalizer` → `policy` → `gate`. They are not live provider runs.

### Third-party resources

External SUTs, examples, and providers are **not** part of this repository. This project’s [Apache-2.0 license](LICENSE) does not grant rights to third-party material. License and terms checks below are **unresolved** (**UNKNOWN / TO VERIFY**).

| Resource | Where referenced | Role |
|---|---|---|
| Rahul Shetty Academy RAG demo (`https://rahulshettyacademy.com/rag-llm/ask`) | `utils.py` `DEFAULT_RAG_API_URL` | External RAG SUT. Not owned by this project. |
| Playwright TodoMVC (`https://demo.playwright.dev/todomvc`) | `TODO_MVC_URL` | Live/browser MCP target |
| Official Playwright MCP `@playwright/mcp@0.0.82` | `PLAYWRIGHT_MCP_PACKAGE` | MCP server for live tests and `demo.web` |
| Official Langfuse `applications/user-feedback` at SHA `77c0d4406ec7cffc211a21d98e2f43e0046a29ae` | `LANGFUSE_EXAMPLES_SHA` | Chatbot SUT cloned in CI only |
| Together AI, OpenRouter, Langfuse | live job env and `.env.example` | Judge, inference, and tracing hosts |
