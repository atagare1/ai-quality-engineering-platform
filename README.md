# AI-QE Evaluation Platform

A provider-aware evaluation platform for testing AI/RAG applications, agents, and tool use.

One provider-agnostic evaluation core, four independently gated SUT paths, and five independently selectable GitHub Actions live suites.

External RAG, Langfuse OpenAI Agent, Playwright MCP, and Langfuse user-feedback Chatbot each produce a request map. `EvaluationRunner` executes injected evaluators, `QualityPolicy` applies thresholds, and `QualityGate` decides **per path**. There is no combined platform score, pass rate, or averaged result.

Playwright MCP is the interactive CLI/web flagship demo. External RAG, Langfuse Agent, and the Chatbot are validated through integration and live tests, not CLI scenarios.

Package: [`src/ai_qe_eval`](src/ai_qe_eval). Historical RAGAS POC scripts remain in the repo as a baseline; they are not the foundation for new work.

Technical companion: [Architecture](docs/architecture.md). Validation strategy: [Platform validation strategy](docs/platform-validation-strategy.md). Reported live outcomes in this README supersede older matrix rows in the strategy document.

| Layer | What it is | Status |
|---|---|---|
| Phase 1 | RAGAS POC (Together AI + external RAG demo) | Historical / retained |
| Phase 2 | Provider-agnostic evaluation core | Implemented |
| Phase 3+ | MCP / Langfuse capture, CLI, flagship demo | Implemented for the scope below |

Future work must extend the Phase 2 contracts. Do not grow new framework behavior by editing the original RAGAS pytest scripts.

### Why this project?

AI systems fail in different domains. A RAG answer can be irrelevant or unfaithful even when retrieval returns context. An agent can choose the wrong tool or trajectory. An MCP workflow can execute every tool correctly and still reach the wrong final state. The evaluator or judge can fail independently of the system under test. This platform scores those failures separately so a judge outage is not reported as an application defect.

### What makes this different?

* **Evidence-driven evaluation** — scores come from captured observations (tools, snapshots, retrieved contexts, conversation turns), not from one opaque model score.
* **Provider-agnostic evaluation core** — domain, policy, gate, and runner do not import RAGAS or DeepEval. Adapters do.
* **SUT / evaluator separation** — the system under test is measured independently of the judge that scores it.
* **Fail-closed quality gates** — `QualityGate` fails the run if any policy fails.
* **Independent gates per SUT path** — RAG, Agent, MCP, and Chatbot are never averaged into one platform score.

---

## Architecture

Four SUT paths produce Design A request maps. `EvaluationRunner` is the only executor. `QualityGate` is fail-closed **per path**. Paths are never folded into one platform score.

```text
SUT interaction → trace / evidence collection → evaluator → normalized result → threshold PASS/FAIL

External RAG ──────────────────► live_rag_demo_request ──────────────► faithfulness
Langfuse OpenAI Agent ─────────► get_many(trace_id) / TOOL ──────────► tool_correctness + final_state
Playwright MCP ────────────────► mcp_p0_request ─────────────────────► tool_correctness + health + final_state
Langfuse user-feedback Chatbot ► get_many(session_id) / root SPAN ───► turn_relevancy + per-turn G-Eval
                                         │
                                         ▼
                                 EvaluationRunner
                                         │
                                         ▼
                                 QualityPolicy.apply     (one policy per result metric)
                                         │
                                         ▼
                                 QualityGate.evaluate    (per path; any-fail)
                                         │
                                         ▼
                    CLI / demo.web (MCP flagship)  |  five opt-in GitHub Actions live suites
```

`EvaluationResult.score` is not pass/fail. `QualityPolicy` owns the threshold comparison. `QualityGate` is an in-process any-fail decision for that run, not a CI plugin and not a combined score across SUTs.

MCP is the interactive flagship: `python -m ai_qe_eval` (scripted) and `python -m ai_qe_eval.demo.web` (live browser). RAG, Agent, and Chatbot use the same runner through pytest, not CLI scenarios.

```mermaid
flowchart TD
    RAG[External RAG]
    AG[Langfuse OpenAI Agent]
    MCP[Playwright MCP]
    CH[Langfuse user-feedback Chatbot]
    RAG --> REQ[Request map]
    AG --> REQ
    MCP --> REQ
    CH --> REQ
    REQ --> R[EvaluationRunner]
    C[EvaluationConfig] --> R
    Reg[EvaluationRegistry] --> R
    Inj[Evaluators and QualityPolicy map] --> R
    R --> E[Evaluator.evaluate]
    E --> F[EvaluationResult list]
    F --> N[normalize_many]
    N --> P[QualityPolicy.apply]
    P --> G[QualityGate per path]
    G --> MCPOut[CLI / demo.web MCP flagship]
    G --> Live[five opt-in GitHub Actions live suites]
```

---

## Quick Start

Install from the repository root, then run the scripted MCP demo. The CLI does not start `@playwright/mcp`, a browser, Langfuse, the external RAG API, or the user-feedback chatbot. RAG and Langfuse SUTs are not CLI scenarios.

```bash
git clone https://github.com/atagare1/llm-rag-evaluation-ragas.git
cd llm-rag-evaluation-ragas
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

PYTHONPATH=src python -m ai_qe_eval
PYTHONPATH=src python -m ai_qe_eval --scenario fail
PYTHONPATH=src python -m ai_qe_eval.demo.web
```

`python -m ai_qe_eval` is scripted (no browser). `python -m ai_qe_eval.demo.web` starts a local page that drives **live** Playwright MCP stdio against TodoMVC. Neither command runs the RAG demo or Langfuse.

Exit codes for the CLI: QualityGate PASS = `0`, FAIL = `1`.

### PASS — correct tools, healthy execution, expected final state

```text
AI-QE Evaluation
────────────────────────────────
Scenario: MCP P0 Demo

Evaluator                  Score    Result
Tool Correctness            1.00     PASS
MCP Execution Health        1.00     PASS
Final State                 1.00     PASS

Quality Gate                         PASS
────────────────────────────────
```

### FAIL — same tool sequence, wrong application outcome

The fail demo types `Buy bread` instead of `Buy milk`. Tool Correctness and MCP Execution Health still pass. Final State fails, so the gate fails. The CLI prints the evaluator reason on failing rows only.

```text
AI-QE Evaluation
────────────────────────────────
Scenario: MCP P0 Demo

Evaluator                  Score    Result
Tool Correctness            1.00     PASS
MCP Execution Health        1.00     PASS
Final State                 0.00     FAIL
  Final-state check failed.

Quality Gate                         FAIL
────────────────────────────────
```

---

## TodoMVC demo

Goal: add `Buy milk` on the [Playwright TodoMVC](https://demo.playwright.dev/todomvc) demo.

Scripted tool order (also the QE-supplied expected names):

`browser_navigate` → `browser_snapshot` → `browser_click` → `browser_type` → `browser_click` → `browser_snapshot`

The CLI demo uses `SequenceToolSelector` and a `ScriptedMcpSession` that returns predetermined `CallToolResult` values. `run_playwright_mcp_agent` still captures observations through the real serialize/capture helpers. `mcp_p0_request()` then builds the Design A map for the three P0 evaluators.

Final State is independent of tool names. It checks whether the last `browser_snapshot` contains a list item for `Buy milk`. Correct tool execution therefore does not imply a correct outcome: the fail scenario keeps the same six tools and `isError=false` results, types `Buy bread`, and the gate fails on Final State only.

The default CLI Tool Correctness path uses DeepEval name/order matching (`evaluation_params=[]`). Expected calls are QE-supplied names; they are not inferred from telemetry.

---

## Implemented evaluators

Adapters return `EvaluationResult`. They do not apply `QualityPolicy` or `QualityGate`.

**Deterministic** (`evaluators/deterministic.py`)

| Evaluator | Metric | Input |
|---|---|---|
| `DeterministicEvaluator` | `exact_match` | `output`, `expected` |
| `MCPExecutionHealthEvaluator` | `mcp_execution_health` | observed `ToolInvocation` results; fails on `isError=true` |
| `FinalStateEvaluator` | `final_state` | caller-supplied `bool` |

**RAGAS** (`evaluators/ragas.py`)

| Evaluator | Metric | Notes |
|---|---|---|
| `RAGASFaithfulnessEvaluator` | `faithfulness` | RAGAS 0.2.15 `Faithfulness` only. Other Phase 1 RAGAS metrics are not this adapter. |

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

The default CLI runs only the three P0 evaluators: `tool_correctness`, `mcp_execution_health`, `final_state`.

---

## Integrations

### Playwright MCP

Implemented in `integrations/playwright_mcp.py`, `playwright_mcp_agent.py`, and `playwright_mcp_selector.py`.

* Official package pin: `@playwright/mcp@0.0.82` over Python `mcp` stdio.
* Serializes `CallToolResult` to plain dicts, including snapshot sidecar resolution.
* `run_playwright_mcp_agent` executes an injected selector and captures `ToolInvocation` values.
* `mcp_p0_request()` is the Design A builder for the P0 evaluators.

**Default CLI:** scripted session, no `npx`, no browser. **Live browser demo:** `python -m ai_qe_eval.demo.web`. **Live validation:** `@pytest.mark.live` tests start the real Playwright MCP server against TodoMVC. Those tests are not the CLI path.

### Langfuse

Implemented in `integrations/langfuse_observations.py` and `capture/langfuse_trace.py`. Injected client. Paginates Observations API v2 `get_many`. Does not use `langfuse.trace()` or `api.trace.list`. The CLI does not call Langfuse.

**Agent path** (`spikes/langfuse_openai_agents`): `get_many(trace_id=...)`. `TOOL` observations become `ToolInvocation` values. `langfuse_tool_correctness_request()` builds a Tool Correctness request map. Expected tool calls stay QE-supplied. The live Tool Correctness judge is OpenRouter (`live_deepeval_local_model()`). A GENERATION mapper can build a G-Eval Correctness request (first user content, last assistant text; expected stays QE-supplied). That GENERATION path is unit-tested; it is not live-validated through Langfuse.

**Chatbot path** (official Langfuse `applications/user-feedback` example, not in this repo): `get_many(session_id=..., fields including io)`. Root `handle-chat-message` observations with `is_root_observation=true` are sorted by `start_time`. Non-empty root `input`/`output` strings become `ConversationTurn` pairs, then Turn Relevancy and per-turn G-Eval request maps. Local live tests require the app already listening at `CHAT_BASE_URL` (default `http://127.0.0.1:3000`). The GitHub Actions chatbot job clones a pinned official example, adapts the chat route to OpenRouter, starts the app, evaluates, and stops it.

---

## Live GitHub Actions suites

Live evaluation is **manual only**. `.github/workflows/live-tests.yml` is `workflow_dispatch` and never runs on push or pull request. Five independent boolean checkboxes all default to `false`. Selecting none fails the `select-jobs` gate.

Jobs do not share a combined score. Select only the suite you intend to pay for and observe.

| Checkbox | Job | Selected tests | Judge / SUT provider | JUnit minimum |
|---|---|---|---|---|
| `run_ragas` | RAGAS Together | 3 | Together AI judge (`RAGAS_LLM_PROVIDER=together`, default `meta-llama/Llama-3.3-70B-Instruct-Turbo`) + external RAG demo | `--min-tests 3` |
| `run_deepeval` | DeepEval OpenRouter | 18 | OpenRouter `LocalModel` judge (default `meta-llama/llama-3.3-70b-instruct`) | `--min-tests 18` |
| `run_playwright_mcp` | Playwright MCP | 6 | Live `@playwright/mcp@0.0.82` SUT; OpenRouter-backed Tool Correctness judge | `--min-tests 6` |
| `run_langfuse_agent` | Langfuse agent spike | 5 | Spike SUT traces to Langfuse; OpenRouter-backed DeepEval Tool Correctness judge; deterministic Final State | `--min-tests 5` |
| `run_langfuse_chatbot` | Langfuse chatbot | 2 | Official Langfuse `applications/user-feedback` example, CI-adapted to OpenRouter for inference; OpenRouter DeepEval judge | `--min-tests 2` |

Inventory when every box is checked: **34** selected live tests (`3+18+6+5+2`). That is a dispatch inventory, not a claim that the whole workflow passed.

The chatbot inference model (`CHAT_MODEL`, default `openai/gpt-4o-mini`) is separate from the DeepEval judge. CI does **not** map `OPENROUTER_API_KEY` to `OPENAI_API_KEY`. Phase 1 Mixtral defaults in `utils.py` are unchanged.

### Required secrets and environment (no values)

Repository secrets are never printed. Copy `.env.example` for local placeholders (Together, OpenRouter, Langfuse, `CHAT_BASE_URL`, and `CHAT_MODEL`).

| Suite | Repository secrets | Job-set environment (not secrets) |
|---|---|---|
| RAGAS | `TOGETHER_API_KEY` | `RAGAS_LLM_PROVIDER=together`, `OPENAI_BASE_URL=https://api.together.xyz/v1`. The job also sets `OPENAI_API_KEY` from the Together secret so the existing RAGAS OpenAI-compatible client can call Together. That is not an OpenRouter mapping. |
| DeepEval | `OPENROUTER_API_KEY` | `OPENAI_BASE_URL=https://openrouter.ai/api/v1`. Optional `DEEPEVAL_JUDGE_MODEL`. |
| Playwright MCP | `OPENROUTER_API_KEY` | Same OpenRouter base URL. Requires `npx` / Node 20 in CI. |
| Langfuse Agent | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `OPENROUTER_API_KEY` | OpenRouter base URL. CI also installs `spikes/langfuse_openai_agents/requirements.txt` (OpenInference instrumentor). |
| Langfuse Chatbot | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `OPENROUTER_API_KEY` | `CHAT_BASE_URL=http://127.0.0.1:3000`, `CHAT_MODEL=openai/gpt-4o-mini`, pinned `LANGFUSE_EXAMPLES_SHA=77c0d4406ec7cffc211a21d98e2f43e0046a29ae`. The official app process is started with `OPENAI_API_KEY` and `OPENAI_BASE_URL` unset. |

### How to run offline tests

Push/PR CI (`.github/workflows/tests.yml`) runs the deterministic suite:

```bash
pytest tests -q          # Phase 2 baseline (no root Phase 1 files; pytest.ini applies -m "not live")
```

`python -m pytest` also collects unmarked Phase 1 root tests. Those are not `@pytest.mark.live` and they are not `EvaluationRunner` tests. With an OpenRouter `OPENAI_BASE_URL` they have previously failed 401 (`Missing Authentication header`). That is a Phase 1 provider-auth mismatch, not a Phase 2 regression.

Local live tests, when credentials are configured:

```bash
pytest -m live --override-ini="addopts=-p no:deepeval"
```

### How to select a GitHub Actions suite

1. Open **Actions → live-tests → Run workflow**.
2. Check only the suite(s) to run. Leave the others `false`.
3. Confirm the matching repository secrets exist.
4. Do not treat an unselected job as passed. An unselected job does not run.

### JUnit safeguards

Each live job writes a JUnit XML file and then runs `.github/scripts/require_live_junit.py` with that suite's `--min-tests` value. The helper fails the job if:

* the report cannot be read
* collected tests are fewer than the minimum
* any required test was skipped
* executed (non-skipped) tests are fewer than the minimum

A missing, skipped, or uncollected required live test cannot produce a false-green job. A real assertion failure (for example chatbot G-Eval below threshold) still fails pytest; the JUnit helper does not hide that.

---

## Validated SUT paths

These go through capture helpers and `EvaluationRunner`. Playwright MCP is the CLI/web flagship. External RAG, Langfuse Agent, and the Chatbot are not CLI scenarios.

| SUT | Evidence | Live-validated through Runner | Unit-tested only |
|---|---|---|---|
| External RAG demo (`RAG_API_URL`) | `live_rag_demo_request`: live answer + retrieved contexts | Together AI RAGAS Faithfulness suite (3 selected tests). See reported results below. | Adapter maps for answer relevancy, contextual relevancy/precision/recall, hallucination, and G-Eval correctness |
| Langfuse OpenAI Agent | `get_many(trace_id=...)`; `TOOL` rows | Tool Correctness (OpenRouter judge) and Final State, including expected-negative cases | G-Eval from GENERATION observations |
| Playwright MCP | Serialized MCP tools; last snapshot for state | Tool Correctness (OpenRouter judge), Execution Health, Final State, including expected-negative cases | Scripted CLI / doubles |
| Langfuse user-feedback chatbot | `get_many(session_id=...)`; root `handle-chat-message` SPAN `input`/`output` | Turn Relevancy and per-turn G-Eval. Latest reported G-Eval is below threshold. | Extraction and request-shape tests |

---

## Validation

Results in this section are **reported from prior runs**. They were not re-executed for this documentation update. Attribution of those live suite outcomes to commit `c095dd5` is **UNKNOWN / TO VERIFY**. Do not treat the table as a new live execution, and do not claim that the entire `live-tests` workflow passed.

### Reported live suite outcomes

| Suite | Selected tests | Reported outcome |
|---|---|---|
| RAGAS (Together AI) | 3 | **Passed** |
| DeepEval (OpenRouter) | 18 | **Passed** |
| Playwright MCP | 6 | **Passed**, including expected-negative tool-order, final-state, and execution-error cases |
| Langfuse Agent | 5 | **Passed**, including expected-negative tool-order and wrong-expected-title cases. Tool Correctness uses the OpenRouter judge. |
| Langfuse Chatbot | 2 | Both tests **executed** (0 skipped). Chatbot returned **HTTP 200**. Langfuse observations were retrieved. Turn Relevancy **passed**. G-Eval **failed** with score **0.6** against threshold **0.8**. |

**Known unresolved quality issue:** Langfuse Chatbot G-Eval scored **0.6 < 0.8**. The threshold and assertions were not lowered. Connectivity and tracing succeeded; the gate failed on the evaluator score. This is not recorded as a workflow-wide pass.

### Reported offline contract suite

After the Langfuse agent OpenRouter judge fix, the focused offline contract suite most recently reported **38 passed**. That suite covers judge-config, agent spike install, live-workflow, and Tool Correctness adapter contracts. It is not a live-provider run and not a full `pytest tests -q` recount.

A previously documented Phase 2 baseline was **392 passed, 35 deselected, 1 warning** (`pytest tests -q`). That count was not re-measured for this update. DeepEval is pinned at 4.2.6. The warning is a `HallucinationMetric` score-direction notice; platform hallucination scoring is already higher-is-better with policy `>= 0.8`.

Agent GENERATION G-Eval and non-faithfulness RAG maps remain unit-tested only.

**Historical RAG judge-block (not the current Together suite).** An earlier OpenRouter RAGAS Faithfulness attempt was **JUDGE_BLOCKED / NOT SCORED** (`LLMDidNotFinishException` / `max_tokens`) after the RAG API returned HTTP 200 with a valid answer and 4 retrieved contexts. That remains a failure-domain example (SUT succeeded; that judge did not). It is not the reported outcome of the current Together AI RAGAS job.

---

## Roadmap

**Implemented**

* Design A request maps and thin `EvaluationRunner`
* Registry, `EvaluationConfig`, result normalizer, `QualityPolicy`, `QualityGate`
* Deterministic, RAGAS Faithfulness, and DeepEval evaluator adapters listed above
* Playwright MCP capture/agent boundary, scripted CLI, and live `ai_qe_eval.demo.web`
* Langfuse Agent `trace_id`/TOOL path and chatbot `session_id`/root `handle-chat-message` path
* External RAG demo capture (Together AI live RAGAS suite; earlier OpenRouter judge-block retained as history)
* Opt-in `live-tests.yml` with five independent checkboxes and JUnit collection/skip gates
* Concise CLI report (no score aggregation)

**Not implemented / future**

* Mapping `GateDecision.passed` itself into a dedicated CI plugin (live jobs today fail on pytest assertions plus the JUnit collection/skip helper)
* Persistence, lineage, or a dashboard
* Distributed execution
* Security or adversarial evaluation
* CLI coverage beyond the P0 MCP demo (RAG and Langfuse are not CLI scenarios)
* Remaining Phase 1 RAGAS metrics as Phase 2 adapters (context precision, context recall, response relevancy, factual correctness)
* Default-CLI Langfuse or RAG execution
* Live Langfuse G-Eval for the OpenAI Agent GENERATION path

---

## Phase 1 — historical RAGAS POC

The original implementation scores answers from an external RAG demo with [RAGAS](https://github.com/explodinggradients/ragas) 0.2.15. The judge is an open model on [Together AI](https://www.together.ai/) through the OpenAI-compatible client.

It remains in the repository as:

* historical implementation
* baseline / reference
* source of the live evaluation experiments

It is **not** the architectural foundation for later phases. The live tests call RAGAS directly (`utils.py`, `conftest.py`, root `test_*.py`). They do not go through `EvaluationRunner`.

Phase 1 metrics still present: context precision (without reference), context recall, faithfulness, response relevancy, factual correctness.

Phase 1 thresholds in `utils.py` (`RAGAS_THRESHOLD_*`) are experimental POC gates. They are **not** `QualityPolicy` and they are **not** production-validated.

### Known live-integration issue (not a Phase 2 defect)

These four root files are historical Phase 1 RAGAS tests. They are **not** marked `live`, so `python -m pytest` collects them. They call the external RAG API and a judge through `conftest.py` / `utils.py` using `OPENAI_API_KEY` + `OPENAI_BASE_URL`. They do not use `EvaluationRunner`.

The working `.env` points `OPENAI_BASE_URL` at OpenRouter. Phase 1 still sends `OPENAI_API_KEY` (Together-era / OpenAI-key placeholder) with the Mixtral id `mistralai/Mixtral-8x7B-Instruct-v0.1`. That combination currently returns **401 `Missing Authentication header`**. `test_resp_relevancy_factual_correctness.py` then also reports a missing/NaN `answer_relevancy` score.

This is a Phase 1 provider-auth mismatch. Do not treat it as a Phase 2 or P2 regression. `pytest tests -q` does not collect these files.

Historically, with Together as `OPENAI_BASE_URL`, the same Mixtral id returned `model_not_available` for serverless access. That Together wording is retained as history; the current measured failure is the 401 above. Freeze-review counts (`pytest tests -q` — 206 passed) are historical. The current baseline is the Validation section.

### Install and configure (Phase 1)

```bash
git clone https://github.com/atagare1/llm-rag-evaluation-ragas.git
cd llm-rag-evaluation-ragas
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Do not commit `.env`.

```env
OPENAI_API_KEY=your_together_ai_key
OPENAI_BASE_URL=https://api.together.xyz/v1
RAG_API_URL=https://rahulshettyacademy.com/rag-llm/ask
RAGAS_LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
RAGAS_EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
```

Root Phase 1 tests skip the judge fixture when `OPENAI_API_KEY` is unset, but they are unmarked `live`, so a set key is enough for `python -m pytest` to run them. `pytest.ini` puts `src` on `pythonpath` and disables the DeepEval pytest plugin (`-p no:deepeval`) so collection does not require DeepEval’s optional providers.

```bash
pytest tests -q   # Phase 2 baseline; no root Phase 1 files
python -m pytest  # also collects unmarked Phase 1 root tests (currently 401)
```

Experimental Phase 1 gates: context precision `> 0.8`, context recall `> 0.7`, faithfulness `> 0.8`, answer relevancy `> 0.8`, factual correctness `> 0.8`.

Together AI is the OpenAI-compatible host. Model choice is `RAGAS_LLM_MODEL`. This POC does not benchmark multiple models.

---

## Phase 2 contracts

Domain, policy, gate, normalizer, and runner do not import RAGAS or DeepEval. Adapters do.

| Component | Module | Responsibility |
|---|---|---|
| Request map | Runner input | `{"name": {"args": [...], "kwargs": {...}}}`. The runner does not rebuild the map. |
| `EvaluationRun` / `TraceEvaluation` | `domain/run.py` | Execution record. `requests[i]` aligns with `trace_evaluations[i]`. No score aggregation. |
| `EvaluationResult` | `domain/result.py` | `metric`, `evaluator`, `score`, optional `reason`, `raw_result`. No pass/fail. |
| `Evaluator` | `domain/evaluator.py` | `evaluate(*args, **kwargs) -> list[EvaluationResult]` |
| `EvaluationRegistry` | `domain/registry.py` | Name catalog only. Does not store instances or thresholds. |
| `EvaluationConfig` | `domain/config.py` | Ordered capability names. |
| Result normalizer | `normalization/result_normalizer.py` | Copies results. Does not change scores. |
| `QualityPolicy` | `policy/quality_policy.py` | Operators `>`, `>=`, `<`, `<=`, `==`, `!=`. `passed` lives here. |
| `QualityGate` | `gate/quality_gate.py` | Any `passed is False` fails. Empty decision list passes. |
| `EvaluationRunner` | `runner/evaluation_runner.py` | `run` / `run_many`. Resolves catalog names, calls injected evaluators, normalizes, applies policies, gates once. |

`EvaluationRunner.run(request, configuration)` returns `GateDecision` for one request map. `run_many` evaluates several request maps in order. One gate call uses the flat decision list. An empty request list raises `ValueError`. A runner failure clears `last_run` for that call.

### What the core does not decide

* `EvaluationResult.score` is not pass/fail.
* `QualityPolicy` is not the release decision.
* `QualityGate` is not GitHub Actions, Jenkins, or a deployment block.

---

## Tests

| Suite | Role |
|---|---|
| `tests/test_evaluation_*.py`, `test_evaluator_contract.py` | Domain contracts |
| `tests/test_deterministic_evaluator.py` | Exact match, final state, MCP health |
| `tests/test_ragas_evaluator.py`, `tests/test_deepeval_*.py` | Adapter mapping with stubs. No live LLM. |
| `tests/test_evaluation_runner.py`, `test_quality_policy.py`, `test_quality_gate.py` | Thin runner, policy, gate |
| `tests/test_cli.py`, `tests/test_cli_mcp_p0_demo.py` | Scripted CLI report and MCP demo through `EvaluationRunner` |
| `tests/platform/test_pv_*.py` | Platform validation, including `@pytest.mark.live` MCP / Langfuse / RAG / provider tests |
| `tests/test_live_tests_workflow.py`, `test_require_live_junit.py`, `test_deepeval_live_judge_config.py` | Offline contracts for the five live-suite checkboxes, JUnit skip gate, and OpenRouter judge wiring |
| Root `test_*.py` | Unmarked Phase 1 RAGAS. Collected by `python -m pytest`, not by `pytest tests -q`. Separate from the Phase 2 runner. |

End-to-end deterministic tests prove delegation order (`evaluator` → `normalizer` → `policy` → `gate`). They are not live RAGAS, DeepEval-provider, Playwright MCP server, or Langfuse runs.
