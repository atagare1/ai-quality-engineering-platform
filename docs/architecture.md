# Architecture

Companion to `README.md`. README is the measured evidence and operator guide. This document explains how the implemented platform is structured and why.

Source of truth: `src/ai_qe_eval` and README. Do not treat this file as a results report or a Phase 1 history.

---

## 1. Problem and architectural goals

AI systems fail in different domains. A RAG answer can be unfaithful after successful retrieval. An agent can take the wrong tool trajectory. An MCP workflow can execute every tool without error and still reach the wrong application state. A judge can fail independently of the system under test.

The platform is one provider-agnostic evaluation engine for four independently gated SUT paths: External RAG, Langfuse OpenAI Agent, Playwright MCP, and Langfuse user-feedback Chatbot. Live execution is provider-aware: five independently selectable GitHub Actions suites choose Together AI, OpenRouter, or Langfuse only where that suite needs them.

Goals:

* Score captured evidence, not a single opaque model verdict.
* Keep SUT, evaluator, policy, and gate as separate responsibilities.
* Fail closed **per path**. Never average paths into one platform score.
* Isolate a judge outage from an application defect.

---

## 2. Design principles

* **Evidence in, score out.** Capture helpers extract observations. Evaluators score only the arguments they are given.
* **Thin runner.** `EvaluationRunner` does not rebuild evidence, apply thresholds, or invent a gate rule.
* **Provider-agnostic core.** Domain, normalizer, policy, gate, and runner do not import RAGAS or DeepEval. Adapters do.
* **QE-owned expected behavior.** Expected tool names and chatbot expected answers are supplied by the test, not inferred from telemetry.
* **One executor.** Every platform path goes through `EvaluationRunner`. Capture does not evaluate. Adapters do not gate.
* **Deterministic default.** `pytest tests -q` stays non-live. Live provider tests are opt-in.

---

## 3. High-level architecture

```text
SUT interaction
  → trace / evidence collection (tools, snapshots, retrieved contexts, conversation turns)
  → Design A request map   {"capability": {"args": [...], "kwargs": {...}}}
  → EvaluationRunner
  → Evaluator.evaluate(...)
  → EvaluationResult[]     (score is not pass/fail)
  → normalize_many         (copy only; scores unchanged)
  → QualityPolicy.apply    (threshold → PolicyDecision.passed)
  → QualityGate.evaluate   (any-fail for that path)
  → GateDecision
```

`EvaluationConfig` names the ordered capabilities. `EvaluationRegistry` is a catalog of names, not instances or thresholds. Evaluators and the policy map are injected.

`EvaluationRun` records the execution. `requests[i]` aligns with `trace_evaluations[i]`. Flat `results` / `decisions` are the same objects in request order — a view, not a second scoring model.

The runner does not rebuild a request map. Capture helpers sit **outside** the core and produce the map.

---

## 4. Four SUT paths

Each path has its own evidence shape and its own gate. They share the runner.

| Path | Evidence | Request map | Evaluated capabilities (implemented) |
|---|---|---|---|
| External RAG | Live HTTP answer + retrieved contexts | `live_rag_demo_request` | Faithfulness through the runner. Other RAG maps exist as adapters. |
| Langfuse OpenAI Agent | `get_many(trace_id)` `TOOL` rows → `ToolInvocation` | `langfuse_tool_correctness_request` | Tool Correctness (OpenRouter judge); Final State. GENERATION G-Eval is unit-mapped only. |
| Playwright MCP | Serialized MCP tool results → `ToolInvocation`; last snapshot for state | `mcp_p0_request` | Tool Correctness (OpenRouter judge); MCP Execution Health; Final State. CLI/web flagship. |
| Langfuse Chatbot | `get_many(session_id)` root `handle-chat-message` SPAN input/output → `ConversationTurn` | turn-relevancy and per-turn G-Eval maps | Turn Relevancy; per-turn G-Eval. Official example app is not in this repo. |

MCP is the interactive demonstration (`python -m ai_qe_eval`, `python -m ai_qe_eval.demo.web`). RAG, Agent, and Chatbot use the same engine through integration and live tests, not CLI scenarios.

These four paths are evaluation concerns. GitHub Actions exposes **five** independently selectable live jobs so RAGAS, DeepEval smoke/through-runner tests, Playwright MCP, Langfuse Agent, and Langfuse Chatbot can be paid for and observed separately.

---

## 5. Evidence-driven evaluation — no universal `EvaluationTrace`

Evaluators accept `*args, **kwargs` and return `EvaluationResult` lists. Evidence is evaluator-specific:

* RAG metrics need question, answer, and contexts (and a reference when the metric requires one).
* Tool Correctness needs observed and QE-expected invocations.
* Execution Health needs observed `isError` outcomes.
* Final State needs a caller-computed boolean (for example, whether the last snapshot contains `Buy milk`).
* Turn Relevancy and chatbot G-Eval need ordered `ConversationTurn` pairs.

A universal `EvaluationTrace` was removed from the production package. One bag-of-fields object either leaves most fields unused or forces every adapter to pretend the SUT looks the same. Design A request maps keep evidence at the capture boundary and keep the runner generic.

`ToolInvocation` and `ConversationTurn` remain as small, provider-neutral domain types. They are not a substitute trace.

---

## 6. Evaluator categories

Adapters return scores. They do not apply `QualityPolicy` or `QualityGate`.

**Deterministic** — no LLM judge. `exact_match`, `mcp_execution_health` (fail on `isError=true`), `final_state` (caller bool). Used to prove that tool success ≠ outcome success.

**RAG / LLM** — RAGAS Faithfulness; DeepEval faithfulness, answer relevancy, contextual relevancy/precision/recall, hallucination, G-Eval correctness. RAGAS Faithfulness is the only Phase 1 RAGAS metric with a Phase 2 adapter.

**Conversational** — DeepEval Turn Relevancy and per-turn G-Eval on chatbot root-SPAN turns.

**Agent / tool / MCP** — DeepEval Tool Correctness (name/order; expected names are QE-supplied). Combined with Execution Health and Final State on MCP so a healthy tool sequence can still fail the gate.

The default CLI runs only the three MCP P0 evaluators.

---

## 7. Failure-domain isolation and fail-closed behavior

Isolation means a failure is attributed to the layer that actually failed:

| Domain | Example from this platform |
|---|---|
| SUT / application | MCP types `Buy bread`; Final State fails while tools and health pass. |
| Trajectory | Agent or MCP wrong tool order; Tool Correctness fails. |
| Execution | MCP `isError=true`; Execution Health fails. |
| Extraction | Capture could not build a request (missing root input/output, unset session). No score is invented. |
| Judge | Historical OpenRouter RAGAS Faithfulness raised `LLMDidNotFinishException` after a successful RAG HTTP 200. **JUDGE_BLOCKED / NOT SCORED** — not FAIL `0.0`, not an RAG application failure. The current Together AI RAGAS suite is a separate, reported-pass path. |

Fail-closed: any `PolicyDecision.passed is False` fails that path’s `QualityGate`. A missing score is not coerced to `0.0`. A direct `QualityGate.evaluate([])` call currently passes. `EvaluationRunner` no longer reaches the gate with an empty `EvaluationConfig` (it raises `ValueError`).

---

## 8. QualityPolicy vs QualityGate

| | `QualityPolicy` | `QualityGate` |
|---|---|---|
| Input | One `EvaluationResult` | The list of `PolicyDecision`s |
| Owns | Operator and threshold (`>`, `>=`, `<`, `<=`, `==`, `!=`) | Any-fail aggregation |
| Output | `PolicyDecision.passed` | `GateDecision.passed` |
| Does not | Execute evaluators, mutate scores, release software | Re-check scores, average paths, talk to CI |

`EvaluationResult.score` is not pass/fail. Policy is not the release decision. The gate is an in-process decision for **one** run, not GitHub Actions, Jenkins, or a combined score across SUTs.

---

## 9. Live validation approach

Evidence classes stay distinct: unit / doubles, deterministic smoke, live provider, and judge-blocked.

* `pytest tests -q` is the Phase 2 regression gate (non-live). Push/PR CI runs that command only.
* `@pytest.mark.live` is opt-in. GitHub Actions `live-tests.yml` is `workflow_dispatch` only; every checkbox defaults to `false`.
* Positive cases show a path can pass its policies.
* Negative cases are first-class: wrong tool order, wrong expected title, wrong final state, execution error. They prove the gate can fail for the intended reason.
* Phase 1 root RAGAS tests do not enter `EvaluationRunner` and are not platform proof.

Measured outcomes live in README and are labeled as **reported**, not newly executed. This document does not restate them as new results.

### 9.1 Provider boundaries

| Concern | Provider | Not used for |
|---|---|---|
| RAGAS live judge | Together AI (`TOGETHER_API_KEY`, `OPENAI_BASE_URL=https://api.together.xyz/v1`, default Turbo instruct model) | DeepEval scoring, chatbot inference, Langfuse tracing |
| DeepEval live judge (DeepEval, Playwright MCP Tool Correctness, Langfuse Agent Tool Correctness, chatbot Turn Relevancy / G-Eval) | OpenRouter (`OPENROUTER_API_KEY`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `live_deepeval_local_model()`) | RAGAS Together scoring |
| Chatbot inference | OpenRouter via CI-side adapt of the official example (`CHAT_MODEL`, default `openai/gpt-4o-mini`) | DeepEval judge model; no `OPENROUTER_API_KEY` → `OPENAI_API_KEY` mapping |
| Tracing | Langfuse (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`) | Scoring |

Models and free tiers are not interchangeable. Together Turbo, OpenRouter Llama 3.3 70B instruct, and `openai/gpt-4o-mini` are separate configured roles.

### 9.2 Why each live suite is independently selectable

Live jobs have different credentials, runtimes, and cost. RAGAS needs Together. DeepEval and Playwright need OpenRouter. Agent and chatbot also need Langfuse. The chatbot job clones Node 22, starts a Next app, and holds a 50-minute timeout. Forcing every dispatch to run all 34 selected tests would couple unrelated failures and spend unused provider quota. Independent checkboxes keep failure-domain isolation at the CI boundary: an unselected job does not run and is not a pass.

`select-jobs` fails when every checkbox is left `false`. JUnit `--min-tests` on each job fails if required tests are missing, skipped, or uncollected.

### 9.3 Chatbot CI lifecycle

The official Langfuse `applications/user-feedback` example is not in this repository. The chatbot job:

1. Clones `langfuse/langfuse-examples` at pinned SHA `77c0d4406ec7cffc211a21d98e2f43e0046a29ae` (sparse-checkout of `applications/user-feedback`).
2. Adapts only `app/api/chat/route.ts` to `@openrouter/ai-sdk-provider@3.1.0` + `OPENROUTER_API_KEY`.
3. Verifies `CHAT_MODEL` against the OpenRouter catalog and optionally probes a completion. The chat model must differ from the DeepEval judge.
4. Writes a job-local `.env` (OpenRouter + Langfuse + `CHAT_MODEL`; no `OPENAI_API_KEY`).
5. Starts `npm run dev` on `127.0.0.1:3000` with `OPENAI_API_KEY` and `OPENAI_BASE_URL` unset.
6. Readiness-checks the TCP port, then runs the two live pytest files.
7. Stops the Next process in an `if: always()` step.

Local pytest still expects an already-listening `CHAT_BASE_URL`. It does not start the official app.

### 9.4 Tool-correctness expected-pass / expected-fail

Tool Correctness scores captured `ToolInvocation` evidence against QE-supplied expected calls. Final State scores a caller-computed boolean from captured tool results or snapshots. Neither infers expected behavior from telemetry.

Preserved semantics:

* **Expected pass:** observed tool names (and, where selected, `INPUT_PARAMETERS`) match the QE spec; Final State boolean is true.
* **Expected fail:** wrong expected tool order, wrong expected argument/title, or wrong final-state predicate. The score stays below the policy threshold (`tool_correctness >= 0.80` or `final_state == 1.0`) and `QualityGate` fails for that path.

Playwright MCP adds Execution Health (`isError`) so a healthy tool sequence can still fail Final State. The Langfuse Agent suite uses the same pass/fail split: three Tool Correctness cases (correct order, wrong order, clock arguments) plus two Final State cases (expected slideshow title vs wrong title).

---

## 10. Key architectural decisions

**Request maps instead of `EvaluationTrace`**

* Problem: four SUTs produce incompatible evidence.
* Options considered: one universal trace; per-evaluator request maps (Design A).
* Decision: Design A. `EvaluationTrace` is gone from production.
* Trade-off: each path needs a capture helper. Gain: the runner never special-cases RAG vs MCP vs chat.

**Provider-agnostic core**

* Problem: RAGAS and DeepEval APIs must not leak into policy or gate.
* Decision: adapters only at the evaluator boundary.
* Trade-off: more mapping code; the core stays testable without a live model.

**Independent per-path gates and independent live suites**

* Problem: MCP can pass while a RAG judge is blocked or a chatbot G-Eval score is below threshold.
* Decision: no combined platform score, pass rate, or average. Live CI jobs are independently selectable.
* Trade-off: no single headline number for a dashboard. Gain: honest attribution and isolated provider cost.

**QE-supplied expected tools**

* Problem: inferring expected calls from telemetry makes Tool Correctness tautological.
* Decision: expected names come from the test/spec (`evaluation_params=[]` name/order on the default CLI path).
* Trade-off: tests must state intent. Gain: wrong trajectory is detectable.

**MCP as the only interactive flagship**

* Problem: wiring RAG, Agent, and Chatbot into CLI would add process orchestration (browser, Langfuse, Next.js), not evaluation value.
* Decision: CLI/web stay MCP; other paths stay pytest.
* Trade-off: visitors do not click through all four SUTs. Gain: one coherent demo of “tools pass, state fails.”

**Thin runner, injected dependencies**

* Problem: a “smart” orchestrator tends to hide scoring and gating.
* Decision: `run` / `run_many` resolve names, call evaluators, normalize, apply policies, gate once.
* Consequence: a runner exception clears `last_run` for that call. Empty `run_many([])` raises `ValueError`.

---

## 11. Current limitations

Documented in README; not defects to paper over. Outcomes below are **reported**, not newly executed here. Attribution to commit `c095dd5` is **UNKNOWN / TO VERIFY**.

* Langfuse Chatbot G-Eval is a known unresolved quality issue: reported score **0.6** against threshold **0.8**. Both chatbot tests executed without skips; the SUT returned HTTP 200 and Langfuse observations were retrieved. The threshold was not lowered.
* An earlier OpenRouter RAGAS Faithfulness attempt was **JUDGE_BLOCKED / NOT SCORED**. The current Together AI RAGAS suite is reported as passed and is a different judge path.
* Agent GENERATION G-Eval and non-faithfulness RAG maps are unit-tested only.
* Remaining Phase 1 RAGAS metrics have no Phase 2 adapter.
* `QualityGate` is not a CI plugin. Live jobs fail on pytest plus JUnit collection/skip minima. Those minima prove tests executed; they do not establish semantic quality. `EvaluationRunner` raises if `evaluations` is empty. A direct `QualityGate.evaluate([])` call still passes.
* No persistence, lineage, or dashboard. `EvaluationRun.to_dict` exists; nothing aggregates separate SUT runs.
* CLI/web do not run RAG or Langfuse.
* Historical Phase 1 root tests remain a second path and have previously 401'd on OpenRouter. They are not this architecture.

Extend Phase 2 contracts. Do not grow new framework behavior by editing the original RAGAS scripts.
