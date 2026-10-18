"""Second live RAGAS faithfulness smoke with an injected judge model.

Reuses Phase 1 faithfulness test data and the get_test_data fixture.
The judge is selected by RAGAS_LLM_PROVIDER. Official OpenAI uses
gpt-4o-mini with JSON mode. Together uses
meta-llama/Llama-3.3-70B-Instruct-Turbo with JSON mode. OpenRouter uses
meta-llama/llama-3.3-70b-instruct without JSON mode. RAGAS_LLM_MODEL
overrides the model id. The judge is passed only into this test's
RAGASFaithfulnessEvaluator. File defaults stay Mixtral.

Temporary diagnostic: prints per-generation finish/usage scalars and a
truncated response preview plus schema-parse flags. Does not log
credentials, prompts, contexts, or untruncated generated text.
"""

import json
import math

import pytest
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._faithfulness import StatementGeneratorOutput
from ragas.prompt.utils import extract_json

from ai_qe_eval.evaluators.ragas import (
    FAITHFULNESS_METRIC,
    RAGAS_EVALUATOR_NAME,
    RAGASFaithfulnessEvaluator,
)
from ragas_live import live_ragas_llama_chat, ragas_llama_judge_model_name
from utils import metric_threshold, read_test_data

PROBE_TEXT_LIMIT = 400


def _first_generation(result):
    if result is None or not result.generations or not result.generations[0]:
        return None
    return result.generations[0][0]


def _usage_output_tokens(generation, result):
    message = getattr(generation, "message", None)
    if message is not None:
        usage_meta = getattr(message, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta.get("output_tokens") is not None:
            return usage_meta.get("output_tokens")
        if usage_meta is not None and getattr(usage_meta, "output_tokens", None) is not None:
            return usage_meta.output_tokens
        meta = getattr(message, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            for key in ("completion_tokens", "output_tokens"):
                if token_usage.get(key) is not None:
                    return token_usage.get(key)
    llm_output = getattr(result, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or {}
    if isinstance(token_usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            if token_usage.get(key) is not None:
                return token_usage.get(key)
    return None


def _truncate_probe_text(text: str) -> str:
    if len(text) <= PROBE_TEXT_LIMIT:
        return text
    return f"{text[:PROBE_TEXT_LIMIT]}...[truncated {len(text) - PROBE_TEXT_LIMIT} chars]"


def _statement_schema_diagnostics(text: str) -> dict[str, object]:
    extracted = None
    json_ok = False
    schema_ok = False
    schema_error = None
    try:
        extracted = extract_json(text)
    except Exception as exc:
        schema_error = f"extract_json {type(exc).__name__}"
    if extracted is not None:
        try:
            parsed = json.loads(extracted)
            json_ok = True
            StatementGeneratorOutput.model_validate(parsed)
            schema_ok = True
        except Exception as exc:
            detail = str(exc).replace("\n", " ")
            if len(detail) > 160:
                detail = detail[:160] + "..."
            schema_error = f"{type(exc).__name__}: {detail}"
    return {
        "starts_with_json": text.lstrip().startswith(("{", "[")),
        "json_parse_ok": json_ok,
        "statement_schema_ok": schema_ok,
        "schema_error": schema_error,
    }


def _print_judge_probe(stage, result, is_finished, max_tokens):
    finish_reason = None
    stop_reason = None
    completion_tokens = None
    response_chars = None
    response_preview = None
    schema = {
        "starts_with_json": None,
        "json_parse_ok": None,
        "statement_schema_ok": None,
        "schema_error": None,
    }
    generation = _first_generation(result)
    if generation is not None:
        text = generation.text or ""
        response_chars = len(text)
        response_preview = _truncate_probe_text(text)
        schema = _statement_schema_diagnostics(text)
        info = generation.generation_info or {}
        finish_reason = info.get("finish_reason")
        stop_reason = info.get("stop_reason")
        message = getattr(generation, "message", None)
        if message is not None:
            meta = getattr(message, "response_metadata", None) or {}
            if finish_reason is None:
                finish_reason = meta.get("finish_reason")
            if stop_reason is None:
                stop_reason = meta.get("stop_reason")
        completion_tokens = _usage_output_tokens(generation, result)
    print("judge_stage", stage)
    print("finish_reason", finish_reason)
    print("stop_reason", stop_reason)
    print("completion_tokens", completion_tokens)
    print("response_chars", response_chars)
    print("max_tokens", max_tokens)
    print("is_finished", is_finished)
    print("starts_with_json", schema["starts_with_json"])
    print("json_parse_ok", schema["json_parse_ok"])
    print("statement_schema_ok", schema["statement_schema_ok"])
    print("schema_error", schema["schema_error"])
    print("response_preview", response_preview)


class _DiagnosticLangchainLLMWrapper(LangchainLLMWrapper):
    """Test-only wrapper. Captures RAGAS generate metadata; does not change is_finished."""

    def __init__(self, langchain_llm):
        super().__init__(langchain_llm)
        self._generate_calls = 0
        self._last_llm_result = None

    async def agenerate_text(self, *args, **kwargs):
        result = await super().agenerate_text(*args, **kwargs)
        self._last_llm_result = result
        return result

    async def generate(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        self._generate_calls += 1
        if self._generate_calls == 1:
            stage = "statements"
        else:
            # Call 2 may be RAGAS format-repair or NLI verdicts.
            stage = f"generate_{self._generate_calls}"
        self._last_llm_result = None
        is_finished = None
        try:
            result = await super().generate(
                prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            is_finished = True
            return result
        except Exception:
            if self._last_llm_result is not None:
                is_finished = self.is_finished(self._last_llm_result)
            raise
        finally:
            _print_judge_probe(
                stage,
                self._last_llm_result,
                is_finished,
                getattr(self.langchain_llm, "max_tokens", None),
            )


@pytest.mark.live
@pytest.mark.parametrize(
    "get_test_data",
    [read_test_data("rag_test_data_faithfulness.json")],
    indirect=True,
)
def test_ragas_faithfulness_smoke_with_llama_judge(get_test_data):
    sample = get_test_data
    llm = live_ragas_llama_chat()
    wrapper = _DiagnosticLangchainLLMWrapper(llm)
    print("provider_model", ragas_llama_judge_model_name())

    results = RAGASFaithfulnessEvaluator(llm=wrapper).evaluate(
        sample.user_input,
        sample.response,
        list(sample.retrieved_contexts or []),
    )

    assert len(results) == 1
    result = results[0]
    assert result.metric == FAITHFULNESS_METRIC
    assert result.evaluator == RAGAS_EVALUATOR_NAME
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score >= metric_threshold("faithfulness")
    print("faithfulness_score", result.score)
