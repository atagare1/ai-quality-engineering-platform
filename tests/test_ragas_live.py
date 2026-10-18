"""Focused checks for the live RAGAS judge helper providers."""

import os

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue
from ragas.exceptions import LLMDidNotFinishException
from ragas.llms import LangchainLLMWrapper

from ragas_live import (
    DEFAULT_RAGAS_OPENAI_BASE_URL,
    DEFAULT_RAGAS_OPENAI_JUDGE_MODEL,
    DEFAULT_RAGAS_OPENROUTER_BASE_URL,
    DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL,
    DEFAULT_RAGAS_TOGETHER_BASE_URL,
    DEFAULT_RAGAS_TOGETHER_JUDGE_MODEL,
    RAGAS_JUDGE_JSON_RESPONSE_FORMAT,
    live_ragas_llama_chat,
    load_ragas_live_dotenv,
    ragas_live_env_issues,
    ragas_llama_judge_model_name,
)


def _capture_chat(monkeypatch):
    captured = {}

    class CapturingChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ragas_live.ChatOpenAI", CapturingChatOpenAI)
    return captured


def test_live_ragas_llama_chat_requires_explicit_provider(monkeypatch):
    monkeypatch.delenv("RAGAS_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    with pytest.raises(pytest.fail.Exception, match="RAGAS_LLM_PROVIDER"):
        live_ragas_llama_chat()


def test_live_ragas_llama_chat_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with pytest.raises(pytest.fail.Exception, match="not supported"):
        live_ragas_llama_chat()


def test_openai_provider_defaults_to_official_endpoint(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["model"] == DEFAULT_RAGAS_OPENAI_JUDGE_MODEL
    assert captured["model"] == "gpt-4o-mini"
    assert captured["base_url"] == DEFAULT_RAGAS_OPENAI_BASE_URL
    assert captured["api_key"] == "test-openai-key"
    assert captured["api_key"] != "test-openrouter-key"
    assert captured["temperature"] == 0
    assert "max_tokens" not in captured
    assert captured["model_kwargs"]["response_format"] == RAGAS_JUDGE_JSON_RESPONSE_FORMAT
    assert captured["model_kwargs"]["response_format"] == {"type": "json_object"}


def test_openai_provider_respects_model_override(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("RAGAS_LLM_MODEL", "  gpt-4o  ")

    live_ragas_llama_chat()

    assert ragas_llama_judge_model_name() == "gpt-4o"
    assert captured["model"] == "gpt-4o"
    assert captured["base_url"] == DEFAULT_RAGAS_OPENAI_BASE_URL


def test_openai_provider_accepts_explicit_official_base_url(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["api_key"] == "test-openai-key"


def test_openai_provider_fails_openrouter_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    with pytest.raises(pytest.fail.Exception, match="not compatible"):
        live_ragas_llama_chat()


def test_openai_provider_fails_together_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")

    with pytest.raises(pytest.fail.Exception, match="not compatible"):
        live_ragas_llama_chat()


def test_openai_provider_skips_without_openai_key(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(pytest.skip.Exception, match="OPENAI_API_KEY"):
        live_ragas_llama_chat()


def test_openrouter_provider_uses_openrouter_key_and_endpoint(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["model"] == DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL
    assert captured["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert captured["base_url"] == DEFAULT_RAGAS_OPENROUTER_BASE_URL
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "test-openrouter-key"
    assert captured["api_key"] != "test-openai-key"
    assert "max_tokens" not in captured
    assert "model_kwargs" not in captured or "response_format" not in captured.get(
        "model_kwargs", {}
    )


def test_openrouter_provider_does_not_send_json_object_mode(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["model"] == DEFAULT_RAGAS_OPENROUTER_JUDGE_MODEL
    assert captured.get("model_kwargs", {}).get("response_format") is None


def test_openrouter_provider_respects_model_override(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("RAGAS_LLM_MODEL", "  anthropic/claude-3.5-sonnet  ")

    live_ragas_llama_chat()

    assert ragas_llama_judge_model_name() == "anthropic/claude-3.5-sonnet"
    assert captured["model"] == "anthropic/claude-3.5-sonnet"
    assert captured["base_url"] == DEFAULT_RAGAS_OPENROUTER_BASE_URL


def test_openrouter_provider_fails_official_openai_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    with pytest.raises(pytest.fail.Exception, match="does not use official OpenAI"):
        live_ragas_llama_chat()


def test_openrouter_provider_fails_together_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")

    with pytest.raises(pytest.fail.Exception, match="does not use official OpenAI"):
        live_ragas_llama_chat()


def test_openrouter_provider_skips_without_openrouter_key(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(pytest.skip.Exception, match="OPENROUTER_API_KEY"):
        live_ragas_llama_chat()


def test_together_provider_uses_together_key_endpoint_and_json_mode(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["model"] == DEFAULT_RAGAS_TOGETHER_JUDGE_MODEL
    assert captured["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert captured["base_url"] == DEFAULT_RAGAS_TOGETHER_BASE_URL
    assert captured["base_url"] == "https://api.together.xyz/v1"
    assert captured["api_key"] == "test-together-key"
    assert captured["api_key"] != "test-openai-key"
    assert captured["api_key"] != "test-openrouter-key"
    assert captured["temperature"] == 0
    assert "max_tokens" not in captured
    assert captured["model_kwargs"]["response_format"] == RAGAS_JUDGE_JSON_RESPONSE_FORMAT
    assert captured["model_kwargs"]["response_format"] == {"type": "json_object"}


def test_together_provider_respects_model_override(monkeypatch):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("RAGAS_LLM_MODEL", "  meta-llama/Llama-3.3-70B-Instruct-Turbo  ")

    live_ragas_llama_chat()

    assert ragas_llama_judge_model_name() == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert captured["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert captured["base_url"] == DEFAULT_RAGAS_TOGETHER_BASE_URL


def test_together_provider_accepts_together_base_url_but_keeps_together_endpoint(
    monkeypatch,
):
    captured = _capture_chat(monkeypatch)
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    live_ragas_llama_chat()

    assert captured["base_url"] == DEFAULT_RAGAS_TOGETHER_BASE_URL
    assert captured["api_key"] == "test-together-key"


def test_together_provider_fails_openrouter_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    with pytest.raises(pytest.fail.Exception, match="does not use official OpenAI"):
        live_ragas_llama_chat()


def test_together_provider_fails_official_openai_base_url(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    with pytest.raises(pytest.fail.Exception, match="does not use official OpenAI"):
        live_ragas_llama_chat()


def test_together_provider_skips_without_together_key(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TOGETHER_API_KEY"):
        live_ragas_llama_chat()


def test_together_env_issues_report_names_only(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    issues = ragas_live_env_issues()

    assert issues == ["TOGETHER_API_KEY", "OPENAI_BASE_URL"]
    assert "test-" not in str(issues)
    assert "sk-" not in str(issues)


def test_together_env_issues_empty_when_together_config_is_valid(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    assert ragas_live_env_issues() == []


def test_dotenv_does_not_override_process_provider(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAGAS_LLM_PROVIDER=openrouter\nTOGETHER_API_KEY=from-file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "together")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    loaded = load_ragas_live_dotenv(env_file)

    assert loaded is True
    assert os.getenv("RAGAS_LLM_PROVIDER") == "together"
    assert os.getenv("TOGETHER_API_KEY") == "from-file-key"


def _result_with_finish_reason(finish_reason: str) -> LLMResult:
    return LLMResult(
        generations=[
            [
                Generation(
                    text="partial-json",
                    generation_info={"finish_reason": finish_reason},
                )
            ]
        ]
    )


def test_ragas_treats_openai_length_finish_reason_as_unfinished():
    wrapper = LangchainLLMWrapper(object())
    assert wrapper.is_finished(_result_with_finish_reason("stop")) is True
    assert wrapper.is_finished(_result_with_finish_reason("length")) is False


@pytest.mark.asyncio
async def test_ragas_generate_raises_when_finish_reason_is_length():
    wrapper = LangchainLLMWrapper(object())
    unfinished = _result_with_finish_reason("length")

    async def agenerate_text(**_kwargs):
        return unfinished

    wrapper.agenerate_text = agenerate_text

    with pytest.raises(LLMDidNotFinishException):
        await wrapper.generate(StringPromptValue(text="unused"))


# Captured from the live smoke OutputParserException llm_output. The inner
# JSON is valid StatementGeneratorOutput; the leading prose is what RAGAS
# retries parse without extract_json.
RAGAS_REPAIR_PROSE = (
    "The output string did not satisfy the constraints given in the prompt. "
    "The correct output should be in JSON format and comply with the given schema. "
    "Here is the corrected output: "
    '{"statements": ["The Selenium WebDriver Python course contains 23 articles."]}'
)


def test_statement_schema_accepts_extracted_repair_json():
    from ragas.metrics._faithfulness import StatementGeneratorOutput
    from ragas.prompt.utils import extract_json

    extracted = extract_json(RAGAS_REPAIR_PROSE)
    parsed = StatementGeneratorOutput.model_validate_json(extracted)
    assert parsed.statements == [
        "The Selenium WebDriver Python course contains 23 articles."
    ]


def test_ragas_retry_parse_rejects_prose_wrapped_statement_json():
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import PydanticOutputParser
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    parser = PydanticOutputParser(pydantic_object=StatementGeneratorOutput)
    with pytest.raises(OutputParserException, match="Invalid json output"):
        parser.parse(RAGAS_REPAIR_PROSE)


def test_statement_probe_flags_prose_wrapped_json():
    from tests.platform.test_pv_e_ragas_faithfulness_llama_smoke import (
        _statement_schema_diagnostics,
        _truncate_probe_text,
    )

    flags = _statement_schema_diagnostics(RAGAS_REPAIR_PROSE)
    assert flags["starts_with_json"] is False
    assert flags["json_parse_ok"] is True
    assert flags["statement_schema_ok"] is True
    preview = _truncate_probe_text("x" * 500)
    assert preview.endswith("...[truncated 100 chars]")
    assert len(preview) < 500


VALID_STATEMENT_JSON = (
    '{"statements": ["The Selenium WebDriver Python course contains 23 articles."]}'
)


def test_json_only_statement_output_matches_ragas_schema():
    from langchain_core.output_parsers import PydanticOutputParser
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    parser = PydanticOutputParser(pydantic_object=StatementGeneratorOutput)
    parsed = parser.parse(VALID_STATEMENT_JSON)
    assert parsed.statements == [
        "The Selenium WebDriver Python course contains 23 articles."
    ]


def test_ragas_treats_error_finish_reason_as_unfinished():
    wrapper = LangchainLLMWrapper(object())
    result = LLMResult(
        generations=[
            [
                Generation(
                    text="{\n",
                    generation_info={"finish_reason": "error"},
                )
            ]
        ]
    )
    assert wrapper.is_finished(result) is False


def test_schema_valid_json_is_still_unfinished_when_finish_reason_is_length():
    wrapper = LangchainLLMWrapper(object())
    result = LLMResult(
        generations=[
            [
                Generation(
                    text=VALID_STATEMENT_JSON,
                    generation_info={"finish_reason": "length"},
                )
            ]
        ]
    )
    assert wrapper.is_finished(result) is False


def test_schema_valid_json_is_still_unfinished_when_finish_reason_is_error():
    wrapper = LangchainLLMWrapper(object())
    result = LLMResult(
        generations=[
            [
                Generation(
                    text=VALID_STATEMENT_JSON,
                    generation_info={"finish_reason": "error"},
                )
            ]
        ]
    )
    assert wrapper.is_finished(result) is False
