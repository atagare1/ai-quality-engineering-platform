"""Offline contract checks for the live-tests workflow checkboxes."""

from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "live-tests.yml"
)

PINNED_LANGFUSE_EXAMPLES_SHA = "77c0d4406ec7cffc211a21d98e2f43e0046a29ae"
CHECKBOXES = (
    "run_ragas",
    "run_deepeval",
    "run_playwright_mcp",
    "run_langfuse_agent",
    "run_langfuse_chatbot",
)
MIN_TESTS = (
    ("live-ragas.xml", 3),
    ("live-deepeval.xml", 18),
    ("live-playwright.xml", 6),
    ("live-langfuse-agent.xml", 5),
    ("live-langfuse-chatbot.xml", 2),
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_live_tests_workflow_has_five_independent_checkboxes_default_false():
    text = _workflow_text()
    for name in CHECKBOXES:
        assert f"      {name}:" in text
        block_start = text.index(f"      {name}:")
        block = text[block_start : block_start + 220]
        assert "type: boolean" in block
        assert "default: false" in block


def test_live_tests_workflow_fails_when_no_job_is_selected():
    text = _workflow_text()
    assert "select-jobs:" in text
    assert "select at least one of" in text
    for name in CHECKBOXES:
        assert f"github.event.inputs.{name} != 'true'" in text
        assert f"if: ${{{{ github.event.inputs.{name} == 'true' }}}}" in text


def test_live_tests_workflow_preserves_core_providers_and_junit_gates():
    text = _workflow_text()
    assert "RAGAS_LLM_PROVIDER: together" in text
    assert "OPENAI_BASE_URL: https://api.together.xyz/v1" in text
    assert "TOGETHER_API_KEY: ${{ secrets.TOGETHER_API_KEY }}" in text
    assert "OPENAI_BASE_URL: https://openrouter.ai/api/v1" in text
    assert "OPENAI_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}" not in text
    for xml_name, minimum in MIN_TESTS:
        assert f"--min-tests {minimum}" in text
        assert xml_name in text


def test_live_tests_workflow_installs_agent_spike_openinference():
    text = _workflow_text()
    agent = text[text.index("  langfuse-agent:") : text.index("  langfuse-chatbot:")]
    assert "spikes/langfuse_openai_agents/requirements.txt" in agent
    assert (
        "from openinference.instrumentation.openai_agents import "
        "OpenAIAgentsInstrumentor"
    ) in agent
    ragas = text[text.index("  ragas:") : text.index("  deepeval:")]
    chatbot = text[text.index("  langfuse-chatbot:") :]
    assert "spikes/langfuse_openai_agents/requirements.txt" not in ragas
    assert "spikes/langfuse_openai_agents/requirements.txt" not in chatbot


def test_live_tests_workflow_starts_pinned_official_chatbot_locally():
    text = _workflow_text()
    assert f"LANGFUSE_EXAMPLES_SHA: {PINNED_LANGFUSE_EXAMPLES_SHA}" in text
    assert "https://github.com/langfuse/langfuse-examples.git" in text
    assert "applications/user-feedback" in text
    assert "CHAT_BASE_URL: http://127.0.0.1:3000" in text
    assert "CHAT_BASE_URL: ${{ secrets.CHAT_BASE_URL }}" not in text
    assert "CHAT_MODEL: openai/gpt-4o-mini" in text
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" not in text
    assert "adapt_langfuse_chatbot_openrouter.py" in text
    assert "verify_openrouter_chat_model.py" in text
    assert "npm install @openrouter/ai-sdk-provider@3.1.0" in text
    assert "printf 'OPENROUTER_API_KEY=%s\\n'" in text
    assert "printf 'OPENAI_API_KEY=%s\\n' \"${OPENAI_API_KEY}\"" not in text
    assert "env -u OPENAI_API_KEY -u OPENAI_BASE_URL npm run dev" in text
    assert "Stop official user-feedback app" in text
