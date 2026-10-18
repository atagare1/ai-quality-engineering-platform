"""Offline checks for the Langfuse OpenAI Agents spike install contract."""

from pathlib import Path

SPIKE_REQUIREMENTS = (
    Path(__file__).resolve().parents[1]
    / "spikes"
    / "langfuse_openai_agents"
    / "requirements.txt"
)
PINNED_OPENINFERENCE = "openinference-instrumentation-openai-agents==2.5.3"


def test_agent_spike_requirements_pin_openinference_instrumentor():
    text = SPIKE_REQUIREMENTS.read_text(encoding="utf-8")
    assert PINNED_OPENINFERENCE in text
    assert "openinference-instrumentation-openai-agents" in text
