"""Deterministic tests for captured Playwright snapshot evidence."""

from mcp.types import CallToolResult, TextContent

from ai_qe_eval.capture.mcp_trace import tool_invocation_from_observation
from ai_qe_eval.integrations.playwright_mcp import (
    RESOLVED_SNAPSHOT_KEY,
    serialize_call_tool_result,
    snapshot_body_from_serialized_result,
    snapshot_contains_list_item,
)


def _result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=None,
        isError=False,
    )


def test_serializer_captures_sidecar_evidence_and_preserves_original_pointer(
    tmp_path,
    monkeypatch,
):
    snapshot_body = """
- main:
  - list:
    - listitem [ref=e10]:
      - generic [ref=e11]: Buy milk
""".strip()
    sidecar = tmp_path / "snapshot.yml"
    sidecar.write_text(snapshot_body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    plain = serialize_call_tool_result(_result("[Snapshot](snapshot.yml)"))
    invocation = tool_invocation_from_observation(
        name="browser_snapshot",
        arguments={},
        result=plain,
    )
    sidecar.unlink()

    assert plain["content"][0]["text"] == "[Snapshot](snapshot.yml)"
    assert plain[RESOLVED_SNAPSHOT_KEY] == {
        "path": "snapshot.yml",
        "text": snapshot_body,
    }
    assert snapshot_body_from_serialized_result(invocation.result) == snapshot_body


def test_inline_snapshot_serialization_remains_backward_compatible():
    snapshot_body = "- main:\n  - heading \"todos\""

    plain = serialize_call_tool_result(_result(snapshot_body))

    assert plain == {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": snapshot_body,
                "annotations": None,
                "meta": None,
            }
        ],
        "structuredContent": None,
    }
    assert snapshot_body_from_serialized_result(plain) == snapshot_body


def test_missing_sidecar_preserves_result_and_records_resolution_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    plain = serialize_call_tool_result(_result("[Snapshot](missing.yml)"))

    assert plain["content"][0]["text"] == "[Snapshot](missing.yml)"
    assert plain[RESOLVED_SNAPSHOT_KEY]["path"] == "missing.yml"
    assert plain[RESOLVED_SNAPSHOT_KEY]["text"] is None
    assert "FileNotFoundError" in plain[RESOLVED_SNAPSHOT_KEY]["resolutionError"]
    assert snapshot_body_from_serialized_result(plain) == "[Snapshot](missing.yml)"


def test_strengthened_state_check_accepts_exact_text_inside_list_item():
    snapshot_body = """
- main:
  - textbox "What needs to be done?" [ref=e1]
  - list:
    - listitem [ref=e10]:
      - checkbox "Toggle Todo" [ref=e11]
      - generic [ref=e12]: Buy milk
      - button "Delete" [ref=e13]
""".strip()

    assert snapshot_contains_list_item(snapshot_body, "Buy milk") is True


def test_strengthened_state_check_rejects_text_outside_list_item():
    snapshot_body = """
- main:
  - textbox "What needs to be done?" [ref=e1]: Buy milk
  - paragraph: Buy milk
""".strip()

    assert snapshot_contains_list_item(snapshot_body, "Buy milk") is False


def test_strengthened_state_check_requires_exact_todo_text():
    snapshot_body = """
- main:
  - list:
    - listitem [ref=e10]:
      - generic [ref=e11]: Buy milk and bread
""".strip()

    assert snapshot_contains_list_item(snapshot_body, "Buy milk") is False
