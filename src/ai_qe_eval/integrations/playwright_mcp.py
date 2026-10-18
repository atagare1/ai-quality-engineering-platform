"""Playwright MCP stdio integration boundary.

Starts/uses the official @playwright/mcp package through the installed Python
mcp ClientSession transport helpers. Serializes MCP SDK result objects into
plain Python values for the domain capture layer.

Does not import evaluators, Runner, Policy, or Gate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.types import CallToolResult

PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@0.0.82"
TODO_MVC_URL = "https://demo.playwright.dev/todomvc"
RESOLVED_SNAPSHOT_KEY = "resolvedSnapshot"
_SNAPSHOT_SIDECAR_RE = re.compile(r"\[Snapshot\]\(([^)]+)\)")


def _snapshot_text(content: list[Any]) -> str:
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            return item["text"]
    return ""


def _resolved_snapshot(content: list[Any]) -> dict[str, Any] | None:
    match = _SNAPSHOT_SIDECAR_RE.search(_snapshot_text(content))
    if match is None:
        return None
    referenced_path = match.group(1)
    path = Path(referenced_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "path": referenced_path,
            "text": None,
            "resolutionError": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": referenced_path,
        "text": text,
    }


def snapshot_body_from_serialized_result(plain_result: dict[str, Any]) -> str:
    """Return captured sidecar evidence when present, otherwise inline text."""
    resolved = plain_result.get(RESOLVED_SNAPSHOT_KEY)
    if isinstance(resolved, dict) and isinstance(resolved.get("text"), str):
        return resolved["text"]
    return _snapshot_text(plain_result.get("content") or [])


def snapshot_contains_list_item(snapshot_body: str, expected_text: str) -> bool:
    """Return whether an accessibility snapshot has a listitem with exact text."""
    if not isinstance(snapshot_body, str) or not isinstance(expected_text, str):
        return False
    lines = snapshot_body.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)-\s+listitem\b", line)
        if match is None:
            continue
        item_indent = len(match.group(1))
        item_lines = [line]
        for nested_line in lines[index + 1 :]:
            if nested_line.strip():
                nested_indent = len(nested_line) - len(nested_line.lstrip())
                if nested_indent <= item_indent:
                    break
            item_lines.append(nested_line)
        item_body = "\n".join(item_lines)
        exact_text = re.escape(expected_text)
        if re.search(
            rf"(?m)^\s*-\s+[^:\n]+:\s*{exact_text}\s*$",
            item_body,
        ):
            return True
    return False


def playwright_mcp_stdio_parameters(
    *,
    headless: bool = True,
    isolated: bool = True,
) -> StdioServerParameters:
    """Return stdio launch parameters for the official Playwright MCP server."""
    args = ["--yes", PLAYWRIGHT_MCP_PACKAGE]
    if headless:
        args.append("--headless")
    if isolated:
        args.append("--isolated")
    return StdioServerParameters(command="npx", args=args)


def serialize_call_tool_result(result: CallToolResult) -> dict[str, Any]:
    """Convert an MCP CallToolResult into plain JSON-friendly data.

    Domain ToolInvocation.result must not hold MCP SDK objects.
    """
    if not isinstance(result, CallToolResult):
        raise TypeError(
            "serialize_call_tool_result requires CallToolResult, "
            f"got {type(result).__name__}"
        )
    content: list[Any] = []
    for item in result.content or []:
        if hasattr(item, "model_dump"):
            content.append(item.model_dump(mode="json"))
        else:
            content.append({"type": type(item).__name__, "repr": repr(item)})
    plain_result = {
        "isError": bool(result.isError),
        "content": content,
        "structuredContent": result.structuredContent,
    }
    resolved_snapshot = _resolved_snapshot(content)
    if resolved_snapshot is not None:
        plain_result[RESOLVED_SNAPSHOT_KEY] = resolved_snapshot
    return plain_result
