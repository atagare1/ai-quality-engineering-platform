"""PV: live Playwright MCP stdio connectivity smoke.

Starts the official @playwright/mcp server over stdio using the installed
Python mcp ClientSession API. Lists tools, calls browser_navigate, then
closes the session/process via context managers.

Does not touch EvaluationTrace, capture, evaluators, Runner, Policy, or Gate.
"""

from __future__ import annotations

import importlib.metadata
import shutil

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@0.0.82"
TODO_MVC_URL = "https://demo.playwright.dev/todomvc"


@pytest.mark.live
@pytest.mark.asyncio
async def test_pv_playwright_mcp_stdio_browser_navigate_smoke():
    if shutil.which("npx") is None and shutil.which("npx.cmd") is None:
        pytest.skip("npx is not available on PATH")

    mcp_version = importlib.metadata.version("mcp")
    server = StdioServerParameters(
        command="npx",
        args=[
            "--yes",
            PLAYWRIGHT_MCP_PACKAGE,
            "--headless",
            "--isolated",
        ],
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            listed = await session.list_tools()
            tool_names = [tool.name for tool in listed.tools]
            print("mcp_sdk_version", mcp_version)
            print("playwright_mcp_package", PLAYWRIGHT_MCP_PACKAGE)
            print("tool_count", len(tool_names))
            print("tool_names", tool_names)

            assert tool_names, "Playwright MCP returned no tools"
            assert "browser_navigate" in tool_names

            result = await session.call_tool(
                "browser_navigate",
                {"url": TODO_MVC_URL},
            )
            print("navigate_is_error", result.isError)
            print("navigate_content", result.content)

            assert result.isError is False
            assert result.content

    print("session_closed", True)
