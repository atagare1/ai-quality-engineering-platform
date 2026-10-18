"""External integration adapters.

MCP SDK and transport types stay in this package. Domain capture and
evaluation receive only plain Python values.
"""

from ai_qe_eval.integrations.playwright_mcp import (
    PLAYWRIGHT_MCP_PACKAGE,
    TODO_MVC_URL,
    playwright_mcp_stdio_parameters,
    serialize_call_tool_result,
)

__all__ = [
    "PLAYWRIGHT_MCP_PACKAGE",
    "TODO_MVC_URL",
    "playwright_mcp_stdio_parameters",
    "serialize_call_tool_result",
]
