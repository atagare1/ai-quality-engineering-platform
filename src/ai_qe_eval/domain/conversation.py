"""Conversation and tool-invocation domain types.

Provider-neutral representation of ordered chatbot turns and tool calls.
Does not import DeepEval, MCP, or other evaluation frameworks.

List order is turn order and tool-call order.
Adapters map these types onto framework test cases later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ConversationRole = Literal["user", "assistant"]
_CONVERSATION_ROLES = ("user", "assistant")


def tool_invocations_from_payload(value: Any) -> list[ToolInvocation] | None:
    """Rebuild a tool-invocation list from serialized dicts.

    None stays None. An empty list stays an empty list.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(
            "tool_calls must be a list of ToolInvocation or dict, "
            f"got {type(value).__name__}"
        )
    invocations: list[ToolInvocation] = []
    for index, item in enumerate(value):
        if isinstance(item, ToolInvocation):
            invocations.append(item)
        elif isinstance(item, dict):
            invocations.append(ToolInvocation.from_dict(item))
        else:
            raise TypeError(
                "tool_calls items must be ToolInvocation or dict, "
                f"got {type(item).__name__} at index {index}"
            )
    return invocations


def conversation_turns_from_payload(value: Any) -> list[ConversationTurn] | None:
    """Rebuild a turn list from serialized dicts.

    None stays None. An empty list stays an empty list.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(
            "turns must be a list of ConversationTurn or dict, "
            f"got {type(value).__name__}"
        )
    turns: list[ConversationTurn] = []
    for index, item in enumerate(value):
        if isinstance(item, ConversationTurn):
            turns.append(item)
        elif isinstance(item, dict):
            turns.append(ConversationTurn.from_dict(item))
        else:
            raise TypeError(
                "turns items must be ConversationTurn or dict, "
                f"got {type(item).__name__} at index {index}"
            )
    return turns


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any] | None = None
    result: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "ToolInvocation.name must be a str, "
                f"got {type(self.name).__name__}"
            )
        if self.arguments is not None and not isinstance(self.arguments, dict):
            raise TypeError(
                "ToolInvocation.arguments must be a dict or None, "
                f"got {type(self.arguments).__name__}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolInvocation:
        return cls(
            name=data["name"],
            arguments=data.get("arguments"),
            result=data.get("result"),
        )


@dataclass
class ConversationTurn:
    role: ConversationRole
    content: str
    retrieval: list[Any] | None = None
    tool_calls: list[ToolInvocation] | None = None

    def __post_init__(self) -> None:
        if self.role not in _CONVERSATION_ROLES:
            raise ValueError(
                "ConversationTurn.role must be 'user' or 'assistant', "
                f"got {self.role!r}"
            )
        if not isinstance(self.content, str):
            raise TypeError(
                "ConversationTurn.content must be a str, "
                f"got {type(self.content).__name__}"
            )
        if self.retrieval is not None and not isinstance(self.retrieval, list):
            raise TypeError(
                "ConversationTurn.retrieval must be a list or None, "
                f"got {type(self.retrieval).__name__}"
            )
        if self.tool_calls is not None:
            if not isinstance(self.tool_calls, list):
                raise TypeError(
                    "ConversationTurn.tool_calls must be a list or None, "
                    f"got {type(self.tool_calls).__name__}"
                )
            for index, call in enumerate(self.tool_calls):
                if not isinstance(call, ToolInvocation):
                    raise TypeError(
                        "ConversationTurn.tool_calls items must be ToolInvocation, "
                        f"got {type(call).__name__} at index {index}"
                    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(
            role=data["role"],
            content=data["content"],
            retrieval=data.get("retrieval"),
            tool_calls=tool_invocations_from_payload(data.get("tool_calls")),
        )
