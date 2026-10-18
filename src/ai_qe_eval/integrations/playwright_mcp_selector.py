"""Provider-neutral tool-selector boundary for the Playwright MCP executor.

Selectors choose the next tool. They do not execute tools, build traces, or
evaluate metrics. This module does not import MCP, DeepEval, or Runner types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolSelector(Protocol):
    def select_next(
        self,
        goal: str,
        observed: Sequence[Any],
        last_plain_result: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Return the next (tool_name, arguments), or None to stop."""


class SequenceToolSelector:
    """Choose tools from a predefined sequence. Arguments are returned unchanged."""

    def __init__(
        self,
        steps: Sequence[tuple[str, dict[str, Any] | None]],
    ) -> None:
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise TypeError(
                "steps must be a sequence of (name, arguments) pairs, "
                f"got {type(steps).__name__}"
            )
        copied: list[tuple[str, dict[str, Any] | None]] = []
        for index, step in enumerate(steps):
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError(
                    "each step must be (name, arguments), "
                    f"got {type(step).__name__} at index {index}"
                )
            name, arguments = step
            if not isinstance(name, str) or not name:
                raise TypeError(
                    f"step name must be a non-empty str at index {index}"
                )
            if arguments is not None and not isinstance(arguments, dict):
                raise TypeError(
                    "step arguments must be a dict or None, "
                    f"got {type(arguments).__name__} at index {index}"
                )
            copied.append((name, arguments))
        self._steps = copied
        self._index = 0

    @property
    def index(self) -> int:
        return self._index

    def select_next(
        self,
        goal: str,
        observed: Sequence[Any],
        last_plain_result: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        if self._index >= len(self._steps):
            return None
        step = self._steps[self._index]
        self._index += 1
        return step
