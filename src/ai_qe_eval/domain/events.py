"""Minimal event representation (P2-02).

Events are ordered execution facts, not evaluator results.
Typed event classes (LLMEvent, ToolCall, …) are deferred.

Each event is a dict with a `type` discriminator. Additional keys are
application-specific and stored as provided; they are not interpreted here.
"""

from __future__ import annotations

from typing import Any

EVENT_TYPE_KEY = "type"

TraceEvent = dict[str, Any]


def make_trace_event(event_type: str, /, **payload: Any) -> TraceEvent:
    """Build a generic trace event. Payload fields are copied without transformation."""
    event: TraceEvent = dict(payload)
    event[EVENT_TYPE_KEY] = event_type
    return event
