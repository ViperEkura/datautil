from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    thinking: str | None = None


@dataclass(frozen=True)
class Conversation:
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    conversation: Conversation
    source_location: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_format: str | None = None


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    location: str
    record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "location": self.location, "record_id": self.record_id}


class DataError(Exception):
    def __init__(self, code: str, message: str, location: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.location = location
